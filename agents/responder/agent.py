"""Responder agent: the written answer, when the pipeline needs one
(ARCHITECTURE_NOTES.md section 11).

    async for ev in respond_stream(ctx):      # {"event": ..., "data": {...}}
        ...
    text = await respond(ctx)                  # the same, collected into one string

Events, in order (the /chat SSE contract):
1. status  {"stage": "writing"}
2. card    deterministic cards (how_to_use / safety / compare), built from catalog fields
3. message {"delta": "..."}   R1's text as it streams
   message {"text": "...", "replace": true}   when a guardrail or a fallback changes the text
4. card    the R2 quiz / objection card (sales_training only), when ready in time
5. message {"delta": "<disclaimer>", "disclaimer": true}   when the disclaimer rule applies
6. done    {"latency_ms", "partial", "source", "route", "first_token_ms", "notes"}

The chat never goes silent: if R1 fails on every route, a template answer
built from the data is sent; if the stream breaks mid-answer, the text so
far is kept and done carries partial=true.
"""

import asyncio
import contextlib
import json
import logging
import re
import time
from typing import AsyncIterator, Dict, List, Optional

from agents.responder.answer import complete_answer, stream_answer
from agents.responder.cards import build_cards, disclaimer_for
from agents.responder.context import (
    LANGUAGE_NAMES, ResponderContext, product_lines, reply_language, select_products,
)
from agents.responder.fallback_templates import template_answer
from agents.responder.guardrails import (
    cut_to_limit, language_ok, max_words, ungrounded_amounts, unknown_ingredients, word_count,
)
from agents.responder.prompts import answer_messages, load_store_facts
from agents.responder.sales_card import sales_card
from config import settings
from llm.fallback import StreamBroken, StreamInfo

logger = logging.getLogger("responder")

SOURCE_LLM = "llm"
SOURCE_TEMPLATE = "template"          # every R1 route failed, or strict grounding rejected the answer


def _ev(event: str, data: dict) -> dict:
    return {"event": event, "data": data}


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 1)


class PlainTextFilter:
    """The widget shows plain text, so bold markers (** and __) and emoji are
    removed from the stream as it goes. A chunk ending in * or _ is held back
    until the next chunk, since the marker may be split across the two."""

    def __init__(self) -> None:
        self.carry = ""

    @staticmethod
    def _clean(text: str) -> str:
        return _EMOJI_RE.sub("", text.replace("**", "").replace("__", ""))

    def feed(self, chunk: str) -> str:
        text = self.carry + chunk
        cut = len(text.rstrip("*_"))
        self.carry = text[cut:]
        return self._clean(text[:cut])

    def flush(self) -> str:
        out, self.carry = self._clean(self.carry), ""
        return out


_HEADING_RE = re.compile(r"(?m)^[ \t]*#{1,6}[ \t]+")
# Pictographs, symbols and the variation selector (the same ranges tests/test_web.py bans in widget copy).
_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]")


def fold_message(text: str, data: dict) -> str:
    """Apply one message event to the reply text so far (disclaimer chunks are not part of it)."""
    if data.get("disclaimer"):
        return text
    if data.get("replace"):
        return data.get("text", "")
    if "delta" in data:
        return text + data["delta"]
    return data.get("text", text)


def _check(ctx: ResponderContext, products: List[dict], text: str, notes: List[str]) -> bool:
    """Price and ingredient checks. False when strict grounding rejects the text."""
    allowed = product_lines(products, ctx) + [
        json.dumps(load_store_facts(), ensure_ascii=False), ctx.query_original, ctx.query_en,
    ]
    bad_amounts = ungrounded_amounts(text, allowed)
    if bad_amounts:
        logger.warning("[responder] ungrounded amounts %s in answer: %r", bad_amounts, text[:300])
        notes.append(f"ungrounded_amounts={bad_amounts}")
    unknown = unknown_ingredients(text, products, [ctx.query_original, ctx.query_en])
    if unknown:
        logger.warning("[responder] ingredients not in the products sent: %s", unknown)
        notes.append(f"unknown_ingredients={unknown}")
    return not (bad_amounts and settings.RESPONDER_STRICT_GROUNDING)


async def respond_stream(ctx: ResponderContext) -> AsyncIterator[dict]:
    start = time.perf_counter()
    reply_lang = reply_language(ctx.language, ctx.query_original)
    products = select_products(ctx)
    limit = max_words(int(settings.RESPONDER_WORD_LIMITS.get(ctx.audience, 80)), settings.RESPONDER_LENGTH_FACTOR)
    notes: List[str] = []
    yield _ev("status", {"stage": "writing"})

    for card in build_cards(ctx, products, reply_lang):
        yield _ev("card", card)

    r2: Optional[asyncio.Task] = None
    if ctx.intent == "sales_training" and products:
        r2 = asyncio.create_task(sales_card(ctx, products, reply_lang))

    try:
        messages = answer_messages(ctx, products, reply_lang)
        info = StreamInfo()
        streamed = ""          # what the client has been sent
        text = ""
        partial = False
        source = SOURCE_LLM
        try:
            markdown = PlainTextFilter()
            async with contextlib.aclosing(stream_answer(messages, ctx.audience, info)) as stream:
                async for chunk in stream:
                    piece = markdown.feed(chunk)
                    if not piece:
                        continue
                    text += piece
                    if word_count(text) > limit:
                        notes.append("cut_to_length")
                        break
                    streamed += piece
                    yield _ev("message", {"delta": piece, "path": "responder", "language": reply_lang})
                else:
                    tail = markdown.flush()
                    if tail:
                        text += tail
                        streamed += tail
                        yield _ev("message", {"delta": tail, "path": "responder", "language": reply_lang})
        except StreamBroken as e:
            partial = True
            notes.append(f"stream_broken: {e.error}"[:200])
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 -- AllRoutesFailed or a bug: template answer
            logger.warning("[responder] R1 failed, template answer: %s", f"{type(e).__name__}: {e}"[:300])
            notes.append(f"r1_failed: {type(e).__name__}")
            source = SOURCE_TEMPLATE

        final = text
        if source == SOURCE_TEMPLATE or not final.strip():
            source = SOURCE_TEMPLATE
            final = template_answer(ctx, products, reply_lang)
        elif not partial:
            final = cut_to_limit(_HEADING_RE.sub("", final), limit).strip()
            if not language_ok(final, reply_lang) and settings.RESPONDER_LANGUAGE_RETRY:
                notes.append("language_retry")
                retry = await _language_retry(ctx, products, reply_lang, limit)
                retry = retry and _HEADING_RE.sub("", PlainTextFilter._clean(retry)).strip()
                if retry and language_ok(retry, reply_lang):
                    final = retry
                else:
                    logger.warning("[responder] reply not in %s after one retry: %r", reply_lang, final[:200])
            if not _check(ctx, products, final, notes):
                source = SOURCE_TEMPLATE
                final = template_answer(ctx, products, reply_lang)

        if final != streamed:
            yield _ev("message", {"text": final, "replace": True, "path": "responder", "language": reply_lang})

        if r2 is not None:
            card = None
            try:
                card = await asyncio.wait_for(r2, timeout=settings.RESPONDER_SALES_CARD_GRACE_S)
            except asyncio.TimeoutError:
                notes.append("sales_card_timeout")
            except Exception:  # noqa: BLE001 -- sales_card never raises; contain a bug
                logger.exception("[responder] sales card task failed")
            if card is not None:
                yield _ev("card", card)
            elif "sales_card_timeout" not in notes:
                notes.append("sales_card_skipped")

        disclaimer = disclaimer_for(ctx, products, reply_lang)
        if disclaimer:
            yield _ev("message", {"delta": disclaimer, "disclaimer": True, "path": "responder",
                                  "language": reply_lang})

        yield _ev("done", {"latency_ms": _ms(start), "partial": partial, "source": source,
                           "route": info.route, "first_token_ms": info.first_token_ms,
                           "reply_language": reply_lang, "notes": notes})
    finally:
        if r2 is not None and not r2.done():
            r2.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await r2


async def _language_retry(ctx: ResponderContext, products: List[dict], reply_lang: str, limit: int) -> Optional[str]:
    messages = answer_messages(ctx, products, reply_lang,
                               extra_instruction=f"Write the whole reply in {LANGUAGE_NAMES[reply_lang]} only.")
    try:
        return cut_to_limit(await complete_answer(messages, ctx.audience), limit).strip()
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        logger.warning("[responder] language retry failed: %s", f"{type(e).__name__}: {e}"[:200])
        return None


def warm_up() -> None:
    """Load the prompts, store facts and ingredient vocabulary, and build the
    cached model instances (call inside the serving event loop)."""
    from agents.responder.answer import max_tokens_for, route_kwargs
    from agents.responder.guardrails import ingredient_vocabulary
    from agents.responder.prompts import system_prompt
    from llm.client import RESPONDER_FALLBACK_ORDER, fallback_client
    from llm.llm_models import client_llm

    for lang in LANGUAGE_NAMES:
        system_prompt(lang)
    load_store_facts()
    ingredient_vocabulary()
    for persona in settings.RESPONDER_MAX_TOKENS:
        for route in RESPONDER_FALLBACK_ORDER:
            router, model = fallback_client._resolve(route)
            client_llm.get_cached_model(router, model, **route_kwargs(route, max_tokens_for(persona),
                                                                       settings.RESPONDER_TEMPERATURE))


async def respond(ctx: ResponderContext) -> str:
    """The reply as one string, disclaimer appended (for non-streaming callers)."""
    text, disclaimer = "", None
    async for ev in respond_stream(ctx):
        if ev["event"] == "message":
            if ev["data"].get("disclaimer"):
                disclaimer = ev["data"]["delta"]
            text = fold_message(text, ev["data"])
    return f"{text}\n\n{disclaimer}" if disclaimer else text


def collect(events: List[Dict]) -> Dict:
    """Reply text, disclaimer, cards and the done data from a list of responder events."""
    text, disclaimer, cards, done = "", None, [], {}
    for ev in events:
        data = ev["data"]
        if ev["event"] == "message":
            if data.get("disclaimer"):
                disclaimer = data.get("delta")
            text = fold_message(text, data)
        elif ev["event"] == "card":
            cards.append(data)
        elif ev["event"] == "done":
            done = data
    return {"text": text, "disclaimer": disclaimer, "cards": cards, "done": done}
