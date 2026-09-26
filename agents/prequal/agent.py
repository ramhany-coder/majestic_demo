"""Pre-qualification stage: rewriter and router in parallel -> PrequalResult.

    result = await prequalify("في واحد مفيهوش سيليكون؟", ctx, session_id="abc")
    result.query_en   # "I need a hair serum without silicone."
    result.route      # "products_only"

Both calls always run (asyncio.gather) and each falls back on its own, so
the stage returns a usable result even when both models are down. The
optional `on_rewrite` hook fires as soon as the rewriter returns, before the
router has finished; the orchestrator uses it to start the extractor
speculatively (PREQUAL_SPECULATIVE_EXTRACTOR).

Greetings, thanks and goodbyes end the graph here (PREQUAL_SMALL_TALK_END):
result.end is True and result.reply holds the answer. A message made only of
such words makes no LLM call; a router intent=greeting ends it after the calls.
"""

import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional, Union

from agents.filter_extractor.cache import TTLCache
from agents.prequal.llm_call import STATUS_DEFAULT, STATUS_SKIPPED, CallResult
from agents.prequal.prompts import get_template
from agents.prequal.rewriter import rewrite, rewrite_default
from agents.prequal.router import route
from agents.prequal.schemas import ROUTER_DEFAULT
from agents.prequal.session_context import SessionContext, trim_history
from agents.prequal.small_talk import (is_small_talk, small_talk_language, small_talk_query_en,
                                        small_talk_reply)
from config import settings
from models.prequal import PrequalResult

logger = logging.getLogger("prequal")

RewriteHook = Callable[[dict], Union[None, Awaitable[None]]]

_cache = TTLCache(maxsize=settings.PREQUAL_CACHE_SIZE, ttl_s=settings.PREQUAL_CACHE_TTL_S)


def get_cache() -> TTLCache:
    return _cache


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def _cache_key(session_id: str, message: str, ctx: SessionContext) -> tuple:
    """(session, message, history the prompt sees). The history is part of
    the key so the same words later in the chat ("any cheaper?") are not
    served a stale rewrite. A retry of a turn that already completed finds
    this message and its reply at the end of the history; they are dropped
    so the retry hits the entry the first attempt stored."""
    history = list(ctx.history)
    if len(history) >= 2 and history[-2].get("role") == "user" and _norm(history[-2].get("content")) == _norm(message):
        history = history[:-2]
    fingerprint = tuple((m["role"], m["content"]) for m in trim_history(history))
    return session_id, _norm(message), fingerprint


async def _guard(coro, key: str, default: Callable[[], dict]) -> CallResult:
    """rewrite()/route() never raise; this only contains a bug in them."""
    start = time.perf_counter()
    try:
        return await coro
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("[prequal] %s raised", key)
        return CallResult(key, default(), STATUS_DEFAULT, round((time.perf_counter() - start) * 1000, 1),
                          notes=[repr(e)])


async def _rewrite_then_hook(message: str, ctx: SessionContext, hook: Optional[RewriteHook]) -> CallResult:
    res = await _guard(rewrite(message, ctx.history, ctx.last_products), "rewriter",
                       lambda: rewrite_default(message, ctx.history))
    if hook is not None:
        try:
            out = hook(dict(res.data))
            if asyncio.iscoroutine(out):
                await out
        except Exception:  # noqa: BLE001 -- a broken hook must not break the stage
            logger.exception("[prequal] on_rewrite hook failed")
    return res


def warm_up() -> None:
    """Build both templates and the model instances for every route, so the
    first request doesn't do that synchronous work inside the calls' timeouts.
    Call inside the event loop that will serve requests (the model instances
    bind that loop's shared HTTP client)."""
    from agents.filter_extractor.calls import route_kwargs
    from llm.client import PREQUAL_FALLBACK_ORDER, fallback_client
    from llm.llm_models import client_llm

    for key in ("rewriter", "router"):
        get_template(key)
    for max_tokens in (settings.PREQUAL_REWRITER_MAX_TOKENS, settings.PREQUAL_ROUTER_MAX_TOKENS):
        for r in PREQUAL_FALLBACK_ORDER:
            router, model = fallback_client._resolve(r)
            client_llm.get_cached_model(router, model, **route_kwargs(r, max_tokens))


def resolve_k(k: Optional[int]) -> int:
    """null (not stated, or the router failed) -> RETRIEVAL_K_DEFAULT; capped at RETRIEVAL_K_MAX."""
    return min(k if isinstance(k, int) and k >= 1 else settings.RETRIEVAL_K_DEFAULT, settings.RETRIEVAL_K_MAX)


def _ends_turn(rt: CallResult) -> bool:
    return (settings.PREQUAL_SMALL_TALK_END and rt.data["intent"] == "greeting"
            and not rt.data["needs_retrieval"])


def _build(message: str, rw: CallResult, rt: CallResult, total_ms: float) -> PrequalResult:
    router_k = rt.data.get("k")
    end = _ends_turn(rt)
    return PrequalResult(
        query_original=message,
        query_en=rw.data["query_en"],
        language=rw.data["language"],
        is_follow_up=rw.data["is_follow_up"],
        route=rt.data["route"],
        needs_retrieval=rt.data["needs_retrieval"],
        intent=rt.data["intent"],
        persona=rt.data["persona"],
        k=resolve_k(router_k),
        skip_metadata_filters=rw.data.get("skip_metadata_filters", False),
        end=end,
        reply=small_talk_reply(message, rw.data["language"]) if end else None,
        meta={
            "latency_ms": {"rewriter": rw.latency_ms, "router": rt.latency_ms, "total": total_ms},
            "status": {"rewriter": rw.status, "router": rt.status},
            "k_source": "router" if router_k else "default",
            "routes": {r.key: r.route for r in (rw, rt) if r.route},
            "prompt_tokens": {r.key: r.prompt_tokens for r in (rw, rt) if r.prompt_tokens},
            "notes": [f"{r.key}: {n}" for r in (rw, rt) for n in r.notes],
            "cached": False,
        },
    )


def _small_talk_result(message: str, total_ms: float) -> PrequalResult:
    language = small_talk_language(message)
    return PrequalResult(
        query_original=message, query_en=small_talk_query_en(message), language=language,
        is_follow_up=False, route="needs_response", needs_retrieval=False, intent="greeting", persona="unknown",
        k=resolve_k(None), end=True, reply=small_talk_reply(message, language),
        meta={"latency_ms": {"rewriter": 0.0, "router": 0.0, "total": total_ms},
              "status": {"rewriter": STATUS_SKIPPED, "router": STATUS_SKIPPED}, "k_source": "default",
              "routes": {}, "prompt_tokens": {}, "notes": ["small talk"], "cached": False},
    )


def _empty_message_result(message: str) -> PrequalResult:
    return PrequalResult(
        query_original=message, query_en="", language="en", is_follow_up=False,
        route="needs_response", needs_retrieval=False, intent="other", persona="unknown",
        k=resolve_k(None),
        meta={"latency_ms": {"rewriter": 0.0, "router": 0.0, "total": 0.0},
              "status": {"rewriter": STATUS_SKIPPED, "router": STATUS_SKIPPED}, "k_source": "default",
              "routes": {}, "prompt_tokens": {}, "notes": ["empty message"], "cached": False},
    )


async def prequalify(message: Optional[str], ctx: Optional[SessionContext] = None,
                     session_id: Optional[str] = None, *, on_rewrite: Optional[RewriteHook] = None,
                     use_cache: bool = True) -> PrequalResult:
    """`ctx` carries the chat history and last products (a fresh chat when
    None). With a `session_id`, results are cached per (session, message) so
    a retried request is not billed twice; degraded results are not cached."""
    start = time.perf_counter()
    message = (message or "").strip()[: settings.PREQUAL_MAX_QUERY_CHARS]
    ctx = ctx or SessionContext()
    if not message:
        return _empty_message_result(message)
    if settings.PREQUAL_SMALL_TALK_END and is_small_talk(message):
        res = _small_talk_result(message, round((time.perf_counter() - start) * 1000, 1))
        logger.info("[prequal] small talk, ending turn: %r", message)
        return res

    key = _cache_key(session_id, message, ctx) if session_id and use_cache else None
    if key is not None:
        hit = _cache.get(key)
        if hit is not None:
            res = hit.model_copy(deep=True)
            res.meta["cached"] = True
            res.meta["latency_ms"] = {**res.meta["latency_ms"], "total": round((time.perf_counter() - start) * 1000, 1)}
            return res

    rw, rt = await asyncio.gather(
        _rewrite_then_hook(message, ctx, on_rewrite),
        _guard(route(message, ctx.history, ctx.last_products), "router", lambda: dict(ROUTER_DEFAULT)),
    )
    res = _build(message, rw, rt, round((time.perf_counter() - start) * 1000, 1))

    if rw.degraded and rt.degraded:
        logger.error("[prequal] both calls failed; using defaults. rewriter=%s router=%s", rw.notes, rt.notes)
    else:
        logger.info("[prequal] %.0fms route=%s intent=%s k=%d end=%s status=%s query_en=%r",
                    res.meta["latency_ms"]["total"], res.route, res.intent, res.k, res.end, res.meta["status"],
                    res.query_en)
    if key is not None and not (rw.degraded or rt.degraded):
        _cache.set(key, res.model_copy(deep=True))
    return res
