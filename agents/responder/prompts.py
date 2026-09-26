"""Builds the responder's prompts: the static system message
(prompts/responder/answer.md), then only the blocks that apply, then the
dynamic part with the message last.

    system, human = build_answer_prompt(ctx, products, reply_lang)
    messages = [SystemMessage(system), HumanMessage(human)]
"""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from agents.prequal.session_context import format_history
from agents.responder.context import (
    LANGUAGE_NAMES, NO_PRODUCT_INTENTS, ResponderContext, excluded_hits, product_lines,
)
from config import settings
from llm.prompt_loader import load_prompt

logger = logging.getLogger("responder")

ROOT = Path(__file__).resolve().parents[2]
NO_DATA = "(none)"

PERSONA_BLOCKS = {
    "customer": ("Audience: a customer. Up to ~80 words. Direct answer first, then one recommendation with the "
                 "reason (concern / skin type / hero ingredient), mention an active offer if any, end with one soft "
                 "next step. No jargon."),
    "sales_trainee": ("Audience: a Majestic sales trainee. Coach them, up to ~150 words: 3 bullet selling points "
                      "(benefit → proof from ingredients → who it suits), then one suggested line to say to the "
                      "customer in Egyptian Arabic. Objections: Acknowledge → Reframe value (active %, size, offer) "
                      "→ Close. Never name or attack competitors."),
    "doctor": ("Audience: a doctor or pharmacist. Concise and technical, up to ~150 words: hero ingredient and "
               "concentration, relevant ingredients, form, usage, warnings, suitability flags. No marketing words. "
               "Say clearly when data (full INCI, pregnancy safety, studies) is not provided."),
}

INTENT_BLOCKS = {
    "how_to_use": "Explain when and how to use it in 1–3 sentences; the steps card shows the details.",
    "safety": ("Answer yes/no/not stated from the warnings and suitability data; if not stated, say so and advise "
               "a doctor."),
    "compare": "Give the key difference and who each product suits in 2–4 sentences; the table shows the details.",
    "product_info": "Explain what it is, what it does and who it suits.",
    "price_offer": "State the price, any discount or buy-X-get-Y offer, and stock status.",
    "sales_training": "Follow the sales_trainee persona; a quiz/objection card is generated separately.",
    "greeting": "Greet back in one line and offer help with skin, hair or a product.",
    "out_of_scope": ("Orders, delivery, returns, complaints or unrelated topics: be polite, do not invent policies, "
                     "point to Majestic customer service; use STORE FACTS if they answer it."),
    # Not in the plan: a find request mixed with a question reaches the responder.
    "find_products": ("Recommend the best one or two options shown and say why in 1–2 sentences; the cards show "
                      "the full list."),
}
INTENT_BLOCKS["refine_products"] = INTENT_BLOCKS["find_products"]

KEY_LABELS = {
    "product_type": "type", "product_form": "form", "suitable_for": "suitable for", "concerns": "concern",
    "product_group": "group", "category": "category", "hero_ingredient": "hero ingredient",
    "ingredients.include": "containing", "brand": "brand",
}


def intent_block_key(ctx: ResponderContext) -> str:
    """The intent block to use. The router's catch-all "other" covers orders,
    delivery and complaints when no retrieval ran, else it is a product question."""
    if ctx.intent in INTENT_BLOCKS:
        return ctx.intent
    return "out_of_scope" if ctx.retrieval is None else "product_info"


def readable_keys(ctx: ResponderContext) -> str:
    """"form = roll-on; type = sunscreen": the relaxed keys with their requested
    values; with no products, the keys that blocked the match (else every key)."""
    r = ctx.retrieval
    if r is None:
        return ""
    if r.relaxed_keys:
        values = r.meta.get("relaxed_values") or {}
        keys = r.relaxed_keys
    else:
        values = r.applied_filters
        keys = r.meta.get("near_miss_keys") or [k for k in r.applied_filters if k != "ingredients.exclude"]
    parts = [f"{KEY_LABELS.get(k, k)} = {', '.join(values.get(k) or []) or '?'}" for k in keys]
    return "; ".join(parts) or ctx.query_en


def situation_blocks(ctx: ResponderContext, products: List[dict]) -> List[str]:
    blocks = []
    r = ctx.retrieval
    if ctx.mode == "no_match" and r is not None:
        if products:
            blocks.append(f"No exact match for: {readable_keys(ctx)}. Say so in one sentence, then present the "
                          "closest options shown.")
        else:
            blocks.append(f"No product matches: {readable_keys(ctx)}. Say so in one sentence and suggest a broader "
                          "request, a pharmacist, or Majestic customer service. Do not name products.")
    if r is not None and r.unresolved_names:
        blocks.append(f"Could not find: {', '.join(r.unresolved_names)}. Say so and suggest the closest product "
                      "name if one is shown.")
    for p in products:
        hits = excluded_hits(p, r)
        if (p.get("match") or {}).get("conflict") == "contains_excluded" and hits:
            blocks.append(f"{p['name']} contains {', '.join(hits)}, which the user wants to avoid. Warn clearly and "
                          "suggest an alternative if shown.")
    for p in products:
        if not p.get("available", True):
            blocks.append(f"{p['name']} is currently out of stock. Say so and suggest an in-stock alternative if "
                          "shown.")
    return blocks


@lru_cache(maxsize=1)
def load_store_facts(path: Optional[str] = None) -> Dict:
    """Confirmed store facts only. An unfilled placeholder ("{{...}}") is dropped,
    so the answer says "Majestic customer service" without a number."""
    p = Path(path or settings.RESPONDER_STORE_FACTS_PATH)
    if not p.is_absolute():
        p = ROOT / p
    try:
        facts = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.error("[responder] store facts not loaded from %s: %s", p, e)
        return {}
    return {k: v for k, v in facts.items() if not (isinstance(v, str) and "{{" in v)}


# Store facts go only where a store question is likely. With them in every
# prompt, the model added shipping and payment lines to product answers.
STORE_FACT_INTENTS = frozenset({"out_of_scope", "other", "price_offer", "greeting"})


def uses_store_facts(ctx: ResponderContext) -> bool:
    return ctx.intent in STORE_FACT_INTENTS or ctx.retrieval is None


def system_prompt(reply_lang: str) -> str:
    return load_prompt("responder/answer.md").replace("{{reply_language}}", LANGUAGE_NAMES[reply_lang])


def build_answer_prompt(ctx: ResponderContext, products: List[dict], reply_lang: str) -> Tuple[str, str]:
    persona = PERSONA_BLOCKS[ctx.audience]
    blocks = [persona, INTENT_BLOCKS[intent_block_key(ctx)], *situation_blocks(ctx, products)]
    lines = product_lines(products, ctx) if ctx.intent not in NO_PRODUCT_INTENTS else []
    human = "\n".join([
        *blocks,
        "",
        "STORE FACTS: " + (json.dumps(load_store_facts(), ensure_ascii=False, separators=(",", ":"))
                           if uses_store_facts(ctx) else NO_DATA),
        "HISTORY:",
        format_history(ctx.history),
        "PRODUCT DATA:",
        "\n".join(lines) if lines else NO_DATA,
        f"USER MESSAGE: {ctx.query_original}",
        f"MEANING (English): {ctx.query_en}",
    ])
    return system_prompt(reply_lang), human


def answer_messages(ctx: ResponderContext, products: List[dict], reply_lang: str,
                    extra_instruction: Optional[str] = None) -> list:
    system, human = build_answer_prompt(ctx, products, reply_lang)
    if extra_instruction:
        human = f"{human}\n{extra_instruction}"
    return [SystemMessage(content=system), HumanMessage(content=human)]


def sales_card_messages(ctx: ResponderContext, products: List[dict], reply_lang: str) -> list:
    """R2's prompt is short and fully dynamic: one human message."""
    lines = product_lines(products, ctx)
    text = (load_prompt("responder/sales_card.md")
            .replace("{{reply_language}}", LANGUAGE_NAMES[reply_lang])
            .replace("{{product_json_lines}}", "\n".join(lines) if lines else NO_DATA)
            .replace("{{query_original}}", ctx.query_original))
    return [HumanMessage(content=text)]
