"""R2: a quiz or objection card for sales trainees, one structured call run
alongside R1 (only for intent = sales_training). Validated in code; any
failure or timeout returns None and the card is skipped. Never raises
(except CancelledError).
"""

import asyncio
import logging
from typing import List, Optional

from agents.responder.context import ResponderContext
from agents.responder.prompts import sales_card_messages
from config import settings
from llm.client import RESPONDER_FALLBACK_ORDER, fallback_client, turn_routes

logger = logging.getLogger("responder")

SALES_CARD_SCHEMA = {
    "title": "majestic_sales_card",
    "type": "object", "additionalProperties": False, "required": ["card_type", "quiz", "objection"],
    "properties": {
        "card_type": {"type": "string", "enum": ["quiz", "objection"]},
        "quiz": {"type": "array", "maxItems": 3, "items": {
            "type": "object", "additionalProperties": False, "required": ["q", "options", "answer_index", "explain"],
            "properties": {
                "q": {"type": "string"},
                "options": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "string"}},
                "answer_index": {"type": "integer", "minimum": 0, "maximum": 3},
                "explain": {"type": "string"}}}},
        "objection": {"type": ["object", "null"], "additionalProperties": False,
                      "required": ["objection", "talking_points", "suggested_reply"],
                      "properties": {
                          "objection": {"type": "string"},
                          "talking_points": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
                          "suggested_reply": {"type": "string"}}},
    },
}


def _text(v) -> str:
    return v.strip() if isinstance(v, str) else ""


def validate_card(data: dict) -> Optional[dict]:
    """The widget card, or None when the output is unusable. Bad quiz items are
    dropped one by one; a quiz needs at least one good item."""
    if not isinstance(data, dict):
        return None
    card_type = data.get("card_type")
    if card_type == "quiz":
        items = []
        for q in (data.get("quiz") or [])[:3]:
            if not isinstance(q, dict):
                continue
            options = [_text(o) for o in q.get("options") or []]
            idx = q.get("answer_index")
            if isinstance(idx, float) and idx.is_integer():
                idx = int(idx)
            if (not _text(q.get("q")) or len(options) != 4 or not all(options)
                    or not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx <= 3):
                continue
            items.append({"q": _text(q["q"]), "options": options, "answer_index": idx,
                          "explain": _text(q.get("explain"))})
        return {"type": "quiz", "items": items} if items else None
    if card_type == "objection":
        o = data.get("objection")
        if not isinstance(o, dict):
            return None
        points = [_text(t) for t in (o.get("talking_points") or [])[:3] if _text(t)]
        if not _text(o.get("objection")) or not points or not _text(o.get("suggested_reply")):
            return None
        return {"type": "objection", "objection": _text(o["objection"]), "talking_points": points,
                "suggested_reply": _text(o["suggested_reply"])}
    return None


def _route_kwargs(route: str) -> dict:
    from agents.responder.answer import route_kwargs
    return route_kwargs(route, settings.RESPONDER_SALES_CARD_MAX_TOKENS, settings.RESPONDER_SALES_CARD_TEMPERATURE)


async def sales_card(ctx: ResponderContext, products: List[dict], reply_lang: str,
                     deadline_s: Optional[float] = None) -> Optional[dict]:
    if not products:
        return None
    routes = turn_routes(RESPONDER_FALLBACK_ORDER)
    deadline = deadline_s if deadline_s is not None else settings.RESPONDER_SALES_CARD_DEADLINE_S
    try:
        res = await fallback_client.aconstrained_invoke(
            sales_card_messages(ctx, products, reply_lang), routes, SALES_CARD_SCHEMA,
            deadline_s=deadline, model_kwargs=[_route_kwargs(r) for r in routes],
        )
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 -- AllRoutesFailed, timeouts: skip the card
        logger.warning("[responder] sales card skipped: %s", f"{type(e).__name__}: {e}"[:300])
        return None
    card = validate_card(res.data)
    if card is None:
        logger.warning("[responder] sales card output invalid, skipped: %s", str(res.data)[:300])
    return card
