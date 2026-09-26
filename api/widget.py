"""What the Jamila widget receives: a chat turn as events, and the details
behind an answer card. The /chat SSE endpoint (api/chat.py) and the Streamlit
app (streamlit_app.py) both send exactly these, so web/app.js handles one
event format for both transports.

    turn = await handle_message("عايزة سيروم للشعر", session_id="abc")
    turn_events(turn)
    # [{"event": "message",  "data": {"text": ..., "path": "products_only", "language": "ar", "health": False}},
    #  {"event": "products", "data": {"items": [card, ...], "total": 7, "relaxed": []}},
    #  {"event": "done",     "data": {"persona": "customer", "intent": "find_products", ...}}]

A card carries CARD_FIELDS only. The catalogue's description, key_ingredients,
how_to_use and warnings go to the answer card (product_details), never onto a
card; tags, collections and sku go nowhere.
"""

from typing import Dict, Iterable, List, Optional

from agents.orchestrator.orchestrator import TurnResult
from agents.prequal.llm_call import STATUS_DEFAULT
from agents.retrieval.index_builder import get_index

# Same list as Jamila.CARD_FIELDS in web/components/bundle.js.
CARD_FIELDS = (
    "handle", "variant_id", "url", "url_ar", "image", "brand", "name", "name_ar", "size",
    "price", "compare_at_price", "discount_percent", "promotion", "available", "why", "conflict",
)
# Responder replies to these intents are health answers and carry the
# disclaimer; so does any responder reply when the router fell back to its
# default intent and the question is unknown.
HEALTH_INTENTS = frozenset({"product_info", "how_to_use", "compare", "safety", "sales_training"})
# The "why this product" line names at most this many values per key.
WHY_MAX = 2

INCLUDE_KEY = "ingredients.include"


def _record(handle: str) -> Optional[dict]:
    return get_index().products.get(handle)


def _hero_name(record: dict) -> Optional[str]:
    hero = record.get("hero_ingredient")
    return hero.get("name") if isinstance(hero, dict) else hero


def why(record: dict, applied: Dict[str, List[str]]) -> dict:
    """The requested values this product actually has, per key; else its own
    first concerns, so every card gets a line."""
    out: Dict[str, List[str]] = {}
    for key in ("concerns", "suitable_for"):
        own = set(record.get(key) or [])
        hits = [v for v in applied.get(key, []) if v in own]
        if hits:
            out[key] = hits[:WHY_MAX]
    own_ingredients = set(record.get("ingredients_canonical") or [])
    hero = _hero_name(record)
    contains = [v for v in applied.get(INCLUDE_KEY, []) if v in own_ingredients]
    contains += [v for v in applied.get("hero_ingredient", []) if v == hero and v not in contains]
    if contains:
        out["contains"] = contains[:WHY_MAX]
    if not out:
        if record.get("concerns"):
            out["concerns"] = list(record["concerns"][:WHY_MAX])
        elif record.get("suitable_for"):
            out["suitable_for"] = list(record["suitable_for"][:WHY_MAX])
    return out


def to_card(product: dict, applied: Optional[Dict[str, List[str]]] = None) -> dict:
    """A RetrievedProduct dump -> a widget card: CARD_FIELDS only, with size,
    variant and discount filled in from the catalogue."""
    record = _record(product["handle"]) or {}
    source = {**record, **product}
    card = {key: source.get(key) for key in CARD_FIELDS if key not in ("why", "conflict")}
    card["why"] = why(record, applied or {}) if record else {}
    card["conflict"] = (product.get("match") or {}).get("conflict")
    return card


def cards(products: Iterable[dict], applied: Optional[Dict[str, List[str]]] = None) -> List[dict]:
    return [to_card(p, applied) for p in products]


def is_health_answer(turn: TurnResult) -> bool:
    if turn.path != "responder":
        return False
    router_status = (turn.prequal.meta.get("status") or {}).get("router")
    return turn.prequal.intent in HEALTH_INTENTS or router_status == STATUS_DEFAULT


def turn_events(turn: TurnResult) -> List[dict]:
    pq = turn.prequal
    events = [{"event": "message", "data": {
        "text": turn.reply, "path": turn.path, "language": pq.language, "health": is_health_answer(turn),
    }}]
    if turn.products:
        retrieval = turn.retrieval or {}
        events.append({"event": "products", "data": {
            "items": cards(turn.products, retrieval.get("applied_filters") or {}),
            "total": retrieval.get("total_candidates", len(turn.products)),
            "relaxed": retrieval.get("relaxed_keys", []),
        }})
    retrieval = turn.retrieval
    events.append({"event": "done", "data": {
        "persona": pq.persona, "intent": pq.intent, "route": pq.route, "path": turn.path,
        "timings_ms": turn.timings_ms,
        # What the pipeline did, for the query console's details.
        "pipeline": {
            "query_en": pq.query_en,
            "language": pq.language,
            "needs_retrieval": pq.needs_retrieval,
            "is_follow_up": pq.is_follow_up,
            "model_status": pq.meta.get("status") or {},
            "retrieval_ran": retrieval is not None,
            "applied_filters": (retrieval or {}).get("applied_filters") or {},
            "relaxed_keys": (retrieval or {}).get("relaxed_keys") or [],
            "total_candidates": (retrieval or {}).get("total_candidates"),
        },
    }})
    return events


def error_events() -> List[dict]:
    return [{"event": "error", "data": {"code": "pipeline_failed"}}]


def product_details(handle: str) -> Optional[dict]:
    """Everything an answer card shows. `*_ar` is None where the catalogue has
    no Arabic version; the widget then shows the English field and says so."""
    record = _record(handle)
    if record is None:
        return None
    images = record.get("images") or []
    return {
        "handle": handle,
        "variant_id": record.get("variant_id"),
        "url": record.get("url"),
        "url_ar": record.get("url_ar"),
        "image": images[0].get("src") if images else None,
        "brand": record.get("brand"),
        "name": record.get("name"),
        "name_ar": record.get("name_ar"),
        "size": record.get("size"),
        "price": record.get("price"),
        "compare_at_price": record.get("compare_at_price"),
        "discount_percent": record.get("discount_percent"),
        "promotion": record.get("promotion"),
        "available": record.get("available", True),
        "key_ingredients": [i["name"] for i in record.get("key_ingredients") or [] if i.get("name")],
        "description": record.get("description"),
        "description_ar": record.get("description_ar"),
        "how_to_use": record.get("how_to_use") or [],
        "how_to_use_ar": record.get("how_to_use_ar"),
        "warnings": record.get("warnings") or [],
        "warnings_ar": record.get("warnings_ar"),
    }
