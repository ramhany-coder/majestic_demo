"""What the Jamila widget receives: a chat turn as events, and the details
behind an answer card. The /chat SSE endpoint (api/chat.py) and the Streamlit
app (streamlit_app.py) both send exactly these, so web/app.js handles one
event format for both transports.

    turn = await handle_message("عايزة سيروم للشعر", session_id="abc")
    turn_events(turn)
    # [{"event": "message",  "data": {"text": ..., "path": "products_only", "language": "ar", "health": False}},
    #  {"event": "products", "data": {"items": [card, ...], "total": 7, "relaxed": []}},
    #  {"event": "done",     "data": {"persona": "customer", "intent": "find_products", ...}}]

A responder turn streams (api/chat.py sends each event as it happens):

    products  (when retrieval found any), card* (how_to_use / safety / compare),
    message {"delta"}* (the answer as it is written), message {"text", "replace": true}
    (a guardrail or fallback changed it), card (quiz / objection), message {"delta",
    "disclaimer": true}, done {..., "latency_ms", "partial"}

turn_events() is the same list for transports that deliver a turn at once
(Streamlit): the message chunks are joined into one message event, with the
disclaimer text in `disclaimer` and `health: true`.

A card carries CARD_FIELDS only. The catalogue's description, key_ingredients,
how_to_use and warnings go to the answer card (product_details), never onto a
card; tags, collections and sku go nowhere.
"""

from typing import Dict, Iterable, List, Optional

from agents.orchestrator.orchestrator import TurnResult
from agents.retrieval.index_builder import get_index

# Same list as Jamila.CARD_FIELDS in web/components/bundle.js.
CARD_FIELDS = (
    "handle", "variant_id", "url", "url_ar", "image", "brand", "name", "name_ar", "size",
    "price", "compare_at_price", "discount_percent", "promotion", "available", "why", "conflict",
)
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
    """The responder's code rule decided the disclaimer (agents/responder/cards.py)."""
    return turn.path == "responder" and bool(turn.disclaimer)


def wire_event(name: str, data: dict) -> dict:
    """An orchestrator event -> what the widget receives: product dumps become cards."""
    if name == "products":
        return {"event": "products", "data": {
            "items": cards(data.get("items") or [], data.get("applied") or {}),
            "total": data.get("total", len(data.get("items") or [])),
            "relaxed": data.get("relaxed", []),
        }}
    return {"event": name, "data": data}


def _joined(turn: TurnResult, events: List[dict]) -> List[dict]:
    """Responder message chunks joined into one message event, where the first chunk was."""
    out: List[dict] = []
    placed = False
    for ev in events:
        if ev["event"] != "message" or turn.path != "responder":
            out.append(ev)
        elif not placed:
            placed = True
            out.append({"event": "message", "data": {
                "text": turn.reply, "path": turn.path, "language": ev["data"].get("language", turn.prequal.language),
                "health": is_health_answer(turn), "disclaimer": turn.disclaimer,
            }})
    return out


def done_event(turn: TurnResult) -> dict:
    pq = turn.prequal
    retrieval = turn.retrieval
    return {"event": "done", "data": {
        "persona": pq.persona, "intent": pq.intent, "route": pq.route, "path": turn.path,
        "timings_ms": turn.timings_ms,
        "latency_ms": turn.timings_ms.get("total"),
        "partial": bool(turn.responder.get("partial")),
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
            "responder": turn.responder,
        },
    }}


def turn_events(turn: TurnResult) -> List[dict]:
    """The whole turn as widget events, ending with done (see the module docstring)."""
    if turn.events:
        events = [wire_event(e["event"], e["data"]) for e in _joined(turn, turn.events)]
        return events + [done_event(turn)]
    # A TurnResult built without events (tests, old callers): message, then products.
    events = [{"event": "message", "data": {
        "text": turn.reply, "path": turn.path, "language": turn.prequal.language, "health": is_health_answer(turn),
        "disclaimer": turn.disclaimer,
    }}]
    if turn.products:
        retrieval = turn.retrieval or {}
        events.append(wire_event("products", {
            "items": turn.products, "applied": retrieval.get("applied_filters") or {},
            "total": retrieval.get("total_candidates", len(turn.products)),
            "relaxed": retrieval.get("relaxed_keys", []),
        }))
    for card in turn.cards:
        events.append({"event": "card", "data": card})
    return events + [done_event(turn)]


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
