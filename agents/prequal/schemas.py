"""Structured-output schemas for the two pre-qualification calls, plus the
code-side validation that runs on every response whether or not the provider
enforced the schema. A response that fails validation counts as a failed
call, so the caller falls back to the safe default."""

from typing import List, Optional, Tuple

from config import settings

LANGUAGES = ("ar", "arabizi", "en", "mixed")
ROUTES = ("products_only", "needs_response")
INTENTS = (
    "find_products", "refine_products", "product_info", "how_to_use", "compare", "safety",
    "price_offer", "sales_training", "greeting", "out_of_scope", "other",
)
PERSONAS = ("customer", "sales_trainee", "doctor", "unknown")
PRODUCT_LIST_INTENTS = ("find_products", "refine_products")
QUERY_EN_MAX_CHARS = 400

REWRITER_SCHEMA = {
    "title": "majestic_prequal_rewrite",
    "type": "object",
    "additionalProperties": False,
    "required": ["query_en", "language", "is_follow_up"],
    "properties": {
        "query_en": {"type": "string", "maxLength": QUERY_EN_MAX_CHARS},
        "language": {"type": "string", "enum": list(LANGUAGES)},
        "is_follow_up": {"type": "boolean"},
    },
}

ROUTER_SCHEMA = {
    "title": "majestic_prequal_route",
    "type": "object",
    "additionalProperties": False,
    "required": ["route", "needs_retrieval", "intent", "persona"],
    "properties": {
        "route": {"type": "string", "enum": list(ROUTES)},
        "needs_retrieval": {"type": "boolean"},
        "intent": {"type": "string", "enum": list(INTENTS)},
        "persona": {"type": "string", "enum": list(PERSONAS)},
    },
}

ROUTER_DEFAULT = {"route": "needs_response", "needs_retrieval": True, "intent": "other", "persona": "unknown"}


def _enum(data: dict, key: str, allowed: tuple) -> str:
    value = str(data.get(key, "")).strip().lower()
    if value not in allowed:
        raise ValueError(f"{key}={data.get(key)!r} is not one of {allowed}")
    return value


def _bool(data: dict, key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key}={value!r} is not a boolean")
    return value


def validate_rewrite(data: dict) -> Tuple[dict, List[str]]:
    """Raises ValueError on an unusable response (empty query, bad enum)."""
    notes: List[str] = []
    query_en = " ".join(str(data.get("query_en") or "").split())
    if not query_en:
        raise ValueError("empty query_en")
    if len(query_en) > QUERY_EN_MAX_CHARS:
        notes.append(f"query_en cut from {len(query_en)} chars")
        query_en = query_en[:QUERY_EN_MAX_CHARS].rstrip()
    out = {
        "query_en": query_en,
        "language": _enum(data, "language", LANGUAGES),
        "is_follow_up": _bool(data, "is_follow_up"),
    }
    return out, notes


def validate_route(data: dict, route_from_intent: Optional[bool] = None) -> Tuple[dict, List[str]]:
    """Raises ValueError on a bad enum. Contradictions are corrected (and noted):
    - find_products / refine_products with route=needs_response -> products_only
      (PREQUAL_ROUTE_FROM_INTENT). In the live eval the model's intent was
      right more often than its route, and every such contradiction was a
      find request misrouted to needs_response.
    - products_only always needs retrieval."""
    if route_from_intent is None:
        route_from_intent = settings.PREQUAL_ROUTE_FROM_INTENT
    notes: List[str] = []
    out = {
        "route": _enum(data, "route", ROUTES),
        "needs_retrieval": _bool(data, "needs_retrieval"),
        "intent": _enum(data, "intent", INTENTS),
        "persona": _enum(data, "persona", PERSONAS),
    }
    if route_from_intent and out["intent"] in PRODUCT_LIST_INTENTS and out["route"] == "needs_response":
        out["route"] = "products_only"
        notes.append(f"intent={out['intent']} with route=needs_response -> products_only")
    if out["route"] == "products_only" and not out["needs_retrieval"]:
        out["needs_retrieval"] = True
        notes.append("products_only with needs_retrieval=false -> set true")
    return out, notes
