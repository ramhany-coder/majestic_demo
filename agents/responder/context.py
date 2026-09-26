"""What the responder is given, and the product data it sends to the LLM.

    ctx = ResponderContext(query_original=..., query_en=..., language="ar", persona="customer",
                           intent="safety", history=[...], retrieval=result, mode="answer")
    products = select_products(ctx)          # <= RESPONDER_MAX_PRODUCTS catalog records, best first
    lines = product_lines(products, ctx)     # one compact JSON line per product, intent's fields only

Product data comes from the catalog record behind each RetrievedProduct, so
it carries the fields retrieval does not (description, how_to_use, warnings).
`data_issues`, images, raw HTML and URLs never reach the LLM.
"""

import json
import re
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from agents.retrieval.index_builder import get_index
from config import settings
from models.retrieval import RetrievalResult

Mode = Literal["answer", "no_match", "no_products"]
ReplyLanguage = Literal["ar", "arabizi", "en"]

EXCLUDE_KEY = "ingredients.exclude"


class ResponderContext(BaseModel):
    """Built in code by the orchestrator (ARCHITECTURE_NOTES.md section 11)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    query_original: str
    query_en: str
    language: Literal["ar", "arabizi", "en", "mixed"]
    persona: Literal["customer", "sales_trainee", "doctor", "unknown"]
    intent: str                                               # from the router
    history: List[Dict[str, str]] = Field(default_factory=list)   # oldest first; trimmed when formatted
    retrieval: Optional[RetrievalResult] = None               # None: retrieval did not run
    mode: Mode = "answer"

    @property
    def products(self) -> List[dict]:
        """RetrievedProduct dumps, best first (empty without retrieval)."""
        return [p.model_dump() for p in self.retrieval.products] if self.retrieval else []

    @property
    def audience(self) -> str:
        """Persona used for tone, limits and the disclaimer: unknown is a customer."""
        return "customer" if self.persona == "unknown" else self.persona


def unanchored(retrieval: Optional[RetrievalResult]) -> bool:
    """The user named products that were not found, and nothing else narrowed
    the search: retrieval's list is then the whole pool by boosts, unrelated
    to the question, so the responder sends none of it."""
    if retrieval is None or not retrieval.unresolved_names or retrieval.name_hits:
        return False
    return not any(v for k, v in retrieval.applied_filters.items() if k != EXCLUDE_KEY)


def mode_for(route: str, retrieval: Optional[RetrievalResult]) -> Mode:
    """no_products: retrieval did not run (greeting, delivery, off-topic).
    no_match: retrieval ran but found nothing, relaxed a key to find something,
    or found none of the products the user named.
    answer: otherwise."""
    if retrieval is None:
        return "no_products"
    if not retrieval.products or retrieval.relaxed_keys or unanchored(retrieval):
        return "no_match"
    return "answer"


# ---------------------------------------------------------------- reply language

_ARABIC_RE = re.compile(r"[؀-ۿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")
# A Latin word with a digit standing for an Arabic letter (3ayez, 7aga, 2ol), or a common Arabizi word.
_ARABIZI_DIGIT_RE = re.compile(r"\b[a-z]*[a-z][2357][a-z]*\b|\b[2357][a-z]+\b", re.IGNORECASE)
_ARABIZI_WORDS = frozenset(
    "ana enta enty ezay eih eh leh fe fi mesh msh bta3 beta3 3ayez 3ayza 3andy 3ando momken kda keda "
    "tamam shokran ya wala walla ahsan a7san ma3lesh yenfa3 ynfa3 bas kaman".split()
)


def _is_arabizi(text: str) -> bool:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    return bool(_ARABIZI_DIGIT_RE.search(text)) or sum(w in _ARABIZI_WORDS for w in words) >= 2


def reply_language(language: str, query_original: str = "", arabizi_reply: Optional[str] = None) -> ReplyLanguage:
    """ar / arabizi -> Egyptian Arabic in Arabic script (or Arabizi when
    RESPONDER_ARABIZI_REPLY=arabizi); en -> English; mixed -> the script that
    dominates the original message."""
    arabizi_reply = arabizi_reply or settings.RESPONDER_ARABIZI_REPLY
    arabizi_code: ReplyLanguage = "arabizi" if arabizi_reply == "arabizi" else "ar"
    if language == "ar":
        return "ar"
    if language == "arabizi":
        return arabizi_code
    if language == "en":
        return "en"
    arabic = len(_ARABIC_RE.findall(query_original))
    latin = len(_LATIN_RE.findall(query_original))
    if arabic >= latin:
        return "ar"
    return arabizi_code if _is_arabizi(query_original) else "en"


LANGUAGE_NAMES = {
    "ar": "Egyptian Arabic, in Arabic script",
    "arabizi": "Egyptian Arabic written in Arabizi (Latin letters, digits for Arabic sounds)",
    "en": "English",
}


# ---------------------------------------------------------------- products and fields

# Fields sent per product, by intent (plan section 3). Find / refine / no-match share one list.
_LIST_FIELDS = ("name", "name_ar", "product_type", "product_form", "concerns", "suitable_for", "price",
                "promotion", "available", "conflict")
INTENT_FIELDS: Dict[str, tuple] = {
    # concerns is an addition: the customer persona gives a reason, and without it the model invented one.
    "how_to_use": ("name", "name_ar", "product_form", "concerns", "how_to_use", "warnings"),
    "safety": ("name", "name_ar", "suitable_for", "warnings", "hero_ingredient", "ingredients_canonical"),
    "compare": ("name", "name_ar", "product_type", "product_form", "hero_ingredient", "concerns", "suitable_for",
                "key_features", "size", "price", "promotion"),
    "product_info": ("name", "name_ar", "description", "hero_ingredient", "ingredients_canonical", "concerns",
                     "suitable_for"),
    # product_type and size are additions: without them the model guessed what the product is.
    "price_offer": ("name", "name_ar", "product_type", "size", "price", "compare_at_price", "discount_percent",
                    "promotion", "available"),
    "sales_training": ("name", "name_ar", "description", "hero_ingredient", "concerns", "suitable_for",
                       "key_features", "price", "promotion"),
    "find_products": _LIST_FIELDS,
    "refine_products": _LIST_FIELDS,
    "greeting": (),
    "out_of_scope": (),
}
NO_PRODUCT_INTENTS = frozenset({"greeting", "out_of_scope"})


def fields_for(intent: str, mode: Mode) -> tuple:
    if mode == "no_match":
        return _LIST_FIELDS
    # "other" with products (the router's default) reads like a product question.
    return INTENT_FIELDS.get(intent, INTENT_FIELDS["product_info"])


def _record(product: dict) -> dict:
    """The catalog record behind a RetrievedProduct dump; the dump's own values
    (price, stock, match) win."""
    record = get_index().products.get(product["handle"]) or {}
    return {**record, **product}


def select_products(ctx: ResponderContext) -> List[dict]:
    """At most RESPONDER_MAX_PRODUCTS merged records: retrieval's order already
    puts name hits first, then the best scores. Compare sends only the named
    products when two or more were named. Greetings and off-topic send none."""
    if (ctx.retrieval is None or not ctx.retrieval.products or ctx.intent in NO_PRODUCT_INTENTS
            or unanchored(ctx.retrieval)):
        return []
    products = ctx.retrieval.products
    if ctx.intent == "compare":
        named = [p for p in products if p.handle in set(ctx.retrieval.name_hits)]
        if len(named) >= 2:
            products = named
    return [_record(p.model_dump()) for p in products[: settings.RESPONDER_MAX_PRODUCTS]]


def hero_text(record: dict) -> Optional[str]:
    hero = record.get("hero_ingredient")
    if isinstance(hero, dict):
        if not hero.get("name"):
            return None
        return f"{hero['name']} {hero['concentration']}".strip() if hero.get("concentration") else hero["name"]
    return hero or None


# Catalog promotion codes, spelled out: the model read "buy2get1" as "buy 2 get 3".
PROMOTIONS = {"buy1get1": "buy 1 get 1 free", "buy2get1": "buy 2 get 1 free"}


def _value(record: dict, key: str, description_chars: int):
    if key == "promotion":
        promo = record.get("promotion")
        return PROMOTIONS.get(promo, promo) if promo else None
    if key == "hero_ingredient":
        return hero_text(record)
    if key == "conflict":
        return (record.get("match") or {}).get("conflict")
    if key == "description":
        text = " ".join((record.get("description") or "").split())
        return text if len(text) <= description_chars else text[: description_chars - 1].rstrip() + "…"
    if key == "available":
        return bool(record.get("available", True))
    return record.get(key)


def product_fields(record: dict, fields: tuple, description_chars: Optional[int] = None) -> dict:
    """The intent's fields, empty values dropped (available=false is kept)."""
    chars = description_chars or settings.RESPONDER_DESCRIPTION_CHARS
    out = {}
    for key in fields:
        value = _value(record, key, chars)
        if value is None or value == [] or value == "":
            continue
        if key == "available" and value:
            continue            # in stock is the default; only "false" is worth tokens
        out[key] = value
    return out


def product_lines(products: List[dict], ctx: ResponderContext) -> List[str]:
    """Out-of-stock products always say so, whatever the intent: the
    out_of_stock block tells the model, and the data must back it."""
    fields = fields_for(ctx.intent, ctx.mode)
    fields = fields if "available" in fields else fields + ("available",)
    return [json.dumps(product_fields(p, fields), ensure_ascii=False, separators=(",", ":")) for p in products]


def excluded_hits(record: dict, retrieval: Optional[RetrievalResult]) -> List[str]:
    """The excluded ingredients this product contains."""
    if retrieval is None:
        return []
    excluded = retrieval.applied_filters.get(EXCLUDE_KEY) or []
    own = set(record.get("ingredients_canonical") or [])
    return [i for i in excluded if i in own]
