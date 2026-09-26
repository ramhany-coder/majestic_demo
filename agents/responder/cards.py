"""Cards built in code from catalog fields, so they cannot contain invented
details and are ready before the LLM's first token; plus the disclaimer rule.

    build_cards(ctx, products, reply_lang)   # [{"type": "how_to_use" | "safety" | "compare", ...}]
    disclaimer_for(ctx, products, reply_lang) # the text, or None
"""

import re
from typing import List, Optional

from agents.responder.context import ResponderContext, hero_text

# Step and safety cards cover the products the user named, else the top one.
CARD_PRODUCTS = 2

PREGNANCY_RE = re.compile(r"pregnan|حامل|الحمل|حوامل", re.IGNORECASE)
BREASTFEEDING_RE = re.compile(r"breast.?feed|lactat|nursing|رضاع|مرضع", re.IGNORECASE)

DISCLAIMER = {
    "ar": "المعلومات للتوعية ومش بديلة عن استشارة الطبيب.",
    "arabizi": "El ma3lomat lel taw3eya w mesh badila 3an estesharet el doktor.",
    "en": "This information is for awareness and is not a substitute for medical advice.",
}
DISCLAIMER_INTENTS = frozenset({"safety", "how_to_use", "product_info"})
MEDICAL_GROUPS = frozenset({"Supplements", "Skin Repair & Healing", "Joint & Muscle Care"})

COMPARE_ROWS = {
    "hero_ingredient": {"en": "Hero ingredient", "ar": "المكون الأساسي", "arabizi": "El mokawen el asasy"},
    "suitable_for": {"en": "Suitable for", "ar": "مناسب لـ", "arabizi": "Monaseb le"},
    "product_form": {"en": "Form", "ar": "الشكل", "arabizi": "El shakl"},
    "size": {"en": "Size", "ar": "الحجم", "arabizi": "El 7agm"},
    "price": {"en": "Price (EGP)", "ar": "السعر (ج.م)", "arabizi": "El se3r (EGP)"},
    "promotion": {"en": "Offer", "ar": "العرض", "arabizi": "El 3ard"},
}


def display_name(record: dict, reply_lang: str) -> str:
    return (record.get("name_ar") if reply_lang == "ar" else None) or record.get("name") or record.get("handle", "")


def _card_targets(ctx: ResponderContext, products: List[dict]) -> List[dict]:
    named = set(ctx.retrieval.name_hits) if ctx.retrieval else set()
    hits = [p for p in products if p["handle"] in named]
    return (hits or products[:1])[:CARD_PRODUCTS]


def safety_flag(record: dict, suitable_label: str, pattern: "re.Pattern") -> str:
    """"warning" when a warning mentions it (this wins: it is the safer
    reading), "suitable" when suitable_for lists it, else "not_listed"."""
    if any(pattern.search(w or "") for w in record.get("warnings") or []):
        return "warning"
    if suitable_label in (record.get("suitable_for") or []):
        return "suitable"
    return "not_listed"


def how_to_use_card(record: dict, reply_lang: str) -> Optional[dict]:
    steps = [s for s in record.get("how_to_use") or [] if s]
    if not steps:
        return None
    return {"type": "how_to_use", "handle": record["handle"], "name": display_name(record, reply_lang),
            "steps": steps}


def safety_card(record: dict, reply_lang: str) -> dict:
    return {
        "type": "safety", "handle": record["handle"], "name": display_name(record, reply_lang),
        "warnings": list(record.get("warnings") or []),
        "suitable_for": list(record.get("suitable_for") or []),
        "flags": {"pregnancy": safety_flag(record, "pregnancy", PREGNANCY_RE),
                  "breastfeeding": safety_flag(record, "breastfeeding", BREASTFEEDING_RE)},
    }


def _compare_value(record: dict, key: str):
    if key == "hero_ingredient":
        return hero_text(record)
    if key == "suitable_for":
        return list(record.get("suitable_for") or []) or None    # a list: the widget labels each value
    return record.get(key)


def compare_card(products: List[dict], reply_lang: str) -> Optional[dict]:
    if len(products) < 2:
        return None
    rows = []
    for key, labels in COMPARE_ROWS.items():
        values = [_compare_value(p, key) for p in products]
        if any(v not in (None, "") for v in values):
            rows.append({"key": key, "label": labels[reply_lang], "values": values})
    return {"type": "compare", "columns": [p["handle"] for p in products],
            "names": [display_name(p, reply_lang) for p in products], "rows": rows}


def build_cards(ctx: ResponderContext, products: List[dict], reply_lang: str) -> List[dict]:
    """Price/offer questions get no card: the product cards already show it."""
    if not products:
        return []
    if ctx.intent == "how_to_use":
        return [c for c in (how_to_use_card(p, reply_lang) for p in _card_targets(ctx, products)) if c]
    if ctx.intent == "safety":
        return [safety_card(p, reply_lang) for p in _card_targets(ctx, products)]
    if ctx.intent == "compare":
        card = compare_card(products, reply_lang)
        return [card] if card else []
    return []


def needs_disclaimer(ctx: ResponderContext, products: List[dict]) -> bool:
    """Health intents about medical product groups, and every customer safety answer."""
    if ctx.intent == "safety" and ctx.audience == "customer":
        return True
    return ctx.intent in DISCLAIMER_INTENTS and any(p.get("product_group") in MEDICAL_GROUPS for p in products)


def disclaimer_for(ctx: ResponderContext, products: List[dict], reply_lang: str) -> Optional[str]:
    return DISCLAIMER[reply_lang] if needs_disclaimer(ctx, products) else None
