"""Responder agent interface.

TODO(responder): building the real responder (an LLM call through
`fallback_client` with its own prompt, persona-aware: customer / sales
trainee / doctor) is out of scope for the pre-qualification and retrieval
tasks. The orchestrator already hands it everything it needs in
ResponderContext, including the whole RetrievalResult (relaxed_keys and
meta.relaxed_values for "no spray found, here are gels", unresolved_names,
per-product match.conflict such as "contains_excluded"); replace the body of
`respond` and keep the signature.

Until then, `respond` returns a short templated reply in the user's
language, listing the products it was given, so the pipeline runs end to
end.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models.retrieval import RetrievalResult

REASON_NEEDS_RESPONSE = "needs_response"   # router asked for a response
REASON_NO_RESULTS = "no_results"           # products_only, but retrieval found nothing
REASON_RELAXED = "relaxed"                 # products_only, but some filters were dropped to find products


@dataclass
class ResponderContext:
    message: str                                   # original user message
    query_en: str                                  # standalone English rewrite
    language: str                                  # ar | arabizi | en | mixed
    persona: str                                   # customer | sales_trainee | doctor | unknown
    intent: str
    history: List[Dict[str, str]] = field(default_factory=list)   # recent messages, oldest first
    products: List[dict] = field(default_factory=list)            # RetrievedProduct dumps, best first (may be empty)
    reason: str = REASON_NEEDS_RESPONSE
    retrieval: Optional[RetrievalResult] = None                   # None when retrieval did not run


_TEXT = {
    "ar": {"intro": "دي المنتجات اللي ممكن تفيدك:", "none": "للأسف مش لاقي منتج مطابق.",
           "relaxed": "للأسف مش لاقي منتج مطابق بالظبط.", "closest": "أقرب اختيارات:",
           "fallback": "تمام، ازاي أقدر أساعدك؟"},
    "arabizi": {"intro": "Dy el montagat elly momken tefidak:", "none": "Lel asaf mesh la2y montag motabe2.",
                "relaxed": "Lel asaf mesh la2y montag motabe2 bel zabt.", "closest": "A2rab e5tyarat:",
                "fallback": "Tamam, ezay a2dar asa3dak?"},
    "en": {"intro": "Here are some products that may help:", "none": "Sorry, I couldn't find a matching product.",
           "relaxed": "Sorry, I couldn't find an exact match.", "closest": "Closest options:",
           "fallback": "Sure, how can I help?"},
}


def _names(products: List[dict], language: str) -> str:
    key = "name_ar" if language == "ar" else "name"
    return "\n".join(f"{i}) {p.get(key) or p.get('name')}" for i, p in enumerate(products, 1))


async def respond(ctx: ResponderContext) -> str:
    """TODO(responder): placeholder, see module docstring."""
    t = _TEXT.get("ar" if ctx.language == "mixed" else ctx.language, _TEXT["en"])
    if ctx.reason == REASON_NO_RESULTS:
        return t["none"]
    if ctx.reason == REASON_RELAXED:
        return f"{t['relaxed']}\n{t['closest']}\n{_names(ctx.products, ctx.language)}"
    if ctx.products:
        return f"{t['intro']}\n{_names(ctx.products, ctx.language)}"
    return t["fallback"]
