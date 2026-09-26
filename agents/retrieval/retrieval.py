"""Catalog retrieval: MetadataFilters (+ the English query) -> ranked products.

Implements the hand-off contract in ARCHITECTURE_NOTES.md section 5 over the
in-memory catalog (108 products):

- matched_handles, when present, restrict the result to those products.
- Otherwise AND across keys, OR within a key; ingredients.include = all
  present, ingredients.exclude = none present (never relaxed).
- On zero results, relax product_form, suitable_for, concerns,
  product_group, category -- in that order -- until something matches.
- unmatched.* terms are soft signals on the description.
- No usable filters (extractor found nothing, or the rewriter failed and the
  filters were skipped): token-overlap text search on the query. No
  embedding model is configured in this project, so this is the semantic
  fallback.
"""

import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

from agents.filter_extractor.catalog import PRODUCTS_PATH, _load_json, normalize, normalize_ar
from config import settings
from models.filter_extractor import MetadataFilters

RELAX_ORDER = ("product_form", "suitable_for", "concerns", "product_group", "category")
SINGLE_VALUE_KEYS = ("brand", "category", "product_group", "product_type", "product_form")
LIST_KEYS = ("concerns", "suitable_for")
MIN_TEXT_SCORE = 2.0
UNMATCHED_BOOST = 1.5

_STOPWORDS = {
    "the", "and", "for", "with", "without", "you", "your", "have", "has", "any", "are", "is", "do", "does",
    "can", "need", "want", "looking", "something", "product", "products", "one", "that", "this", "what",
    "which", "there", "please", "show", "some", "from", "about", "safe", "good", "best", "my", "me", "i",
    "also", "use", "using", "how", "does", "during", "it", "its", "of", "to", "in", "on", "a", "an", "or",
}
_TOKEN_RE = re.compile(r"[\w%+-]+", re.UNICODE)


@dataclass
class RetrievalResult:
    products: List[dict] = field(default_factory=list)   # full catalog records, best first
    mode: str = "none"                                    # handles | filters | text | none
    relaxed: List[str] = field(default_factory=list)      # filter keys dropped to find results
    latency_ms: float = 0.0

    def cards(self) -> List[dict]:
        return [to_card(p) for p in self.products]

    def summary(self) -> dict:
        return {"mode": self.mode, "count": len(self.products), "relaxed": self.relaxed,
                "handles": [p["handle"] for p in self.products], "latency_ms": self.latency_ms}


def to_card(p: dict) -> dict:
    images = p.get("images") or []
    return {
        "handle": p["handle"], "name": p["name"], "name_ar": p.get("name_ar"), "brand": p.get("brand"),
        "product_type": p.get("product_type"), "price": p.get("price"),
        "compare_at_price": p.get("compare_at_price"), "available": p.get("available", True),
        "url": p.get("url"), "url_ar": p.get("url_ar"), "image": images[0].get("src") if images else None,
    }


def _tokens(text: str) -> set:
    toks = set()
    for t in _TOKEN_RE.findall(normalize(text) + " " + normalize_ar(text)):
        if len(t) < 3 or t in _STOPWORDS:
            continue
        toks.add(t)
        if t.endswith("s") and len(t) > 3:
            toks.add(t[:-1])   # serums -> serum
    return toks


@dataclass(frozen=True)
class _Indexed:
    product: dict
    name_toks: frozenset
    meta_toks: frozenset       # search_text: name, brand, taxonomy, concerns, ingredients
    desc_toks: frozenset
    desc_norm: str
    ingredients: frozenset
    hero: Optional[str]


@lru_cache(maxsize=1)
def _index() -> tuple:
    out = []
    for p in _load_json(PRODUCTS_PATH)["products"]:
        hero = p.get("hero_ingredient")
        out.append(_Indexed(
            product=p,
            name_toks=frozenset(_tokens(p["name"] + " " + (p.get("name_ar") or ""))),
            meta_toks=frozenset(_tokens(p.get("search_text") or "")),
            desc_toks=frozenset(_tokens(p.get("description") or "")),
            desc_norm=normalize(p.get("description") or ""),
            ingredients=frozenset(p.get("ingredients_canonical") or []),
            hero=hero.get("name") if isinstance(hero, dict) else hero,
        ))
    return tuple(out)


def text_score(ix: _Indexed, query_toks: set) -> float:
    return sum(3.0 if t in ix.name_toks else 2.0 if t in ix.meta_toks else 1.0 if t in ix.desc_toks else 0.0
               for t in query_toks)


def _matches(ix: _Indexed, active: Dict[str, set], include: set, exclude: set) -> bool:
    p = ix.product
    if exclude & ix.ingredients:
        return False
    if include and not include <= ix.ingredients:
        return False
    for key, values in active.items():
        if key in SINGLE_VALUE_KEYS and p.get(key) not in values:
            return False
        if key in LIST_KEYS and not values & set(p.get(key) or []):
            return False
        if key == "hero_ingredient" and ix.hero not in values:
            return False
    return True


def _soft_score(ix: _Indexed, f: MetadataFilters) -> float:
    score = 0.0
    for term in f.unmatched.concerns + f.unmatched.include:
        if normalize(term) and normalize(term) in ix.desc_norm:
            score += UNMATCHED_BOOST
    for term in f.unmatched.exclude:
        t = normalize(term)
        if not t:
            continue
        if f"{t} free" in ix.desc_norm or f"without {t}" in ix.desc_norm or f"free of {t}" in ix.desc_norm:
            score += UNMATCHED_BOOST
        elif t in ix.desc_norm:
            score -= 2 * UNMATCHED_BOOST
    return score


def _rank(items: Sequence[_Indexed], query_toks: set, f: Optional[MetadataFilters]) -> List[dict]:
    scored = [(text_score(ix, query_toks) + (_soft_score(ix, f) if f else 0.0), ix) for ix in items]
    scored.sort(key=lambda s: (-s[0], not s[1].product.get("available", True), not s[1].product.get("best_seller")))
    return [ix.product for _, ix in scored]


def _active_filters(f: MetadataFilters) -> Dict[str, set]:
    active = {k: set(getattr(f, k)) for k in SINGLE_VALUE_KEYS + LIST_KEYS if getattr(f, k)}
    if f.hero_ingredient:
        active["hero_ingredient"] = set(f.hero_ingredient)
    return active


def text_search(query: str, top_k: Optional[int] = None) -> List[dict]:
    top_k = top_k or settings.RETRIEVAL_TOP_K
    q = _tokens(query or "")
    scored = [(text_score(ix, q), ix) for ix in _index()]
    scored = [s for s in scored if s[0] >= MIN_TEXT_SCORE]
    scored.sort(key=lambda s: (-s[0], not s[1].product.get("available", True), not s[1].product.get("best_seller")))
    return [ix.product for _, ix in scored[:top_k]]


def retrieve(filters: Optional[MetadataFilters], query_en: str = "", top_k: Optional[int] = None) -> RetrievalResult:
    start = time.perf_counter()
    top_k = top_k or settings.RETRIEVAL_TOP_K
    q = _tokens(query_en or "")

    def done(products: List[dict], mode: str, relaxed: Optional[List[str]] = None) -> RetrievalResult:
        return RetrievalResult(products[:top_k], mode, relaxed or [], round((time.perf_counter() - start) * 1000, 1))

    if filters is None or filters.is_empty():
        return done(text_search(query_en, top_k), "text")

    include, exclude = set(filters.ingredients.include), set(filters.ingredients.exclude)
    if filters.matched_handles:
        order = {h: i for i, h in enumerate(filters.matched_handles)}
        hits = [ix for ix in _index() if ix.product["handle"] in order and not exclude & ix.ingredients]
        hits.sort(key=lambda ix: order[ix.product["handle"]])
        return done([ix.product for ix in hits], "handles")

    active = _active_filters(filters)
    if not active and not include and not exclude:
        # Only unmatched terms: text search, with the soft signals on top.
        return done(_rank([ix for ix in _index() if text_score(ix, q) >= MIN_TEXT_SCORE], q, filters), "text")

    relaxed: List[str] = []
    while True:
        hits = [ix for ix in _index() if _matches(ix, active, include, exclude)]
        if hits:
            return done(_rank(hits, q, filters), "filters", relaxed)
        key = next((k for k in RELAX_ORDER if k in active), None)
        if key is None:
            return done([], "filters", relaxed)
        del active[key]
        relaxed.append(key)
