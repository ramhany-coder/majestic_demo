"""Exact filter step: extractor output -> candidate set C.

- Active keys: the non-empty lists among brand, category, product_group,
  product_type, product_form, concerns, suitable_for, hero_ingredient and
  ingredients.include. No active keys: C is the whole pool.
- OR within a key, AND across keys. ingredients.include is the exception:
  the product must contain all of them.
- suitable_for naming a specific skin type also matches "all skin types"
  (RETRIEVAL_ALL_SKIN_TYPES_MATCH).
- product_form also matches the catalog forms that contain it as whole words
  ("cream" -> "cream gel", "tinted cream"; RETRIEVAL_FORM_VARIANTS). A form
  that only restates the product_type ("lotion" with "body lotion") is not
  applied, so the Body Milk (type body lotion, form milk) still matches.
- ingredients.exclude removes any product containing any of them. It is
  applied always, including every fallback, and never relaxed.
- Empty C, by default (RETRIEVAL_RELAX_FILTERS off): C stays empty, so only
  products matching every requested key are ever returned. near_miss_keys
  lists the keys that, dropped alone, would give products.
- Empty C with RETRIEVAL_RELAX_FILTERS on (the plan): drop active keys one at
  a time in RELAX_ORDER until C is non-empty, recording relaxed_keys. With
  every key dropped, C is the pool minus excluded products and
  fallback = "semantic_only".
- Bundle rule (RETRIEVAL_BUNDLE_RULE): bundles stay in C only when the request
  asks for them (product_type bundle, group Bundles & Offers, a bundle matched
  by name, or intent price_offer).
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

from agents.retrieval.index_builder import RetrievalIndex
from config import settings
from models.filter_extractor import MetadataFilters

INCLUDE = "ingredients.include"
EXCLUDE = "ingredients.exclude"
FILTER_KEYS = ("brand", "category", "product_group", "product_type", "product_form", "concerns", "suitable_for",
               "hero_ingredient", INCLUDE)
RELAX_ORDER = ("product_form", "suitable_for", "concerns", "product_group", "category", "hero_ingredient", INCLUDE,
               "product_type", "brand")
BUNDLE_TYPE = "bundle"
BUNDLE_GROUP = "Bundles & Offers"
OFFER_INTENT = "price_offer"
FALLBACK_SEMANTIC_ONLY = "semantic_only"
ALL_SKIN_TYPES = "all skin types"
SKIN_TYPES = frozenset({"acne-prone skin", "combination skin", "dry skin", "oily skin", "sensitive skin"})


@dataclass
class FilterOutcome:
    candidates: Set[str]
    requested: Dict[str, List[str]]                        # active keys as extracted
    applied: Dict[str, List[str]]                          # keys actually used, plus ingredients.exclude
    key_sets: Dict[str, Set[str]]                          # requested key -> handles matching it
    relaxed_keys: List[str] = field(default_factory=list)
    near_miss_keys: List[str] = field(default_factory=list)  # strict and empty: keys that, dropped alone, give products
    implied_keys: List[str] = field(default_factory=list)    # requested but not applied: restated by another key
    excluded: Set[str] = field(default_factory=set)        # handles containing an excluded ingredient
    bundles_allowed: bool = True
    fallback: Optional[str] = None

    def matched_keys(self, handle: str) -> List[str]:
        return [k for k in self.requested if k not in self.implied_keys and handle in self.key_sets[k]]

    def mismatched_keys(self, handle: str) -> List[str]:
        return [k for k in self.requested if k not in self.implied_keys and handle not in self.key_sets[k]]


def active_keys(filters: Optional[MetadataFilters]) -> Dict[str, List[str]]:
    if filters is None:
        return {}
    out = {}
    for key in FILTER_KEYS:
        values = filters.ingredients.include if key == INCLUDE else getattr(filters, key)
        if values:
            out[key] = list(values)
    return out


def key_matches(index: RetrievalIndex, key: str, values: List[str]) -> Set[str]:
    """Handles matching one key: any value (OR), or all of them for ingredients.include."""
    if key == INCLUDE:
        sets = [index.inverted["ingredients"].get(v, set()) for v in values]
        return set.intersection(*sets) if sets else set()
    table = index.inverted[key]
    if key == "suitable_for" and settings.RETRIEVAL_ALL_SKIN_TYPES_MATCH and SKIN_TYPES.intersection(values):
        values = [*values, ALL_SKIN_TYPES]
    if key == "product_form" and settings.RETRIEVAL_FORM_VARIANTS:
        values = form_variants(table, values)
    return set().union(*(table.get(v, set()) for v in values))


def form_variants(forms: Iterable[str], values: List[str]) -> List[str]:
    """The requested forms plus every catalog form containing one of them as
    whole words: "cream" -> "cream gel", "jelly cream", "leave-in cream"."""
    wanted = [set(v.split()) for v in values]
    return [*values, *(f for f in forms if f not in values and any(w <= set(f.split()) for w in wanted))]


def bundles_allowed(filters: Optional[MetadataFilters], intent: Optional[str], named_bundle: bool = False) -> bool:
    if not settings.RETRIEVAL_BUNDLE_RULE or named_bundle or intent == OFFER_INTENT:
        return True
    return filters is not None and (BUNDLE_TYPE in filters.product_type or BUNDLE_GROUP in filters.product_group)


def form_restates_type(requested: Dict[str, List[str]]) -> bool:
    """True when every requested form is already words of a requested
    product_type ("lotion" in "body lotion", "cream" in "anti-aging cream")."""
    types = [set(t.split()) for t in requested.get("product_type", [])]
    forms = requested.get("product_form", [])
    return bool(types and forms) and all(any(set(f.split()) <= t for t in types) for f in forms)


def apply_filters(index: RetrievalIndex, filters: Optional[MetadataFilters], intent: Optional[str] = None,
                  named_bundle: bool = False) -> FilterOutcome:
    requested = active_keys(filters)
    exclude = set(filters.ingredients.exclude) if filters is not None else set()
    excluded = {h for h in index.order if index.ingredients[h] & exclude}
    allow_bundles = bundles_allowed(filters, intent, named_bundle)
    base = index.pool - excluded - (set() if allow_bundles else index.bundles)
    key_sets = {k: key_matches(index, k, v) for k, v in requested.items()}

    implied = ["product_form"] if settings.RETRIEVAL_FORM_VARIANTS and form_restates_type(requested) else []
    active = {k: v for k, v in requested.items() if k not in implied}
    relaxed: List[str] = []
    while True:
        candidates = set(base)
        for key in active:
            candidates &= key_sets[key]
        if candidates or not active or not settings.RETRIEVAL_RELAX_FILTERS:
            break
        drop = next(k for k in RELAX_ORDER if k in active)
        del active[drop]
        relaxed.append(drop)

    near_miss = []
    if not candidates and active:
        near_miss = [k for k in active if base.intersection(*(key_sets[o] for o in active if o != k))]

    applied = dict(active)
    if exclude:
        applied[EXCLUDE] = sorted(exclude)
    return FilterOutcome(
        candidates=candidates,
        requested=requested,
        applied=applied,
        key_sets=key_sets,
        relaxed_keys=relaxed,
        near_miss_keys=near_miss,
        implied_keys=implied,
        excluded=excluded,
        bundles_allowed=allow_bundles,
        fallback=FALLBACK_SEMANTIC_ONLY if requested and not active else None,
    )
