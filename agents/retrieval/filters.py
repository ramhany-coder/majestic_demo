"""Exact filter step: extractor output -> candidate set C.

- Active keys: the non-empty lists among brand, category, product_group,
  product_type, product_form, concerns, suitable_for, hero_ingredient and
  ingredients.include. No active keys: C is the whole pool.
- OR within a key, AND across keys. ingredients.include is the exception:
  the product must contain all of them.
- ingredients.exclude removes any product containing any of them. It is
  applied always, including every fallback, and never relaxed.
- Empty C: drop active keys one at a time in RELAX_ORDER until C is
  non-empty, recording relaxed_keys. With every key dropped, C is the pool
  minus excluded products and fallback = "semantic_only".
- Bundle rule (RETRIEVAL_BUNDLE_RULE): bundles stay in C only when the request
  asks for them (product_type bundle, group Bundles & Offers, a bundle matched
  by name, or intent price_offer).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

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


@dataclass
class FilterOutcome:
    candidates: Set[str]
    requested: Dict[str, List[str]]                        # active keys as extracted
    applied: Dict[str, List[str]]                          # keys actually used, plus ingredients.exclude
    key_sets: Dict[str, Set[str]]                          # requested key -> handles matching it
    relaxed_keys: List[str] = field(default_factory=list)
    excluded: Set[str] = field(default_factory=set)        # handles containing an excluded ingredient
    bundles_allowed: bool = True
    fallback: Optional[str] = None

    def matched_keys(self, handle: str) -> List[str]:
        return [k for k in self.requested if handle in self.key_sets[k]]

    def mismatched_keys(self, handle: str) -> List[str]:
        return [k for k in self.requested if handle not in self.key_sets[k]]


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
    return set().union(*(table.get(v, set()) for v in values))


def bundles_allowed(filters: Optional[MetadataFilters], intent: Optional[str], named_bundle: bool = False) -> bool:
    if not settings.RETRIEVAL_BUNDLE_RULE or named_bundle or intent == OFFER_INTENT:
        return True
    return filters is not None and (BUNDLE_TYPE in filters.product_type or BUNDLE_GROUP in filters.product_group)


def apply_filters(index: RetrievalIndex, filters: Optional[MetadataFilters], intent: Optional[str] = None,
                  named_bundle: bool = False) -> FilterOutcome:
    requested = active_keys(filters)
    exclude = set(filters.ingredients.exclude) if filters is not None else set()
    excluded = {h for h in index.order if index.ingredients[h] & exclude}
    allow_bundles = bundles_allowed(filters, intent, named_bundle)
    base = index.pool - excluded - (set() if allow_bundles else index.bundles)
    key_sets = {k: key_matches(index, k, v) for k, v in requested.items()}

    active = dict(requested)
    relaxed: List[str] = []
    while True:
        candidates = set(base)
        for key in active:
            candidates &= key_sets[key]
        if candidates or not active:
            break
        drop = next(k for k in RELAX_ORDER if k in active)
        del active[drop]
        relaxed.append(drop)

    applied = dict(active)
    if exclude:
        applied[EXCLUDE] = sorted(exclude)
    return FilterOutcome(
        candidates=candidates,
        requested=requested,
        applied=applied,
        key_sets=key_sets,
        relaxed_keys=relaxed,
        excluded=excluded,
        bundles_allowed=allow_bundles,
        fallback=FALLBACK_SEMANTIC_ONLY if requested and not active else None,
    )
