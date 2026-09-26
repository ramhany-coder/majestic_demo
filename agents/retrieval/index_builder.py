"""Builds everything retrieval needs from data/majestic_catalog.json, once per
process. Nothing is hard-coded: a new scrape only needs the catalog files
replaced (scripts/build_catalogs.py) and a restart.

- Product pool: `products` only (never `non_products` such as cards and
  memberships). Bundles keep product_kind = "bundle".
- Inverted indexes: filter key -> value -> set of handles, one per catalog
  field in metadata_catalog.json's field_map. Bundles are also indexed under
  each of their contains_product_types, so "hair tonic" can find a tonic kit.
- Name indexes: one bigram BM25 index per language (name_search.py).

The product embedding matrix has its own disk-cached registry
(embedding_registry.py), because it needs the embedding model.
"""

import hashlib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set

from agents.filter_extractor.catalog import METADATA_PATH, PRODUCTS_PATH, _load_json
from agents.retrieval.name_search import NameIndex

# Filter key -> product field. hero_ingredient is a {name, ...} object and
# ingredients are the canonical list (see metadata_catalog.json -> field_map).
FILTER_FIELDS = {
    "brand": "brand",
    "category": "category",
    "product_group": "product_group",
    "product_type": "product_type",
    "product_form": "product_form",
    "concerns": "concerns",
    "suitable_for": "suitable_for",
    "hero_ingredient": "hero_ingredient",
    "ingredients": "ingredients_canonical",
}
BUNDLE_KIND = "bundle"
_WS_RE = re.compile(r"\s+")


def hero_name(product: dict) -> Optional[str]:
    hero = product.get("hero_ingredient")
    return hero.get("name") if isinstance(hero, dict) else hero


def field_values(product: dict, key: str) -> List[str]:
    """A product's values for one filter key, as a list."""
    if key == "hero_ingredient":
        value = hero_name(product)
        return [value] if value else []
    value = product.get(FILTER_FIELDS[key])
    if isinstance(value, list):
        return [v for v in value if v]
    return [value] if value else []


@dataclass
class RetrievalIndex:
    products: Dict[str, dict]                       # handle -> catalog record
    order: List[str]                                # catalog order
    bundles: Set[str]
    inverted: Dict[str, Dict[str, Set[str]]]        # key -> value -> handles
    ingredients: Dict[str, frozenset]               # handle -> canonical ingredients
    descriptions: Dict[str, str]                    # handle -> lowercased description, spaces collapsed
    names_en: NameIndex
    names_ar: NameIndex
    catalog_hash: str
    # bundle handle -> product types it is indexed under only through contains_product_types
    via_contains: Dict[str, Set[str]] = field(default_factory=dict)

    @property
    def pool(self) -> Set[str]:
        return set(self.order)

    def has_value(self, handle: str, key: str, value: str) -> bool:
        return handle in self.inverted.get(key, {}).get(value, ())


def catalog_hash(paths=(PRODUCTS_PATH, METADATA_PATH)) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def build_index(products_doc: dict, digest: str = "") -> RetrievalIndex:
    products = {p["handle"]: p for p in products_doc["products"]}
    order = list(products)
    inverted: Dict[str, Dict[str, Set[str]]] = {k: {} for k in FILTER_FIELDS}
    bundles, via_contains = set(), {}
    for handle, p in products.items():
        for key in FILTER_FIELDS:
            for value in field_values(p, key):
                inverted[key].setdefault(value, set()).add(handle)
        if p.get("product_kind") == BUNDLE_KIND:
            bundles.add(handle)
            extra = {t for t in p.get("contains_product_types") or [] if t and t != p.get("product_type")}
            for t in extra:
                inverted["product_type"].setdefault(t, set()).add(handle)
            via_contains[handle] = extra
    return RetrievalIndex(
        products=products,
        order=order,
        bundles=bundles,
        inverted=inverted,
        ingredients={h: frozenset(p.get("ingredients_canonical") or []) for h, p in products.items()},
        descriptions={h: _WS_RE.sub(" ", (p.get("description") or "").lower()).strip() for h, p in products.items()},
        names_en=NameIndex.build("en", [(h, p["name"]) for h, p in products.items()]),
        names_ar=NameIndex.build("ar", [(h, p.get("name_ar") or "") for h, p in products.items()]),
        catalog_hash=digest,
        via_contains=via_contains,
    )


@lru_cache(maxsize=1)
def get_index() -> RetrievalIndex:
    """Built once per process (a few ms)."""
    return build_index(_load_json(PRODUCTS_PATH), catalog_hash())

