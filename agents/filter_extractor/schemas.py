"""The 10 call specs and their JSON schemas, built from the catalog at startup.

Each schema is `type: object`, `additionalProperties: false`, every field
required; enum arrays use that key's allowed values. `validate_output` is the
code-side check that runs on every response regardless of whether the
provider enforced the schema: off-list values are snapped to the closest
allowed value (rapidfuzz >= 90) or dropped, lists are de-duplicated and cut to
maxItems.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process, utils

from agents.filter_extractor.catalog import Catalog, get_catalog

MAX_ITEMS_CATALOG = 8
MAX_ITEMS_NAMES = 5
MAX_ITEMS_UNMATCHED = 3
SNAP_SCORE = 90


@dataclass(frozen=True)
class FieldSpec:
    name: str
    enum_key: Optional[str]  # catalog key whose allowed list is the enum; None = free text
    max_items: int


@dataclass(frozen=True)
class CallSpec:
    key: str              # call id; also the prompt file name and config key
    prompt_file: str
    fields: Tuple[FieldSpec, ...]
    uses_context: bool = False

    @property
    def enum_key(self) -> Optional[str]:
        """The catalog key this call fills (None for the names call)."""
        return next((f.enum_key for f in self.fields if f.enum_key), None)


def _catalog_field(name: str, key: str) -> FieldSpec:
    return FieldSpec(name, key, MAX_ITEMS_CATALOG)


CALL_SPECS: Tuple[CallSpec, ...] = (
    CallSpec("names", "filter_extractor/names.md",
             (FieldSpec("name_en", None, MAX_ITEMS_NAMES), FieldSpec("name_ar", None, MAX_ITEMS_NAMES))),
    CallSpec("brand", "filter_extractor/brand.md", (_catalog_field("brand", "brand"),)),
    CallSpec("category", "filter_extractor/category.md", (_catalog_field("category", "category"),)),
    CallSpec("product_group", "filter_extractor/product_group.md", (_catalog_field("product_group", "product_group"),)),
    CallSpec("product_type", "filter_extractor/product_type.md", (_catalog_field("product_type", "product_type"),)),
    CallSpec("product_form", "filter_extractor/product_form.md", (_catalog_field("product_form", "product_form"),)),
    CallSpec("concerns", "filter_extractor/concerns.md",
             (_catalog_field("concerns", "concerns"), FieldSpec("unmatched", None, MAX_ITEMS_UNMATCHED))),
    CallSpec("suitable_for", "filter_extractor/suitable_for.md", (_catalog_field("suitable_for", "suitable_for"),)),
    CallSpec("hero_ingredient", "filter_extractor/hero_ingredient.md", (_catalog_field("hero_ingredient", "hero_ingredient"),)),
    CallSpec("ingredients", "filter_extractor/ingredients.md", (
        _catalog_field("include", "ingredients"),
        _catalog_field("exclude", "ingredients"),
        FieldSpec("unmatched_include", None, MAX_ITEMS_UNMATCHED),
        FieldSpec("unmatched_exclude", None, MAX_ITEMS_UNMATCHED),
    )),
)
CALLS_BY_KEY: Dict[str, CallSpec] = {c.key: c for c in CALL_SPECS}


def build_schema(spec: CallSpec, catalog: Catalog) -> dict:
    props = {}
    for f in spec.fields:
        items = {"type": "string"}
        if f.enum_key:
            items["enum"] = list(catalog.allowed[f.enum_key])
        props[f.name] = {"type": "array", "items": items, "maxItems": f.max_items}
    return {
        "title": f"majestic_{spec.key}",
        "type": "object",
        "additionalProperties": False,
        "required": [f.name for f in spec.fields],
        "properties": props,
    }


@lru_cache(maxsize=None)
def get_schema(key: str) -> dict:
    return build_schema(CALLS_BY_KEY[key], get_catalog())


def empty_output(spec: CallSpec) -> dict:
    return {f.name: [] for f in spec.fields}


def snap_value(value: str, allowed: List[str], lower_map: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Exact (case-insensitive) match, else the closest allowed value scoring >= 90."""
    if not isinstance(value, str) or not value.strip():
        return None
    v = value.strip()
    lower_map = lower_map or {a.lower(): a for a in allowed}
    if v.lower() in lower_map:
        return lower_map[v.lower()]
    hit = process.extractOne(v, allowed, scorer=fuzz.ratio, processor=utils.default_process, score_cutoff=SNAP_SCORE)
    return hit[0] if hit else None


def _dedupe(values):
    seen, out = set(), []
    for v in values:
        k = v.lower()
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out


def validate_output(spec: CallSpec, data: dict, catalog: Optional[Catalog] = None) -> Tuple[dict, List[str]]:
    """Return (clean output, list of values that were snapped or dropped)."""
    catalog = catalog or get_catalog()
    data = data if isinstance(data, dict) else {}
    out, notes = {}, []
    for f in spec.fields:
        raw = data.get(f.name) or []
        if isinstance(raw, str):
            raw = [raw]
        values = []
        if f.enum_key:
            allowed = catalog.allowed[f.enum_key]
            for v in raw:
                snapped = snap_value(v, allowed)
                if snapped is None:
                    notes.append(f"dropped {f.name}={v!r}")
                    continue
                if snapped != v:
                    notes.append(f"snapped {f.name}={v!r}->{snapped!r}")
                values.append(snapped)
        else:
            values = [v.strip() for v in raw if isinstance(v, str) and v.strip()]
        out[f.name] = _dedupe(values)[: f.max_items]
    return out, notes
