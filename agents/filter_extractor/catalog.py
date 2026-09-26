"""Loads data/majestic_catalog.json + data/metadata_catalog.json once and derives
everything the extractor needs from them: allowed lists, schema enums,
hierarchy, hero -> ingredient map, name indexes and the rule-based alias tables.

Nothing here hard-codes a catalog value. A re-scrape only needs the two JSON
files replaced (see scripts/build_catalogs.py) and a restart.
"""

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("MAJESTIC_DATA_DIR", ROOT / "data"))
PRODUCTS_PATH = DATA_DIR / "majestic_catalog.json"
METADATA_PATH = DATA_DIR / "metadata_catalog.json"
ALIASES_PATH = DATA_DIR / "aliases.json"

# Catalog keys whose values come from an allowed list (everything but names).
CATALOG_KEYS = (
    "brand", "category", "product_group", "product_type", "product_form",
    "concerns", "suitable_for", "hero_ingredient", "ingredients",
)

_SIZE_RE = re.compile(r"\b\d+(\.\d+)?\s*(?:ml|gm|g|mg|l)\b", re.IGNORECASE)
_AR_SIZE_RE = re.compile(r"[\d٠-٩]+\s*(مل|جم|جرام)?")
_PUNCT_RE = re.compile(r"[^\w\s+%&/-]", re.UNICODE)
_WS_RE = re.compile(r"\s+")

# Arabic letter variants seen in the scrape (ڤ / ڈ / ڄ are all used for "v").
_AR_FOLD = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي",
    "ڤ": "ف", "ڈ": "ف", "ڄ": "ف", "ـ": "",
})


def normalize(text: str) -> str:
    """Lowercase, strip accents/punctuation (keeps + % & / -), collapse spaces."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    text = text.replace("’", "'").replace("'", "")
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def normalize_ar(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = "".join(ch for ch in text if not ("ً" <= ch <= "ٟ"))  # harakat
    text = text.translate(_AR_FOLD)
    text = _AR_SIZE_RE.sub(" ", text)
    text = re.sub(r"[×+–\-()،,]", " ", text)
    return _WS_RE.sub(" ", text).strip()


_AR_UNIT_RE = re.compile(r"[\d٠-٩]+(\.[\d٠-٩]+)?\s*(?:مل|جم|جرام)")


def strip_size(name: str) -> str:
    """'Capixy Hair Serum 120ml' -> 'Capixy Hair Serum' (also '120 مل' in Arabic)."""
    name = _AR_UNIT_RE.sub(" ", _SIZE_RE.sub(" ", name))
    return _WS_RE.sub(" ", name).strip(" -+–")


@dataclass
class ProductName:
    handle: str
    name: str
    name_ar: Optional[str]
    brand: str
    is_bundle: bool
    norm: str           # normalized English name without size
    norm_ar: str        # normalized Arabic name without size
    display: str        # English name without size
    display_ar: Optional[str]


@dataclass
class Catalog:
    allowed: Dict[str, List[str]]
    hierarchy: Dict[str, Dict[str, Dict[str, List[str]]]]
    product_line_to_brand: Dict[str, str]
    brand_aliases: Dict[str, List[str]]
    hero_to_ingredients: Dict[str, List[str]]
    field_map: Dict[str, str]
    products: List[dict]
    names: List[ProductName]
    aliases: Dict[str, Dict[str, str]] = field(default_factory=dict)
    typos: Dict[str, str] = field(default_factory=dict)
    unmatched_terms: Dict[str, str] = field(default_factory=dict)
    scraped_at: Optional[str] = None

    def product_lines_text(self) -> str:
        """'Sebio-Control / Aqua-Repair → Vacation; Advance Repair / Intense → Capixy'."""
        by_brand: Dict[str, List[str]] = {}
        for line, brand in self.product_line_to_brand.items():
            by_brand.setdefault(brand, []).append(line)
        return "; ".join(f"{' / '.join(lines)} → {brand}" for brand, lines in by_brand.items())


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_catalog(products_doc: dict, metadata: dict, aliases_doc: Optional[dict] = None) -> Catalog:
    allowed = {k: list(metadata["allowed_values"][k]) for k in CATALOG_KEYS}
    products = products_doc["products"]
    names = []
    for p in products:
        display = strip_size(p["name"])
        display_ar = strip_size(p["name_ar"]) if p.get("name_ar") else None
        names.append(ProductName(
            handle=p["handle"],
            name=p["name"],
            name_ar=p.get("name_ar"),
            brand=p["brand"],
            is_bundle=p.get("product_kind") == "bundle",
            norm=normalize(display).replace("%", ""),
            norm_ar=normalize_ar(p.get("name_ar") or ""),
            display=display,
            display_ar=display_ar,
        ))

    aliases_doc = aliases_doc or {}
    aliases = {k: {normalize(a): v for a, v in (aliases_doc.get(k) or {}).items()} for k in CATALOG_KEYS}
    return Catalog(
        allowed=allowed,
        hierarchy=metadata["hierarchy"],
        product_line_to_brand=metadata["product_line_to_brand"],
        brand_aliases=metadata["brand_aliases"],
        hero_to_ingredients=metadata["hero_to_ingredients"],
        field_map=metadata["field_map"],
        products=products,
        names=names,
        aliases=aliases,
        typos={normalize(k): normalize(v) for k, v in (aliases_doc.get("_typos") or {}).items()},
        unmatched_terms={normalize(k): v for k, v in (aliases_doc.get("_unmatched_ingredients") or {}).items()},
        scraped_at=metadata.get("scraped_at"),
    )


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    aliases_doc = _load_json(ALIASES_PATH) if ALIASES_PATH.exists() else {}
    return build_catalog(_load_json(PRODUCTS_PATH), _load_json(METADATA_PATH), aliases_doc)
