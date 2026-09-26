"""Builds data/majestic_catalog.json and data/metadata_catalog.json from the raw
scrape (data/data.json) plus the hand-kept rules in data/catalog_rules.json.

Run after every re-scrape:

    python -m scripts.build_catalogs            # writes both files
    python -m scripts.build_catalogs --check    # exit 1 if anything needs review

Nothing else in the project hard-codes an allowed value: the extractor reads the
two generated files at startup, so replacing them and restarting is enough.
"""

import argparse
import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "data.json"
RULES_PATH = DATA_DIR / "catalog_rules.json"
PRODUCTS_OUT = DATA_DIR / "majestic_catalog.json"
METADATA_OUT = DATA_DIR / "metadata_catalog.json"

# Filter key -> product field it filters on (dotted path for nested values).
FIELD_MAP = OrderedDict([
    ("name_en", "name"),
    ("name_ar", "name_ar"),
    ("brand", "brand"),
    ("category", "category"),
    ("product_group", "product_group"),
    ("product_type", "product_type"),
    ("product_form", "product_form"),
    ("concerns", "concerns"),
    ("suitable_for", "suitable_for"),
    ("hero_ingredient", "hero_ingredient.name"),
    ("ingredients", "ingredients_canonical"),
])

_TM_RE = re.compile(r"[™®]|(?<=[a-z])TM\b")
_PCT_RE = re.compile(r"\d+(\.\d+)?\s*%")
_WS_RE = re.compile(r"\s+")


def raw_key(name: str) -> str:
    """Lookup key for catalog_rules.ingredient_canon (lowercase, no ™/®/%)."""
    s = _TM_RE.sub("", name)
    s = _PCT_RE.sub("", s)
    return _WS_RE.sub(" ", s).strip().lower()


def clean_display(name: str) -> str:
    s = _TM_RE.sub("", name)
    s = _PCT_RE.sub("", s)
    s = _WS_RE.sub(" ", s).strip()
    return s[:1].upper() + s[1:]


class IngredientCanon:
    def __init__(self, rules: dict):
        self.rules = {k: v for k, v in rules.items() if not k.startswith("_")}
        self.unreviewed: Counter = Counter()
        self.known_case: dict = {}

    def register_case(self, canonical: str) -> None:
        self.known_case.setdefault(canonical.lower(), canonical)

    def __call__(self, raw: str) -> list:
        key = raw_key(raw)
        if key in self.rules:
            return list(self.rules[key])
        # Case-only variants ("Argan oil" vs "Argan Oil") collapse onto whichever
        # spelling was seen first, preferring title case.
        display = clean_display(raw)
        return [display]


def smart_title(name: str) -> str:
    """'Rose oil' -> 'Rose Oil', 'EYELISS' -> 'Eyeliss'; short acronyms (DEA, MSM,
    PCA) and mixed-case brand spellings (PatcH2O, HyaCare) are left alone."""
    words = []
    for w in name.split(" "):
        if w.islower():
            w = w[:1].upper() + w[1:]
        elif w.isupper() and len(w) > 4 and not any(ch.isdigit() for ch in w):
            w = w[:1] + w[1:].lower()
        words.append(w)
    return " ".join(words)


def _spelling_rank(s: str):
    shouting = any(w.isupper() and len(w) > 4 for w in s.split(" "))
    return (not shouting, sum(ch.isupper() for ch in s))


def _ordered_unique(values):
    seen, out = set(), []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def build(raw: dict, rules: dict):
    canon = IngredientCanon(rules["ingredient_canon"])
    products = raw["products"]

    # Pass 1: pick one spelling per case-insensitive name, preferring the one
    # with more capital letters ("Argan Oil" over "Argan oil").
    spellings: dict = {}
    for p in products:
        for ing in p.get("key_ingredients") or []:
            for c in canon(ing["name"]):
                cur = spellings.get(c.lower())
                if cur is None or _spelling_rank(c) > _spelling_rank(cur):
                    spellings[c.lower()] = c
    spellings = {k: smart_title(v) for k, v in spellings.items()}
    for mapped in canon.rules.values():
        for c in mapped:
            spellings[c.lower()] = c  # rule targets always win

    ruled_keys = set(canon.rules)
    unreviewed = Counter()
    out_products = []
    for p in products:
        names = []
        for ing in p.get("key_ingredients") or []:
            if raw_key(ing["name"]) not in ruled_keys:
                unreviewed[ing["name"]] += 1
            names.extend(spellings[c.lower()] for c in canon(ing["name"]))
        q = dict(p)
        q["ingredients_canonical"] = _ordered_unique(names)
        out_products.append(q)

    def values(field):
        c = Counter()
        for p in out_products:
            v = p.get(field)
            for x in (v if isinstance(v, list) else [v]):
                if x:
                    c[x] += 1
        return sorted(c, key=str.lower)

    heroes = sorted({p["hero_ingredient"]["name"] for p in out_products if p.get("hero_ingredient")}, key=str.lower)
    ingredients = sorted({i for p in out_products for i in p["ingredients_canonical"]}, key=str.lower)
    ingredient_set = set(ingredients)

    # Hero -> ingredient mapping: explicit rule, else split the hero name.
    hero_rules = {k: v for k, v in rules["hero_to_ingredients"].items() if not k.startswith("_")}
    hero_to_ingredients = {}
    for h in heroes:
        if h in hero_rules:
            mapped = hero_rules[h]
        else:
            parts = re.split(r"\s*\+\s*|\s*[()]\s*|,\s*", h)
            mapped = [spellings.get(c.lower(), c) for part in parts if part.strip() for c in canon(part)]
        hero_to_ingredients[h] = [m for m in _ordered_unique(mapped) if m in ingredient_set]

    # Hierarchy: which groups/categories each type (and each group) appears under.
    type_h, group_h = {}, {}
    for p in out_products:
        t, g, c = p.get("product_type"), p.get("product_group"), p.get("category")
        if t:
            e = type_h.setdefault(t, {"product_group": set(), "category": set()})
            if g:
                e["product_group"].add(g)
            if c:
                e["category"].add(c)
        if g and c:
            group_h.setdefault(g, {"category": set()})["category"].add(c)
    hierarchy = {
        "product_type": {t: {k: sorted(v) for k, v in e.items()} for t, e in sorted(type_h.items())},
        "product_group": {g: {k: sorted(v) for k, v in e.items()} for g, e in sorted(group_h.items())},
    }

    # Product lines must still exist in the catalog for their brand.
    problems = []
    product_lines = {}
    for line, brand in rules["product_lines"].items():
        hit = any(p["brand"] == brand and line.lower().replace("-", " ") in p["name"].lower().replace("-", " ")
                  for p in out_products)
        if hit:
            product_lines[line] = brand
        else:
            problems.append(f"product line '{line}' ({brand}) no longer appears in any product name")

    brands = values("brand")
    brand_aliases = {b: rules["brand_aliases"].get(b, [b.lower()]) for b in brands}
    for b in brands:
        if b not in rules["brand_aliases"]:
            problems.append(f"new brand '{b}' has no entry in catalog_rules.brand_aliases")

    # Arabic hints (kept for a later Arabic-aware extractor; prompts don't use them).
    ar_brand = {}
    for p in out_products:
        if p.get("name_ar"):
            first = re.sub(r"^\d+\s*×\s*", "", p["name_ar"]).split()[0]
            ar_brand.setdefault(p["brand"], Counter())[first] += 1
    arabic_hints = {
        "brand": {b: [w for w, _ in c.most_common()] for b, c in sorted(ar_brand.items())},
        "product_name": {p["name"]: p["name_ar"] for p in out_products if p.get("name_ar")},
    }

    allowed = OrderedDict([
        ("brand", brands),
        ("category", values("category")),
        ("product_group", values("product_group")),
        ("product_type", values("product_type")),
        ("product_form", values("product_form")),
        ("concerns", values("concerns")),
        ("suitable_for", values("suitable_for")),
        ("hero_ingredient", heroes),
        ("ingredients", ingredients),
    ])

    metadata = OrderedDict([
        ("source", raw.get("source")),
        ("scraped_at", raw.get("scraped_at")),
        ("product_count", len(out_products)),
        ("allowed_values", allowed),
        ("counts", {k: len(v) for k, v in allowed.items()}),
        ("hierarchy", hierarchy),
        ("product_line_to_brand", product_lines),
        ("brand_aliases", brand_aliases),
        ("hero_to_ingredients", hero_to_ingredients),
        ("field_map", FIELD_MAP),
        ("arabic_hints", arabic_hints),
    ])

    catalog = OrderedDict([
        ("source", raw.get("source")),
        ("company", raw.get("company")),
        ("scraped_at", raw.get("scraped_at")),
        ("currency", raw.get("currency")),
        ("products", out_products),
    ])
    return catalog, metadata, unreviewed, problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=RAW_PATH)
    ap.add_argument("--check", action="store_true", help="don't write; exit 1 if review is needed")
    args = ap.parse_args(argv)

    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    catalog, metadata, unreviewed, problems = build(raw, rules)

    print(f"products: {metadata['product_count']}")
    for k, n in metadata["counts"].items():
        print(f"  {k:16s} {n}")
    if problems:
        print("\nproblems:")
        for p in problems:
            print("  -", p)
    new_names = [n for n in unreviewed if raw_key(n) not in {i.lower() for i in metadata["allowed_values"]["ingredients"]}]
    if new_names:
        print(f"\n{len(new_names)} raw ingredient names kept as-is (add a rule to catalog_rules.json to merge them):")
        for n in sorted(new_names):
            print("  -", n)

    if args.check:
        return 1 if (problems or new_names) else 0

    PRODUCTS_OUT.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    METADATA_OUT.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {PRODUCTS_OUT.relative_to(ROOT)} and {METADATA_OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
