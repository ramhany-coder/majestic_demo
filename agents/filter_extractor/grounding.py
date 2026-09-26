"""Code-side grounding checks run by merge.py when the query is known.

Small models tend to *recommend* instead of *extract*: they name a product
the user never mentioned, add the product type they would suggest, or assume
a body area. The prompts already forbid this. These checks enforce it
deterministically, keeping a value only when something in the query (or,
for names, the recent context) supports it. Every drop is recorded in
meta.notes.
"""

from typing import List, Optional, Sequence, Tuple

from rapidfuzz import fuzz

from agents.filter_extractor.catalog import Catalog, normalize
from agents.filter_extractor.rule_based import EDGE_STOPWORDS, RuleExtractor

HEAD_NOUN_SCORE = 75
# suitable_for values that name a body area (the rest are skin types / groups).
AREA_VALUES = {"underarm", "scalp", "lips", "hands", "eye area", "body", "intimate area"}
_GENERIC_NAME_WORDS = {"and", "for", "the", "with", "of", "&", "+", "-", "in", "on", "x"}


def _fuzzy_in(word: str, tokens: Sequence[str], score: int = HEAD_NOUN_SCORE) -> bool:
    return any(word == t or (len(word) >= 4 and len(t) >= 3 and fuzz.ratio(word, t) >= score) for t in tokens)


class Grounder:
    def __init__(self, catalog: Catalog, rules: RuleExtractor):
        self.catalog = catalog
        self.rules = rules
        self.brand_aliases = {normalize(a): b for b, al in catalog.brand_aliases.items() for a in al}
        self.line_words = {normalize(l.replace("-", " ")): b for l, b in catalog.product_line_to_brand.items()}

    def _mentioned_brands(self, text: str) -> set:
        """Brands whose alias or product line appears verbatim (as whole words) in text."""
        padded = f" {' '.join(self.rules.tokens(text))} "
        found = {b for a, b in self.brand_aliases.items() if f" {a} " in padded}
        found |= {b for l, b in self.line_words.items() if f" {l} " in padded}
        return found

    # ---- names --------------------------------------------------------------
    def names(self, name_en: List[str], name_ar: List[str], query: str,
              context: Sequence[str], notes: List[str]) -> Tuple[List[str], List[str]]:
        grounded_brands = self._mentioned_brands(query + " " + " ".join(context or []))
        keep_en, keep_ar = [], []
        for i, n in enumerate(name_en):
            toks = [t for t in self.rules.tokens(n) if t not in _GENERIC_NAME_WORDS]
            name_brands = {self.brand_aliases[t] for t in toks if t in self.brand_aliases}
            name_brands |= {b for l, b in self.line_words.items() if l in " ".join(toks)}
            rest = [t for t in toks if t not in self.brand_aliases]
            if not rest and name_brands:
                notes.append(f"names: dropped {n!r} (brand only, not a product)")
                continue
            if not (name_brands & grounded_brands):
                notes.append(f"names: dropped {n!r} (no brand or product line of it in query/context)")
                continue
            keep_en.append(n)
            if i < len(name_ar):
                keep_ar.append(name_ar[i])
        # Arabic names beyond the English list (should not happen) are kept as-is.
        keep_ar += [a for a in name_ar[len(name_en):]]
        return keep_en, keep_ar

    # ---- ingredients ------------------------------------------------------
    def ingredient_values(self, values: List[str], query: str, field: str, notes: List[str]) -> List[str]:
        toks = self.rules.tokens(query)
        brand_toks = [t for t in toks if t in self.brand_aliases]
        text = " ".join(toks)
        out = []
        for v in values:
            vn = normalize(v)
            looks_like_brand = any(fuzz.ratio(vn.split()[0], b) >= 85 for b in brand_toks)
            if looks_like_brand and vn.split()[0] not in text.split():
                notes.append(f"ingredients: dropped {field} {v!r} (only the brand word is in the query)")
                continue
            out.append(v)
        return out

    # ---- product type -----------------------------------------------------
    def product_types(self, values: List[str], query: str, notes: List[str]) -> List[str]:
        toks = self.rules.tokens(query)
        by_rules = set(self.rules.product_type(query)["product_type"])
        out = []
        for v in values:
            words = [w for w in normalize(v.replace(",", " ")).split() if w not in EDGE_STOPWORDS and w != "&"]
            head = words[-1] if words else v
            if v in by_rules or _fuzzy_in(head, toks) or any(_fuzzy_in(w, toks, 90) for w in words if len(w) >= 5):
                out.append(v)
            else:
                notes.append(f"product_type: dropped {v!r} (not asked for: no {head!r} in query)")
        return out

    # ---- suitable_for areas -----------------------------------------------
    def suitable_for(self, values: List[str], query: str, notes: List[str]) -> List[str]:
        toks = self.rules.tokens(query)
        by_rules = set(self.rules.suitable_for(query)["suitable_for"])
        out = []
        for v in values:
            words = [w for w in normalize(v).split() if w != "area"]
            if v not in AREA_VALUES or v in by_rules or any(
                    _fuzzy_in(w, toks) or any(t.startswith(w) for t in toks) for w in words):
                out.append(v)
            else:
                notes.append(f"suitable_for: dropped area {v!r} (no area word in query)")
        return out


_GROUNDERS: dict = {}


def ground_brands(brands: List[str], query: str, context: Optional[Sequence[str]], matched_brands: set,
                  rules: RuleExtractor, notes: List[str]) -> List[str]:
    """Keep a brand the query mentions (per the rule-based brand matcher, which
    knows aliases, typos and product lines), the context mentions, or a matched
    product implies. Drops e.g. Capixy read out of the ingredient "capixyl"."""
    said = set(rules.brand(query)["brand"]) | set(rules.brand(" ".join(context or []))["brand"])
    out = []
    for b in brands:
        if b in said or b in matched_brands:
            out.append(b)
        else:
            notes.append(f"brand: dropped {b!r} (not mentioned in query/context)")
    return out


def ground(f, query: Optional[str], context: Optional[Sequence[str]], catalog: Catalog, rules: RuleExtractor,
           notes: List[str]) -> None:
    """Apply every check to a MetadataFilters in place (before name matching)."""
    if not query:
        return
    g = _GROUNDERS.get(id(catalog))
    if g is None or g.catalog is not catalog or g.rules is not rules:
        g = _GROUNDERS[id(catalog)] = Grounder(catalog, rules)
    f.name_en, f.name_ar = g.names(f.name_en, f.name_ar, query, context or [], notes)
    f.ingredients.include = g.ingredient_values(f.ingredients.include, query, "include", notes)
    f.ingredients.exclude = g.ingredient_values(f.ingredients.exclude, query, "exclude", notes)
    f.product_type = g.product_types(f.product_type, query, notes)
    f.suitable_for = g.suitable_for(f.suitable_for, query, notes)
