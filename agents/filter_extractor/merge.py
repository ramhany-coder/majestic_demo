"""Merges the 10 per-call outputs into one MetadataFilters and post-processes it
in code (never in the LLM):

1. Hierarchy check: a product_group / category that contradicts the chosen
   product_type(s) is dropped; so is a category that contradicts the chosen
   product_group(s). Parent values are never added.
2. Hero ingredients add their mapped ingredients to ingredients.include,
   unless an ingredient is excluded.
3. Exclude wins: an ingredient in both lists stays only in exclude.
4. Names are matched to product handles (IDF-weighted fuzzy token overlap,
   see NameMatcher; plain token_set_ratio >= 80 over-matched subsets), and a
   matched product's brand is added to `brand` when missing.
5. Grounding (when the query is passed in, right after step 1): grounding.py
   drops values nothing in the query supports (invented names, recommended
   product types, assumed body areas, brand words read as ingredients, a
   brand neither mentioned nor implied by a matched product).
6. Duplicates are removed; first-mention order is kept.
"""

import math
from typing import Dict, List, Optional, Sequence

from rapidfuzz import fuzz

from agents.filter_extractor.catalog import Catalog, get_catalog, normalize, normalize_ar
from agents.filter_extractor.grounding import ground, ground_brands
from agents.filter_extractor.rule_based import RuleExtractor, get_rule_extractor
from models.filter_extractor import IngredientFilter, MetadataFilters, UnmatchedTerms

NAME_COVERAGE = 0.8   # share of the name's IDF-weighted words a product must contain
NAME_MARGIN = 0.1     # keep products scoring within this of the best match
_NAME_STOP = {"and", "for", "the", "with", "of", "&", "+", "-", "in", "on", "x", "bundle", "kit", "offer", "pack", "set"}
_BUNDLE_HINTS = ("bundle", "kit", "+", " x ", "offer", "set of", "pack")


def dedupe(values) -> List[str]:
    seen, out = set(), []
    for v in values or []:
        if not v:
            continue
        k = v.casefold()
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out


def _allowed_parents(values: List[str], table: Dict[str, Dict[str, List[str]]], parent: str) -> Optional[set]:
    """Union of `parent` values permitted by the chosen child values (None = no constraint)."""
    known = [v for v in values if v in table]
    if not known:
        return None
    allowed = set()
    for v in known:
        allowed.update(table[v].get(parent, []))
    return allowed


def apply_hierarchy(f: MetadataFilters, catalog: Catalog, notes: List[str]) -> None:
    h = catalog.hierarchy
    by_type_group = _allowed_parents(f.product_type, h["product_type"], "product_group")
    by_type_cat = _allowed_parents(f.product_type, h["product_type"], "category")
    if by_type_group is not None:
        notes += [f"hierarchy: dropped product_group {g!r} (contradicts product_type)"
                  for g in f.product_group if g not in by_type_group]
        f.product_group = [g for g in f.product_group if g in by_type_group]
    if by_type_cat is not None:
        notes += [f"hierarchy: dropped category {c!r} (contradicts product_type)"
                  for c in f.category if c not in by_type_cat]
        f.category = [c for c in f.category if c in by_type_cat]
    by_group_cat = _allowed_parents(f.product_group, h["product_group"], "category")
    if by_group_cat is not None:
        notes += [f"hierarchy: dropped category {c!r} (contradicts product_group)"
                  for c in f.category if c not in by_group_cat]
        f.category = [c for c in f.category if c in by_group_cat]


class NameMatcher:
    """Matches a free-text product name to catalog handles.

    Token-set ratio alone scores "Capixy Tonic Spray" as a perfect match for
    "Capixy Intense Tonic Spray" (a subset), and puts "Capixy Lash Serum"
    closer to the hair serums than to "Capixy Lashes Treatment Serum". So each
    name is scored by IDF-weighted soft token overlap in both directions:
    rare words (tonic, lashes, hyalu) count more than brand or form words. A
    product must cover NAME_COVERAGE of the name's weight, and only products
    within NAME_MARGIN of the best score are kept. A name that is a product
    line ("Sebio-Control") matches every product of that line.
    """

    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.en = [(p, self._toks(p.norm)) for p in catalog.names]
        self.ar = [(p, self._toks(p.norm_ar)) for p in catalog.names if p.norm_ar]
        self.idf_en = self._idf([t for _, t in self.en])
        self.idf_ar = self._idf([t for _, t in self.ar])
        self.lines = {normalize(l.replace("-", " ")) for l in catalog.product_line_to_brand}
        self.brand_words = {normalize(a) for al in catalog.brand_aliases.values() for a in al}

    @staticmethod
    def _toks(text: str) -> List[str]:
        return [t for t in text.replace("%", " ").replace("-", " ").split() if t not in _NAME_STOP]

    @staticmethod
    def _idf(docs: List[List[str]]) -> Dict[str, float]:
        n = len(docs)
        df: Dict[str, int] = {}
        for d in docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        return {t: math.log((n + 1) / (c + 0.5)) for t, c in df.items()}

    @staticmethod
    def _sim(a: str, b: str) -> float:
        if a == b:
            return 1.0
        if min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)):
            return 0.9
        r = fuzz.ratio(a, b) / 100
        return r if r >= 0.8 else 0.0

    def _score(self, q: List[str], p: List[str], idf: Dict[str, float]) -> tuple:
        rare = max(idf.values()) if idf else 1.0

        def w(t):
            return idf.get(t, rare)  # an unseen word counts as rare

        qw = sum(w(t) for t in q) or 1.0
        pw = sum(w(t) for t in p) or 1.0
        name_cov = sum(w(t) * max((self._sim(t, x) for x in p), default=0.0) for t in q) / qw
        prod_cov = sum(w(x) * max((self._sim(x, t) for t in q), default=0.0) for x in p) / pw
        f1 = 2 * name_cov * prod_cov / (name_cov + prod_cov) if name_cov + prod_cov else 0.0
        return name_cov, f1

    def match(self, name: str, arabic: bool = False) -> List[str]:
        wants_bundle = any(hint in f" {name.lower()} " for hint in _BUNDLE_HINTS)
        if not arabic:
            n = normalize(name).replace("-", " ")
            core = " ".join(t for t in n.split() if t not in self.brand_words)
            if core in self.lines:
                return [p.handle for p, toks in self.en
                        if core in " ".join(toks) and (wants_bundle or not p.is_bundle)]
        q = self._toks(normalize_ar(name) if arabic else normalize(name))
        if not q:
            return []
        pool, idf = (self.ar, self.idf_ar) if arabic else (self.en, self.idf_en)
        scored = []
        for p, toks in pool:
            if p.is_bundle and not wants_bundle:
                continue
            cov, f1 = self._score(q, toks, idf)
            if cov >= NAME_COVERAGE:
                scored.append((f1, p.handle))
        if not scored:
            return []
        best = max(s for s, _ in scored)
        return [h for s, h in sorted(scored, key=lambda x: -x[0]) if s >= best - NAME_MARGIN]


_MATCHERS: Dict[int, NameMatcher] = {}


def _name_matcher(catalog: Catalog) -> NameMatcher:
    m = _MATCHERS.get(id(catalog))
    if m is None or m.catalog is not catalog:
        m = _MATCHERS[id(catalog)] = NameMatcher(catalog)
    return m


def match_names(name_en: List[str], name_ar: List[str], catalog: Catalog) -> List[str]:
    m = _name_matcher(catalog)
    handles: List[str] = []
    for n in name_en:
        handles += m.match(n)
    if not handles:
        # Arabic names only decide when the English ones matched nothing.
        for n in name_ar:
            handles += m.match(n, arabic=True)
    return dedupe(handles)


def merge_outputs(outputs: Dict[str, dict], catalog: Optional[Catalog] = None,
                  query: Optional[str] = None, context: Optional[Sequence[str]] = None,
                  rules: Optional[RuleExtractor] = None) -> MetadataFilters:
    """`outputs` maps call key -> validated output dict (missing key = empty).
    Passing `query` (and `context`) also runs the grounding checks."""
    if catalog is None:
        catalog = get_catalog()
        rules = rules or get_rule_extractor()
    elif rules is None and query:
        rules = RuleExtractor(catalog)
    o = {k: v or {} for k, v in outputs.items()}
    names = o.get("names", {})
    ing = o.get("ingredients", {})
    concerns = o.get("concerns", {})
    notes: List[str] = []

    name_en = [n for n in names.get("name_en", []) if n]
    name_ar = list(names.get("name_ar", []))[: len(name_en)] + list(names.get("name_ar", []))[len(name_en):]
    f = MetadataFilters(
        name_en=name_en,
        name_ar=name_ar,
        brand=dedupe(o.get("brand", {}).get("brand")),
        category=dedupe(o.get("category", {}).get("category")),
        product_group=dedupe(o.get("product_group", {}).get("product_group")),
        product_type=dedupe(o.get("product_type", {}).get("product_type")),
        product_form=dedupe(o.get("product_form", {}).get("product_form")),
        concerns=dedupe(concerns.get("concerns")),
        suitable_for=dedupe(o.get("suitable_for", {}).get("suitable_for")),
        hero_ingredient=dedupe(o.get("hero_ingredient", {}).get("hero_ingredient")),
        ingredients=IngredientFilter(include=dedupe(ing.get("include")), exclude=dedupe(ing.get("exclude"))),
        unmatched=UnmatchedTerms(
            concerns=dedupe(concerns.get("unmatched")),
            include=dedupe(ing.get("unmatched_include")),
            exclude=dedupe(ing.get("unmatched_exclude")),
        ),
    )

    # Hierarchy first, on the types the model chose: grounding may later drop a
    # recommended type, and that must not take a stated group down with it.
    apply_hierarchy(f, catalog, notes)
    if query:
        ground(f, query, context, catalog, rules, notes)
    f.name_en = dedupe(f.name_en)
    f.name_ar = dedupe(f.name_ar)

    excluded = {e.casefold() for e in f.ingredients.exclude}
    for hero in f.hero_ingredient:
        for i in catalog.hero_to_ingredients.get(hero, []):
            if i.casefold() not in excluded:
                f.ingredients.include.append(i)
    f.ingredients.include = [i for i in dedupe(f.ingredients.include) if i.casefold() not in excluded]
    un_ex = {e.casefold() for e in f.unmatched.exclude}
    f.unmatched.include = [i for i in f.unmatched.include if i.casefold() not in un_ex]

    f.matched_handles = match_names(f.name_en, f.name_ar, catalog)
    by_handle = {p.handle: p for p in catalog.names}
    if query:
        matched_brands = {by_handle[h].brand for h in f.matched_handles}
        f.brand = ground_brands(f.brand, query, context, matched_brands, rules, notes)
    for hnd in f.matched_handles:
        b = by_handle[hnd].brand
        if b not in f.brand:
            f.brand.append(b)

    f.meta.notes = notes
    return f
