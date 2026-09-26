"""Deterministic fallback extractor: alias dictionary + rapidfuzz over the
normalized query. Used per key when that key's LLM call fails or times out,
and usable offline (tests, eval baseline). One function per call key; each
returns exactly the shape the matching LLM call returns.

Matching: every 1-3 token n-gram of the normalized query is compared with the
key's aliases (allowed values, their parenthesised / '+' parts, brand aliases,
product lines, data/aliases.json). Exact match only for strings shorter than 4
characters, else rapidfuzz.fuzz.ratio >= 88. Longest n-grams win and matches
never overlap. Results keep the order the user mentioned things in.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from rapidfuzz import fuzz, process

from agents.filter_extractor.catalog import Catalog, get_catalog, normalize, normalize_ar

FUZZY_SCORE = 88
EXACT_BELOW = 4
MAX_NGRAM = 3

# Words that turn the next ingredient (or form) into an exclusion.
NEGATION_MARKERS = (
    ("without",), ("no",), ("not",), ("avoid",), ("avoiding",), ("except",), ("non",),
    ("free", "from"), ("free", "of"), ("allergic", "to"), ("allergy", "to"), ("allergic",),
    ("sensitive", "to"), ("cant", "use"), ("cannot", "use"), ("dont", "want"), ("hate",),
)
NEGATION_REACH = 4          # a marker applies to an ingredient at most this many tokens later
PRODUCT_WORDS = {"serum", "cream", "gel", "spray", "roll", "roll-on", "balm", "lotion", "foam",
                 "moisturizer", "cleanser", "toner", "oil", "mask", "stick"}
# An n-gram (n > 1) may not start or end on one of these, so "no salicylic acid"
# cannot fuzzy-match "salicylic acid" and swallow its negation marker.
EDGE_STOPWORDS = {"no", "not", "without", "free", "avoid", "a", "an", "the", "and", "or", "for", "with",
                  "of", "my", "i", "is", "to", "in", "on", "it", "its", "me", "some", "any", "that", "this"}
# For forms / types a marker only reaches over filler words ("not a heavy cream").
NEGATION_FILLERS = {"a", "an", "the", "too", "very", "heavy", "thick", "greasy", "sticky", "oily",
                    "any", "as", "like", "in", "some", "regular", "normal"}
NAME_GENERIC = {"ml", "gm", "and", "for", "the", "with", "of", "&", "+", "-", "x", "in", "on"}


@dataclass(frozen=True)
class Hit:
    start: int
    end: int          # exclusive token index
    value: str
    alias: str


class AliasTable:
    def __init__(self, entries: Dict[str, str]):
        self.exact = {a: v for a, v in entries.items() if a}
        self.fuzzy_keys = [a for a in self.exact if len(a) >= EXACT_BELOW]

    def lookup(self, gram: str) -> Optional[Tuple[str, str]]:
        if gram in self.exact:
            return self.exact[gram], gram
        if len(gram) < EXACT_BELOW or not self.fuzzy_keys:
            return None
        hit = process.extractOne(gram, self.fuzzy_keys, scorer=fuzz.ratio, score_cutoff=FUZZY_SCORE)
        return (self.exact[hit[0]], hit[0]) if hit else None


def _value_parts(value: str) -> List[str]:
    """'Niacinamide (Vitamin B3)' -> ['niacinamide (vitamin b3)', 'niacinamide', 'vitamin b3']."""
    parts = [value]
    inner = re.findall(r"\(([^)]*)\)", value)
    outer = re.sub(r"\([^)]*\)", " ", value)
    parts.append(outer)
    for chunk in inner:
        parts.extend(re.split(r",\s*", chunk))
    return [normalize(p) for p in parts if normalize(p)]


class RuleExtractor:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.tables: Dict[str, AliasTable] = {}
        for key, values in catalog.allowed.items():
            entries: Dict[str, str] = {}
            for v in values:
                for part in _value_parts(v):
                    entries.setdefault(part, v)
            if key == "brand":
                for brand, aliases in catalog.brand_aliases.items():
                    for a in aliases:
                        entries[normalize(a)] = brand
                for line, brand in catalog.product_line_to_brand.items():
                    entries[normalize(line)] = brand
                    entries[normalize(line.replace("-", " "))] = brand
            if key == "hero_ingredient":
                # Hero parts like "egcg" or "zinc pca" alone are too weak; keep full
                # names plus the curated aliases.
                entries = {normalize(v): v for v in values}
            for alias, v in catalog.aliases.get(key, {}).items():
                if v in values:
                    entries[alias] = v
            self.tables[key] = AliasTable(entries)
        self.unmatched_table = AliasTable(dict(catalog.unmatched_terms))
        self.form_table = self.tables["product_form"]
        self.line_table = AliasTable({normalize(l.replace("-", " ")): l for l in catalog.product_line_to_brand})
        self._name_df = self._build_name_df()
        # Brand words never match another key's aliases ("capixy" vs "Capixyl").
        brand_words = set(self.tables["brand"].exact)
        self.reserved = {id(t): brand_words for k, t in self.tables.items() if k != "brand"}
        self.reserved[id(self.unmatched_table)] = brand_words
        # ...and ingredient words never match a brand ("capixyl" vs "Capixy").
        self.reserved[id(self.tables["brand"])] = set(self.tables["ingredients"].exact) - brand_words

    # ----- helpers --------------------------------------------------------
    def tokens(self, query: str) -> List[str]:
        text = normalize(query).replace("-free", " free").replace("-", " ")
        out: List[str] = []
        for tok in text.split():
            out.extend(self.catalog.typos.get(tok, tok).split())
        return out

    def match(self, table: AliasTable, toks: Sequence[str]) -> List[Hit]:
        taken = [False] * len(toks)
        hits: List[Hit] = []
        for n in range(min(MAX_NGRAM, len(toks)), 0, -1):
            for i in range(len(toks) - n + 1):
                if any(taken[i:i + n]):
                    continue
                if n > 1 and (toks[i] in EDGE_STOPWORDS or toks[i + n - 1] in EDGE_STOPWORDS):
                    if " ".join(toks[i:i + n]) not in table.exact:
                        continue
                gram = " ".join(toks[i:i + n])
                if gram in self.reserved.get(id(table), ()):
                    continue
                found = table.lookup(gram)
                if found:
                    hits.append(Hit(i, i + n, found[0], found[1]))
                    for j in range(i, i + n):
                        taken[j] = True
        return sorted(hits, key=lambda h: h.start)

    @staticmethod
    def _unique(values):
        seen, out = set(), []
        for v in values:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    def _values(self, key: str, toks: Sequence[str]) -> List[str]:
        return self._unique(h.value for h in self.match(self.tables[key], toks))

    @staticmethod
    def _marker_ends(toks: Sequence[str]) -> List[int]:
        """Token index right after each negation marker."""
        ends = []
        for i in range(len(toks)):
            for m in NEGATION_MARKERS:
                if tuple(toks[i:i + len(m)]) == m:
                    ends.append(i + len(m))
        return ends

    @staticmethod
    def _negated_hits(hits: List[Hit], marker_ends: List[int], toks: Sequence[str],
                      reach: int = NEGATION_REACH, fillers_only: bool = False) -> set:
        """Hits turned into exclusions. A marker flips only the next hit after it
        (within NEGATION_REACH tokens); the flip then runs along "X, Y and Z"
        chains. "alcohol free" / "paraben-free" flips the hit before "free"."""
        ordered = sorted(hits, key=lambda h: h.start)
        negated = set()
        for end in marker_ends:
            nxt = next((h for h in ordered if h.start >= end), None)
            if nxt is not None and nxt.start - end <= reach and (
                    not fillers_only or all(t in NEGATION_FILLERS for t in toks[end:nxt.start])):
                negated.add(nxt)
        for h in ordered:
            if h.end < len(toks) and toks[h.end] == "free" and (
                    h.end + 1 >= len(toks) or toks[h.end + 1] not in {"from", "of"}):
                negated.add(h)
        for prev, h in zip(ordered, ordered[1:]):
            if prev in negated and h not in negated:
                between = toks[prev.end:h.start]
                if len(between) <= 1 and all(t in {"and", "or", "nor", "&"} for t in between):
                    negated.add(h)
        return negated

    # ----- one function per call ------------------------------------------
    def names(self, query: str, context: Optional[Sequence[str]] = None) -> dict:
        toks = self.tokens(query)
        brand_hits = self.match(self.tables["brand"], toks)
        brands = {h.value for h in brand_hits}
        lines = self._unique(h.value for h in self.match(self.line_table, toks))
        tokset = set(toks)
        chosen = []
        for brand in brands:
            candidates = [n for n in self.catalog.names if n.brand == brand and not n.is_bundle]
            scored = []
            for n in candidates:
                distinct = [t for t in n.norm.split() if t not in NAME_GENERIC and normalize(brand) != t]
                if not distinct:
                    continue
                matched = [t for t in distinct if t in tokset or any(
                    len(t) >= EXACT_BELOW and fuzz.ratio(t, q) >= 85 for q in tokset)]
                if not any(self._name_df.get((brand, t), 99) <= 2 for t in matched):
                    continue
                pos = min((i for i, q in enumerate(toks) if q in matched or any(
                    len(t) >= EXACT_BELOW and fuzz.ratio(t, q) >= 85 for t in matched)), default=len(toks))
                scored.append((frozenset(matched), len(matched) / len(distinct), n, pos))
            best: Dict[frozenset, tuple] = {}
            for key, cov, n, pos in scored:
                if cov >= 0.2 and (key not in best or cov > best[key][0]):
                    best[key] = (cov, n, pos)
            # A match whose tokens are a strict subset of another match's is the same mention.
            keys = list(best)
            for key in keys:
                if not any(key < other for other in keys):
                    chosen.append((best[key][2], best[key][1]))
        chosen = [n for _, n in sorted(chosen, key=lambda x: x[0])]
        name_en = [n.display for n in chosen]
        name_ar = [n.display_ar or "" for n in chosen]
        for line in lines:
            if not any(normalize(line.replace("-", " ")) in normalize(n.replace("-", " ")) for n in name_en):
                name_en.append(line)
                name_ar.append("")
        return {"name_en": name_en[:5], "name_ar": name_ar[:5]}

    def brand(self, query: str) -> dict:
        return {"brand": self._values("brand", self.tokens(query))}

    def category(self, query: str) -> dict:
        toks = self.tokens(query)
        cats = self._values("category", toks)
        h = self.catalog.hierarchy
        for t in self._values("product_type", toks):
            implied = h["product_type"].get(t, {}).get("category", [])
            if len(implied) == 1:
                cats.append(implied[0])
        for g in self._values("product_group", toks):
            implied = h["product_group"].get(g, {}).get("category", [])
            if len(implied) == 1:
                cats.append(implied[0])
        return {"category": self._unique(cats)}

    def product_group(self, query: str) -> dict:
        return {"product_group": self._values("product_group", self.tokens(query))}

    def product_type(self, query: str) -> dict:
        toks = self.tokens(query)
        markers = self._marker_ends(toks)
        hits = self.match(self.tables["product_type"], toks)
        negated = self._negated_hits(hits, markers, toks, reach=3, fillers_only=True)
        return {"product_type": self._unique(h.value for h in hits if h not in negated)}

    def product_form(self, query: str) -> dict:
        toks = self.tokens(query)
        markers = self._marker_ends(toks)
        hits = self.match(self.form_table, toks)
        # A marker inside the alias itself ("no rinse") is part of the phrase, not a negation.
        own = {h for h in hits if set(h.alias.split()) & {"no", "without", "dont"}}
        negated = self._negated_hits([h for h in hits if h not in own], markers, toks, reach=3, fillers_only=True)
        keep = [h.value for h in hits if h in own or h not in negated]
        return {"product_form": self._unique(keep)}

    def concerns(self, query: str) -> dict:
        return {"concerns": self._values("concerns", self.tokens(query)), "unmatched": []}

    def suitable_for(self, query: str) -> dict:
        return {"suitable_for": self._values("suitable_for", self.tokens(query))}

    def hero_ingredient(self, query: str) -> dict:
        toks = self.tokens(query)
        out = []
        for h in self.match(self.tables["hero_ingredient"], toks):
            window = toks[h.end:h.end + 2]
            if any(w in PRODUCT_WORDS for w in window):
                out.append(h.value)
        return {"hero_ingredient": self._unique(out)}

    def ingredients(self, query: str) -> dict:
        toks = self.tokens(query)
        markers = self._marker_ends(toks)
        ing_hits = self.match(self.tables["ingredients"], toks)
        taken = {i for h in ing_hits for i in range(h.start, h.end)}
        un_hits = [h for h in self.match(self.unmatched_table, toks) if not (set(range(h.start, h.end)) & taken)]
        all_hits = sorted(ing_hits + un_hits, key=lambda h: h.start)
        negated = self._negated_hits(all_hits, markers, toks)
        out = {"include": [], "exclude": [], "unmatched_include": [], "unmatched_exclude": []}
        for h in all_hits:
            is_un = h in un_hits
            bucket = ("unmatched_" if is_un else "") + ("exclude" if h in negated else "include")
            out[bucket].append(h.value)
        for k in out:
            out[k] = self._unique(out[k])[: (3 if k.startswith("unmatched") else 8)]
        # exclude wins inside a single call too
        out["include"] = [v for v in out["include"] if v not in out["exclude"]]
        return out

    # ----- name document frequency (per brand) -----------------------------
    def _build_name_df(self) -> Dict[tuple, int]:
        df: Dict[tuple, int] = {}
        for n in self.catalog.names:
            if n.is_bundle:
                continue
            for t in set(n.norm.split()):
                df[(n.brand, t)] = df.get((n.brand, t), 0) + 1
        return df

    def run(self, key: str, query: str, context: Optional[Sequence[str]] = None) -> dict:
        fn: Callable = getattr(self, key)
        return fn(query, context) if key == "names" else fn(query)


@lru_cache(maxsize=1)
def get_rule_extractor() -> RuleExtractor:
    return RuleExtractor(get_catalog())


def extract_rule_based(key: str, query: str, context: Optional[Sequence[str]] = None) -> dict:
    return get_rule_extractor().run(key, query, context)
