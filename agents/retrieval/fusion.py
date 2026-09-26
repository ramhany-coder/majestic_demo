"""Fusion: ranks the candidate set C.

    final = sem_score (cosine, 0..1)
          + available
          + best_seller
          + unmatched_include   per unmatched include term found in the description
          + unmatched_concern   per unmatched concern found in the description
          + extra_concern       per requested concern matched beyond the first
          + unmatched_exclude   per unmatched exclude term mentioned in the description (negative),
            or exclude_free instead when the text says "<term>-free" / "free from <term>" / "contains no <term>"
          + bundle              if the product is a bundle (negative)

Weights are RETRIEVAL_FUSION_WEIGHTS. Out-of-stock products always rank below
in-stock ones; ties break on name score, then lower price, then handle.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Optional

from agents.retrieval.index_builder import RetrievalIndex
from config import settings
from models.filter_extractor import MetadataFilters

_GAP = r"[^.;:!?]{0,80}?"   # a short run within one sentence, e.g. the rest of an ingredient list


@dataclass
class Ranked:
    handle: str
    score: float
    sem: float
    boosts: List[str] = field(default_factory=list)


def _term(term: str) -> str:
    """Regex for a term, singular or plural: "parabens" also finds "paraben"."""
    words = term.lower().split()
    last = words[-1]
    if len(last) > 3 and last.endswith("s") and not last.endswith("ss"):
        words[-1] = last[:-1]
    return r"(?<!\w)" + r"[\s-]+".join(re.escape(w) for w in words) + r"(?:s|es)?(?!\w)"


@lru_cache(maxsize=512)
def _patterns(term: str) -> tuple:
    t = _term(term)
    mention = re.compile(t)
    free = re.compile("|".join([
        t + r"[\s-]*free(?!\w)",                                   # silicone-free, alcohol free
        r"(?<!\w)free\s+(?:from|of)(?!\w)" + _GAP + t,             # free from ammonia, aluminum, and alcohol
        r"(?<!\w)(?:contains?\s+no|without)(?!\w)" + _GAP + t,     # contains no silicone, paraben, ... sulfate
        r"(?<!\w)(?:no|zero|0%)\s+(?:added\s+)?" + t,              # no parabens, 0% alcohol
    ]))
    return mention, free


def _stem(term: str) -> str:
    """A substring every match must contain: the first word, singular."""
    first = term.lower().split()[0]
    return first[:-1] if len(first) > 3 and first.endswith("s") and not first.endswith("ss") else first


def mentions(description: str, term: str) -> bool:
    term = term.strip()
    return bool(term) and _stem(term) in description and bool(_patterns(term)[0].search(description))


def says_free_of(description: str, term: str) -> bool:
    term = term.strip()
    return bool(term) and _stem(term) in description and bool(_patterns(term)[1].search(description))


def score_product(index: RetrievalIndex, handle: str, filters: Optional[MetadataFilters], sem: float,
                  weights: Dict[str, float]) -> Ranked:
    p = index.products[handle]
    desc = index.descriptions[handle]
    score, boosts = sem, []

    def add(key: str, label: str, times: int = 1) -> None:
        nonlocal score
        score += weights.get(key, 0.0) * times
        boosts.append(label)

    if p.get("available", True):
        add("available", "available")
    if p.get("best_seller"):
        add("best_seller", "best_seller")
    if handle in index.bundles:
        add("bundle", "bundle")
    if filters is not None:
        for term in filters.unmatched.include:
            if mentions(desc, term):
                add("unmatched_include", f"include:{term}")
        for term in filters.unmatched.concerns:
            if mentions(desc, term):
                add("unmatched_concern", f"concern:{term}")
        matched = len(set(filters.concerns) & set(p.get("concerns") or []))
        if matched > 1:
            add("extra_concern", f"extra_concerns:{matched - 1}", matched - 1)
        for term in filters.unmatched.exclude:
            if says_free_of(desc, term):
                add("exclude_free", f"free_of:{term}")
            elif mentions(desc, term):
                add("unmatched_exclude", f"mentions:{term}")
    return Ranked(handle, round(score, 6), round(sem, 6), boosts)


def rank(index: RetrievalIndex, candidates: Iterable[str], filters: Optional[MetadataFilters],
         sem_scores: Dict[str, float], name_scores: Optional[Dict[str, float]] = None,
         weights: Optional[Dict[str, float]] = None) -> List[Ranked]:
    weights = settings.RETRIEVAL_FUSION_WEIGHTS if weights is None else weights
    name_scores = name_scores or {}
    ranked = [score_product(index, h, filters, sem_scores.get(h, 0.0), weights) for h in candidates]

    def key(r: Ranked) -> tuple:
        p = index.products[r.handle]
        return (not p.get("available", True), -r.score, -name_scores.get(r.handle, 0.0),
                p.get("price") or 0.0, r.handle)

    return sorted(ranked, key=key)
