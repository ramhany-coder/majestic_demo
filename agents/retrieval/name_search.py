"""Name step: resolve the extractor's name_en / name_ar to products, even when
they are misspelled ("capixi dray foom" -> Capixy Intense Dry Foam).

- Names are normalized (sizes, punctuation and, for Arabic, letter variants
  and diacritics removed) and split into character bigrams per word, with
  boundary markers: "capixy foam" -> ^c ca ap pi ix xy y$ ^f fo oa am m$.
- One BM25 index per language scores a queried name against every product
  name. English names go to the English index and Arabic names to the Arabic
  one (by script, so an Arabic string in name_en still finds its product).
- A product is accepted as a hit only if its BM25 score is at least
  RETRIEVAL_NAME_REL_SCORE x the top score for that queried name AND its
  bigram Dice coefficient is at least RETRIEVAL_NAME_DICE_MIN. BM25 scores are
  unbounded, so the Dice check rejects the case where every candidate is poor.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from config import settings

_EN_SIZE_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:ml|gm|g|caps|tabs)\b")
_AR_SIZE_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:مل|جم|جرام|ml|gm|g)\b")
_NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)
_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_TASHKEEL_RE = re.compile(r"[ً-ٰٟـ]")   # harakat, superscript alef, tatweel

_AR_FOLD = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ة": "ه", "ى": "ي", "ؤ": "و", "ئ": "ي",
    # "v" is written ڤ, ڨ, ڈ or ڄ across the scrape.
    "ڤ": "ف", "ڨ": "ف", "ڈ": "ف", "ڄ": "ف",
    **{chr(0x0660 + d): str(d) for d in range(10)},   # Arabic-Indic digits
    **{chr(0x06F0 + d): str(d) for d in range(10)},   # Persian digits
})


def _strip_symbols(text: str) -> str:
    """®, ™, ×, + and other symbol characters -> spaces. Runs before Unicode
    normalization, which would otherwise turn ™ into the letters "TM"."""
    return "".join(" " if unicodedata.category(ch).startswith("S") else ch for ch in text or "")


def is_arabic(text: str) -> bool:
    return bool(_ARABIC_RE.search(text or ""))


def normalize_name_en(name: str) -> str:
    """'Capixy Intense Dry Foam 120 ml' -> 'capixy intense dry foam';
    'Vacation Piña Colada ... 50 ml' -> 'vacation pina colada ...'."""
    text = unicodedata.normalize("NFKD", _strip_symbols(name)).lower()
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _EN_SIZE_RE.sub(" ", text)
    return " ".join(_NON_WORD_RE.sub(" ", text).split())


def normalize_name_ar(name: str) -> str:
    """Unify alef/teh-marbuta/yeh and the "v" letters, drop tashkeel, tatweel,
    sizes (مل / جم) and symbols: 'ڤاكيشن سيروم ڤيتامين سي 10% – 30 مل' ->
    'فاكيشن سيروم فيتامين سي 10'."""
    text = _TASHKEEL_RE.sub("", unicodedata.normalize("NFKC", _strip_symbols(name)).lower())
    text = _AR_SIZE_RE.sub(" ", text.translate(_AR_FOLD))
    return " ".join(_NON_WORD_RE.sub(" ", text).split())


def normalize_name(name: str, lang: str) -> str:
    return normalize_name_ar(name) if lang == "ar" else normalize_name_en(name)


def bigrams(text: str) -> List[str]:
    """Character bigrams per word with boundary markers (the input must
    already be normalized)."""
    out: List[str] = []
    for word in text.split():
        padded = f"^{word}$"
        out.extend(padded[i:i + 2] for i in range(len(padded) - 1))
    return out


def dice(a: frozenset, b: frozenset) -> float:
    return 2 * len(a & b) / (len(a) + len(b)) if a and b else 0.0


@dataclass
class NameIndex:
    """BM25 over the bigrams of one language's product names."""
    lang: str
    handles: List[str]
    texts: List[str]              # normalized names
    grams: List[frozenset]        # bigram set per name, for the Dice check
    bm25: Optional[BM25Okapi]
    # bigram -> its BM25 contribution to every name, precomputed from the fitted
    # rank_bm25 model. get_scores() loops over every document in Python for each
    # query bigram (~1.3 ms per name); summing these vectors gives the same
    # scores in a few microseconds.
    term_scores: Dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def build(cls, lang: str, items: Sequence[Tuple[str, str]], k1: Optional[float] = None,
              b: Optional[float] = None) -> "NameIndex":
        """`items` are (handle, raw name) pairs; empty names are skipped."""
        handles, texts, docs = [], [], []
        for handle, raw in items:
            text = normalize_name(raw, lang)
            if text:
                handles.append(handle)
                texts.append(text)
                docs.append(bigrams(text))
        bm25 = BM25Okapi(docs, k1=settings.RETRIEVAL_BM25_K1 if k1 is None else k1,
                         b=settings.RETRIEVAL_BM25_B if b is None else b) if docs else None
        return cls(lang, handles, texts, [frozenset(d) for d in docs], bm25, _term_scores(bm25))

    def score(self, query_text: str) -> np.ndarray:
        """Same result as bm25.get_scores(bigrams(query_text)): a repeated
        query bigram counts once per occurrence, an unknown one adds nothing."""
        out = np.zeros(len(self.handles))
        for gram in bigrams(query_text) if self.bm25 is not None else ():
            vec = self.term_scores.get(gram)
            if vec is not None:
                out += vec
        return out


def _term_scores(bm25: Optional[BM25Okapi]) -> Dict[str, np.ndarray]:
    if bm25 is None:
        return {}
    doc_len = np.asarray(bm25.doc_len, dtype=float)
    norm = bm25.k1 * (1 - bm25.b + bm25.b * doc_len / bm25.avgdl)
    out = {}
    for gram, idf in bm25.idf.items():
        tf = np.array([doc.get(gram, 0) for doc in bm25.doc_freqs], dtype=float)
        out[gram] = idf * tf * (bm25.k1 + 1) / (tf + norm)
    return out


@dataclass
class NameHit:
    handle: str
    score: float     # BM25 / top BM25 for the queried name, so 1.0 is the best match
    bm25: float
    dice: float
    query: str
    lang: str


@dataclass
class NameSearchResult:
    hits: List[NameHit] = field(default_factory=list)            # accepted, best per product, best first
    unresolved: List[str] = field(default_factory=list)          # queried names with no accepted hit
    soft: Dict[str, float] = field(default_factory=dict)         # every product's best normalized score
    per_query: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def handles(self) -> List[str]:
        return [h.handle for h in self.hits]


def match_name(query: str, index: NameIndex, *, rel: Optional[float] = None, dice_min: Optional[float] = None,
               max_hits: Optional[int] = None) -> Tuple[List[NameHit], Dict[str, float]]:
    """Accepted hits for one queried name (best first) plus every product's
    normalized score, for tie-breaking."""
    rel = settings.RETRIEVAL_NAME_REL_SCORE if rel is None else rel
    dice_min = settings.RETRIEVAL_NAME_DICE_MIN if dice_min is None else dice_min
    max_hits = settings.RETRIEVAL_NAME_MAX_HITS if max_hits is None else max_hits
    text = normalize_name(query, index.lang)
    scores = index.score(text)
    top = float(scores.max()) if len(scores) else 0.0
    if top <= 0:
        return [], {}
    qgrams = frozenset(bigrams(text))
    soft = {h: float(s) / top for h, s in zip(index.handles, scores) if s > 0}
    hits = []
    for i in np.argsort(-scores, kind="stable"):
        s = float(scores[i])
        if s < rel * top:
            break
        d = dice(qgrams, index.grams[i])
        if d >= dice_min:
            hits.append(NameHit(index.handles[i], s / top, s, d, query, index.lang))
    hits.sort(key=lambda h: (-h.score, -h.dice, h.handle))
    return hits[:max_hits], soft


def search_names(names_en: Sequence[str], names_ar: Sequence[str], en_index: NameIndex, ar_index: NameIndex,
                 **kw) -> NameSearchResult:
    """name_en[i] and name_ar[i] are the same name in two scripts (the names
    call keeps them in the same order). A name is unresolved only when
    neither form has an accepted hit."""
    res = NameSearchResult()
    best: Dict[str, NameHit] = {}
    n = max(len(names_en), len(names_ar))
    for i in range(n):
        forms = [q for q in (names_en[i] if i < len(names_en) else "", names_ar[i] if i < len(names_ar) else "")
                 if q and q.strip()]
        resolved = False
        for q in forms:
            if q in res.per_query:
                resolved = resolved or bool(res.per_query[q])
                continue
            hits, soft = match_name(q, ar_index if is_arabic(q) else en_index, **kw)
            res.per_query[q] = [h.handle for h in hits]
            resolved = resolved or bool(hits)
            for h, s in soft.items():
                res.soft[h] = max(res.soft.get(h, 0.0), s)
            for h in hits:
                if h.handle not in best or (h.score, h.dice) > (best[h.handle].score, best[h.handle].dice):
                    best[h.handle] = h
        if forms and not resolved:
            res.unresolved.append(forms[0])
    res.hits = sorted(best.values(), key=lambda h: (-h.score, -h.dice, h.handle))
    return res
