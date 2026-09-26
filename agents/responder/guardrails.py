"""Checks run in code on R1's answer (plan section 9).

- prices_ok:        every EGP amount and % in the answer appears in the data sent (or the query / store facts)
- unknown_ingredients: ingredients named in the answer that no product sent contains and the query doesn't name
- language_ok:      the answer is written in the reply language
- cut_to_limit:     over RESPONDER_LENGTH_FACTOR x the word limit -> cut at the last complete sentence
"""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Set

from agents.responder.context import hero_text

ROOT = Path(__file__).resolve().parents[2]

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫٬", "01234567890123456789.,")
_NUMBER = r"\d+(?:[.,]\d+)*"
_CURRENCY = r"(?:EGP|L\.?E\.?|pounds?|جنيه(?:ات|اً|ا)?|ج\.\s?م\.?|ج\s?م|ج\.)"
_PERCENT = r"(?:%|٪|\s?percent|\s?في\s?الم[يئ]ة|\s?بالم[يئ]ة)"
AMOUNT_RE = re.compile(
    rf"(?:{_CURRENCY}\s?({_NUMBER}))|(?:({_NUMBER})\s?(?:{_CURRENCY}|{_PERCENT}))", re.IGNORECASE)
ANY_NUMBER_RE = re.compile(_NUMBER)

_ARABIC_LETTER = re.compile(r"[ء-يٱ-ۓڤگ]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_SENTENCE_END = re.compile(r"[.!?؟…]\s|[.!?؟…]$|\n")


def _num(text: str) -> Optional[float]:
    text = text.replace(",", "")
    try:
        return round(float(text), 2)
    except ValueError:
        return None


def numbers_in(texts: Iterable[str]) -> Set[float]:
    out: Set[float] = set()
    for t in texts:
        for m in ANY_NUMBER_RE.findall((t or "").translate(_DIGITS)):
            n = _num(m)
            if n is not None:
                out.add(n)
    return out


def amounts_in(answer: str) -> List[float]:
    """EGP amounts and percentages written in the answer."""
    found = []
    for m in AMOUNT_RE.finditer((answer or "").translate(_DIGITS)):
        n = _num(m.group(1) or m.group(2))
        if n is not None:
            found.append(n)
    return found


def ungrounded_amounts(answer: str, allowed_texts: Iterable[str]) -> List[float]:
    allowed = numbers_in(allowed_texts)
    return [n for n in amounts_in(answer) if n not in allowed]


# ---------------------------------------------------------------- ingredients

@lru_cache(maxsize=1)
def ingredient_vocabulary() -> tuple:
    meta = json.loads((ROOT / "data" / "metadata_catalog.json").read_text(encoding="utf-8"))
    names = meta["allowed_values"].get("ingredients") or []
    # Longest first, so "Hyaluronic Acid" is matched before "Acid"-like short names.
    return tuple(sorted({n for n in names if len(n) >= 4}, key=len, reverse=True))


def _mentions(text: str, name: str) -> bool:
    return re.search(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", text, re.IGNORECASE) is not None


def unknown_ingredients(answer: str, products: List[dict], query_texts: Iterable[str]) -> List[str]:
    """Catalog ingredient names (English) in the answer that are not in any
    sent product's ingredients_canonical or hero ingredient, nor in the query.
    Arabic ingredient names are not checked."""
    own = {i.lower() for p in products for i in p.get("ingredients_canonical") or []}
    heroes = " ".join(hero_text(p) or "" for p in products).lower()
    query = " ".join(query_texts)
    out = []
    remaining = answer or ""
    for name in ingredient_vocabulary():
        if not _mentions(remaining, name):
            continue
        remaining = re.sub(re.escape(name), " ", remaining, flags=re.IGNORECASE)
        if name.lower() in own or name.lower() in heroes or _mentions(query, name):
            continue
        out.append(name)
    return out


# ---------------------------------------------------------------- language and length

def language_ok(answer: str, reply_lang: str) -> bool:
    """Arabic replies are mostly Arabic letters (product names may stay Latin).
    English and Arabizi replies must not be mostly Arabic: an English reply to
    a sales trainee carries a suggested line in Egyptian Arabic by design."""
    arabic = len(_ARABIC_LETTER.findall(answer or ""))
    latin = len(_LATIN_LETTER.findall(answer or ""))
    letters = arabic + latin
    if letters < 8:
        return True
    share = arabic / letters
    return share >= 0.4 if reply_lang == "ar" else share < 0.5


def word_count(text: str) -> int:
    return len((text or "").split())


def cut_to_limit(text: str, max_words: int) -> str:
    """Text within max_words, cut after the last complete sentence (else at the
    word limit, with an ellipsis)."""
    if word_count(text) <= max_words:
        return text
    spans = list(re.finditer(r"\S+", text))
    prefix = text[: spans[max_words - 1].end()]
    ends = [m.end() for m in _SENTENCE_END.finditer(prefix)]
    if ends and ends[-1] > len(prefix) // 3:
        return prefix[: ends[-1]].rstrip()
    return prefix.rstrip(" ,،;:-") + "…"


def max_words(word_limit: int, factor: float) -> int:
    return max(1, int(round(word_limit * factor)))
