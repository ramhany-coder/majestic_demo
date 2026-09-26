"""Call A: rewrite the latest message (+ history) into one standalone English
request for the metadata extractor."""

import re
from typing import Dict, Optional, Sequence

from agents.prequal.llm_call import CallResult, run_structured
from agents.prequal.prompts import get_template
from agents.prequal.schemas import REWRITER_SCHEMA, validate_rewrite
from config import settings

_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
# A token mixing letters with the Arabizi digits (3ayza, 7amel, sha3r, wa2e3).
_ARABIZI_DIGIT_RE = re.compile(r"^(?=.*[a-z])(?=.*[235789])[a-z235789]+$")
# Letter+digit tokens that are English, not Arabizi (b3, spf50, 5ml, 3d, q10).
_NOT_ARABIZI_RE = re.compile(r"^(b\d+|spf\d+|q\d+|\d+(ml|gm|g|mg|l|d|x|st|nd|rd|th|pcs))$")
_ARABIZI_WORDS = {
    "ana", "enta", "enty", "enti", "howa", "heya", "ezay", "ezzay", "ezai", "leh", "leih", "feh", "fih", "fe",
    "fi", "mafish", "mafeesh", "mesh", "msh", "mish", "keda", "kda", "bta3", "beta3", "momken", "mumkin",
    "lel", "lil", "wel", "ely", "elly", "eli", "dah", "deh", "dih", "bas", "bs", "kaman", "tayeb",
    "ya3ni", "yaani", "awi", "awy", "khales", "gamed", "wesh", "shaar", "bashra", "el", "dy", "wala",
    "walla", "habibi", "shokran", "sokran", "ahlan", "zay", "zai", "gheir", "ghir", "gher", "3ayez",
}  # words that are also English ("men", "min", "tab", "da") are left out
_LETTER_SHARE = 0.8


def detect_language(text: str) -> str:
    """Script-based guess used when the rewriter fails: ar | arabizi | en | mixed."""
    arabic = len(_ARABIC_RE.findall(text or ""))
    latin = len(_LATIN_RE.findall(text or ""))
    if not arabic and not latin:
        return "en"
    if arabic / (arabic + latin) >= _LETTER_SHARE:
        return "ar"
    if latin / (arabic + latin) < _LETTER_SHARE:
        return "mixed"
    tokens = _TOKEN_RE.findall(text.lower())
    digit_hits = sum(1 for t in tokens if _ARABIZI_DIGIT_RE.match(t) and not _NOT_ARABIZI_RE.match(t))
    word_hits = sum(1 for t in tokens if t in _ARABIZI_WORDS)
    return "arabizi" if digit_hits >= 1 or word_hits >= 2 else "en"


def rewrite_default(message: str, history: Sequence[Dict[str, str]] = ()) -> dict:
    """Rewriter failed: pass the raw message on. The metadata filters are only
    trusted for a standalone English message; anything else (Arabic, Arabizi,
    or a follow-up whose meaning lives in the history) skips them and
    retrieval falls back to text search."""
    language = detect_language(message)
    return {
        "query_en": message,
        "language": language,
        "is_follow_up": False,
        "skip_metadata_filters": not (language == "en" and not history),
    }


def _validate(data: dict) -> tuple:
    out, notes = validate_rewrite(data)
    out["skip_metadata_filters"] = False
    return out, notes


async def rewrite(message: str, history: Sequence[Dict[str, str]] = (),
                  last_products: Sequence[str] = (), routes: Optional[list] = None) -> CallResult:
    return await run_structured(
        "rewriter",
        get_template("rewriter").messages(message, history, last_products),
        REWRITER_SCHEMA,
        _validate,
        lambda: rewrite_default(message, history),
        max_tokens=settings.PREQUAL_REWRITER_MAX_TOKENS,
        primary_timeout_s=settings.PREQUAL_REWRITER_TIMEOUT_S,
        deadline_s=settings.PREQUAL_REWRITER_DEADLINE_S,
        routes=routes,
    )
