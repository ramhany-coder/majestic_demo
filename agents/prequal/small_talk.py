"""Greetings, thanks and goodbyes: detect them and reply without the rest of the graph.

    is_small_talk("السلام عليكم")          # True
    small_talk_reply("thanks a lot!")      # "You're welcome. I'm here if you need anything else."

A message made only of these words (or only emoji / punctuation) ends the
turn in prequal with no LLM call. When the router tags a longer message as
intent=greeting, the same templated reply is used and the turn ends too.
"""

import re
from typing import Optional

from agents.filter_extractor.catalog import normalize_ar

KIND_GREETING = "greeting"
KIND_THANKS = "thanks"
KIND_BYE = "bye"
KIND_ACK = "ack"          # only filler ("ok", "tamam") or emoji

_GREETING_WORDS = {
    # English
    "hi", "hii", "hiii", "hello", "helo", "hey", "heyy", "hiya", "yo", "good", "morning", "evening",
    "afternoon", "howdy", "greetings",
    # Arabizi
    "ahlan", "ahln", "salam", "salaam", "salamo", "alaykom", "3aleko", "3alekom", "alaikum", "marhaba",
    "sabah", "masa", "kheir", "kher", "elkheir", "ezayak", "ezayek", "ezzayak", "ezzayek", "ezay",
    "3amel", "3amla", "eh", "eih", "hay",
    # Arabic (after normalize_ar: أإآ -> ا, ة -> ه, ى -> ي)
    "السلام", "عليكم", "وعليكم", "سلام", "اهلا", "مرحبا", "هاي", "هلو", "صباح", "مساء", "الخير", "النور",
    "ازيك", "ازيكم", "عامل", "عامله", "ايه",
}
_THANKS_WORDS = {
    "thanks", "thank", "thx", "thnx", "ty", "cheers", "appreciate", "appreciated",
    "shokran", "shukran", "sokran", "merci", "mersi", "tislam", "teslam", "teslami",
    "شكرا", "متشكر", "متشكره", "متشكرين", "تسلم", "تسلمي", "ميرسي", "مرسي", "الف",
}
_BYE_WORDS = {
    "bye", "goodbye", "cya", "later", "night", "salama", "ma3a", "ma3", "باي", "السلامه", "مع",
}
# Neutral filler: allowed in small talk but does not decide its kind.
_FILLER_WORDS = {
    "you", "u", "so", "very", "much", "a", "lot", "alot", "there", "ok", "okay", "k", "great", "perfect",
    "nice", "cool", "welcome", "dear", "all", "everyone", "sir", "doctor", "doc", "again", "and", "awesome",
    "fine", "got", "it", "sure", "how", "are", "r", "ya", "el", "w", "tamam", "gamed", "habibi", "ya3ni",
    "يا", "دكتور", "دكتوره", "تمام", "جدا", "اوي", "و", "كتير", "حبيبي", "جميل",
}
_SMALL_TALK_WORDS = _GREETING_WORDS | _THANKS_WORDS | _BYE_WORDS | _FILLER_WORDS

_WORD_RE = re.compile(r"[a-z0-9]+|[ء-ي]+")
_ARABIC_RE = re.compile(r"[؀-ۿ]")
_LATIN_ARABIZI = {
    "ahlan", "ahln", "salamo", "3aleko", "3alekom", "sabah", "masa", "kheir", "kher", "elkheir", "ezayak",
    "ezayek", "ezzayak", "ezzayek", "ezay", "3amel", "3amla", "shokran", "shukran", "sokran", "tislam",
    "teslam", "teslami", "salama", "ma3a", "ma3", "tamam", "gamed", "habibi", "ya3ni", "marhaba",
}

# The widget speaking on its own: no exclamation marks, gender-neutral
# wording, and Jamila in the feminine (web/README.md, Voice).
REPLIES = {
    "ar": {
        KIND_GREETING: "أهلاً، أنا جميلة. أقدر أساعدك في إيه النهارده؟",
        KIND_THANKS: "العفو. لو فيه أي حاجة تانية، أنا موجودة.",
        KIND_BYE: "مع السلامة، نتكلم تاني في أي وقت.",
        KIND_ACK: "تمام، أقدر أساعدك في إيه كمان؟",
    },
    "arabizi": {
        KIND_GREETING: "Ahlan, ana Jamila. A2dar asa3dak fe eh el naharda?",
        KIND_THANKS: "El 3afw. Law fe ay 7aga tanya, ana mawgooda.",
        KIND_BYE: "Ma3a el salama, netkallem tany fe ay wa2t.",
        KIND_ACK: "Tamam, a2dar asa3dak fe eh kaman?",
    },
    "en": {
        KIND_GREETING: "Hello, I'm Jamila. How can I help today?",
        KIND_THANKS: "You're welcome. I'm here if you need anything else.",
        KIND_BYE: "Goodbye, take care.",
        KIND_ACK: "Sure, what else can I help you with?",
    },
}


def _words(text: str) -> list:
    """Arabic words are folded with normalize_ar; Latin ones only lowercased
    (normalize_ar would drop the Arabizi digits: "3aleko" -> "aleko")."""
    return [normalize_ar(w) if _ARABIC_RE.match(w) else w for w in _WORD_RE.findall((text or "").lower())]


def is_small_talk(text: str) -> bool:
    """Only greeting / thanks / goodbye / filler words, or only emoji and
    punctuation. Anything else ("hi, any sunscreen?") goes through the graph."""
    text = (text or "").strip()
    if not text:
        return False
    words = _words(text)
    if not words:
        return not any(ch.isalnum() for ch in text)
    return all(w in _SMALL_TALK_WORDS for w in words)


def small_talk_kind(text: str) -> str:
    """bye > thanks > greeting > ack ("thanks, bye" is a goodbye). A longer
    message the router tagged as a greeting and that has none of these words
    gets the greeting reply."""
    words = set(_words(text))
    if words & _BYE_WORDS:
        return KIND_BYE
    if words & _THANKS_WORDS:
        return KIND_THANKS
    if words & _GREETING_WORDS or not is_small_talk(text):
        return KIND_GREETING
    return KIND_ACK


def small_talk_language(text: str) -> str:
    """ar | arabizi | en for a small-talk message (the rewriter's detector
    needs two Arabizi words, too strict for a lone "ahlan")."""
    if _ARABIC_RE.search(text or ""):
        return "ar"
    return "arabizi" if set(_words(text)) & _LATIN_ARABIZI else "en"


# Standalone English for query_en when no rewriter ran.
QUERY_EN = {KIND_GREETING: "Hello.", KIND_THANKS: "Thank you.", KIND_BYE: "Goodbye.", KIND_ACK: "OK."}


def small_talk_query_en(text: str) -> str:
    return QUERY_EN[small_talk_kind(text)]


def small_talk_reply(text: str, language: Optional[str] = None) -> str:
    language = language or small_talk_language(text)
    replies = REPLIES.get("ar" if language == "mixed" else language, REPLIES["en"])
    return replies[small_talk_kind(text)]
