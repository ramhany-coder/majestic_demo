"""Per-chat state for the pre-qualification stage, and how it is formatted
into the two prompts.

    store = get_session_store()
    ctx = store.get(session_id)            # a copy; edits don't leak until save()
    ...
    ctx.add_turn(message, reply, shown_product_names)
    store.save(session_id, ctx)

The store is in-process (the extractor's TTLCache). Swap it for Redis or
similar behind the same get/save interface when running several workers.
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from agents.filter_extractor.cache import TTLCache
from config import settings

# Messages kept per session. Only the last PREQUAL_HISTORY_MESSAGES are sent
# to the LLM; a few more are kept for the responder.
MAX_STORED_MESSAGES = 20
NO_HISTORY = "(none)"
NO_PRODUCTS = "none"


@dataclass
class SessionContext:
    history: List[Dict[str, str]] = field(default_factory=list)   # {"role": "user"|"assistant", "content"}
    last_products: List[str] = field(default_factory=list)         # English names shown last turn
    last_filters: Optional[dict] = None                            # debug only, never sent to the LLM

    def add_turn(self, user_message: str, reply: str, shown_products: Optional[Sequence[str]] = None) -> None:
        """Append the exchange. `shown_products=None` means the turn showed no
        product list (e.g. "thanks"), so last_products keeps pointing at what
        is still on screen; a list (even empty) replaces it."""
        self.history.append({"role": "user", "content": user_message or ""})
        self.history.append({"role": "assistant", "content": reply or ""})
        self.history = self.history[-MAX_STORED_MESSAGES:]
        if shown_products is not None:
            self.last_products = [p for p in shown_products if p][: settings.PREQUAL_LAST_PRODUCTS]


def _cut(text: str, limit: int) -> str:
    text = " ".join((text or "").split())      # one line per message
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def trim_history(history: Sequence[Dict[str, str]], n_messages: Optional[int] = None,
                 user_chars: Optional[int] = None, assistant_chars: Optional[int] = None) -> List[Dict[str, str]]:
    """Last `n_messages`, oldest first, each cut to its role's character limit."""
    n = settings.PREQUAL_HISTORY_MESSAGES if n_messages is None else n_messages
    u = settings.PREQUAL_HISTORY_USER_CHARS if user_chars is None else user_chars
    a = settings.PREQUAL_HISTORY_ASSISTANT_CHARS if assistant_chars is None else assistant_chars
    recent = list(history)[-n:] if n > 0 else []
    return [
        {"role": m.get("role", "user"), "content": _cut(m.get("content", ""), a if m.get("role") == "assistant" else u)}
        for m in recent
    ]


def format_history(history: Sequence[Dict[str, str]]) -> str:
    lines = [f"{'A' if m['role'] == 'assistant' else 'U'}: {m['content']}" for m in trim_history(history)]
    return "\n".join(lines) if lines else NO_HISTORY


def format_last_products(names: Sequence[str]) -> str:
    names = [n for n in names if n][: settings.PREQUAL_LAST_PRODUCTS]
    return " ".join(f"{i}) {n}" for i, n in enumerate(names, 1)) if names else NO_PRODUCTS


class SessionStore:
    def __init__(self, ttl_s: Optional[float] = None, maxsize: Optional[int] = None):
        self._cache = TTLCache(maxsize=maxsize or settings.SESSION_MAX_SESSIONS,
                               ttl_s=ttl_s or settings.SESSION_TTL_S)

    def get(self, session_id: str) -> SessionContext:
        ctx = self._cache.get(session_id)
        return copy.deepcopy(ctx) if ctx is not None else SessionContext()

    def save(self, session_id: str, ctx: SessionContext) -> None:
        self._cache.set(session_id, copy.deepcopy(ctx))

    def clear(self) -> None:
        self._cache.clear()


_store = SessionStore()


def get_session_store() -> SessionStore:
    return _store
