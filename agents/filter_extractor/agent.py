"""Orchestrator: one English user message -> MetadataFilters.

    filters = await extract_filters("need a sun blok spray for the beach")

Flow: trim to 500 chars -> (optional translate-to-English) -> greeting
short-circuit -> cache lookup -> 10 calls in parallel with
asyncio.gather(return_exceptions=True) -> merge + post-process -> cache.

`filter_extractor(state)` is the async graph-node form (same contract as the
drug-assistant nodes: read from state, return a partial state dict).
"""

import asyncio
import logging
import re
import time
from typing import Dict, List, Optional, Sequence

from agents.filter_extractor.cache import TTLCache
from agents.filter_extractor.calls import (
    CALL_FUNCTIONS, STATUS_FAILED, STATUS_SKIPPED, CallOutcome, translate_to_english,
)
from agents.filter_extractor.catalog import normalize
from agents.filter_extractor.merge import merge_outputs
from agents.filter_extractor.schemas import CALLS_BY_KEY, empty_output
from config import settings
from llm.client import EXTRACTOR_FALLBACK_ORDER
from llm.fallback import parse_route
from llm.llm_models import shared_async_http_client
from models.filter_extractor import MetadataFilters

logger = logging.getLogger("filter_extractor")

# A message made only of these words is small talk: no LLM call at all.
_SMALL_TALK = {
    "hi", "hii", "hiii", "hello", "helo", "hey", "heyy", "hiya", "yo", "salam", "salaam", "marhaba",
    "thanks", "thank", "thx", "thnx", "ty", "you", "u", "so", "very", "much", "a", "lot", "alot",
    "good", "morning", "evening", "afternoon", "night", "there", "ok", "okay", "k", "great",
    "perfect", "nice", "cool", "bye", "goodbye", "cheers", "welcome", "dear", "all", "everyone",
    "sir", "doctor", "doc", "again", "and", "awesome", "fine", "got", "it", "sure",
}
_WORD_RE = re.compile(r"[a-z]+")

MAX_CONTEXT_NAMES = 10

_cache = TTLCache(maxsize=settings.EXTRACTOR_CACHE_SIZE, ttl_s=settings.EXTRACTOR_CACHE_TTL_S)


def get_cache() -> TTLCache:
    return _cache


def is_small_talk(query: str) -> bool:
    text = query.lower()
    words = _WORD_RE.findall(text)
    if not words:
        return not any(ch.isalnum() for ch in text)  # only emoji / punctuation
    return all(w in _SMALL_TALK for w in words)


def _cache_key(query: str, context: Sequence[str]) -> tuple:
    """Case, punctuation and hyphen/space differences share one cache entry."""
    return " ".join(normalize(query).replace("-", " ").split()), tuple(context)


async def _gather_calls(query: str, context: Sequence[str]) -> List[CallOutcome]:
    keys = list(CALL_FUNCTIONS)
    results = await asyncio.gather(
        *(CALL_FUNCTIONS[k](query, context) for k in keys),
        return_exceptions=True,
    )
    outcomes = []
    for key, res in zip(keys, results):
        if isinstance(res, BaseException):
            # run_call never raises; this only guards against a bug in it.
            logger.error("[filter_extractor] call '%s' raised: %r", key, res)
            outcomes.append(CallOutcome(key, empty_output(CALLS_BY_KEY[key]), STATUS_FAILED, 0.0, notes=[repr(res)]))
        else:
            outcomes.append(res)
    return outcomes


async def extract_filters(query: Optional[str], context: Optional[Sequence[str]] = None,
                          use_cache: bool = True) -> MetadataFilters:
    """`query` is the user's message (English, or any language when
    EXTRACTOR_TRANSLATE is on). In the chat pipeline it is the prequal
    rewriter's `query_en`, which already names any product the user referred
    to, and no context is passed. `context` (product names from the last
    turns) is optional; no LLM call sees it, only grounding and the
    rule-based fallback read it."""
    start = time.perf_counter()
    query = (query or "").strip()[: settings.EXTRACTOR_MAX_QUERY_CHARS]
    context = [c for c in (context or []) if c][-MAX_CONTEXT_NAMES:]

    def _elapsed() -> float:
        return round((time.perf_counter() - start) * 1000, 1)

    if not query or is_small_talk(query):
        f = MetadataFilters()
        f.meta.short_circuit = "empty" if not query else "small_talk"
        f.meta.calls = {k: STATUS_SKIPPED for k in CALL_FUNCTIONS}
        f.meta.latency_ms = _elapsed()
        return f

    translate_status = None
    if settings.EXTRACTOR_TRANSLATE:
        cached_en = _cache.get(("__translate__", query)) if use_cache else None
        if cached_en is None:
            query_en, translate_status = await translate_to_english(query)
            if use_cache and translate_status != STATUS_FAILED:
                _cache.set(("__translate__", query), query_en)
        else:
            query_en, translate_status = cached_en, "cached"
        query = query_en[: settings.EXTRACTOR_MAX_QUERY_CHARS]

    key = _cache_key(query, context)
    if use_cache:
        hit = _cache.get(key)
        if hit is not None:
            f = hit.model_copy(deep=True)
            f.meta.cached = True
            f.meta.latency_ms = _elapsed()
            return f

    outcomes = await _gather_calls(query, context)
    f = merge_outputs({o.key: o.data for o in outcomes}, query=query, context=context)
    f.meta.calls = {o.key: o.status for o in outcomes}
    if translate_status:
        f.meta.calls["translate"] = translate_status
        f.meta.query_en = query
    f.meta.call_latency_ms = {o.key: o.latency_ms for o in outcomes}
    f.meta.call_outputs = {o.key: o.data for o in outcomes}
    f.meta.routes = {o.key: o.route for o in outcomes if o.route}
    f.meta.prompt_tokens = {o.key: o.prompt_tokens for o in outcomes if o.prompt_tokens}
    f.meta.notes = [f"{o.key}: {n}" for o in outcomes for n in o.notes] + f.meta.notes
    f.meta.latency_ms = _elapsed()

    degraded = {k: s for k, s in f.meta.calls.items() if s not in ("ok", "cached")}
    logger.info("[filter_extractor] %.0fms calls=%s", f.meta.latency_ms, degraded or "all ok")

    # Don't cache a result built from degraded calls: the next request may get the LLM back.
    if use_cache and not degraded:
        _cache.set(key, f.model_copy(deep=True))
    return f


_WARMUP_URLS = {"groq": "https://api.groq.com/openai/v1/models"}


async def warm_up(connections: int = len(CALL_FUNCTIONS)) -> int:
    """Pre-open `connections` pooled HTTPS connections to the primary provider
    with a cheap model-list request (no completion quota used), so the first
    real fan-out doesn't pay 10 TLS handshakes. Call once at app startup, inside
    the event loop that will serve requests. Returns how many succeeded."""
    router, _ = parse_route(EXTRACTOR_FALLBACK_ORDER[0])
    url = _WARMUP_URLS.get(router)
    client = shared_async_http_client()
    if not url or client is None or not settings.GROQ_API:
        return 0
    headers = {"Authorization": f"Bearer {settings.GROQ_API}"}
    results = await asyncio.gather(*(client.get(url, headers=headers) for _ in range(connections)),
                                   return_exceptions=True)
    return sum(1 for r in results if not isinstance(r, BaseException) and r.status_code == 200)


def _recent_product_names(state: Dict) -> List[str]:
    names = list(state.get("recent_product_names") or [])
    if not names:
        for rec in (state.get("context") or [])[:MAX_CONTEXT_NAMES]:
            if isinstance(rec, dict) and rec.get("name"):
                names.append(rec["name"])
    return names


async def filter_extractor(state: Dict) -> Dict:
    """Async graph node: reads `eng_query` (or `query`) and the recent product
    names, writes `metadata_filters` (a plain dict)."""
    query = state.get("eng_query") or state.get("query")
    filters = await extract_filters(query, _recent_product_names(state))
    return {"metadata_filters": filters.model_dump()}
