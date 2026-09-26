"""R1: the conversational answer, streamed through the project's fallback
chain (quality tier, then the fast tier once, same prompt).

    info = StreamInfo()
    async for chunk in stream_answer(messages, persona="customer", info=info):
        ...
    info.route, info.first_token_ms

Raises AllRoutesFailed when no route produced text, and StreamBroken when a
stream fails after it started (llm/fallback.py).
"""

import contextlib
from typing import AsyncIterator, Optional

from agents.filter_extractor.calls import _is_reasoning_model
from config import settings
from llm.client import RESPONDER_FALLBACK_ORDER, fallback_client, turn_routes
from llm.fallback import StreamInfo


def route_kwargs(route: str, max_tokens: int, temperature: float) -> dict:
    kw = {"temperature": temperature, "max_tokens": max_tokens, "max_retries": 0}
    if _is_reasoning_model(route):
        kw["max_tokens"] = max_tokens + settings.RESPONDER_REASONING_TOKEN_ALLOWANCE
        kw["reasoning_effort"] = "low"
    return kw


def max_tokens_for(persona: str) -> int:
    return int(settings.RESPONDER_MAX_TOKENS.get(persona, settings.RESPONDER_MAX_TOKENS["customer"]))


async def stream_answer(messages: list, persona: str, info: Optional[StreamInfo] = None) -> AsyncIterator[str]:
    routes = turn_routes(RESPONDER_FALLBACK_ORDER)
    max_tokens = max_tokens_for(persona)
    stream = fallback_client.astream(
        messages, routes,
        first_token_timeouts=[settings.RESPONDER_FIRST_TOKEN_TIMEOUT_S] * len(routes),
        idle_timeout_s=settings.RESPONDER_IDLE_TIMEOUT_S,
        deadline_s=settings.RESPONDER_DEADLINE_S,
        model_kwargs=[route_kwargs(r, max_tokens, settings.RESPONDER_TEMPERATURE) for r in routes],
        info=info,
    )
    async with contextlib.aclosing(stream):
        async for chunk in stream:
            yield chunk


async def complete_answer(messages: list, persona: str) -> str:
    """The whole answer as one string (used for the language retry)."""
    return "".join([chunk async for chunk in stream_answer(messages, persona)])
