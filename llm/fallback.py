import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from langchain_core.messages import SystemMessage
from pydantic import BaseModel
from llm.helpers import Helpers
from llm.llm_models import client_llm

logger = logging.getLogger("llm.fallback")


def parse_route(route: str) -> tuple:
    """'zai' -> ('zai', None); 'zai:glm-5.3-flash' -> ('zai', 'glm-5.3-flash').
    A route with an explicit model overrides the model configured for that router,
    so one fallback chain can try two models on the same provider."""
    router, _, model = route.partition(":")
    return Helpers.validate_router(router), (model or None)


# Routers whose API has no json_schema response_format and no forced tool
# choice (Z.ai: response_format json_object only, tool_choice "auto" only).
# They get JSON mode, with the schema written into the system prompt instead.
JSON_MODE_ROUTERS = {"zai"}


def with_schema_prompt(message: Any, schema: Union[type, dict]) -> Any:
    """Add the output schema to the system message, for JSON-mode routers."""
    as_dict = schema.model_json_schema() if isinstance(schema, type) else schema
    note = ("Reply with one JSON object only, no prose, matching this JSON schema:\n"
            + json.dumps(as_dict, ensure_ascii=False, separators=(",", ":")))
    if not isinstance(message, list):
        return message
    if message and isinstance(message[0], SystemMessage):
        return [SystemMessage(content=f"{message[0].content}\n\n{note}")] + list(message[1:])
    return [SystemMessage(content=note)] + list(message)


class AllRoutesFailed(RuntimeError):
    """Every route in the chain failed. `errors` has one entry per attempt and
    `timed_out` is True when every attempt failed by timing out."""

    def __init__(self, errors: List[str], timed_out: bool):
        self.errors = errors
        self.timed_out = timed_out
        super().__init__(f"All fallback models failed to generate valid constrained output. Details: {errors}")


class StreamBroken(RuntimeError):
    """A stream failed after it had already produced text, so the chain cannot
    move on to the next route without repeating itself. `route` names it."""

    def __init__(self, route: str, error: str):
        self.route = route
        self.error = error
        super().__init__(f"stream from '{route}' broke mid-answer: {error}")


@dataclass
class StreamInfo:
    """Filled in by astream as it runs: which route is streaming, and the
    failed attempts before it."""
    route: Optional[str] = None
    attempt: Optional[int] = None
    first_token_ms: Optional[float] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class ConstrainedResult:
    """What aconstrained_invoke returns: the parsed output plus which route
    produced it, so callers can tell a primary answer from a fallback one."""
    data: dict
    route: str
    attempt: int
    latency_ms: float
    usage: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class FallBack:
    def __init__(
        self,
        llm_anthropic: Optional[str] = None,
        llm_ollama: Optional[str] = None,
        llm_gpt: Optional[str] = None,
        llm_gemini: Optional[str] = None,
        llm_groq: Optional[str] = None,
        llm_zai: Optional[str] = None,
    ):
        # Map the router names to their specific model strings
        self.llms: Dict[str, str] = {}

        if llm_anthropic:
            self.llms["anthropic"] = llm_anthropic
        if llm_ollama:
            self.llms["ollama"] = llm_ollama
        if llm_gpt:
            self.llms["gpt"] = llm_gpt
        if llm_gemini:
            self.llms["gemini"] = llm_gemini
        if llm_groq:
            self.llms["groq"] = llm_groq
        if llm_zai:
            self.llms["zai"] = llm_zai

    def _resolve(self, route: str) -> tuple:
        router, model = parse_route(route)
        model = model or self.llms.get(router)
        if not model:
            raise ValueError(f"No model string configured for '{router}' during initialization.")
        return router, model

    def invoke(self, message: Any, fallback_order: List[str]) -> str:
        """
        Entry point 1: Regular text generation.
        Attempts to invoke models in the provided sequence.
        Falls back to the next router if one fails.
        """
        errors = []

        for router in fallback_order:
            try:
                print(f"Attempting regular generation using: {router}...")
                router, model_name = self._resolve(router)
                llm = client_llm.get_model(router, model_name)

                response = llm.invoke(message)
                return response.content

            except Exception as e:
                print(f"Router '{router}' failed: {e}")
                errors.append(f"{router} error: {str(e)}")
                continue

        # If the loop finishes without returning, all models failed
        raise RuntimeError(f"All fallback models failed. Details: {errors}")

    def constrained_invoke(self, message: Any, fallback_order: List[str], constraine_model: Optional[BaseModel] = None) -> dict:
        """
        Entry point 2: Constrained/structured generation.
        Forces the output to match the Pydantic schema, trying models in sequence.
        Returns the parsed attributes as a dictionary.
        """
        if not constraine_model:
            raise ValueError("Cannot perform constrained invoke: 'constraine_model' was not provided.")

        errors = []

        for router in fallback_order:
            try:
                print(f"Attempting constrained generation using: {router}...")
                router, model_name = self._resolve(router)
                llm = client_llm.get_model(router, model_name)

                # json_schema uses native constrained decoding (response_format), not
                # tool-calling -- avoids Groq's "model did not call a tool" failures
                # that forced tool_choice can produce.
                if router in JSON_MODE_ROUTERS:
                    structured_llm = llm.with_structured_output(constraine_model, method="json_mode")
                    pydantic_response = structured_llm.invoke(with_schema_prompt(message, constraine_model))
                else:
                    structured_llm = llm.with_structured_output(constraine_model, method="json_schema")
                    pydantic_response = structured_llm.invoke(message)

                # Return as a dictionary
                return pydantic_response.model_dump()

            except Exception as e:
                print(f"Constrained router '{router}' failed: {e}")
                errors.append(f"{router} error: {str(e)}")
                continue

        # If the loop finishes without returning, all models failed
        raise RuntimeError(f"All fallback models failed to generate valid constrained output. Details: {errors}")

    async def aconstrained_invoke(
        self,
        message: Any,
        fallback_order: List[str],
        constraine_model: Union[type, dict, None] = None,
        *,
        timeouts: Optional[List[float]] = None,
        deadline_s: Optional[float] = None,
        model_kwargs: Optional[List[Dict[str, Any]]] = None,
        strict: Optional[bool] = True,
    ) -> ConstrainedResult:
        """
        Entry point 3: async constrained generation, same chain as entry point 2.

        `constraine_model` is a Pydantic class or a JSON-schema dict (a dict
        needs a "title"). Route i gets its own timeout `timeouts[i]`, and
        `deadline_s` caps the whole chain so a slow primary cannot push the
        fallback past the caller's budget. `model_kwargs[i]` are constructor
        overrides for route i (max_tokens, reasoning_effort, ...). Model
        instances are cached, so their HTTP connection pools are reused.

        Raises AllRoutesFailed (a RuntimeError) when every route fails; a
        timeout counts as a failed attempt and the chain moves on.
        """
        if not constraine_model:
            raise ValueError("Cannot perform constrained invoke: 'constraine_model' was not provided.")

        errors: List[str] = []
        timeouts_hit = 0
        loop = asyncio.get_running_loop()
        start = loop.time()

        for i, route in enumerate(fallback_order):
            remaining = None if deadline_s is None else deadline_s - (loop.time() - start)
            if remaining is not None and remaining <= 0.05:
                errors.append(f"{route} timeout: call deadline reached before attempt")
                timeouts_hit += 1
                break
            timeout = timeouts[i] if timeouts and i < len(timeouts) else None
            if remaining is not None:
                timeout = remaining if timeout is None else min(timeout, remaining)
            attempt_start = loop.time()
            try:
                router, model_name = self._resolve(route)
                kwargs = dict(model_kwargs[i]) if model_kwargs and i < len(model_kwargs) and model_kwargs[i] else {}
                llm = client_llm.get_cached_model(router, model_name, **kwargs)
                if router in JSON_MODE_ROUTERS:
                    structured = llm.with_structured_output(constraine_model, method="json_mode", include_raw=True)
                    prompt = with_schema_prompt(message, constraine_model)
                else:
                    structured = llm.with_structured_output(
                        constraine_model, method="json_schema", include_raw=True, strict=strict,
                    )
                    prompt = message
                out = await asyncio.wait_for(structured.ainvoke(prompt), timeout=timeout)
                if out.get("parsing_error") or out.get("parsed") is None:
                    raise ValueError(f"unparseable structured output: {out.get('parsing_error')}")
                parsed = out["parsed"]
                data = parsed.model_dump() if isinstance(parsed, BaseModel) else dict(parsed)
                usage = dict(getattr(out.get("raw"), "usage_metadata", None) or {})
                return ConstrainedResult(
                    data=data,
                    route=f"{router}:{model_name}",
                    attempt=i,
                    latency_ms=round((loop.time() - attempt_start) * 1000, 1),
                    usage=usage,
                    errors=errors,
                )
            except asyncio.TimeoutError:
                elapsed = loop.time() - attempt_start
                logger.warning("constrained route '%s' timeout after %.2fs", route, elapsed)
                errors.append(f"{route} timeout after {elapsed:.2f}s")
                timeouts_hit += 1
            except Exception as e:  # noqa: BLE001 -- any provider error moves on to the next route
                logger.warning("constrained route '%s' failed: %s", route, e)
                errors.append(f"{route} error: {e}")

        raise AllRoutesFailed(errors, timed_out=bool(errors) and timeouts_hit == len(errors))

    async def astream(
        self,
        message: Any,
        fallback_order: List[str],
        *,
        first_token_timeouts: Optional[List[float]] = None,
        idle_timeout_s: Optional[float] = None,
        deadline_s: Optional[float] = None,
        model_kwargs: Optional[List[Dict[str, Any]]] = None,
        info: Optional[StreamInfo] = None,
    ) -> AsyncIterator[str]:
        """
        Entry point 4: async streamed text generation, same chain.

        Yields text chunks. A route that errors or times out *before its first
        text chunk* moves on to the next route, as in the other entry points
        (reasoning models stream empty chunks while thinking; those don't
        count). Once text has been yielded, a failure or a gap longer than
        `idle_timeout_s` raises StreamBroken, so the caller can close the answer
        cleanly. `deadline_s` caps the whole call, and hitting it after text has
        started also raises StreamBroken. `info` is filled in as the call runs.

        Raises AllRoutesFailed when no route produced any text.
        """
        info = info if info is not None else StreamInfo()
        errors = info.errors
        timeouts_hit = 0
        loop = asyncio.get_running_loop()
        start = loop.time()

        def remaining() -> Optional[float]:
            return None if deadline_s is None else deadline_s - (loop.time() - start)

        for i, route in enumerate(fallback_order):
            left = remaining()
            if left is not None and left <= 0.05:
                errors.append(f"{route} timeout: call deadline reached before attempt")
                timeouts_hit += 1
                break
            first_timeout = first_token_timeouts[i] if first_token_timeouts and i < len(first_token_timeouts) else None
            if left is not None:
                first_timeout = left if first_timeout is None else min(first_timeout, left)
            attempt_start = loop.time()
            started = False
            stream = None
            try:
                router, model_name = self._resolve(route)
                kwargs = dict(model_kwargs[i]) if model_kwargs and i < len(model_kwargs) and model_kwargs[i] else {}
                llm = client_llm.get_cached_model(router, model_name, **kwargs)
                stream = llm.astream(message).__aiter__()
                while True:
                    if started:
                        wait = idle_timeout_s
                        left = remaining()
                        if left is not None:
                            wait = left if wait is None else min(wait, left)
                    else:
                        wait = None if first_timeout is None else max(0.0, first_timeout - (loop.time() - attempt_start))
                    try:
                        chunk = await asyncio.wait_for(stream.__anext__(), timeout=wait)
                    except StopAsyncIteration:
                        break
                    text = _chunk_text(chunk)
                    if not text:
                        continue
                    if not started:
                        started = True
                        info.route = f"{router}:{model_name}"
                        info.attempt = i
                        info.first_token_ms = round((loop.time() - attempt_start) * 1000, 1)
                    yield text
                if started:
                    return
                raise ValueError("stream ended without any text")
            except asyncio.CancelledError:
                raise
            except GeneratorExit:
                raise
            except Exception as e:  # noqa: BLE001 -- provider errors, timeouts, empty streams
                timed_out = isinstance(e, asyncio.TimeoutError)
                detail = f"timeout after {loop.time() - attempt_start:.2f}s" if timed_out else f"error: {e}"
                if started:
                    logger.warning("stream route '%s' broke mid-answer: %s", route, detail)
                    raise StreamBroken(info.route or route, detail) from e
                logger.warning("stream route '%s' failed before its first token: %s", route, detail)
                errors.append(f"{route} {detail}")
                timeouts_hit += int(timed_out)
            finally:
                if stream is not None and hasattr(stream, "aclose"):
                    with contextlib.suppress(Exception):
                        await stream.aclose()

        raise AllRoutesFailed(errors, timed_out=bool(errors) and timeouts_hit == len(errors))


def _chunk_text(chunk: Any) -> str:
    """Text of one streamed message chunk (content may be a list of parts)."""
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return ""
