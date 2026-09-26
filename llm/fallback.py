import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union

from pydantic import BaseModel
from llm.helpers import Helpers
from llm.llm_models import client_llm

logger = logging.getLogger("llm.fallback")


def parse_route(route: str) -> tuple:
    """'groq' -> ('groq', None); 'groq:openai/gpt-oss-20b' -> ('groq', 'openai/gpt-oss-20b').
    A route with an explicit model overrides the model configured for that router,
    so one fallback chain can try two models on the same provider."""
    router, _, model = route.partition(":")
    return Helpers.validate_router(router), (model or None)


class AllRoutesFailed(RuntimeError):
    """Every route in the chain failed. `errors` has one entry per attempt and
    `timed_out` is True when every attempt failed by timing out."""

    def __init__(self, errors: List[str], timed_out: bool):
        self.errors = errors
        self.timed_out = timed_out
        super().__init__(f"All fallback models failed to generate valid constrained output. Details: {errors}")


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
                structured = llm.with_structured_output(
                    constraine_model, method="json_schema", include_raw=True, strict=strict,
                )
                out = await asyncio.wait_for(structured.ainvoke(message), timeout=timeout)
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
