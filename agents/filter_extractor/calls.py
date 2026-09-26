"""The 10 async extractor calls. Each one: build its own short prompt ->
constrained LLM call through the project's fallback chain (primary model,
then the fallback model once) -> validate against its schema -> on total
failure, the deterministic extractor for that key only.

A call never raises: whatever happens, it returns a CallOutcome, so one
failure can never block or empty the other nine.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from agents.filter_extractor.prompts import get_template, get_translate_template
from agents.filter_extractor.rule_based import extract_rule_based
from agents.filter_extractor.schemas import CALLS_BY_KEY, empty_output, get_schema, validate_output
from config import settings
from llm.client import EXTRACTOR_FALLBACK_ORDER, fallback_client
from llm.fallback import AllRoutesFailed

logger = logging.getLogger("filter_extractor")

# Model-name substrings of reasoning models, which need token headroom to think.
REASONING_MODEL_HINTS = ("gpt-oss", "deepseek-r1", "o1", "o3", "o4")

STATUS_OK = "ok"
STATUS_FALLBACK = "fallback_model"
STATUS_RULE = "rule_based"
STATUS_TIMEOUT = "timeout"       # every LLM attempt timed out; rule-based values used
STATUS_FAILED = "failed"         # LLM failed and rule fallback is off -> empty output
STATUS_SKIPPED = "skipped"


@dataclass
class CallOutcome:
    key: str
    data: dict
    status: str
    latency_ms: float
    route: Optional[str] = None
    prompt_tokens: Optional[int] = None
    notes: List[str] = field(default_factory=list)


def _is_reasoning_model(route: str) -> bool:
    return any(h in route.split(":", 1)[-1] for h in REASONING_MODEL_HINTS)


def route_kwargs(route: str, max_tokens: int) -> dict:
    kw = {"temperature": 0, "max_tokens": max_tokens, "max_retries": 0}
    if _is_reasoning_model(route):
        kw["max_tokens"] = max_tokens + settings.EXTRACTOR_REASONING_TOKEN_ALLOWANCE
        kw["reasoning_effort"] = "low"
    return kw


def _timeouts(key: str, n_routes: int) -> List[float]:
    primary = float(settings.EXTRACTOR_CALL_TIMEOUTS.get(key, settings.EXTRACTOR_TIMEOUT_S))
    return [primary] + [settings.EXTRACTOR_FALLBACK_TIMEOUT_S] * (n_routes - 1)


async def run_call(key: str, query: str, context: Optional[Sequence[str]] = None,
                   routes: Optional[List[str]] = None) -> CallOutcome:
    spec = CALLS_BY_KEY[key]
    routes = list(routes if routes is not None else EXTRACTOR_FALLBACK_ORDER)
    start = time.perf_counter()
    messages = get_template(key).messages(query, context if spec.uses_context else None)
    max_tokens = int(settings.EXTRACTOR_MAX_TOKENS[key])
    errors: List[str] = []
    timed_out = False
    try:
        res = await fallback_client.aconstrained_invoke(
            messages,
            routes,
            get_schema(key),
            timeouts=_timeouts(key, len(routes)),
            deadline_s=settings.EXTRACTOR_CALL_DEADLINE_S,
            model_kwargs=[route_kwargs(r, max_tokens) for r in routes],
        )
        data, notes = validate_output(spec, res.data)
        return CallOutcome(
            key=key,
            data=data,
            status=STATUS_OK if res.attempt == 0 else STATUS_FALLBACK,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            route=res.route,
            prompt_tokens=res.usage.get("input_tokens"),
            notes=res.errors + notes,
        )
    except asyncio.CancelledError:
        raise
    except AllRoutesFailed as e:
        errors, timed_out = e.errors or [str(e)], e.timed_out
    except Exception as e:  # noqa: BLE001 -- e.g. validation bug; degrade to rules
        errors.append(f"{type(e).__name__}: {e}")
    if settings.EXTRACTOR_USE_RULE_FALLBACK:
        try:
            data, notes = validate_output(spec, extract_rule_based(key, query, context))
            status = STATUS_TIMEOUT if timed_out else STATUS_RULE
        except Exception as e:  # noqa: BLE001
            logger.exception("rule-based fallback failed for %s", key)
            data, notes, status = empty_output(spec), [f"rule-based error: {e}"], STATUS_FAILED
    else:
        data, notes, status = empty_output(spec), [], STATUS_FAILED
    summary = "; ".join(errors)[:300]
    logger.warning("[filter_extractor] call '%s' degraded to %s: %s", key, status, summary)
    return CallOutcome(
        key=key,
        data=data,
        status=status,
        latency_ms=round((time.perf_counter() - start) * 1000, 1),
        route="rule_based" if status in (STATUS_RULE, STATUS_TIMEOUT) else None,
        notes=[summary] + notes,
    )


# ---- the 10 calls -----------------------------------------------------------

async def extract_names(query: str, context: Optional[Sequence[str]] = None) -> CallOutcome:
    return await run_call("names", query, context)


async def extract_brand(query: str, context=None) -> CallOutcome:
    return await run_call("brand", query)


async def extract_category(query: str, context=None) -> CallOutcome:
    return await run_call("category", query)


async def extract_product_group(query: str, context=None) -> CallOutcome:
    return await run_call("product_group", query)


async def extract_product_type(query: str, context=None) -> CallOutcome:
    return await run_call("product_type", query)


async def extract_product_form(query: str, context=None) -> CallOutcome:
    return await run_call("product_form", query)


async def extract_concerns(query: str, context=None) -> CallOutcome:
    return await run_call("concerns", query)


async def extract_suitable_for(query: str, context=None) -> CallOutcome:
    return await run_call("suitable_for", query)


async def extract_hero_ingredient(query: str, context=None) -> CallOutcome:
    return await run_call("hero_ingredient", query)


async def extract_ingredients(query: str, context=None) -> CallOutcome:
    return await run_call("ingredients", query)


CALL_FUNCTIONS: Dict[str, object] = {
    "names": extract_names,
    "brand": extract_brand,
    "category": extract_category,
    "product_group": extract_product_group,
    "product_type": extract_product_type,
    "product_form": extract_product_form,
    "concerns": extract_concerns,
    "suitable_for": extract_suitable_for,
    "hero_ingredient": extract_hero_ingredient,
    "ingredients": extract_ingredients,
}

TRANSLATE_SCHEMA = {
    "title": "majestic_translate",
    "type": "object",
    "additionalProperties": False,
    "required": ["query_en"],
    "properties": {"query_en": {"type": "string"}},
}


async def translate_to_english(query: str) -> tuple:
    """Optional pre-step (EXTRACTOR_TRANSLATE). Returns (english_query, status);
    on any failure the original query is used unchanged."""
    routes = list(EXTRACTOR_FALLBACK_ORDER)
    try:
        res = await fallback_client.aconstrained_invoke(
            get_translate_template().messages(query),
            routes,
            TRANSLATE_SCHEMA,
            timeouts=_timeouts("translate", len(routes)),
            deadline_s=settings.EXTRACTOR_CALL_DEADLINE_S,
            model_kwargs=[route_kwargs(r, settings.EXTRACTOR_TRANSLATE_MAX_TOKENS) for r in routes],
        )
        text = (res.data.get("query_en") or "").strip()
        return (text or query), (STATUS_OK if res.attempt == 0 else STATUS_FALLBACK)
    except Exception as e:  # noqa: BLE001
        logger.warning("[filter_extractor] translate step failed, using original query: %s", e)
        return query, STATUS_FAILED
