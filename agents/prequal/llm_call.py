"""One pre-qualification call: constrained LLM call through the project's
fallback chain (primary model, then the fallback model once) -> code-side
validation -> on total failure, the call's safe default.

Never raises (except CancelledError), so a failure in one call can never
cancel or empty the other.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from agents.filter_extractor.calls import route_kwargs
from config import settings
from llm.client import PREQUAL_FALLBACK_ORDER, fallback_client

logger = logging.getLogger("prequal")

STATUS_OK = "ok"
STATUS_FALLBACK = "fallback_model"
STATUS_DEFAULT = "default"
STATUS_SKIPPED = "skipped"


@dataclass
class CallResult:
    key: str
    data: dict
    status: str
    latency_ms: float
    route: Optional[str] = None
    prompt_tokens: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        return self.status not in (STATUS_OK, STATUS_FALLBACK)


async def run_structured(
    key: str,
    messages: list,
    schema: dict,
    validator: Callable[[dict], Tuple[dict, List[str]]],
    default: Callable[[], dict],
    *,
    max_tokens: int,
    primary_timeout_s: float,
    deadline_s: float,
    routes: Optional[List[str]] = None,
) -> CallResult:
    routes = list(routes if routes is not None else PREQUAL_FALLBACK_ORDER)
    start = time.perf_counter()
    timeouts = [primary_timeout_s] + [settings.PREQUAL_FALLBACK_TIMEOUT_S] * (len(routes) - 1)
    try:
        res = await fallback_client.aconstrained_invoke(
            messages,
            routes,
            schema,
            timeouts=timeouts,
            deadline_s=deadline_s,
            model_kwargs=[route_kwargs(r, max_tokens) for r in routes],
        )
        data, notes = validator(res.data)
        return CallResult(
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
    except Exception as e:  # noqa: BLE001 -- AllRoutesFailed, invalid output, or a bug: use the default
        error = f"{type(e).__name__}: {e}"[:300]
    logger.warning("[prequal] %s degraded to default: %s", key, error)
    return CallResult(
        key=key,
        data=default(),
        status=STATUS_DEFAULT,
        latency_ms=round((time.perf_counter() - start) * 1000, 1),
        notes=[error],
    )
