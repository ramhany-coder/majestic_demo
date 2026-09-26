"""Call B: decide what happens after pre-qualification: a product list only
(retrieval is the final stage) or a response from the responder agent."""

from typing import Dict, Optional, Sequence

from agents.prequal.llm_call import CallResult, run_structured
from agents.prequal.prompts import get_template
from agents.prequal.schemas import ROUTER_DEFAULT, ROUTER_SCHEMA, validate_route
from config import settings


async def route(message: str, history: Sequence[Dict[str, str]] = (),
                last_products: Sequence[str] = (), routes: Optional[list] = None) -> CallResult:
    # On failure: needs_response + needs_retrieval, so the responder can still
    # show products -- the safest choice.
    return await run_structured(
        "router",
        get_template("router").messages(message, history, last_products),
        ROUTER_SCHEMA,
        validate_route,
        lambda: dict(ROUTER_DEFAULT),
        max_tokens=settings.PREQUAL_ROUTER_MAX_TOKENS,
        primary_timeout_s=settings.PREQUAL_ROUTER_TIMEOUT_S,
        deadline_s=settings.PREQUAL_ROUTER_DEADLINE_S,
        routes=routes,
    )
