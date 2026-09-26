"""Chat turn orchestrator: prequal -> extractor -> retrieval -> product list or responder.

    turn = await handle_message("عايزة سيروم للشعر", session_id="abc")
    turn.path       # "products_only": templated intro + cards, no LLM after prequal
    turn.products   # product cards

Routing (ARCHITECTURE_NOTES.md section 9):
- needs_retrieval = false             -> responder (history + message, no products)
- products_only and retrieval found   -> templated intro in the user's language (FINAL, no LLM)
- products_only and nothing found     -> responder with the closest text-search matches
- needs_response and needs_retrieval  -> responder with the top products
"""

import asyncio
import contextlib
import logging
import time
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from agents.filter_extractor.agent import extract_filters
from agents.prequal.agent import prequalify
from agents.prequal.session_context import SessionStore, get_session_store
from agents.responder.responder import REASON_NEEDS_RESPONSE, REASON_NO_RESULTS, ResponderContext, respond
from agents.retrieval.retrieval import RetrievalResult, retrieve, text_search, to_card
from config import settings
from models.filter_extractor import MetadataFilters
from models.prequal import PrequalResult

logger = logging.getLogger("orchestrator")

PRODUCTS_INTRO = {
    "ar": "دي المنتجات المناسبة ليكي:",
    "mixed": "دي المنتجات المناسبة ليكي:",
    "arabizi": "Dy el montagat el monasba lik:",
    "en": "Here are matching products:",
}
RESPONDER_ERROR = {
    "ar": "معلش، حصلت مشكلة. ممكن تبعت رسالتك تاني؟",
    "mixed": "معلش، حصلت مشكلة. ممكن تبعت رسالتك تاني؟",
    "arabizi": "Ma3lesh, 7asalet moshkela. Momken teb3at resaltak tany?",
    "en": "Sorry, something went wrong. Could you send your message again?",
}


class TurnResult(BaseModel):
    reply: str
    products: List[dict] = Field(default_factory=list)      # cards shown with the reply
    path: Literal["products_only", "responder"]
    prequal: PrequalResult
    filters: Optional[dict] = None                           # MetadataFilters.model_dump(), when extracted
    retrieval: Optional[dict] = None                         # RetrievalResult.summary()
    timings_ms: Dict[str, float] = Field(default_factory=dict)


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 1)


async def _cancel(task: Optional[asyncio.Task]) -> None:
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


async def _safe_respond(ctx: ResponderContext) -> str:
    try:
        return await respond(ctx)
    except Exception:  # noqa: BLE001 -- the chat must still reply
        logger.exception("[orchestrator] responder failed")
        return RESPONDER_ERROR.get(ctx.language, RESPONDER_ERROR["en"])


async def warm_up() -> int:
    """Call once at startup inside the serving event loop. Loads the catalog,
    every prompt template and the retrieval index, builds the cached model
    instances, and pre-opens pooled HTTPS connections, so the first user
    doesn't pay for it. Returns how many connections were opened."""
    from agents.filter_extractor.agent import warm_up as extractor_warm_up
    from agents.filter_extractor.calls import CALL_FUNCTIONS, route_kwargs
    from agents.filter_extractor.prompts import all_templates
    from agents.prequal.agent import warm_up as prequal_warm_up
    from agents.retrieval.retrieval import _index
    from llm.client import EXTRACTOR_FALLBACK_ORDER, fallback_client
    from llm.llm_models import client_llm

    prequal_warm_up()
    all_templates()
    _index()
    for key in CALL_FUNCTIONS:
        for r in EXTRACTOR_FALLBACK_ORDER:
            router, model = fallback_client._resolve(r)
            client_llm.get_cached_model(router, model, **route_kwargs(r, int(settings.EXTRACTOR_MAX_TOKENS[key])))
    return await extractor_warm_up(len(CALL_FUNCTIONS) + 2)


async def handle_message(message: str, session_id: str = "default",
                         store: Optional[SessionStore] = None) -> TurnResult:
    store = store or get_session_store()
    start = time.perf_counter()
    timings: Dict[str, float] = {}
    ctx = store.get(session_id)

    # Optional speculative extraction: starts when the rewriter returns,
    # cancelled below if the router says no retrieval is needed.
    spec_task: Optional[asyncio.Task] = None

    def on_rewrite(rw: dict) -> None:
        nonlocal spec_task
        if not rw.get("skip_metadata_filters"):
            spec_task = asyncio.create_task(extract_filters(rw["query_en"]))

    t = time.perf_counter()
    try:
        pq = await prequalify(message, ctx, session_id,
                              on_rewrite=on_rewrite if settings.PREQUAL_SPECULATIVE_EXTRACTOR else None)
    except BaseException:
        await _cancel(spec_task)
        raise
    timings["prequal"] = _ms(t)

    filters: Optional[MetadataFilters] = None
    found: Optional[RetrievalResult] = None
    if pq.needs_retrieval:
        if pq.skip_metadata_filters:
            await _cancel(spec_task)
        else:
            t = time.perf_counter()
            # The rewriter already resolved references into product names, so
            # the extractor gets no context.
            filters = await spec_task if spec_task is not None else await extract_filters(pq.query_en)
            timings["extractor"] = _ms(t)
        t = time.perf_counter()
        found = retrieve(filters, pq.query_en, settings.RETRIEVAL_TOP_K)
        timings["retrieval"] = _ms(t)
    else:
        await _cancel(spec_task)

    shown: Optional[List[dict]] = None   # None: this turn showed no product list
    if pq.route == "products_only" and found is not None and found.products:
        path = "products_only"
        shown = found.products
        reply = PRODUCTS_INTRO.get(pq.language, PRODUCTS_INTRO["en"])
    else:
        path = "responder"
        reason = REASON_NEEDS_RESPONSE
        products = found.products if found is not None else []
        if pq.route == "products_only" and found is not None and not found.products:
            reason = REASON_NO_RESULTS
            products = text_search(pq.query_en, settings.RETRIEVAL_TOP_K)   # closest options
        if found is not None:
            shown = products
        t = time.perf_counter()
        reply = await _safe_respond(ResponderContext(
            message=pq.query_original, query_en=pq.query_en, language=pq.language, persona=pq.persona,
            intent=pq.intent, history=list(ctx.history), products=products, reason=reason,
        ))
        timings["responder"] = _ms(t)

    ctx.add_turn(message, reply, [p["name"] for p in shown] if shown is not None else None)
    if filters is not None:
        ctx.last_filters = filters.model_dump(exclude={"meta"})
    store.save(session_id, ctx)

    timings["total"] = _ms(start)
    logger.info("[orchestrator] session=%s path=%s timings=%s last_filters=%s",
                session_id, path, timings, ctx.last_filters)
    return TurnResult(
        reply=reply,
        products=[to_card(p) for p in (shown or [])],
        path=path,
        prequal=pq,
        filters=filters.model_dump() if filters is not None else None,
        retrieval=found.summary() if found is not None else None,
        timings_ms=timings,
    )
