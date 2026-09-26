"""Chat turn orchestrator: prequal -> (extractor || semantic search) -> retrieval
-> product list or responder.

    turn = await handle_message("عايزة سيروم للشعر", session_id="abc")
    turn.path       # "products_only": templated intro + cards, no LLM after prequal
    turn.products   # RetrievedProduct cards

Routing (ARCHITECTURE_NOTES.md sections 9 and 10):
- needs_retrieval = false                     -> responder (history + message, no products)
- products_only, products found, none relaxed -> templated intro in the user's language (FINAL, no LLM)
- products_only, nothing found or relaxed     -> responder with the whole RetrievalResult, so it can explain
- needs_response and needs_retrieval          -> responder with the whole RetrievalResult

Streaming: `on_event(event, data)` receives the turn's events as they happen
(the /chat SSE contract, before api/widget.py turns product dumps into cards):
message, products (right after retrieval), card, and the responder's message
chunks. The same events are kept in TurnResult.events.
"""

import asyncio
import contextlib
import logging
import time
from typing import Awaitable, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from agents.filter_extractor.agent import extract_filters
from agents.prequal.agent import prequalify
from agents.prequal.session_context import SessionStore, get_session_store
from agents.responder.agent import fold_message, respond_stream
from agents.responder.context import ResponderContext, mode_for
from agents.retrieval.agent import retrieve
from agents.retrieval.semantic import semantic_search
from config import settings
from models.filter_extractor import MetadataFilters
from models.prequal import PrequalResult
from models.retrieval import RetrievalResult, RetrievedProduct

logger = logging.getLogger("orchestrator")

# Templated replies are the widget speaking on its own: Egyptian Arabic,
# gender-neutral, no exclamation marks (see web/README.md, Voice).
PRODUCTS_INTRO = {
    "ar": "دي المنتجات اللي تناسب طلبك:",
    "mixed": "دي المنتجات اللي تناسب طلبك:",
    "arabizi": "Dy el montagat el monasba lik:",
    "en": "Here are matching products:",
}
RESPONDER_ERROR = {
    "ar": "معلش، حصلت مشكلة. ممكن تبعت رسالتك تاني؟",
    "mixed": "معلش، حصلت مشكلة. ممكن تبعت رسالتك تاني؟",
    "arabizi": "Ma3lesh, 7asalet moshkela. Momken teb3at resaltak tany?",
    "en": "Sorry, something went wrong. Could you send your message again?",
}


# Progress stages reported through handle_message(on_stage=...), for a
# streaming client's status line.
STAGE_UNDERSTANDING = "understanding"
STAGE_SEARCHING = "searching"
STAGE_WRITING = "writing"


class TurnResult(BaseModel):
    reply: str
    products: List[dict] = Field(default_factory=list)      # RetrievedProduct cards shown with the reply
    path: Literal["products_only", "responder", "small_talk"]
    prequal: PrequalResult
    filters: Optional[dict] = None                           # MetadataFilters.model_dump(), when extracted
    retrieval: Optional[dict] = None                         # RetrievalResult.model_dump() without products, plus handles
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    # Everything the turn sent besides status: message (whole text or responder
    # chunks), products (RetrievedProduct dumps), card. See api/widget.py.
    events: List[dict] = Field(default_factory=list)
    cards: List[dict] = Field(default_factory=list)          # responder cards: how_to_use, safety, compare, quiz, objection
    disclaimer: Optional[str] = None                         # added by the responder's code rule; not part of `reply`
    responder: Dict = Field(default_factory=dict)            # the responder's done data: latency_ms, partial, source, route...


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 1)


async def _cancel(task: Optional[asyncio.Task]) -> None:
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


async def _run_responder(ctx: ResponderContext, emit: Callable[[str, dict], None],
                         on_stage: Optional[Callable[[str], None]]) -> dict:
    """Stream the responder's events through `emit`. Returns the reply text,
    the disclaimer, the cards and the responder's done data. A responder that
    raises still leaves a reply: the text so far, or a fixed apology."""
    text, disclaimer, cards, done = "", None, [], {}
    try:
        async for ev in respond_stream(ctx):
            name, data = ev["event"], ev["data"]
            if name == "status":
                _notify(on_stage, data.get("stage", STAGE_WRITING))
                continue
            if name == "done":
                done = data
                continue
            if name == "message":
                if data.get("disclaimer"):
                    disclaimer = data.get("delta")
                text = fold_message(text, data)
            elif name == "card":
                cards.append(data)
            emit(name, data)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 -- the chat must still reply
        logger.exception("[orchestrator] responder failed")
        done = {**done, "partial": bool(text.strip()), "source": "error"}
        if not text.strip():
            text = RESPONDER_ERROR.get(ctx.language, RESPONDER_ERROR["en"])
            emit("message", {"text": text, "replace": True, "path": "responder", "language": ctx.language})
    return {"text": text, "disclaimer": disclaimer, "cards": cards, "done": done}


async def _safe_extract(work: Awaitable[MetadataFilters]) -> Optional[MetadataFilters]:
    """extract_filters never raises by design; this only contains a bug in it,
    so retrieval still runs (on the name-free, filter-free semantic path)."""
    try:
        return await work
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        logger.exception("[orchestrator] extractor raised; retrieving without filters")
        return None


async def _none() -> None:
    return None


def _notify(on_stage: Optional[Callable[[str], None]], stage: str) -> None:
    if on_stage is None:
        return
    try:
        on_stage(stage)
    except Exception:  # noqa: BLE001 -- a progress listener never breaks the turn
        logger.exception("[orchestrator] on_stage listener failed for %s", stage)


async def _timed(timings: Dict[str, float], key: str, work: Awaitable):
    t = time.perf_counter()
    try:
        return await work
    finally:
        timings[key] = _ms(t)


async def warm_up() -> int:
    """Call once at startup inside the serving event loop. Loads the catalog,
    every prompt template, the retrieval indexes and the embedding model,
    builds the cached model instances, and pre-opens pooled HTTPS connections,
    so the first user doesn't pay for it. Returns how many connections were
    opened."""
    from agents.filter_extractor.agent import warm_up as extractor_warm_up
    from agents.filter_extractor.calls import CALL_FUNCTIONS, route_kwargs
    from agents.filter_extractor.prompts import all_templates
    from agents.prequal.agent import warm_up as prequal_warm_up
    from agents.responder.agent import warm_up as responder_warm_up
    from agents.retrieval import semantic
    from agents.retrieval.index_builder import get_index
    from llm.client import EXTRACTOR_FALLBACK_ORDER, fallback_client
    from llm.llm_models import client_llm

    prequal_warm_up()
    responder_warm_up()
    all_templates()
    get_index()
    for key in CALL_FUNCTIONS:
        for r in EXTRACTOR_FALLBACK_ORDER:
            router, model = fallback_client._resolve(r)
            client_llm.get_cached_model(router, model, **route_kwargs(r, int(settings.EXTRACTOR_MAX_TOKENS[key])))
    sem_warm, opened = await asyncio.gather(asyncio.to_thread(semantic.warm_up),
                                            extractor_warm_up(len(CALL_FUNCTIONS) + 2), return_exceptions=True)
    if isinstance(sem_warm, BaseException):
        logger.error("[orchestrator] semantic warm-up failed, retrieval runs without it: %r", sem_warm)
    if isinstance(opened, BaseException):
        raise opened
    return opened


async def handle_message(*args, llm: Optional[str] = None, **kwargs) -> TurnResult:
    """Run one turn (arguments as _handle_message). `llm` picks the provider
    for every LLM call in the turn ("glm" | "groq", see
    settings.LLM_PROVIDER_ROUTES); None or unknown uses the configured routes."""
    from llm.client import use_provider

    with use_provider(llm):
        return await _handle_message(*args, **kwargs)


async def _handle_message(message: str, session_id: str = "default",
                          store: Optional[SessionStore] = None,
                          on_stage: Optional[Callable[[str], None]] = None,
                          on_event: Optional[Callable[[str, dict], None]] = None) -> TurnResult:
    """`on_stage`, when given, is called with STAGE_UNDERSTANDING, then
    STAGE_SEARCHING if retrieval runs and STAGE_WRITING if the responder runs.
    `on_event`, when given, receives every other event as it happens."""
    store = store or get_session_store()
    start = time.perf_counter()
    timings: Dict[str, float] = {}
    ctx = store.get(session_id)
    events: List[dict] = []

    def emit(name: str, data: dict) -> None:
        events.append({"event": name, "data": data})
        if on_event is None:
            return
        try:
            on_event(name, data)
        except Exception:  # noqa: BLE001 -- a listener never breaks the turn
            logger.exception("[orchestrator] on_event listener failed for %s", name)

    # Optional speculative start: the extractor and the semantic search begin
    # when the rewriter returns, and are cancelled below if the router says
    # no retrieval is needed.
    spec: Dict[str, asyncio.Task] = {}

    def on_rewrite(rw: dict) -> None:
        if not rw.get("skip_metadata_filters"):
            spec["extractor"] = asyncio.create_task(extract_filters(rw["query_en"]))
        spec["semantic"] = asyncio.create_task(semantic_search(rw["query_en"]))

    async def cancel_spec() -> None:
        for task in spec.values():
            await _cancel(task)

    _notify(on_stage, STAGE_UNDERSTANDING)
    t = time.perf_counter()
    try:
        pq = await prequalify(message, ctx, session_id,
                              on_rewrite=on_rewrite if settings.PREQUAL_SPECULATIVE_EXTRACTOR else None)
    except BaseException:
        await cancel_spec()
        raise
    timings["prequal"] = _ms(t)

    if pq.end:
        await cancel_spec()
        reply = pq.reply or RESPONDER_ERROR.get(pq.language, RESPONDER_ERROR["en"])
        emit("message", {"text": reply, "path": "small_talk", "language": pq.language, "health": False})
        ctx.add_turn(message, reply, None)      # no list shown: last_products stay
        store.save(session_id, ctx)
        timings["total"] = _ms(start)
        logger.info("[orchestrator] session=%s path=small_talk timings=%s", session_id, timings)
        return TurnResult(reply=reply, path="small_talk", prequal=pq, timings_ms=timings, events=events)

    filters: Optional[MetadataFilters] = None
    found: Optional[RetrievalResult] = None
    if pq.needs_retrieval:
        _notify(on_stage, STAGE_SEARCHING)
        if pq.skip_metadata_filters:
            await _cancel(spec.pop("extractor", None))
        # The rewriter already resolved references into product names, so the
        # extractor gets no context. It and the semantic search run in parallel.
        extract = None if pq.skip_metadata_filters else spec.get("extractor") or extract_filters(pq.query_en)
        search = spec.get("semantic") or semantic_search(pq.query_en)
        t = time.perf_counter()
        try:
            filters, sem = await asyncio.gather(
                _timed(timings, "extractor", _safe_extract(extract)) if extract is not None else _none(),
                _timed(timings, "semantic", search),
            )
        except BaseException:
            await cancel_spec()
            raise
        timings["extractor_semantic"] = _ms(t)
        t = time.perf_counter()
        found = retrieve(filters, sem, pq.query_en, pq.intent, pq.k)
        timings["retrieval"] = _ms(t)
    else:
        await cancel_spec()

    shown: Optional[List[RetrievedProduct]] = None   # None: this turn showed no product list
    answer: dict = {"disclaimer": None, "cards": [], "done": {}}
    if pq.route == "products_only" and found is not None and found.products and not found.relaxed_keys:
        path = "products_only"
        shown = found.products
        reply = PRODUCTS_INTRO.get(pq.language, PRODUCTS_INTRO["en"])
        emit("message", {"text": reply, "path": path, "language": pq.language, "health": False})
        _emit_products(emit, found)
    else:
        path = "responder"
        if found is not None:
            shown = found.products
            if found.products:
                _emit_products(emit, found)     # retrieval's list goes out before the answer is written
        t = time.perf_counter()
        answer = await _run_responder(ResponderContext(
            query_original=pq.query_original, query_en=pq.query_en, language=pq.language, persona=pq.persona,
            intent=pq.intent, history=list(ctx.history), retrieval=found, mode=mode_for(pq.route, found),
        ), emit, on_stage)
        reply = answer["text"]
        timings["responder"] = _ms(t)

    ctx.add_turn(message, reply, [p.name for p in shown] if shown is not None else None)
    if filters is not None:
        ctx.last_filters = filters.model_dump(exclude={"meta"})
    store.save(session_id, ctx)

    timings["total"] = _ms(start)
    logger.info("[orchestrator] session=%s path=%s k=%d timings=%s last_filters=%s",
                session_id, path, pq.k, timings, ctx.last_filters)
    return TurnResult(
        reply=reply,
        products=[p.model_dump() for p in shown or []],
        path=path,
        prequal=pq,
        filters=filters.model_dump() if filters is not None else None,
        retrieval={**found.model_dump(exclude={"products"}), "handles": found.handles} if found is not None else None,
        timings_ms=timings,
        events=events,
        cards=answer["cards"],
        disclaimer=answer["disclaimer"],
        responder=answer["done"],
    )


def _emit_products(emit: Callable[[str, dict], None], found: RetrievalResult) -> None:
    emit("products", {"items": [p.model_dump() for p in found.products],
                      "total": found.total_candidates or len(found.products),
                      "relaxed": list(found.relaxed_keys), "applied": dict(found.applied_filters)})
