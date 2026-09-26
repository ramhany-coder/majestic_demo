import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agents.filter_extractor.agent import extract_filters
from agents.orchestrator.orchestrator import TurnResult, handle_message
from agents.prequal.agent import prequalify
from agents.prequal.session_context import SessionContext
from agents.responder.agent import collect, respond_stream
from agents.responder.context import ResponderContext, mode_for
from agents.retrieval.agent import retrieve
from agents.retrieval.semantic import semantic_search
from models.filter_extractor import MetadataFilters
from models.prequal import PrequalResult
from models.retrieval import RetrievalResult, RetrievedProduct

router = APIRouter()
logger = logging.getLogger("pipeline")


async def _run_stage(name: str, fn: Callable[[], Awaitable[Any]]) -> Any:
    """Time a single-stage call; log and raise a 500 with the stage name + latency on failure."""
    start = time.perf_counter()
    try:
        result = await fn()
    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error("[api] stage '%s' failed after %.2fs: %s", name, elapsed, e)
        raise HTTPException(
            status_code=500,
            detail={
                "failed_stage": name,
                "stage_latency_seconds": round(elapsed, 3),
                "error": str(e),
            },
        )

    logger.info("[api] stage '%s' completed in %.2fs", name, time.perf_counter() - start)
    return result


# ---- stage 1: pre-qualification (rewriter + router) ----

class PrequalRequest(BaseModel):
    message: str
    # Stateless: the caller passes the chat context. Only /pipeline/run reads
    # and writes the server-side session store.
    history: List[Dict[str, str]] = []
    last_products: List[str] = []


@router.post("/prequal", response_model=PrequalResult)
async def prequal(payload: PrequalRequest):
    ctx = SessionContext(history=payload.history, last_products=payload.last_products)
    return await _run_stage("prequal", lambda: prequalify(payload.message, ctx, use_cache=False))


# ---- stage 2: metadata filter extractor ----

class ExtractRequest(BaseModel):
    query: str
    context: List[str] = []


@router.post("/extract", response_model=MetadataFilters)
async def extract(payload: ExtractRequest):
    return await _run_stage("extractor", lambda: extract_filters(payload.query, payload.context))


# ---- stage 3: retrieval ----

class RetrieveRequest(BaseModel):
    filters: Optional[MetadataFilters] = None   # None -> metadata filters skipped (name + semantic only)
    query_en: str = ""
    intent: Optional[str] = None
    k: Optional[int] = None                     # None -> RETRIEVAL_K_DEFAULT, capped at RETRIEVAL_K_MAX
    semantic: bool = True                       # run semantic search over query_en first


@router.post("/retrieve", response_model=RetrievalResult)
async def retrieve_products(payload: RetrieveRequest):
    async def run():
        # semantic_search never raises; a timeout or failure ranks without it.
        sem = await semantic_search(payload.query_en) if payload.semantic and payload.query_en else None
        return retrieve(payload.filters, sem, payload.query_en, payload.intent, payload.k)

    return await _run_stage("retrieval", run)


# ---- stage 4: responder ----

class RespondRequest(BaseModel):
    message: str
    query_en: str
    language: Literal["ar", "arabizi", "en", "mixed"] = "en"
    persona: Literal["customer", "sales_trainee", "doctor", "unknown"] = "unknown"
    intent: str = "other"
    history: List[Dict[str, str]] = []
    # A RetrievalResult from /retrieve (preferred), or bare RetrievedProduct dumps.
    retrieval: Optional[RetrievalResult] = None
    products: List[Dict[str, Any]] = []
    route: Literal["products_only", "needs_response"] = "needs_response"


class RespondResponse(BaseModel):
    reply: str
    disclaimer: Optional[str] = None
    cards: List[Dict[str, Any]] = []
    partial: bool = False
    source: Optional[str] = None           # llm | template
    latency_seconds: float


@router.post("/respond", response_model=RespondResponse)
async def respond_endpoint(payload: RespondRequest):
    """The responder alone, collected into one JSON body (the SSE stream is POST /chat)."""
    start = time.perf_counter()
    found = payload.retrieval
    if found is None and payload.products:
        products = [RetrievedProduct(**p) for p in payload.products]
        found = RetrievalResult(products=products, k=len(products), name_hits=[p.handle for p in products])
    ctx = ResponderContext(query_original=payload.message, query_en=payload.query_en, language=payload.language,
                           persona=payload.persona, intent=payload.intent, history=payload.history,
                           retrieval=found, mode=mode_for(payload.route, found))

    async def run():
        return collect([ev async for ev in respond_stream(ctx)])

    out = await _run_stage("responder", run)
    return {"reply": out["text"], "disclaimer": out["disclaimer"], "cards": out["cards"],
            "partial": bool(out["done"].get("partial")), "source": out["done"].get("source"),
            "latency_seconds": round(time.perf_counter() - start, 3)}


# ---- full pipeline ----

class PipelineRequest(BaseModel):
    message: str
    session_id: str = Field(default="default", min_length=1)


@router.post("/pipeline/run", response_model=TurnResult)
async def run_full_pipeline(payload: PipelineRequest):
    try:
        return await handle_message(payload.message, payload.session_id)
    except Exception as e:
        logger.exception("[api] pipeline failed for session=%s", payload.session_id)
        raise HTTPException(status_code=500, detail=str(e))
