"""Semantic step: cosine similarity between query_en and every product's
description embedding. The orchestrator runs it in parallel with the
extractor, so it adds no wall-clock time on top of the extractor.

    sem = await semantic_search("I need a sunscreen spray for oily skin.")
    sem.scores["vacation-sunscreen-lotion-spray-200ml"]   # cosine, clipped to 0..1

Model loading follows the drug assistant's agents/image_pii/helpers.py:

- The embedding model (model_manager.py: downloaded once, then loaded from
  a local folder) and the product matrix (embedding_registry.py: cached on
  disk) load lazily in a background thread, started by the first call.
- Each request waits at most its own timeout. If the model is not ready yet,
  that request skips semantic search (status "loading") without giving up,
  so a slow first load is used by a later request.
- A load that fails is cached as a permanent failure until restart (status
  "failed"), so a broken model never costs every request a new attempt.

orchestrator.warm_up() calls warm_up() so all of this happens at startup.
semantic_search never raises (except CancelledError); on any problem it
returns empty scores and fusion ranks with sem_score = 0 ("no_semantic").
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from agents.filter_extractor.cache import TTLCache
from agents.retrieval import embedding_registry
from config import settings
from llm.embeddings import get_embeddings

logger = logging.getLogger("retrieval")

STATUS_OK = "ok"
STATUS_CACHED = "cached"      # query embedding came from the LRU cache
STATUS_LOADING = "loading"    # the model is still initializing; this request skips it
STATUS_TIMEOUT = "timeout"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"    # empty query, semantic search disabled, or the caller did not run it


@dataclass
class SemanticScores:
    scores: Dict[str, float] = field(default_factory=dict)   # handle -> cosine in 0..1
    status: str = STATUS_SKIPPED
    latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status in (STATUS_OK, STATUS_CACHED)


@dataclass
class SemanticIndex:
    route: str
    embedder: object          # LangChain Embeddings
    matrix: np.ndarray        # unit rows, one per product
    handles: List[str]

    def cosine(self, query_vec) -> Dict[str, float]:
        q = np.asarray(query_vec, dtype=np.float32)
        q /= max(float(np.linalg.norm(q)), 1e-12)
        sims = np.clip(self.matrix @ q, 0.0, 1.0)
        return {h: float(s) for h, s in zip(self.handles, sims)}


class _ModelLoader:
    """One lazy background init per embedding route. `index` stays None
    while loading and after a failure; `done` tells the two apart."""

    def __init__(self, route: str):
        self.route = route
        self.started = threading.Event()
        self.done = threading.Event()
        self.index: Optional[SemanticIndex] = None
        self.error: Optional[str] = None

    def _worker(self) -> None:
        try:
            embedder = get_embeddings(self.route)
            emb = embedding_registry.get_product_embeddings(self.route, embedder.embed_documents)
            self.index = SemanticIndex(self.route, embedder, emb.matrix, emb.handles)
            logger.info("[retrieval] semantic index ready: %s, %d products", self.route, len(emb.handles))
        except Exception as e:  # noqa: BLE001 -- recorded as a permanent failure
            self.error = f"{type(e).__name__}: {e}"[:300]
            logger.error("[retrieval] embedding model %s failed to load: %s", self.route, self.error)
        finally:
            self.done.set()

    def get(self, timeout_s: float) -> Optional[SemanticIndex]:
        """Start the init on the first call, then wait up to `timeout_s`."""
        with _lock:
            if not self.started.is_set():
                self.started.set()
                threading.Thread(target=self._worker, name="embedding-model-init", daemon=True).start()
        self.done.wait(timeout_s)
        return self.index


_lock = threading.Lock()
_loaders: Dict[str, _ModelLoader] = {}
_query_cache = TTLCache(maxsize=settings.RETRIEVAL_QUERY_CACHE_SIZE, ttl_s=7 * 24 * 3600)


def _loader(route: Optional[str] = None) -> _ModelLoader:
    route = route or settings.RETRIEVAL_EMBEDDING_ROUTE
    with _lock:
        return _loaders.setdefault(route, _ModelLoader(route))


def reset() -> None:
    """Forget loaded models, cached query embeddings and the in-memory product
    matrix, as in a fresh process (tests)."""
    with _lock:
        _loaders.clear()
    _query_cache.clear()
    embedding_registry._embeddings = embedding_registry._UNSET


def warm_up(route: Optional[str] = None, timeout_s: Optional[float] = None) -> None:
    """Start loading the model and product matrix now, instead of on the first
    request, and run one query so the first user doesn't pay for lazy
    initialization. Blocking (call it in a thread). Waits up to
    RETRIEVAL_MODEL_INIT_TIMEOUT_S; if the load is still running after that
    it carries on in the background. Raises if the load failed."""
    if not settings.RETRIEVAL_SEMANTIC_ENABLED:
        logger.info("[retrieval] semantic search disabled (RETRIEVAL_SEMANTIC_ENABLED=false); no model loaded")
        return
    loader = _loader(route)
    idx = loader.get(settings.RETRIEVAL_MODEL_INIT_TIMEOUT_S if timeout_s is None else timeout_s)
    if idx is not None:
        idx.embedder.embed_query("warm up")
    elif loader.done.is_set():
        raise RuntimeError(f"embedding model {loader.route} failed to load: {loader.error}")
    else:
        logger.warning("[retrieval] embedding model %s still loading after warm-up wait; requests skip "
                       "semantic search until it is ready", loader.route)


async def _query_scores(idx: SemanticIndex, text: str) -> tuple:
    key = (idx.route, text.lower())
    vec = _query_cache.get(key)
    status = STATUS_CACHED
    if vec is None:
        vec = np.asarray(await idx.embedder.aembed_query(text), dtype=np.float32)
        _query_cache.set(key, vec)
        status = STATUS_OK
    return idx.cosine(vec), status


async def semantic_search(query_en: str, *, timeout_s: Optional[float] = None,
                          route: Optional[str] = None) -> SemanticScores:
    if not settings.RETRIEVAL_SEMANTIC_ENABLED:
        return SemanticScores(status=STATUS_SKIPPED, error="disabled (RETRIEVAL_SEMANTIC_ENABLED=false)")
    start = time.perf_counter()
    text = " ".join((query_en or "").split())
    if not text:
        return SemanticScores(status=STATUS_SKIPPED)
    timeout_s = settings.RETRIEVAL_SEMANTIC_TIMEOUT_S if timeout_s is None else timeout_s
    loader = _loader(route)

    def elapsed() -> float:
        return round((time.perf_counter() - start) * 1000, 1)

    try:
        idx = loader.index if loader.done.is_set() else await asyncio.to_thread(loader.get, timeout_s)
        if idx is None:
            if loader.done.is_set():
                return SemanticScores(status=STATUS_FAILED, latency_ms=elapsed(), error=loader.error)
            logger.warning("[retrieval] semantic: model %s still loading; skipped for this request", loader.route)
            return SemanticScores(status=STATUS_LOADING, latency_ms=elapsed(), error="embedding model still loading")
        remaining = max(timeout_s - (time.perf_counter() - start), 0.05)
        scores, status = await asyncio.wait_for(_query_scores(idx, text), timeout=remaining)
        return SemanticScores(scores, status, elapsed())
    except asyncio.TimeoutError:
        logger.warning("[retrieval] semantic: timeout after %.2fs (%s)", timeout_s, loader.route)
        return SemanticScores(status=STATUS_TIMEOUT, latency_ms=elapsed(), error=f"timeout after {timeout_s}s")
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 -- retrieval must still work without the semantic signal
        logger.warning("[retrieval] semantic: failed (%s): %s", loader.route, e)
        return SemanticScores(status=STATUS_FAILED, latency_ms=elapsed(), error=f"{type(e).__name__}: {e}"[:300])
