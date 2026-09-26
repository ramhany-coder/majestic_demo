"""Semantic step with a fake embedder (no model download, no torch): cosine
scores, background model init (loading / permanent failure), timeouts, the
query cache, and the on-disk product embedding cache."""

import asyncio
import threading

import pytest

import agents.retrieval.embedding_registry as registry
import agents.retrieval.semantic as semantic
from agents.retrieval.index_builder import get_index
from config import settings
from llm.embeddings import parse_embedding_route

VOCAB = ["sunscreen", "hair", "serum", "deodorant", "joint", "shampoo", "cream", "spray", "lip", "acne"]


class FakeEmbedder:
    """Bag of words over VOCAB: good enough for cosine to rank by topic."""

    def __init__(self, delay=0.0, fail=False):
        self.delay, self.fail = delay, fail
        self.doc_calls = 0
        self.query_calls = 0

    def _vec(self, text):
        t = text.lower()
        return [float(t.count(w)) for w in VOCAB] + [0.01]

    def embed_documents(self, texts):
        self.doc_calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    async def aembed_query(self, text):
        self.query_calls += 1
        if self.fail:
            raise RuntimeError("provider down")
        await asyncio.sleep(self.delay)
        return self._vec(text)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    emb = FakeEmbedder()
    monkeypatch.setattr(semantic, "get_embeddings", lambda route: emb)
    monkeypatch.setattr(registry, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(settings, "RETRIEVAL_EMBEDDING_ROUTE", "hf:fake-model")
    semantic.reset()
    yield emb
    semantic.reset()


def search(query, **kw):
    return asyncio.run(semantic.semantic_search(query, **kw))


def test_scores_every_product_by_cosine(fake):
    s = search("sunscreen")
    assert s.status == "ok" and s.ok and len(s.scores) == 108
    assert all(0.0 <= v <= 1.0 for v in s.scores.values())
    assert s.scores["vacation-sunscreen-cream-60ml"] > s.scores["capixy-hair-serum"]


def test_query_embeddings_are_cached(fake):
    search("hair serum")
    s = search("  Hair   Serum ")
    assert s.status == "cached" and fake.query_calls == 1


def test_query_timeout_returns_empty_scores(fake):
    search("warm")                                      # model loaded
    fake.delay = 1.0
    s = search("hair serum", timeout_s=0.05)
    assert s.status == "timeout" and s.scores == {} and not s.ok


def test_provider_error_returns_failed(fake):
    fake.fail = True
    s = search("hair serum")
    assert s.status == "failed" and "provider down" in s.error and s.scores == {}


def test_empty_query_or_disabled_is_skipped(fake, monkeypatch):
    assert search("   ").status == "skipped" and fake.query_calls == 0
    monkeypatch.setattr(settings, "RETRIEVAL_SEMANTIC_ENABLED", False)
    assert search("hair serum").status == "skipped" and fake.query_calls == 0


def test_slow_model_load_skips_the_request_then_is_used_later(monkeypatch, tmp_path):
    release = threading.Event()
    emb = FakeEmbedder()

    def slow_model(route):
        release.wait(5)
        return emb

    monkeypatch.setattr(semantic, "get_embeddings", slow_model)
    monkeypatch.setattr(registry, "CACHE_DIR", tmp_path)
    semantic.reset()
    try:
        assert search("hair serum", timeout_s=0.05).status == "loading"      # not ready: skipped, not failed
        release.set()
        semantic._loader().done.wait(5)
        assert search("hair serum").status == "ok"                           # the same load, now ready
    finally:
        release.set()
        semantic.reset()


def test_failed_model_load_is_cached_as_permanent(monkeypatch, tmp_path):
    calls = []

    def broken(route):
        calls.append(route)
        raise OSError("model not found")

    monkeypatch.setattr(semantic, "get_embeddings", broken)
    monkeypatch.setattr(registry, "CACHE_DIR", tmp_path)
    semantic.reset()
    try:
        assert search("hair serum").status == "failed"
        s = search("hair serum")
        assert s.status == "failed" and "model not found" in s.error
        assert len(calls) == 1                          # never retried until restart
        with pytest.raises(RuntimeError, match="failed to load"):
            semantic.warm_up()
    finally:
        semantic.reset()


def test_warm_up_loads_the_model_and_runs_a_query(fake):
    semantic.warm_up()
    loader = semantic._loader()
    assert loader.done.is_set() and loader.index is not None and fake.doc_calls == 1


def test_product_embeddings_come_from_the_registry_cache(fake, tmp_path):
    search("hair serum")
    assert len(list(tmp_path.glob("product_embeddings_*.npz"))) == 1
    semantic.reset()                                    # a fresh process
    search("sunscreen")
    assert fake.doc_calls == 1                          # loaded from disk, not embedded again
    assert semantic._loader().index.handles == get_index().order


@pytest.mark.parametrize("route,ok", [("hf:sentence-transformers/all-MiniLM-L6-v2", True),
                                      ("gpt:text-embedding-3-small", True), ("ollama:nomic-embed-text", True),
                                      ("groq:whatever", False), ("hf:", False), ("nonsense", False)])
def test_embedding_routes(route, ok):
    if ok:
        assert parse_embedding_route(route)[0] == route.split(":")[0]
    else:
        with pytest.raises(ValueError):
            parse_embedding_route(route)
