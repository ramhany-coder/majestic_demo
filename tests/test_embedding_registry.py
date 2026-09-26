"""Product embedding registry (same cases as the drug assistant's
tests/test_engine_registry.py): build + cache file on first use, the same
in-memory object afterwards, a fresh process loads the file instead of
re-embedding, and a changed catalog, model or schema version rebuilds and
drops the stale file."""

import json

import numpy as np
import pytest

import agents.retrieval.embedding_registry as registry

ROUTE = "hf:fake-model"


def _catalog(names):
    return {"products": [{"handle": n.lower(), "name": n, "description": f"{n} description",
                          "bundle_components": []} for n in names]}


class Embedder:
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return [[float(len(t)), 1.0, float(i)] for i, t in enumerate(texts)]


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """Points DATA_PATH / CACHE_DIR at a tiny temp catalog and resets the
    module-level singleton, which would otherwise leak across tests."""
    data_path = tmp_path / "majestic_catalog.json"
    data_path.write_text(json.dumps(_catalog(["Alpha", "Beta"])), encoding="utf-8")
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(registry, "DATA_PATH", data_path)
    monkeypatch.setattr(registry, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(registry, "_embeddings", registry._UNSET)
    return data_path, cache_dir


def cache_files(cache_dir):
    return sorted(cache_dir.glob("product_embeddings_*.npz"))


def test_first_call_builds_and_writes_one_cache_file(isolated_registry):
    _, cache_dir = isolated_registry
    emb = Embedder()
    out = registry.get_product_embeddings(ROUTE, emb.embed_documents)
    assert out.handles == ["alpha", "beta"] and out.matrix.shape == (2, 3) and emb.calls == 1
    assert np.allclose(np.linalg.norm(out.matrix, axis=1), 1.0)
    files = cache_files(cache_dir)
    assert len(files) == 1 and files[0].name.startswith(f"product_embeddings_v{registry._SCHEMA_VERSION}_hf-fake-model_")


def test_second_call_reuses_the_same_in_memory_object():
    emb = Embedder()
    first = registry.get_product_embeddings(ROUTE, emb.embed_documents)
    assert registry.get_product_embeddings(ROUTE, emb.embed_documents) is first and emb.calls == 1


def test_fresh_process_loads_from_the_cache_file(monkeypatch):
    registry.get_product_embeddings(ROUTE, Embedder().embed_documents)
    monkeypatch.setattr(registry, "_embeddings", registry._UNSET)     # a fresh process

    def must_not_embed(texts):
        raise AssertionError("embed_documents should not run again -- the cache file should be loaded")

    out = registry.get_product_embeddings(ROUTE, must_not_embed)
    assert out.handles == ["alpha", "beta"] and out.matrix.dtype == np.float32


def test_catalog_change_rebuilds_and_drops_the_stale_file(isolated_registry, monkeypatch):
    data_path, cache_dir = isolated_registry
    registry.get_product_embeddings(ROUTE, Embedder().embed_documents)
    old = cache_files(cache_dir)
    data_path.write_text(json.dumps(_catalog(["Alpha", "Beta", "Gamma"])), encoding="utf-8")
    monkeypatch.setattr(registry, "_embeddings", registry._UNSET)

    emb = Embedder()
    out = registry.get_product_embeddings(ROUTE, emb.embed_documents)
    assert emb.calls == 1 and out.handles == ["alpha", "beta", "gamma"]
    new = cache_files(cache_dir)
    assert len(new) == 1 and new != old


def test_model_change_rebuilds():
    registry.get_product_embeddings(ROUTE, Embedder().embed_documents)
    emb = Embedder()
    out = registry.get_product_embeddings("hf:other-model", emb.embed_documents)
    assert emb.calls == 1 and out.route == "hf:other-model"


def test_schema_version_bump_rebuilds(isolated_registry, monkeypatch):
    _, cache_dir = isolated_registry
    registry.get_product_embeddings(ROUTE, Embedder().embed_documents)
    monkeypatch.setattr(registry, "_SCHEMA_VERSION", registry._SCHEMA_VERSION + 1)
    monkeypatch.setattr(registry, "_embeddings", registry._UNSET)
    emb = Embedder()
    registry.get_product_embeddings(ROUTE, emb.embed_documents)
    assert emb.calls == 1 and len(cache_files(cache_dir)) == 1


def test_corrupt_cache_file_rebuilds_instead_of_crashing(isolated_registry, monkeypatch):
    _, cache_dir = isolated_registry
    registry.get_product_embeddings(ROUTE, Embedder().embed_documents)
    cache_files(cache_dir)[0].write_bytes(b"not an npz file")
    monkeypatch.setattr(registry, "_embeddings", registry._UNSET)
    emb = Embedder()
    out = registry.get_product_embeddings(ROUTE, emb.embed_documents)
    assert emb.calls == 1 and out.handles == ["alpha", "beta"]


def test_embedding_text_falls_back_to_bundle_components():
    p = {"handle": "y", "name": "Duo Kit", "description": "", "bundle_components": [{"name": "Serum A"}, {"name": "Cream B"}]}
    assert registry.embedding_text(p) == "Duo Kit | Serum A | Cream B"
    assert registry.embedding_text({"name": "Z", "description": "  Real   text. "}) == "Real text."
