"""API routes and lifespan with the LLM stages stubbed out (retrieval, the
responder and the session store are real)."""

import json

import pytest
from fastapi.testclient import TestClient

import agents.orchestrator.orchestrator as orch
import api.app as app_module
import api.chat as chat_module
import api.endpoints as endpoints
from agents.prequal.session_context import SessionStore
from agents.retrieval.semantic import SemanticScores
from models.filter_extractor import MetadataFilters
from models.prequal import PrequalResult
from api.widget import CARD_FIELDS


def pq(message, query_en="hair serum", route="products_only", needs_retrieval=True):
    return PrequalResult(query_original=message, query_en=query_en, language="en", is_follow_up=False,
                         route=route, needs_retrieval=needs_retrieval, intent="find_products",
                         persona="customer", meta={})


@pytest.fixture
def client(monkeypatch):
    state = {"warm_ups": 0}

    async def fake_warm_up():
        state["warm_ups"] += 1
        return 0

    async def fake_prequalify(message, ctx=None, session_id=None, on_rewrite=None, use_cache=True):
        state["prequal_ctx"] = ctx
        return pq(message)

    async def fake_extract(query, context=None, use_cache=True):
        return MetadataFilters()

    async def fake_semantic(query_en, **kwargs):     # no embedding model load in tests
        state.setdefault("semantic_calls", []).append(query_en)
        return SemanticScores(status="failed", error="stubbed")

    monkeypatch.setattr(app_module, "warm_up", fake_warm_up)
    monkeypatch.setattr(endpoints, "prequalify", fake_prequalify)
    monkeypatch.setattr(endpoints, "extract_filters", fake_extract)
    monkeypatch.setattr(endpoints, "semantic_search", fake_semantic)
    monkeypatch.setattr(orch, "semantic_search", fake_semantic)
    monkeypatch.setattr(orch, "prequalify", fake_prequalify)
    monkeypatch.setattr(orch, "extract_filters", fake_extract)
    store = SessionStore()
    monkeypatch.setattr(orch, "get_session_store", lambda: store)

    with TestClient(app_module.app) as c:
        c.state = state
        c.store = store
        yield c


def test_lifespan_runs_warm_up(client):
    assert client.state["warm_ups"] == 1


def test_lifespan_skips_warm_up_when_disabled(monkeypatch, client):
    monkeypatch.setattr(app_module.settings, "WARM_UP_ON_STARTUP", False)
    with TestClient(app_module.app):
        pass
    assert client.state["warm_ups"] == 1


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_prequal_passes_caller_context(client):
    r = client.post("/api/prequal", json={"message": "and the price?",
                                          "history": [{"role": "user", "content": "hi"}],
                                          "last_products": ["Serum A"]})
    assert r.status_code == 200
    assert r.json()["route"] == "products_only"
    assert client.state["prequal_ctx"].last_products == ["Serum A"]


def test_extract(client):
    r = client.post("/api/extract", json={"query": "hair serum"})
    assert r.status_code == 200
    assert r.json()["brand"] == []


def test_retrieve_without_filters(client):
    r = client.post("/api/retrieve", json={"query_en": "hair serum", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["k"] == 3 and len(body["products"]) <= 3
    assert body["meta"]["filters_skipped"] is True
    assert client.state["semantic_calls"] == ["hair serum"]


def test_retrieve_with_filters_and_no_semantic(client):
    r = client.post("/api/retrieve", json={"filters": {"product_type": ["hair serum"]}, "query_en": "hair serum",
                                           "intent": "find_products", "semantic": False})
    assert r.status_code == 200
    body = r.json()
    assert body["products"] and all(p["handle"] for p in body["products"])
    assert "semantic_calls" not in client.state


def test_respond(client):
    r = client.post("/api/respond", json={"message": "thanks", "query_en": "thanks"})
    assert r.status_code == 200
    assert r.json()["reply"]


def test_stage_failure_reports_stage(monkeypatch, client):
    async def boom(query, context=None, use_cache=True):
        raise RuntimeError("provider down")

    monkeypatch.setattr(endpoints, "extract_filters", boom)
    r = client.post("/api/extract", json={"query": "hair serum"})
    assert r.status_code == 500
    assert r.json()["detail"]["failed_stage"] == "extractor"
    assert r.json()["detail"]["error"] == "provider down"


def test_pipeline_run_saves_session(client):
    r = client.post("/api/pipeline/run", json={"message": "hair serum", "session_id": "s1"})
    assert r.status_code == 200
    body = r.json()
    assert body["path"] in ("products_only", "responder")
    assert "total" in body["timings_ms"]
    assert len(client.store.get("s1").history) == 2


# ---- the Jamila widget: POST /chat (SSE), GET /api/products/{handle}, the page ----

def sse_events(body: str):
    events = []
    for frame in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_chat_streams_stages_then_message_products_done(client):
    r = client.post("/chat", json={"message": "hair serum", "session_id": "w1", "locale": "en"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = sse_events(r.text)
    names = [name for name, _ in events]
    assert names == ["status", "status", "message", "products", "done"]
    assert [data["stage"] for name, data in events if name == "status"] == ["understanding", "searching"]
    message, products = events[2][1], events[3][1]
    assert message["path"] == "products_only" and message["health"] is False
    card = products["items"][0]
    assert set(card) == set(CARD_FIELDS) and card["variant_id"]
    assert len(client.store.get("w1").history) == 2


def test_chat_ends_with_an_error_event_when_the_pipeline_fails(monkeypatch, client):
    async def boom(message, session_id, on_stage=None):
        raise RuntimeError("provider down")

    monkeypatch.setattr(chat_module, "handle_message", boom)
    events = sse_events(client.post("/chat", json={"message": "hi"}).text)
    assert events == [("error", {"code": "pipeline_failed"})]


def test_chat_rejects_an_empty_message(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_product_details(client):
    r = client.get("/api/products/capixy-lashes-treatment-serum-10ml")
    assert r.status_code == 200 and r.json()["how_to_use"]
    assert client.get("/api/products/nope").status_code == 404


def test_widget_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "components/bundle.js" in r.text
    assert client.get("/components/bundle.js").status_code == 200
    assert client.get("/tokens.css").status_code == 200
