"""The hidden per-request LLM switch: /chat's `llm` field (and the frontends'
?llm= parameter) routes every LLM call in the turn to one provider. No network."""

import asyncio

import agents.orchestrator.orchestrator as orch
from agents.filter_extractor.agent import _cache_key as extractor_cache_key
from config import settings
from llm.client import current_provider, turn_routes, use_provider
from tests.test_api import client, pq, sse_events  # noqa: F401 -- `client` is a fixture

DEFAULT = ["zai:glm-5.3-flash", "zai:glm-5.3-flash"]


def test_no_provider_keeps_the_configured_routes():
    assert current_provider() is None
    assert turn_routes(DEFAULT) == DEFAULT


def test_provider_picks_its_routes_and_resets_after_the_turn():
    with use_provider("groq"):
        assert current_provider() == "groq"
        assert turn_routes(DEFAULT) == settings.LLM_PROVIDER_ROUTES["groq"]
        assert turn_routes(DEFAULT[:1]) == settings.LLM_PROVIDER_ROUTES["groq"][:1]   # fallback switched off
    assert current_provider() is None


def test_unknown_provider_is_ignored():
    with use_provider("nonsense"):
        assert current_provider() is None and turn_routes(DEFAULT) == DEFAULT


def test_tasks_started_in_the_turn_see_the_provider():
    async def main():
        with use_provider("groq"):
            return await asyncio.create_task(asyncio.sleep(0, result=current_provider()))
    assert asyncio.run(main()) == "groq"


def test_cache_entries_are_per_provider():
    plain = extractor_cache_key("hair serum", ())
    with use_provider("groq"):
        assert extractor_cache_key("hair serum", ()) != plain


def test_chat_passes_llm_to_the_whole_turn(monkeypatch, client):  # noqa: F811
    seen = []

    async def fake_prequalify(message, ctx=None, session_id=None, on_rewrite=None, use_cache=True):
        seen.append(current_provider())
        return pq(message)

    monkeypatch.setattr(orch, "prequalify", fake_prequalify)
    for llm in ("groq", "glm", None):
        body = {"message": "hair serum", "session_id": f"s-{llm}"}
        if llm:
            body["llm"] = llm
        events = sse_events(client.post("/chat", json=body).text)
        assert events[-1][0] == "done"
    assert seen == ["groq", "glm", None]


def test_chat_rejects_an_unknown_llm(client):  # noqa: F811
    assert client.post("/chat", json={"message": "hi", "llm": "gpt"}).status_code == 422
