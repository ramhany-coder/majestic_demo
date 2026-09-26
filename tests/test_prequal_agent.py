"""prequalify() with a fake LLM plugged in at the model-factory level, so the
real async fallback chain (timeouts, deadline, fallback route) runs."""

import asyncio
import time

import pytest

import agents.prequal.agent as agent_module
import llm.fallback as fallback_module
from agents.prequal.agent import get_cache, prequalify
from agents.prequal.session_context import SessionContext
from config import settings

PRIMARY = "groq:qwen/qwen3.8-27b"
FALLBACK = "groq:openai/gpt-oss-20b"
REWRITE = "majestic_prequal_rewrite"
ROUTE = "majestic_prequal_route"

ANSWERS = {
    REWRITE: {"query_en": "I need a hair serum without silicone.", "language": "ar", "is_follow_up": True},
    ROUTE: {"route": "products_only", "needs_retrieval": True, "intent": "refine_products", "persona": "customer",
            "k": None},
}


class FakeStructured:
    def __init__(self, model, schema, behaviour):
        self.model, self.schema, self.behaviour = model, schema, behaviour

    async def ainvoke(self, messages):
        title = self.schema["title"]
        self.behaviour["calls"].append((self.model, title, time.perf_counter()))
        self.behaviour["messages"][title] = messages
        action = self.behaviour.get((self.model, title)) or self.behaviour.get(title)
        if action == "hang":
            await asyncio.sleep(30)
        if action == "error":
            raise RuntimeError("provider exploded")
        if action == "bad":
            return {"parsed": {**ANSWERS[title], "language": "klingon", "persona": "alien"}, "raw": None,
                    "parsing_error": None}
        await asyncio.sleep(self.behaviour.get("delay", 0))
        return {"parsed": dict(ANSWERS[title]), "raw": None, "parsing_error": None}


class FakeModel:
    def __init__(self, model, behaviour):
        self.model, self.behaviour = model, behaviour

    def with_structured_output(self, schema, **kwargs):
        return FakeStructured(self.model, schema, self.behaviour)


@pytest.fixture
def behaviour(monkeypatch):
    b = {"calls": [], "messages": {}}
    monkeypatch.setattr(fallback_module.client_llm, "get_cached_model",
                        lambda router, model, **kw: FakeModel(model, b))
    monkeypatch.setattr(settings, "PREQUAL_REWRITER_TIMEOUT_S", 0.2)
    monkeypatch.setattr(settings, "PREQUAL_ROUTER_TIMEOUT_S", 0.2)
    monkeypatch.setattr(settings, "PREQUAL_FALLBACK_TIMEOUT_S", 0.2)
    monkeypatch.setattr(settings, "PREQUAL_REWRITER_DEADLINE_S", 0.5)
    monkeypatch.setattr(settings, "PREQUAL_ROUTER_DEADLINE_S", 0.5)
    get_cache().clear()
    yield b
    get_cache().clear()


def hair_serum_ctx():
    ctx = SessionContext()
    ctx.add_turn("عايزة سيروم للشعر", "دي المنتجات المناسبة ليكي:",
                 ["Capixy Hair Serum 120ml", "Capixy Anti- Dandruff Serum Spray 120ml"])
    return ctx


def run(coro):
    return asyncio.run(coro)


def test_both_calls_run_in_parallel_and_merge(behaviour):
    behaviour["delay"] = 0.1
    start = time.perf_counter()
    r = run(prequalify("في واحد مفيهوش سيليكون؟", hair_serum_ctx()))
    assert time.perf_counter() - start < 0.19          # ~0.1 s, not 0.2 s
    assert {c[1] for c in behaviour["calls"]} == {REWRITE, ROUTE}
    assert r.query_en == "I need a hair serum without silicone."
    assert r.route == "products_only" and r.needs_retrieval and r.intent == "refine_products"
    assert r.query_original == "في واحد مفيهوش سيليكون؟"
    assert r.meta["status"] == {"rewriter": "ok", "router": "ok"}
    assert set(r.meta["latency_ms"]) == {"rewriter", "router", "total"}
    assert r.skip_metadata_filters is False


def test_both_calls_see_history_and_last_products(behaviour):
    run(prequalify("التاني ينفع للحامل؟", hair_serum_ctx()))
    for title in (REWRITE, ROUTE):
        human = behaviour["messages"][title][1].content
        assert "U: عايزة سيروم للشعر" in human
        assert "LAST_PRODUCTS: 1) Capixy Hair Serum 120ml 2) Capixy Anti- Dandruff Serum Spray 120ml" in human
        assert human.endswith("QUERY: التاني ينفع للحامل؟")


def test_rewriter_timeout_does_not_block_router(behaviour):
    behaviour[REWRITE] = "hang"                         # primary and fallback both hang
    start = time.perf_counter()
    r = run(prequalify("عايزة صن بلوك للبشرة الدهنية"))
    assert time.perf_counter() - start < 1.0            # bounded by the rewriter deadline
    assert r.meta["status"] == {"rewriter": "default", "router": "ok"}
    assert r.route == "products_only"                   # router answer kept
    assert r.query_en == "عايزة صن بلوك للبشرة الدهنية" and r.language == "ar"
    assert r.skip_metadata_filters is True              # non-English raw text: skip filters


def test_rewriter_failure_on_standalone_english_keeps_filters(behaviour):
    behaviour[REWRITE] = "error"
    r = run(prequalify("sunscreen for oily skin"))
    assert r.query_en == "sunscreen for oily skin" and r.language == "en"
    assert r.skip_metadata_filters is False


def test_router_timeout_does_not_block_rewriter(behaviour):
    behaviour[ROUTE] = "hang"
    r = run(prequalify("في واحد مفيهوش سيليكون؟", hair_serum_ctx()))
    assert r.meta["status"] == {"rewriter": "ok", "router": "default"}
    assert r.query_en == "I need a hair serum without silicone."
    assert (r.route, r.needs_retrieval, r.intent, r.persona) == ("needs_response", True, "other", "unknown")


def test_both_fail_still_returns_defaults(behaviour, caplog):
    behaviour[REWRITE] = "error"
    behaviour[ROUTE] = "error"
    r = run(prequalify("عايزة سيروم للشعر"))
    assert r.meta["status"] == {"rewriter": "default", "router": "default"}
    assert r.route == "needs_response" and r.needs_retrieval
    assert any(rec.levelname == "ERROR" for rec in caplog.records)


def test_invalid_output_falls_back_to_default(behaviour):
    behaviour[ROUTE] = "bad"
    behaviour[REWRITE] = "bad"
    r = run(prequalify("show me sunscreens"))
    assert r.meta["status"] == {"rewriter": "default", "router": "default"}


def test_primary_error_goes_to_fallback_model(behaviour):
    behaviour[("qwen/qwen3.8-27b", ROUTE)] = "error"
    r = run(prequalify("show me sunscreens"))
    assert r.meta["status"]["router"] == "fallback_model"
    assert r.meta["routes"]["router"] == FALLBACK


def test_a_bug_inside_one_call_is_contained(behaviour, monkeypatch):
    async def boom(*a, **kw):
        raise ValueError("bug")
    monkeypatch.setattr(agent_module, "route", boom)
    r = run(prequalify("show me sunscreens"))
    assert r.meta["status"] == {"rewriter": "ok", "router": "default"}


def test_cache_by_session_and_message(behaviour):
    run(prequalify("Show me sunscreens", session_id="s1"))
    n = len(behaviour["calls"])
    r = run(prequalify("show me  sunscreens", session_id="s1"))
    assert len(behaviour["calls"]) == n and r.meta["cached"]
    run(prequalify("show me sunscreens", session_id="s2"))
    assert len(behaviour["calls"]) == n + 2               # other session: billed


def test_same_words_in_a_new_context_are_not_served_from_cache(behaviour):
    ctx = hair_serum_ctx()
    run(prequalify("any cheaper?", ctx, session_id="s1"))
    n = len(behaviour["calls"])
    ctx.add_turn("show me sunscreens", "Here are matching products:", ["Vacation Sunscreen Cream 60ml"])
    r = run(prequalify("any cheaper?", ctx, session_id="s1"))
    assert len(behaviour["calls"]) == n + 2 and not r.meta["cached"]


def test_retry_of_a_completed_turn_hits_the_cache(behaviour):
    ctx = hair_serum_ctx()
    run(prequalify("في واحد مفيهوش سيليكون؟", ctx, session_id="s1"))
    n = len(behaviour["calls"])
    # The first attempt finished and was saved; the client retries the same message.
    ctx.add_turn("في واحد مفيهوش سيليكون؟", "دي المنتجات المناسبة ليكي:", ["Capixy Hair Serum 120ml"])
    r = run(prequalify("في واحد مفيهوش سيليكون؟", ctx, session_id="s1"))
    assert len(behaviour["calls"]) == n and r.meta["cached"]


def test_degraded_results_are_not_cached(behaviour):
    behaviour[ROUTE] = "error"
    run(prequalify("show me sunscreens", session_id="s1"))
    assert len(get_cache()) == 0


def test_empty_message_makes_no_call(behaviour):
    r = run(prequalify("   "))
    assert behaviour["calls"] == [] and r.needs_retrieval is False


def test_on_rewrite_hook_fires_before_router_finishes(behaviour):
    behaviour[("qwen/qwen3.8-27b", ROUTE)] = "hang"      # router takes the timeout + fallback path
    seen = {}

    async def main():
        async def hook(rw):
            seen["query_en"] = rw["query_en"]
            seen["t"] = time.perf_counter()
        start = time.perf_counter()
        r = await prequalify("في واحد مفيهوش سيليكون؟", hair_serum_ctx(), on_rewrite=hook)
        return start, r

    start, r = run(main())
    assert seen["query_en"] == "I need a hair serum without silicone."
    assert seen["t"] - start < 0.15                      # before the router's 0.2 s timeout
    assert r.meta["status"]["router"] == "fallback_model"


def test_broken_hook_does_not_break_the_stage(behaviour):
    def hook(rw):
        raise RuntimeError("hook bug")
    r = run(prequalify("show me sunscreens", on_rewrite=hook))
    assert r.meta["status"] == {"rewriter": "ok", "router": "ok"}


@pytest.mark.parametrize("message,lang", [("hi", "en"), ("السلام عليكم", "ar"), ("thanks a lot!", "en")])
def test_small_talk_ends_without_llm_calls(behaviour, message, lang):
    r = run(prequalify(message, session_id="s1"))
    assert behaviour["calls"] == []
    assert r.end and r.reply and r.intent == "greeting" and not r.needs_retrieval and r.language == lang
    assert r.meta["status"] == {"rewriter": "skipped", "router": "skipped"}


def test_small_talk_end_can_be_disabled(behaviour, monkeypatch):
    monkeypatch.setattr(settings, "PREQUAL_SMALL_TALK_END", False)
    r = run(prequalify("hi"))
    assert len(behaviour["calls"]) == 2 and not r.end and r.reply is None


def test_router_greeting_ends_the_turn(behaviour, monkeypatch):
    monkeypatch.setitem(ANSWERS, REWRITE, {"query_en": "Hello, how is your day going?", "language": "en",
                                          "is_follow_up": False})
    monkeypatch.setitem(ANSWERS, ROUTE, {"route": "needs_response", "needs_retrieval": False,
                                        "intent": "greeting", "persona": "unknown", "k": None})
    r = run(prequalify("hey, how is your day going my friend"))
    assert len(behaviour["calls"]) == 2
    assert r.end and r.reply == "Hello, I'm Jamila. How can I help today?"


def test_product_request_does_not_end(behaviour):
    r = run(prequalify("show me sunscreens"))
    assert not r.end and r.reply is None


@pytest.mark.parametrize("router_k,expected,source", [(None, 10, "default"), (3, 3, "router"), (1, 1, "router"),
                                                      (50, 20, "router"), (0, 10, "default"), ("2", 2, "router")])
def test_k_defaults_to_10_and_is_capped_at_20(behaviour, monkeypatch, router_k, expected, source):
    monkeypatch.setitem(ANSWERS, ROUTE, {**ANSWERS[ROUTE], "k": router_k})
    r = run(prequalify("warreeni sun blok spray", SessionContext()))
    assert r.meta["status"]["router"] == "ok"          # a bad k never fails the router call
    assert r.k == expected and r.meta["k_source"] == source


def test_router_failure_gives_default_k(behaviour):
    behaviour[ROUTE] = "error"
    r = run(prequalify("show me 3 sunscreens", SessionContext()))
    assert r.meta["status"]["router"] == "default" and r.k == 10
