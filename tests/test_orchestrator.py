"""Orchestrator routing with prequal and the extractor stubbed out (retrieval
and the session store are real)."""

import asyncio

import pytest

import agents.orchestrator.orchestrator as orch
from agents.prequal.session_context import SessionStore
from config import settings
from models.filter_extractor import IngredientFilter, MetadataFilters
from models.prequal import PrequalResult


def pq(message, query_en, route="products_only", needs_retrieval=True, intent="find_products",
       language="ar", skip=False):
    return PrequalResult(query_original=message, query_en=query_en, language=language, is_follow_up=False,
                         route=route, needs_retrieval=needs_retrieval, intent=intent, persona="customer",
                         skip_metadata_filters=skip, meta={})


@pytest.fixture
def env(monkeypatch):
    state = {"prequal": None, "filters": MetadataFilters(), "extract_calls": [], "respond_calls": [],
             "extract_delay": 0.0, "extract_cancelled": False}

    async def fake_prequalify(message, ctx, session_id, on_rewrite=None, use_cache=True):
        state["ctx_seen"] = ctx
        result = state["prequal"]
        if on_rewrite:
            on_rewrite({"query_en": result.query_en, "skip_metadata_filters": result.skip_metadata_filters})
            await asyncio.sleep(0.05)           # router still "running"
        return result

    async def fake_extract(query, context=None, use_cache=True):
        state["extract_calls"].append((query, context))
        try:
            await asyncio.sleep(state["extract_delay"])
        except asyncio.CancelledError:
            state["extract_cancelled"] = True
            raise
        return state["filters"]

    async def fake_respond(ctx):
        state["respond_calls"].append(ctx)
        return "responder reply"

    monkeypatch.setattr(orch, "prequalify", fake_prequalify)
    monkeypatch.setattr(orch, "extract_filters", fake_extract)
    monkeypatch.setattr(orch, "respond", fake_respond)
    monkeypatch.setattr(settings, "PREQUAL_SPECULATIVE_EXTRACTOR", False)
    state["store"] = SessionStore()
    return state


def turn(env, message, session="s1"):
    return asyncio.run(orch.handle_message(message, session, env["store"]))


def test_products_only_ends_at_retrieval_without_responder(env):
    env["prequal"] = pq("في واحد مفيهوش سيليكون؟", "I need a hair serum without silicone.")
    env["filters"] = MetadataFilters(product_type=["hair serum"], ingredients=IngredientFilter(exclude=["Silicone"]))
    t = turn(env, "في واحد مفيهوش سيليكون؟")
    assert t.path == "products_only" and env["respond_calls"] == []
    assert t.reply == "دي المنتجات المناسبة ليكي:"
    assert {p["handle"] for p in t.products} == {"capixy-hair-serum", "capixy-anti-dandruff-serum-spray-120ml"}
    assert env["extract_calls"] == [("I need a hair serum without silicone.", None)]   # no context
    assert set(t.timings_ms) >= {"prequal", "extractor", "retrieval", "total"}


def test_intro_follows_language(env):
    env["prequal"] = pq("show me sunscreens", "Show me sunscreens.", language="en")
    env["filters"] = MetadataFilters(product_type=["sunscreen"])
    assert turn(env, "show me sunscreens").reply == "Here are matching products:"


def test_products_only_with_zero_results_goes_to_responder(env):
    env["prequal"] = pq("retinol serum no retinol", "I need a retinol serum without retinol.", language="en")
    env["filters"] = MetadataFilters(hero_ingredient=["Retinol"], ingredients=IngredientFilter(exclude=["Retinol"]))
    t = turn(env, "retinol serum no retinol")
    assert t.path == "responder" and t.reply == "responder reply"
    ctx = env["respond_calls"][0]
    assert ctx.reason == "no_results" and ctx.products      # closest options from text search
    assert t.retrieval["count"] == 0


def test_needs_response_with_retrieval_passes_products_and_context(env):
    env["prequal"] = pq("التاني ينفع للحامل؟", "Is Vacation Vitamin C Serum 10% safe during pregnancy?",
                        route="needs_response", intent="safety")
    env["filters"] = MetadataFilters(matched_handles=["vacation-vitamin-c-10-30-ml"])
    t = turn(env, "التاني ينفع للحامل؟")
    ctx = env["respond_calls"][0]
    assert t.path == "responder"
    assert ctx.reason == "needs_response" and ctx.intent == "safety" and ctx.persona == "customer"
    assert ctx.message == "التاني ينفع للحامل؟" and ctx.query_en.startswith("Is Vacation Vitamin C")
    assert [p["handle"] for p in ctx.products] == ["vacation-vitamin-c-10-30-ml"]


def test_no_retrieval_skips_extractor_and_keeps_last_products(env):
    env["prequal"] = pq("show me sunscreens", "Show me sunscreens.", language="en")
    env["filters"] = MetadataFilters(product_type=["sunscreen"])
    first = turn(env, "show me sunscreens")
    env["extract_calls"].clear()
    env["prequal"] = pq("thanks!", "Thanks!", route="needs_response", needs_retrieval=False, intent="greeting",
                        language="en")
    t = turn(env, "thanks!")
    assert env["extract_calls"] == [] and t.retrieval is None and t.products == []
    assert env["respond_calls"][-1].products == []
    saved = env["store"].get("s1")
    assert saved.last_products == [p["name"] for p in first.products]
    assert [m["content"] for m in saved.history] == ["show me sunscreens", "Here are matching products:",
                                                     "thanks!", "responder reply"]


def test_session_passes_history_to_next_turn(env):
    env["prequal"] = pq("عايزة سيروم للشعر", "I need a hair serum.")
    env["filters"] = MetadataFilters(product_type=["hair serum"])
    turn(env, "عايزة سيروم للشعر")
    env["prequal"] = pq("في واحد مفيهوش سيليكون؟", "I need a hair serum without silicone.")
    turn(env, "في واحد مفيهوش سيليكون؟")
    ctx_seen = env["ctx_seen"]
    assert ctx_seen.history[0]["content"] == "عايزة سيروم للشعر"
    assert "Capixy Hair Serum 120ml" in ctx_seen.last_products
    assert env["store"].get("s1").last_filters["product_type"] == ["hair serum"]


def test_skip_metadata_filters_uses_text_search(env):
    env["prequal"] = pq("عايزة سيروم للشعر", "عايزة سيروم للشعر", skip=True)
    t = turn(env, "عايزة سيروم للشعر")
    assert env["extract_calls"] == [] and t.filters is None
    assert t.retrieval["mode"] == "text"


def test_responder_error_still_replies(env, monkeypatch):
    async def broken(ctx):
        raise RuntimeError("down")
    monkeypatch.setattr(orch, "respond", broken)
    env["prequal"] = pq("hi", "Hi.", route="needs_response", needs_retrieval=False, intent="greeting", language="en")
    t = turn(env, "hi")
    assert t.path == "responder" and t.reply.startswith("Sorry")


def test_speculative_extractor_is_used(env, monkeypatch):
    monkeypatch.setattr(settings, "PREQUAL_SPECULATIVE_EXTRACTOR", True)
    env["prequal"] = pq("show me sunscreens", "Show me sunscreens.", language="en")
    env["filters"] = MetadataFilters(product_type=["sunscreen"])
    t = turn(env, "show me sunscreens")
    assert len(env["extract_calls"]) == 1 and t.path == "products_only"


def test_speculative_extractor_cancelled_when_no_retrieval(env, monkeypatch):
    monkeypatch.setattr(settings, "PREQUAL_SPECULATIVE_EXTRACTOR", True)
    env["extract_delay"] = 5.0
    env["prequal"] = pq("thanks", "Thanks.", route="needs_response", needs_retrieval=False, intent="greeting",
                        language="en")
    t = turn(env, "thanks")
    assert len(env["extract_calls"]) == 1 and env["extract_cancelled"]
    assert t.filters is None and t.path == "responder"
