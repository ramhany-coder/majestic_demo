"""Orchestrator routing with prequal, the extractor and the semantic search
stubbed out (retrieval and the session store are real)."""

import asyncio

import pytest

import agents.orchestrator.orchestrator as orch
from agents.prequal.session_context import SessionStore
from agents.retrieval.semantic import SemanticScores
from config import settings
from models.filter_extractor import IngredientFilter, MetadataFilters
from models.prequal import PrequalResult
from models.retrieval import RetrievalResult


def pq(message, query_en, route="products_only", needs_retrieval=True, intent="find_products",
       language="ar", skip=False, k=10):
    return PrequalResult(query_original=message, query_en=query_en, language=language, is_follow_up=False,
                         route=route, needs_retrieval=needs_retrieval, intent=intent, persona="customer",
                         k=k, skip_metadata_filters=skip, meta={})


@pytest.fixture
def env(monkeypatch):
    state = {"prequal": None, "filters": MetadataFilters(), "extract_calls": [], "respond_calls": [],
             "extract_delay": 0.0, "extract_cancelled": False, "semantic_calls": [], "semantic_cancelled": False,
             "semantic": SemanticScores(status="ok")}

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

    async def fake_semantic(query):
        state["semantic_calls"].append(query)
        try:
            await asyncio.sleep(state["extract_delay"])
        except asyncio.CancelledError:
            state["semantic_cancelled"] = True
            raise
        return state["semantic"]

    async def fake_respond(ctx):
        state["respond_calls"].append(ctx)
        return "responder reply"

    monkeypatch.setattr(orch, "prequalify", fake_prequalify)
    monkeypatch.setattr(orch, "extract_filters", fake_extract)
    monkeypatch.setattr(orch, "semantic_search", fake_semantic)
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
    assert t.reply == "دي المنتجات اللي تناسب طلبك:"
    assert {p["handle"] for p in t.products} == {"capixy-hair-serum", "capixy-anti-dandruff-serum-spray-120ml"}
    assert env["extract_calls"] == [("I need a hair serum without silicone.", None)]   # no context
    assert env["semantic_calls"] == ["I need a hair serum without silicone."]         # in parallel
    assert set(t.timings_ms) >= {"prequal", "extractor", "semantic", "retrieval", "total"}
    assert t.retrieval["k"] == 10 and t.retrieval["relaxed_keys"] == []


def test_router_k_cuts_the_list(env):
    env["prequal"] = pq("warreeni 3 deodorant", "Show me 3 deodorants.", language="arabizi", k=3)
    env["filters"] = MetadataFilters(product_type=["deodorant"])
    t = turn(env, "warreeni 3 deodorant")
    assert t.path == "products_only" and len(t.products) == 3 and t.retrieval["k"] == 3


def test_intro_follows_language(env):
    env["prequal"] = pq("show me sunscreens", "Show me sunscreens.", language="en")
    env["filters"] = MetadataFilters(product_type=["sunscreen"])
    assert turn(env, "show me sunscreens").reply == "Here are matching products:"


def test_products_only_with_relaxed_filters_goes_to_responder(env):
    # No sunscreen is a roll-on: product_form is relaxed, so the responder explains.
    env["prequal"] = pq("sunscreen roll on", "I need a sunscreen roll-on.", language="en")
    env["filters"] = MetadataFilters(product_type=["sunscreen"], product_form=["roll-on"])
    t = turn(env, "sunscreen roll on")
    assert t.path == "responder" and t.reply == "responder reply"
    ctx = env["respond_calls"][0]
    assert ctx.reason == "relaxed" and ctx.products and all(p["product_type"] == "sunscreen" for p in ctx.products)
    assert ctx.retrieval.relaxed_keys == ["product_form"]
    assert ctx.retrieval.meta["relaxed_values"] == {"product_form": ["roll-on"]}
    assert t.retrieval["relaxed_keys"] == ["product_form"] and t.products == ctx.products


def test_products_only_with_zero_results_goes_to_responder(env, monkeypatch):
    monkeypatch.setattr(orch, "retrieve", lambda *a, **kw: RetrievalResult(products=[], k=10))
    env["prequal"] = pq("show me sunscreens", "Show me sunscreens.", language="en")
    env["filters"] = MetadataFilters(product_type=["sunscreen"])
    t = turn(env, "show me sunscreens")
    assert t.path == "responder" and env["respond_calls"][0].reason == "no_results"
    assert t.products == [] and t.retrieval["handles"] == []


def test_needs_response_with_retrieval_passes_products_and_context(env):
    env["prequal"] = pq("التاني ينفع للحامل؟", "Is Vacation Vitamin C Serum 10% safe during pregnancy?",
                        route="needs_response", intent="safety")
    env["filters"] = MetadataFilters(name_en=["Vacation Vitamin C Serum 10%"], suitable_for=["pregnancy"])
    t = turn(env, "التاني ينفع للحامل؟")
    ctx = env["respond_calls"][0]
    assert t.path == "responder"
    assert ctx.reason == "needs_response" and ctx.intent == "safety" and ctx.persona == "customer"
    assert ctx.message == "التاني ينفع للحامل؟" and ctx.query_en.startswith("Is Vacation Vitamin C")
    # A safety question returns only the named product; it isn't labeled for pregnancy.
    assert [p["handle"] for p in ctx.products] == ["vacation-vitamin-c-10-30-ml"]
    assert ctx.products[0]["match"]["conflict"] == "filter_mismatch"
    assert ctx.retrieval.name_hits == ["vacation-vitamin-c-10-30-ml"]


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
    assert saved.last_products == [p["name"] for p in first.products][: settings.PREQUAL_LAST_PRODUCTS]
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


def test_skip_metadata_filters_uses_semantic_search_only(env):
    env["prequal"] = pq("عايزة سيروم للشعر", "عايزة سيروم للشعر", skip=True)
    env["semantic"] = SemanticScores({"capixy-hair-serum": 0.9}, status="ok")
    t = turn(env, "عايزة سيروم للشعر")
    assert env["extract_calls"] == [] and t.filters is None
    assert env["semantic_calls"] == ["عايزة سيروم للشعر"]
    assert t.retrieval["meta"]["filters_skipped"] and t.products[0]["handle"] == "capixy-hair-serum"


def test_semantic_failure_still_returns_products(env):
    env["prequal"] = pq("show me sunscreens", "Show me sunscreens.", language="en")
    env["filters"] = MetadataFilters(product_type=["sunscreen"])
    env["semantic"] = SemanticScores(status="timeout", error="timeout after 1.5s")
    t = turn(env, "show me sunscreens")
    assert t.path == "products_only" and t.products
    assert t.retrieval["fallback"] == "no_semantic" and t.retrieval["meta"]["semantic"] == "timeout"


def test_extractor_bug_falls_back_to_semantic_path(env, monkeypatch):
    async def broken(query, context=None, use_cache=True):
        raise RuntimeError("bug")
    monkeypatch.setattr(orch, "extract_filters", broken)
    env["prequal"] = pq("show me sunscreens", "Show me sunscreens.", language="en")
    env["semantic"] = SemanticScores({"vacation-sunscreen-cream-60ml": 0.8}, status="ok")
    t = turn(env, "show me sunscreens")
    assert t.filters is None and t.products[0]["handle"] == "vacation-sunscreen-cream-60ml"


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
    assert len(env["extract_calls"]) == 1 and len(env["semantic_calls"]) == 1 and t.path == "products_only"


def test_speculative_extractor_cancelled_when_no_retrieval(env, monkeypatch):
    monkeypatch.setattr(settings, "PREQUAL_SPECULATIVE_EXTRACTOR", True)
    env["extract_delay"] = 5.0
    env["prequal"] = pq("thanks", "Thanks.", route="needs_response", needs_retrieval=False, intent="greeting",
                        language="en")
    t = turn(env, "thanks")
    assert len(env["extract_calls"]) == 1 and env["extract_cancelled"]
    assert len(env["semantic_calls"]) == 1 and env["semantic_cancelled"]
    assert t.filters is None and t.path == "responder"


def test_prequal_end_stops_the_graph(env):
    env["prequal"] = pq("show me sunscreens", "Show me sunscreens.", language="en")
    env["filters"] = MetadataFilters(product_type=["sunscreen"])
    turn(env, "show me sunscreens")
    shown_before = env["store"].get("s1").last_products
    env["extract_calls"].clear()
    greeting = pq("thanks!", "Thank you.", route="needs_response", needs_retrieval=False, intent="greeting",
                  language="en")
    greeting.end, greeting.reply = True, "You're welcome!"
    env["prequal"] = greeting
    t = turn(env, "thanks!")
    assert t.path == "small_talk" and t.reply == "You're welcome!"
    assert env["extract_calls"] == [] and env["respond_calls"] == []
    assert t.products == [] and t.retrieval is None and t.filters is None
    assert set(t.timings_ms) == {"prequal", "total"}
    saved = env["store"].get("s1")
    assert saved.last_products == shown_before
    assert saved.history[-2:] == [{"role": "user", "content": "thanks!"},
                                  {"role": "assistant", "content": "You're welcome!"}]


def test_prequal_end_cancels_speculative_extractor(env, monkeypatch):
    monkeypatch.setattr(settings, "PREQUAL_SPECULATIVE_EXTRACTOR", True)
    env["extract_delay"] = 5.0
    greeting = pq("hi there friend", "Hello.", route="needs_response", needs_retrieval=False,
                  intent="greeting", language="en")
    greeting.end, greeting.reply = True, "Hi!"
    env["prequal"] = greeting
    t = turn(env, "hi there friend")
    assert env["extract_cancelled"] and t.path == "small_talk"


def test_on_stage_reports_each_stage_that_runs(env):
    env["prequal"] = pq("عايزة سيروم للشعر", "I need a hair serum.")
    env["filters"] = MetadataFilters(product_type=["hair serum"])
    stages = []
    t = asyncio.run(orch.handle_message("عايزة سيروم للشعر", "s1", env["store"], on_stage=stages.append))
    assert t.path == "products_only" and stages == ["understanding", "searching"]

    env["prequal"] = pq("ده بيتحط ازاي؟", "How do I use Capixy Hair Serum?", route="needs_response",
                        intent="how_to_use")
    stages.clear()
    t = asyncio.run(orch.handle_message("ده بيتحط ازاي؟", "s1", env["store"], on_stage=stages.append))
    assert t.path == "responder" and stages == ["understanding", "searching", "writing"]


def test_a_failing_stage_listener_does_not_break_the_turn(env):
    env["prequal"] = pq("عايزة سيروم للشعر", "I need a hair serum.")
    env["filters"] = MetadataFilters(product_type=["hair serum"])

    def broken(stage):
        raise RuntimeError("listener bug")

    t = asyncio.run(orch.handle_message("عايزة سيروم للشعر", "s1", env["store"], on_stage=broken))
    assert t.path == "products_only" and t.products
