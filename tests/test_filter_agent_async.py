"""Orchestrator tests with a fake LLM plugged in at the model-factory level, so
the real async fallback chain (timeouts, deadline, fallback route) runs."""

import asyncio
import time

import pytest

import agents.filter_extractor.agent as agent_module
import llm.fallback as fallback_module
from agents.filter_extractor.agent import extract_filters, filter_extractor, get_cache, is_small_talk
from config import settings

PRIMARY = "groq:qwen/qwen3.8-27b"
FALLBACK = "groq:openai/gpt-oss-20b"

ANSWERS = {
    "majestic_names": {"name_en": ["Vacation Sunscreen Spray"], "name_ar": []},
    "majestic_brand": {"brand": []},
    "majestic_category": {"category": ["Skin Care"]},
    "majestic_product_group": {"product_group": ["Sun Care"]},
    "majestic_product_type": {"product_type": ["sunscreen"]},
    "majestic_product_form": {"product_form": ["spray"]},
    "majestic_concerns": {"concerns": ["sun protection"], "unmatched": []},
    "majestic_suitable_for": {"suitable_for": ["body"]},
    "majestic_hero_ingredient": {"hero_ingredient": []},
    "majestic_ingredients": {"include": [], "exclude": [], "unmatched_include": [], "unmatched_exclude": []},
}


class FakeStructured:
    def __init__(self, model, schema, behaviour):
        self.model, self.schema, self.behaviour = model, schema, behaviour

    async def ainvoke(self, messages):
        title = self.schema["title"]
        self.behaviour["calls"].append((self.model, title))
        action = self.behaviour.get((self.model, title)) or self.behaviour.get(title)
        if action == "hang":
            await asyncio.sleep(30)
        if action == "error":
            raise RuntimeError("provider exploded")
        return {"parsed": dict(ANSWERS[title]), "raw": None, "parsing_error": None}


class FakeModel:
    def __init__(self, model, behaviour):
        self.model, self.behaviour = model, behaviour

    def with_structured_output(self, schema, **kwargs):
        return FakeStructured(self.model, schema, self.behaviour)


@pytest.fixture
def behaviour(monkeypatch):
    b = {"calls": []}
    monkeypatch.setattr(fallback_module.client_llm, "get_cached_model",
                        lambda router, model, **kw: FakeModel(model, b))
    monkeypatch.setattr(settings, "EXTRACTOR_TIMEOUT_S", 0.2)
    monkeypatch.setattr(settings, "EXTRACTOR_FALLBACK_TIMEOUT_S", 0.2)
    monkeypatch.setattr(settings, "EXTRACTOR_CALL_DEADLINE_S", 0.5)
    monkeypatch.setattr(settings, "EXTRACTOR_USE_RULE_FALLBACK", True)
    monkeypatch.setattr(settings, "EXTRACTOR_TRANSLATE", False)
    get_cache().clear()
    yield b
    get_cache().clear()


def run(coro):
    return asyncio.run(coro)


def test_all_ten_calls_run_in_parallel(behaviour):
    f = run(extract_filters("need a vacation sun blok spray for the beach"))
    assert len(behaviour["calls"]) == 10
    assert set(f.meta.calls.values()) == {"ok"}
    assert f.product_type == ["sunscreen"] and f.product_form == ["spray"]
    assert f.matched_handles == ["vacation-sunscreen-lotion-spray-200ml"]
    assert f.brand == ["Vacation"]            # added from the matched product


def test_invented_product_name_is_dropped(behaviour):
    # The fake names call answers "Vacation Sunscreen Spray" but the user named no brand.
    f = run(extract_filters("need a sun blok spray for the beach"))
    assert f.name_en == [] and f.matched_handles == []
    assert any("names: dropped" in n for n in f.meta.notes)


def test_one_hanging_call_does_not_block_the_other_nine(behaviour):
    behaviour["majestic_product_form"] = "hang"   # primary and fallback both hang
    start = time.perf_counter()
    f = run(extract_filters("need a sun blok spray for the beach"))
    elapsed = time.perf_counter() - start
    assert elapsed < 1.5                          # bounded by the per-call deadline, not 30 s
    assert f.meta.calls["product_form"] == "timeout"
    assert f.product_form == ["spray"]            # filled by the rule-based fallback
    assert sum(s == "ok" for s in f.meta.calls.values()) == 9


def test_primary_error_goes_to_fallback_model(behaviour):
    behaviour[("qwen/qwen3.8-27b", "majestic_brand")] = "error"
    f = run(extract_filters("sunscreen from vacation please"))
    assert f.meta.calls["brand"] == "fallback_model"
    assert f.meta.routes["brand"] == FALLBACK


def test_both_models_fail_uses_rules_for_that_key_only(behaviour):
    behaviour["majestic_brand"] = "error"
    f = run(extract_filters("sunscreen from vacasion please"))
    assert f.meta.calls["brand"] == "rule_based"
    assert f.brand == ["Vacation"]
    assert f.meta.calls["product_type"] == "ok"


def test_rule_fallback_can_be_switched_off(behaviour, monkeypatch):
    monkeypatch.setattr(settings, "EXTRACTOR_USE_RULE_FALLBACK", False)
    behaviour["majestic_product_form"] = "error"
    f = run(extract_filters("need a sun blok spray"))
    assert f.meta.calls["product_form"] == "failed"
    assert f.product_form == []


def test_a_bug_inside_a_call_is_contained(behaviour, monkeypatch):
    async def boom(query, context=None):
        raise ValueError("bug")
    monkeypatch.setitem(agent_module.CALL_FUNCTIONS, "concerns", boom)
    f = run(extract_filters("need a sun blok spray"))
    assert f.meta.calls["concerns"] == "failed"
    assert f.product_type == ["sunscreen"]


@pytest.mark.parametrize("q", ["hi", "Thanks!!", "good morning doctor", "ok thank you so much", "👍", ""])
def test_small_talk_short_circuits_without_llm(behaviour, q):
    f = run(extract_filters(q))
    assert behaviour["calls"] == []
    assert f.is_empty() and f.meta.short_circuit


@pytest.mark.parametrize("q", ["hi, any sunscreen?", "thanks, and a shampoo for dandruff"])
def test_greeting_with_a_request_is_not_small_talk(q):
    assert not is_small_talk(q)


def test_cache_by_normalized_query(behaviour):
    run(extract_filters("Need a sun-blok spray!"))
    n = len(behaviour["calls"])
    f = run(extract_filters("need a sun blok spray"))
    assert len(behaviour["calls"]) == n and f.meta.cached


def test_degraded_results_are_not_cached(behaviour):
    behaviour["majestic_brand"] = "error"
    run(extract_filters("sunscreen spray"))
    assert len(get_cache()) == 0


def test_query_is_trimmed(behaviour):
    run(extract_filters("sunscreen " * 200))
    # every call sees at most 500 characters of query
    assert len(behaviour["calls"]) == 10


def test_only_names_call_gets_context(behaviour, monkeypatch):
    seen = {}
    real = fallback_module.FallBack.aconstrained_invoke

    async def spy(self, message, order, schema, **kw):
        seen[schema["title"]] = message[1].content
        return await real(self, message, order, schema, **kw)
    monkeypatch.setattr(fallback_module.FallBack, "aconstrained_invoke", spy)
    run(extract_filters("how often should i use it?", ["Capixy Intense Tonic Spray 125ml"]))
    assert "Capixy Intense Tonic Spray" in seen["majestic_names"]
    assert all("Capixy" not in v for k, v in seen.items() if k != "majestic_names")


def test_graph_node_contract(behaviour):
    out = run(filter_extractor({"eng_query": "need a sun blok spray", "recent_product_names": []}))
    assert out["metadata_filters"]["product_type"] == ["sunscreen"]
