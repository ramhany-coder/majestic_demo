"""Schemas, code-side validation, prompt split / budget and script detection."""

import pytest

from agents.filter_extractor.prompts import count_tokens
from agents.prequal.prompts import get_template, split_template
from agents.prequal.rewriter import detect_language, rewrite_default
from agents.prequal.schemas import (
    INTENTS, PERSONAS, REWRITER_SCHEMA, ROUTER_SCHEMA, validate_rewrite, validate_route,
)

# Offline tiktoken estimate. The plan estimated ~450 / ~400; the rewriter is
# larger because its Arabic examples tokenize heavily, and the router grew
# ~100 tokens when it was tuned after the first live eval (ARCHITECTURE_NOTES
# section 9). Live counts are reported by scripts/eval_prequal.py.
PROMPT_BUDGETS = {"rewriter": 600, "router": 500}


def test_schemas_are_strict_objects_with_titles():
    for schema in (REWRITER_SCHEMA, ROUTER_SCHEMA):
        assert schema["title"] and schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
    assert REWRITER_SCHEMA["properties"]["query_en"]["maxLength"] == 400
    assert ROUTER_SCHEMA["properties"]["intent"]["enum"] == list(INTENTS)
    assert ROUTER_SCHEMA["properties"]["persona"]["enum"] == list(PERSONAS)


def test_validate_rewrite_ok_and_normalizes():
    out, notes = validate_rewrite({"query_en": "  I need a hair serum\nwithout silicone. ", "language": "AR",
                                   "is_follow_up": True})
    assert out == {"query_en": "I need a hair serum without silicone.", "language": "ar", "is_follow_up": True}
    assert notes == []


def test_validate_rewrite_cuts_long_query():
    out, notes = validate_rewrite({"query_en": "x" * 450, "language": "en", "is_follow_up": False})
    assert len(out["query_en"]) == 400 and notes


@pytest.mark.parametrize("bad", [
    {"query_en": "", "language": "en", "is_follow_up": False},
    {"query_en": "sunscreen", "language": "french", "is_follow_up": False},
    {"query_en": "sunscreen", "language": "en", "is_follow_up": "yes"},
])
def test_validate_rewrite_rejects(bad):
    with pytest.raises(ValueError):
        validate_rewrite(bad)


def test_validate_route_fixes_products_only_without_retrieval():
    out, notes = validate_route({"route": "products_only", "needs_retrieval": False,
                                 "intent": "find_products", "persona": "customer"})
    assert out["needs_retrieval"] is True and notes


def test_validate_route_follows_a_product_list_intent():
    data = {"route": "needs_response", "needs_retrieval": True, "intent": "refine_products", "persona": "customer"}
    out, notes = validate_route(data)
    assert out["route"] == "products_only" and notes
    assert validate_route(data, route_from_intent=False)[0]["route"] == "needs_response"
    # A question intent keeps needs_response.
    q = {"route": "needs_response", "needs_retrieval": True, "intent": "how_to_use", "persona": "customer"}
    assert validate_route(q)[0]["route"] == "needs_response"


@pytest.mark.parametrize("bad", [
    {"route": "maybe", "needs_retrieval": True, "intent": "other", "persona": "unknown"},
    {"route": "needs_response", "needs_retrieval": True, "intent": "shopping", "persona": "unknown"},
    {"route": "needs_response", "needs_retrieval": True, "intent": "other", "persona": "pharmacist"},
])
def test_validate_route_rejects(bad):
    with pytest.raises(ValueError):
        validate_route(bad)


@pytest.mark.parametrize("key", ["rewriter", "router"])
def test_static_prefix_is_query_independent_and_query_is_last(key):
    t = get_template(key)
    a = t.messages("q one", [{"role": "user", "content": "hi"}], ["P1"])
    b = t.messages("q two", [], [])
    assert a[0].content == b[0].content                      # cacheable prefix
    assert "{{" not in a[0].content and "{{" not in a[1].content
    assert a[1].content.startswith("HISTORY:\nU: hi")
    assert a[1].content.endswith("QUERY: q one")
    assert "LAST_PRODUCTS: 1) P1" in a[1].content
    assert "HISTORY:\n(none)\nLAST_PRODUCTS: none" in b[1].content


def test_split_template_cuts_at_history_label():
    static, dynamic = split_template("rules\nExample\nHISTORY: U: x\n\nHISTORY:\n{{history}}\nQUERY: {{query}}")
    assert static == "rules\nExample\nHISTORY: U: x"
    assert dynamic == "HISTORY:\n{{history}}\nQUERY: {{query}}"


@pytest.mark.parametrize("key", list(PROMPT_BUDGETS))
def test_prompt_within_budget(key):
    assert count_tokens(get_template(key).full_text("")) <= PROMPT_BUDGETS[key]


@pytest.mark.parametrize("text,lang", [
    ("عايزة سيروم للشعر", "ar"),
    ("في واحد مفيهوش سيليكون؟", "ar"),
    ("3ayza serum lel wesh", "arabizi"),
    ("el tany yenfa3 lel 7amel?", "arabizi"),
    ("ana 3ayez shampoo", "arabizi"),
    ("need a sunscreen with spf50 and vitamin b3", "en"),
    ("serum for men 30ml", "en"),
    ("عايز sunscreen للبشرة الدهنية", "mixed"),
    ("👍", "en"),
])
def test_detect_language(text, lang):
    assert detect_language(text) == lang


def test_rewrite_default_uses_filters_only_for_standalone_english():
    assert rewrite_default("sunscreen for oily skin")["skip_metadata_filters"] is False
    assert rewrite_default("any cheaper one?", [{"role": "user", "content": "x"}])["skip_metadata_filters"] is True
    ar = rewrite_default("عايزة صن بلوك")
    assert ar == {"query_en": "عايزة صن بلوك", "language": "ar", "is_follow_up": False, "skip_metadata_filters": True}
