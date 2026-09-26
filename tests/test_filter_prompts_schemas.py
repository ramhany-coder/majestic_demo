import copy
import json

import pytest

from agents.filter_extractor.catalog import (
    CATALOG_KEYS, METADATA_PATH, PRODUCTS_PATH, build_catalog, get_catalog,
)
from agents.filter_extractor.prompts import (
    ALLOWED_SEP, build_template, count_tokens, get_template, split_template,
)
from agents.filter_extractor.schemas import (
    CALL_SPECS, CALLS_BY_KEY, MAX_ITEMS_CATALOG, build_schema, get_schema, snap_value, validate_output,
)

# Plan section 3 (rough estimates). Measured with tiktoken o200k_base here;
# real provider counts are printed by `python -m scripts.eval_extractor`.
PROMPT_BUDGETS = {"names": 261, "brand": 207, "category": 187, "product_group": 231, "product_type": 342,
                  "product_form": 216, "concerns": 334, "suitable_for": 267, "hero_ingredient": 284,
                  "ingredients": 1185}
EXPECTED_COUNTS = {"brand": 9, "category": 4, "product_group": 12, "product_type": 39, "product_form": 21,
                   "concerns": 36, "suitable_for": 17, "hero_ingredient": 11, "ingredients": 236}


def _load():
    return (json.loads(PRODUCTS_PATH.read_text(encoding="utf-8")),
            json.loads(METADATA_PATH.read_text(encoding="utf-8")))


def test_catalog_counts_match_the_plan():
    cat = get_catalog()
    assert len(cat.products) == 108
    assert {k: len(v) for k, v in cat.allowed.items()} == EXPECTED_COUNTS


@pytest.mark.parametrize("spec", [s for s in CALL_SPECS if s.enum_key], ids=lambda s: s.key)
def test_prompt_allowed_line_is_built_from_catalog(spec):
    cat = get_catalog()
    system = get_template(spec.key).system
    assert f"ALLOWED ({spec.enum_key}): " + ALLOWED_SEP.join(cat.allowed[spec.enum_key]) in system


def test_prompts_follow_a_changed_catalog_without_code_changes():
    products, metadata = _load()
    metadata = copy.deepcopy(metadata)
    metadata["allowed_values"]["brand"].append("NewBrand")
    metadata["product_line_to_brand"]["Glow-Max"] = "NewBrand"
    cat = build_catalog(products, metadata)
    tpl = build_template(CALLS_BY_KEY["brand"], cat)
    assert "NewBrand" in tpl.system
    assert "Glow-Max → NewBrand" in tpl.system
    assert "NewBrand" in build_schema(CALLS_BY_KEY["brand"], cat)["properties"]["brand"]["items"]["enum"]


def test_allowed_line_can_be_left_to_the_schema():
    tpl = build_template(CALLS_BY_KEY["brand"], get_catalog(), allowed_in_prompt=False)
    assert "ALLOWED" not in tpl.system


@pytest.mark.parametrize("spec", CALL_SPECS, ids=lambda s: s.key)
def test_static_prefix_first_and_query_last(spec):
    tpl = get_template(spec.key)
    assert "{{" not in tpl.system
    msgs = tpl.messages("QUERY_MARKER", ["Capixy Cream"])
    assert msgs[0].content == tpl.system                     # identical across requests -> cacheable
    assert msgs[1].content.rstrip().endswith("Q: QUERY_MARKER")
    assert ("CONTEXT: Capixy Cream" in msgs[1].content) == spec.uses_context


def test_split_template_cuts_at_first_dynamic_line():
    static, dynamic = split_template("a\nb\nCONTEXT: {{context}}\nQ: {{query}}")
    assert static == "a\nb" and dynamic == "CONTEXT: {{context}}\nQ: {{query}}"


@pytest.mark.parametrize("key", [k for k in PROMPT_BUDGETS if k != "ingredients"])
def test_prompt_within_10_percent_of_budget(key):
    assert count_tokens(get_template(key).full_text("")) <= PROMPT_BUDGETS[key] * 1.10


@pytest.mark.xfail(strict=True, reason="236-ingredient list alone is ~1150 tokens; the plan's 1185 "
                                       "estimate leaves no room for the instructions (see ARCHITECTURE_NOTES.md)")
def test_ingredients_prompt_within_10_percent_of_budget():
    assert count_tokens(get_template("ingredients").full_text("")) <= PROMPT_BUDGETS["ingredients"] * 1.10


@pytest.mark.parametrize("spec", CALL_SPECS, ids=lambda s: s.key)
def test_schema_shape(spec):
    schema = get_schema(spec.key)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == [f.name for f in spec.fields]
    assert schema["title"].replace("_", "").isalnum()
    cat = get_catalog()
    for f in spec.fields:
        prop = schema["properties"][f.name]
        assert prop["type"] == "array" and prop["maxItems"] == f.max_items
        if f.enum_key:
            assert prop["items"]["enum"] == cat.allowed[f.enum_key]
            assert f.max_items == MAX_ITEMS_CATALOG
        else:
            assert "enum" not in prop["items"]


def test_names_and_unmatched_limits():
    names = get_schema("names")["properties"]
    assert names["name_en"]["maxItems"] == 5 and names["name_ar"]["maxItems"] == 5
    assert get_schema("concerns")["properties"]["unmatched"]["maxItems"] == 3
    ing = get_schema("ingredients")["properties"]
    assert ing["unmatched_include"]["maxItems"] == 3 and ing["unmatched_exclude"]["maxItems"] == 3


def test_snap_value():
    allowed = get_catalog().allowed["product_type"]
    assert snap_value("face serum", allowed) == "face serum"
    assert snap_value("Face Serum", allowed) == "face serum"
    assert snap_value("sunscren", allowed) == "sunscreen"
    assert snap_value("face mist", allowed) is None
    assert snap_value("", allowed) is None


def test_validate_output_snaps_drops_dedupes_and_truncates():
    spec = CALLS_BY_KEY["concerns"]
    data = {"concerns": ["Acne", "acne", "hyperpigmentaton", "made up concern"] + ["dryness"] * 3,
            "unmatched": ["a", "b", "c", "d"]}
    out, notes = validate_output(spec, data)
    assert out["concerns"] == ["acne", "hyperpigmentation", "dryness"]
    assert out["unmatched"] == ["a", "b", "c"]
    assert any("dropped" in n for n in notes)


def test_validate_output_tolerates_missing_and_wrong_types():
    out, _ = validate_output(CALLS_BY_KEY["ingredients"], {"include": "Retinol", "exclude": None})
    assert out == {"include": ["Retinol"], "exclude": [], "unmatched_include": [], "unmatched_exclude": []}
    out, _ = validate_output(CALLS_BY_KEY["brand"], "not a dict")
    assert out == {"brand": []}
