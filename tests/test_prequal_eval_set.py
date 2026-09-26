"""The eval set is well formed, its labels use the schema enums, and every
expected filter value exists in the catalog. Also the rewrite-check scorer."""

import json
from pathlib import Path

import pytest

from agents.filter_extractor.catalog import get_catalog
from agents.prequal.schemas import INTENTS, LANGUAGES, PERSONAS, ROUTES
from scripts.eval_prequal import rewrite_check

def _load(name):
    return [json.loads(line) for line in (Path(__file__).parent / name).read_text(encoding="utf-8").splitlines()
            if line.strip()]


DEV = _load("prequal_eval.jsonl")
ROWS = DEV + _load("prequal_holdout.jsonl")
REQUIRED_TAGS = {"follow_up_constraint", "reference", "topic_change", "doctor", "sales_trainee", "greeting",
                 "order", "arabizi", "typos"}
KEY_TO_ALLOWED = {"include": "ingredients", "exclude": "ingredients"}


def test_at_least_40_conversations_covering_every_case():
    assert len(DEV) >= 40
    assert len({r["id"] for r in ROWS}) == len(ROWS)
    assert REQUIRED_TAGS <= {t for r in DEV for t in r["tags"]}


@pytest.mark.parametrize("row", ROWS, ids=[r["id"] for r in ROWS])
def test_row_labels(row):
    e = row["expected"]
    assert e["route"] in ROUTES and isinstance(e["needs_retrieval"], bool)
    assert set(e["intent"]) <= set(INTENTS) and set(e["persona"]) <= set(PERSONAS)
    assert set(e["language"]) <= set(LANGUAGES)
    if e["route"] == "products_only":
        assert e["needs_retrieval"]
    assert (row["filters"] is not None) == e["needs_retrieval"]
    assert not rewrite_check(e["query_en"], e), "the reference rewrite must pass its own check"


@pytest.mark.parametrize("row", [r for r in ROWS if r["filters"]], ids=[r["id"] for r in ROWS if r["filters"]])
def test_expected_filters_exist_in_catalog(row):
    cat = get_catalog()
    handles = {n.handle for n in cat.names}
    for bucket in (row["filters"], row["optional_filters"]):
        for key, values in bucket.items():
            allowed = handles if key == "handles" else set(cat.allowed[KEY_TO_ALLOWED.get(key, key)])
            assert set(values) <= allowed, (key, set(values) - allowed)


def test_rewrite_check():
    exp = {"must_include": [["hair"], ["sunscreen", "sun block"]], "must_not_include": ["silicone"]}
    assert rewrite_check("Do you have a Sun Block for hair?", exp) == []
    assert rewrite_check("Do you have sunscreen?", exp) == ["missing ['hair']"]
    assert rewrite_check("hair sunscreen without silicone", exp) == ["has 'silicone'"]
