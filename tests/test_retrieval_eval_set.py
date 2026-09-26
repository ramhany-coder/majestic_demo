"""tests/retrieval_eval.jsonl is well formed and uses only catalog values, and
the plan's required cases behave as specified. Runs offline without the
embedding model (semantic "failed"), so it checks the deterministic part."""

import json
from pathlib import Path

import pytest

from agents.filter_extractor.catalog import get_catalog
from agents.prequal.schemas import INTENTS
from agents.retrieval.agent import retrieve
from agents.retrieval.index_builder import get_index
from agents.retrieval.semantic import SemanticScores
from models.filter_extractor import MetadataFilters
from scripts.eval_retrieval import score_row

ROWS = [json.loads(line) for line in (Path(__file__).parent / "retrieval_eval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()]
BY_ID = {r["id"]: r for r in ROWS}
NO_SEM = SemanticScores(status="failed", error="offline test")
ALLOWED_KEY = {"hero_ingredient": "hero_ingredient", "include": "ingredients", "exclude": "ingredients"}


def run(row):
    f = MetadataFilters(**row["filters"]) if row["filters"] is not None else None
    return retrieve(f, NO_SEM, row["query_en"], row["intent"], row["k"])


def test_at_least_40_cases_with_the_plan_cases():
    assert len(ROWS) >= 40 and len(BY_ID) == len(ROWS)
    tags = {t for r in ROWS for t in r["tags"]}
    assert {"plan", "exclude", "relaxation", "compare", "product_line", "k", "names", "typos", "arabic"} <= tags


@pytest.mark.parametrize("row", ROWS, ids=[r["id"] for r in ROWS])
def test_row_is_well_formed(row):
    cat, handles = get_catalog(), set(get_index().order)
    assert row["intent"] in INTENTS and row["relevant"]
    assert row["k"] is None or 1 <= row["k"] <= 20
    assert set(row["relevant"]) <= handles and set(row.get("forbidden", [])) <= handles
    assert not set(row["relevant"]) & set(row.get("forbidden", []))
    if row["filters"] is None:
        return
    MetadataFilters(**row["filters"])                     # parses
    for key, values in row["filters"].items():
        if key in ("name_en", "name_ar", "unmatched"):
            continue
        if key == "ingredients":
            for sub, vals in values.items():
                assert set(vals) <= set(cat.allowed["ingredients"]), (sub, vals)
        else:
            assert set(values) <= set(cat.allowed[ALLOWED_KEY.get(key, key)]), (key, values)


@pytest.mark.parametrize("row", [r for r in ROWS if r.get("forbidden")], ids=[r["id"] for r in ROWS if r.get("forbidden")])
def test_never_returns_a_forbidden_product(row):
    assert score_row(row, run(row).handles, 20)["forbidden"] == []


def test_plan_case_hair_serum_without_silicone():
    res = run(BY_ID["r01"])
    assert set(res.handles) == set(BY_ID["r01"]["relevant"])


def test_plan_case_sunscreen_spray_relaxes_the_form():
    res = run(BY_ID["r02"])
    assert res.relaxed_keys == ["product_form"] and all(p.product_type == "sunscreen" for p in res.products)


def test_plan_case_pregnant_without_retinol_or_salicylic():
    res = run(BY_ID["r03"])
    assert res.relaxed_keys == ["suitable_for"]
    assert set(res.handles) == set(BY_ID["r03"]["relevant"])
    assert res.applied_filters["ingredients.exclude"] == ["Retinol", "Salicylic Acid"]


def test_plan_case_compare_returns_only_name_hits():
    res = run(BY_ID["r04"])
    assert set(BY_ID["r04"]["relevant"]) <= set(res.name_hits) and res.handles == res.name_hits[: res.k]


def test_plan_case_deodorant_line_variants():
    res = run(BY_ID["r05"])
    assert set(BY_ID["r05"]["relevant"]) <= set(res.handles[:7])


def test_plan_case_k_3():
    assert len(run(BY_ID["r06"]).products) == 3
