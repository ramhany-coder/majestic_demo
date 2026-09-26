"""tests/responder_eval.jsonl is well formed, covers the plan's cases, and its
deterministic expectations (mode, cards, disclaimer, reply language) hold.
Offline: R1 and R2 are replaced by failures, so answers are templates."""

import asyncio
import json
from collections import Counter
from pathlib import Path

import pytest

import agents.responder.agent as agent
from agents.prequal.schemas import INTENTS, LANGUAGES, PERSONAS
from llm.fallback import AllRoutesFailed
from scripts.eval_responder import run_row, score_row

ROWS = [json.loads(line) for line in (Path(__file__).parent / "responder_eval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()]


def test_at_least_50_cases_covering_the_plan():
    assert len(ROWS) >= 50 and len({r["id"] for r in ROWS}) == len(ROWS)
    intents = {r["intent"] for r in ROWS}
    assert {"how_to_use", "safety", "compare", "product_info", "price_offer", "sales_training", "greeting",
            "out_of_scope", "other", "find_products"} <= intents
    assert {"customer", "sales_trainee", "doctor"} <= {r["persona"] for r in ROWS}
    assert {"ar", "arabizi", "en", "mixed"} <= {r["language"] for r in ROWS}
    tags = Counter(t for r in ROWS for t in r["tags"])
    for tag in ("no_match", "unresolved_names", "contains_excluded", "out_of_stock", "greeting", "delivery"):
        assert tags[tag] >= 1, tag


@pytest.fixture
def offline(monkeypatch):
    async def no_llm(*a, **kw):
        raise AllRoutesFailed(["offline"], timed_out=False)
        yield  # noqa

    async def no_card(*a, **kw):
        return None
    monkeypatch.setattr(agent, "stream_answer", no_llm)
    monkeypatch.setattr(agent, "sales_card", no_card)


@pytest.mark.parametrize("row", ROWS, ids=[r["id"] for r in ROWS])
def test_row_expectations_hold_offline(row, offline):
    assert row["intent"] in INTENTS and row["persona"] in PERSONAS and row["language"] in LANGUAGES
    res = asyncio.run(run_row(row))
    s = score_row(row, res)
    assert s["mode_ok"] and s["cards_ok"] and s["disclaimer_ok"] and s["reply_language_ok"], s
    assert s["non_empty"] and res["done"]["source"] == "template"
