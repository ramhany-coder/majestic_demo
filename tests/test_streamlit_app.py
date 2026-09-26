"""streamlit_app.py's query console, run headless with AppTest and the
pipeline stubbed out (the catalogue is real)."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import agents.orchestrator.orchestrator as orch
from agents.orchestrator.orchestrator import TurnResult
from api.widget import CARD_FIELDS
from models.prequal import PrequalResult

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")
PRODUCTS = [
    {"handle": "capixy-hair-serum", "name": "Capixy Hair Serum 120ml", "brand": "Capixy", "price": 333.0,
     "url": "https://e-majestic.com/products/capixy-hair-serum", "match": {"source": "filters", "conflict": None}},
    {"handle": "capixy-anti-dandruff-serum-spray-120ml", "name": "Capixy Anti- Dandruff Serum Spray 120ml",
     "brand": "Capixy", "price": 319.0, "url": "https://e-majestic.com/products/capixy-anti-dandruff-serum-spray-120ml",
     "match": {"source": "filters", "conflict": None}},
]


def card_views(at):
    """The args of every product-cards component on the page."""
    return [json.loads(el.proto.json_args) for el in at.get("component_instance")]


@pytest.fixture
def calls(monkeypatch):
    seen = {"turns": [], "fail": False}

    async def fake_warm_up():
        return 0

    async def fake_handle(message, session_id="default", store=None, on_stage=None):
        if seen["fail"]:
            raise RuntimeError("provider down")
        seen["turns"].append((message, session_id))
        pq = PrequalResult(query_original=message, query_en="I need a hair serum.", language="en",
                           is_follow_up=False, route="products_only", needs_retrieval=True,
                           intent="find_products", persona="customer", meta={})
        return TurnResult(reply="Here are matching products:", products=PRODUCTS, path="products_only", prequal=pq,
                          retrieval={"applied_filters": {"product_type": ["hair serum"]}, "total_candidates": 2,
                                     "relaxed_keys": []},
                          timings_ms={"prequal": 5.0, "total": 12.0})

    monkeypatch.setattr(orch, "warm_up", fake_warm_up)
    monkeypatch.setattr(orch, "handle_message", fake_handle)
    return seen


@pytest.fixture
def app(calls):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def test_console_is_the_default_and_shows_products_as_the_widget_cards(app):
    assert app.sidebar.radio[0].value == "Query console"
    assert app.sidebar.toggle[0].value is True            # "Show matched products" starts on
    app.chat_input[0].set_value("hair serum").run()
    assert any("Here are matching products:" in m.value for m in app.markdown)
    [view] = card_views(app)
    assert view["view"] == "cards" and view["locale"] == "en"
    assert [c["handle"] for c in view["products"]] == [p["handle"] for p in PRODUCTS]
    assert all(set(c) == set(CARD_FIELDS) for c in view["products"])
    assert view["products"][0]["size"] and view["products"][0]["variant_id"]    # filled from the catalogue
    assert view["details"]["capixy-hair-serum"]["how_to_use"]                   # for the Details answer card


def test_turning_products_off_leaves_the_count(app):
    app.chat_input[0].set_value("hair serum").run()
    app.sidebar.toggle[0].set_value(False).run()
    assert card_views(app) == []
    assert any("2 products matched. Turn on" in c.value for c in app.caption)


def test_follow_ups_share_a_session_until_new_session(app, calls):
    app.chat_input[0].set_value("hair serum").run()
    app.chat_input[0].set_value("without silicone?").run()
    first, second = calls["turns"]
    assert first[1] == second[1]
    app.sidebar.button[0].click().run()
    assert not any("Here are matching products:" in m.value for m in app.markdown)
    app.chat_input[0].set_value("sunscreen").run()
    assert calls["turns"][2][1] != first[1]


def test_a_failing_turn_is_shown_not_raised(app, calls):
    calls["fail"] = True
    app.chat_input[0].set_value("hair serum").run()
    assert not app.exception
    assert "The pipeline failed: RuntimeError: provider down" in app.error[0].value
