"""api/widget.py: cards, the "why this product" line, turn events and answer
card details, against the real catalogue."""

from agents.orchestrator.orchestrator import TurnResult
from agents.retrieval.index_builder import get_index
from api.widget import CARD_FIELDS, product_details, to_card, turn_events, why
from models.prequal import PrequalResult

HANDLE = "capixy-lashes-treatment-serum-10ml"


def pq(intent="find_products", route="products_only", language="ar"):
    return PrequalResult(query_original="x", query_en="x", language=language, is_follow_up=False, route=route,
                         needs_retrieval=True, intent=intent, persona="customer", meta={})


def product(handle=HANDLE, **extra):
    p = get_index().products[handle]
    return {"handle": handle, "name": p["name"], "name_ar": p["name_ar"], "brand": p["brand"],
            "product_type": p["product_type"], "price": p["price"], "compare_at_price": p["compare_at_price"],
            "url": p["url"], "url_ar": p["url_ar"], "image": "img.png", "match": {"conflict": None}, **extra}


def test_card_carries_the_allow_list_only_filled_from_the_catalogue():
    card = to_card(product())
    assert set(card) == set(CARD_FIELDS)
    assert card["size"] == "10 ml" and card["variant_id"] == 41242316996679
    assert card["discount_percent"] == 30 and card["image"] == "img.png"   # the retrieved card wins
    for hidden in ("description", "key_ingredients", "how_to_use", "warnings", "sku", "tags"):
        assert hidden not in card


def test_why_names_the_requested_values_the_product_has():
    record = get_index().products[HANDLE]
    assert why(record, {"concerns": ["dandruff", "hair loss"]}) == {"concerns": ["hair loss"]}


def test_why_falls_back_to_the_products_own_concerns():
    record = get_index().products[HANDLE]
    assert why(record, {}) == {"concerns": ["hair loss", "lash growth"]}


def test_why_names_requested_ingredients_it_contains():
    record = get_index().products[HANDLE]
    assert why(record, {"ingredients.include": ["Biotin", "Retinol"]}) == {"contains": ["Biotin"]}


def test_conflict_comes_from_the_match():
    assert to_card(product(match={"conflict": "contains_excluded"}))["conflict"] == "contains_excluded"


def test_products_turn_streams_message_products_done():
    turn = TurnResult(reply="intro", products=[product()], path="products_only", prequal=pq(),
                      retrieval={"applied_filters": {"concerns": ["hair loss"]}, "total_candidates": 4,
                                 "relaxed_keys": []})
    events = turn_events(turn)
    assert [e["event"] for e in events] == ["message", "products", "done"]
    assert events[0]["data"] == {"text": "intro", "path": "products_only", "language": "ar", "health": False,
                                 "disclaimer": None}
    assert events[1]["data"]["total"] == 4
    assert events[1]["data"]["items"][0]["why"] == {"concerns": ["hair loss"]}
    pipeline = events[2]["data"]["pipeline"]
    assert pipeline["retrieval_ran"] is True and pipeline["total_candidates"] == 4
    assert pipeline["applied_filters"] == {"concerns": ["hair loss"]}


def test_the_responders_disclaimer_flags_the_answer():
    disclaimer = "المعلومات للتوعية ومش بديلة عن استشارة الطبيب."
    turn = TurnResult(reply="Use it at night.", path="responder", prequal=pq("how_to_use", "needs_response"),
                      disclaimer=disclaimer)
    events = turn_events(turn)
    assert [e["event"] for e in events] == ["message", "done"]
    assert events[0]["data"]["health"] is True and events[0]["data"]["disclaimer"] == disclaimer
    small_talk = TurnResult(reply="Hello", path="small_talk", prequal=pq("greeting", "needs_response"))
    assert turn_events(small_talk)[0]["data"]["health"] is False
    no_rule = TurnResult(reply="Use it at night.", path="responder", prequal=pq("how_to_use", "needs_response"))
    assert turn_events(no_rule)[0]["data"]["health"] is False


def test_streamed_responder_events_are_joined_for_one_shot_transports():
    turn = TurnResult(
        reply="Apply at night.", path="responder", prequal=pq("how_to_use", "needs_response"),
        products=[product()], disclaimer="Awareness only.", responder={"partial": False},
        timings_ms={"total": 1234.0},
        events=[
            {"event": "products", "data": {"items": [product()], "total": 1, "relaxed": [], "applied": {}}},
            {"event": "card", "data": {"type": "how_to_use", "handle": HANDLE, "name": "x", "steps": ["a"]}},
            {"event": "message", "data": {"delta": "Apply ", "language": "en"}},
            {"event": "message", "data": {"delta": "at night.", "language": "en"}},
            {"event": "message", "data": {"delta": "Awareness only.", "disclaimer": True, "language": "en"}},
        ])
    events = turn_events(turn)
    assert [e["event"] for e in events] == ["products", "card", "message", "done"]
    assert set(events[0]["data"]["items"][0]) == set(CARD_FIELDS)          # dumps became cards
    assert events[2]["data"]["text"] == "Apply at night." and events[2]["data"]["disclaimer"] == "Awareness only."
    assert events[3]["data"]["latency_ms"] == 1234.0 and events[3]["data"]["partial"] is False


def test_details_carry_the_answer_card_fields_and_mark_missing_arabic():
    d = product_details(HANDLE)
    assert d["key_ingredients"][:2] == ["Capixyl", "Aminexil"]
    assert d["how_to_use"] and d["warnings"] and d["description_ar"]
    assert d["how_to_use_ar"] is None and d["warnings_ar"] is None   # the widget shows English and says so
    assert "sku" not in d and "tags" not in d
    assert product_details("no-such-product") is None
