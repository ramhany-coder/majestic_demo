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
    assert events[0]["data"] == {"text": "intro", "path": "products_only", "language": "ar", "health": False}
    assert events[1]["data"]["total"] == 4
    assert events[1]["data"]["items"][0]["why"] == {"concerns": ["hair loss"]}


def test_health_answers_are_flagged_for_the_disclaimer():
    turn = TurnResult(reply="Use it at night.", path="responder", prequal=pq("how_to_use", "needs_response"))
    events = turn_events(turn)
    assert [e["event"] for e in events] == ["message", "done"]
    assert events[0]["data"]["health"] is True
    small_talk = TurnResult(reply="Hello", path="small_talk", prequal=pq("greeting", "needs_response"))
    assert turn_events(small_talk)[0]["data"]["health"] is False
    delivery = TurnResult(reply="Delivery takes 3 days.", path="responder", prequal=pq("other", "needs_response"))
    assert turn_events(delivery)[0]["data"]["health"] is False


def test_an_unknown_question_gets_the_disclaimer_when_the_router_fell_back():
    unsure = pq("other", "needs_response")
    unsure.meta = {"status": {"rewriter": "ok", "router": "default"}}
    turn = TurnResult(reply="...", path="responder", prequal=unsure)
    assert turn_events(turn)[0]["data"]["health"] is True


def test_details_carry_the_answer_card_fields_and_mark_missing_arabic():
    d = product_details(HANDLE)
    assert d["key_ingredients"][:2] == ["Capixyl", "Aminexil"]
    assert d["how_to_use"] and d["warnings"] and d["description_ar"]
    assert d["how_to_use_ar"] is None and d["warnings_ar"] is None   # the widget shows English and says so
    assert "sku" not in d and "tags" not in d
    assert product_details("no-such-product") is None
