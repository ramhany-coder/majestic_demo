"""Responder agent (agents/responder): prompt blocks, product fields, reply
language, deterministic cards, the disclaimer rule, guardrails, fallback
templates, the event stream and the streaming fallback chain. No network:
R1 and R2 are faked at the module level, the streaming chain at the model."""

import asyncio
import json

import pytest

import agents.responder.agent as agent
import llm.fallback as fallback_module
from agents.responder.cards import build_cards, disclaimer_for, safety_flag, PREGNANCY_RE
from agents.responder.context import (
    ResponderContext, fields_for, mode_for, product_fields, product_lines, reply_language, select_products,
)
from agents.responder.fallback_templates import template_answer
from agents.responder.guardrails import (
    amounts_in, cut_to_limit, language_ok, ungrounded_amounts, unknown_ingredients,
)
from agents.responder.prompts import build_answer_prompt, intent_block_key, load_store_facts, situation_blocks
from agents.responder.sales_card import validate_card
from agents.retrieval.agent import retrieve
from agents.retrieval.index_builder import get_index
from agents.retrieval.semantic import SemanticScores
from config import settings
from llm.client import fallback_client
from llm.fallback import AllRoutesFailed, StreamBroken, StreamInfo
from models.filter_extractor import IngredientFilter, MetadataFilters
from models.retrieval import RetrievalResult

VIT_C = "vacation-vitamin-c-10-30-ml"
DRY_FOAM = "capixy-dry-foam-120-ml"


def run(coro):
    return asyncio.run(coro)


def make_ctx(message="x", intent="how_to_use", persona="customer", language="en", route="needs_response",
             names=None, filters=None, retrieval="auto", query_en=None, history=None):
    if retrieval == "auto":
        f = filters if filters is not None else MetadataFilters(name_en=names or [])
        retrieval = retrieve(f, SemanticScores(status="ok"), query_en or message, intent, 10)
    return ResponderContext(query_original=message, query_en=query_en or message, language=language,
                            persona=persona, intent=intent, history=history or [], retrieval=retrieval,
                            mode=mode_for(route, retrieval))


def record(handle):
    return {**get_index().products[handle], "match": {}}


# ---------------------------------------------------------------- context


def test_mode_follows_the_retrieval_outcome():
    found = RetrievalResult(products=[], k=10)
    assert mode_for("needs_response", None) == "no_products"
    assert mode_for("products_only", found) == "no_match"
    ctx = make_ctx(names=["Vacation Vitamin C Serum"])
    assert ctx.mode == "answer"
    relaxed = ctx.retrieval.model_copy(update={"relaxed_keys": ["product_form"]})
    assert mode_for("products_only", relaxed) == "no_match"


@pytest.mark.parametrize("language,message,expected", [
    ("ar", "", "ar"), ("arabizi", "", "ar"), ("en", "", "en"),
    ("mixed", "عايزة serum للشعر الجاف", "ar"),
    ("mixed", "which serum is better for dry hair ya Jamila", "en"),
    ("mixed", "3ayza serum lel sha3r el gaf", "ar"),
])
def test_reply_language_rule(language, message, expected):
    assert reply_language(language, message, arabizi_reply="arabic") == expected


def test_arabizi_reply_setting():
    assert reply_language("arabizi", "3ayez", arabizi_reply="arabizi") == "arabizi"
    assert reply_language("mixed", "3ayez serum a7san", arabizi_reply="arabizi") == "arabizi"


def test_fields_follow_the_intent():
    assert fields_for("how_to_use", "answer") == ("name", "name_ar", "product_form", "concerns", "how_to_use",
                                                  "warnings")
    assert "price" in fields_for("price_offer", "answer") and "description" not in fields_for("price_offer", "answer")
    assert "conflict" in fields_for("safety", "no_match")                 # no-match mode uses the list fields
    assert fields_for("other", "answer") == fields_for("product_info", "answer")


def test_product_fields_are_small_and_clean():
    rec = record(DRY_FOAM)
    out = product_fields(rec, fields_for("product_info", "answer"), description_chars=100)
    assert len(out["description"]) <= 100 and out["description"].endswith("…")
    assert out["hero_ingredient"].startswith("R2CP Complex")
    for hidden in ("data_issues", "images", "url", "sku", "tags"):
        assert hidden not in out
    assert "available" not in product_fields(rec, ("available",))          # in stock is the default
    assert product_fields({**rec, "available": False}, ("available",)) == {"available": False}


def test_hero_concentration_is_kept():
    assert product_fields(record(VIT_C), ("hero_ingredient",)) == {"hero_ingredient": "Vitamin C 10%"}


def test_at_most_five_products_and_none_for_greetings(monkeypatch):
    ctx = make_ctx("deodorants", intent="find_products", filters=MetadataFilters(product_type=["deodorant"]))
    assert len(ctx.retrieval.products) > 5 and len(select_products(ctx)) == 5
    assert select_products(ctx.model_copy(update={"intent": "greeting"})) == []
    assert select_products(make_ctx("hi", intent="greeting", retrieval=None)) == []


def test_compare_sends_only_the_named_products():
    ctx = make_ctx("compare", intent="compare", names=["Capixy Intense Dry Foam", "Capixy Hair Serum"])
    handles = [p["handle"] for p in select_products(ctx)]
    assert set(handles) == set(ctx.retrieval.name_hits) and len(handles) == 2


def test_product_lines_are_one_json_object_each():
    ctx = make_ctx(names=["Vacation Vitamin C Serum"])
    lines = product_lines(select_products(ctx), ctx)
    assert len(lines) == 1 and json.loads(lines[0])["how_to_use"]


# ---------------------------------------------------------------- prompt blocks


def test_blocks_follow_persona_intent_and_situation():
    ctx = make_ctx("how do I use it", persona="doctor", names=["Vacation Vitamin C Serum"])
    system, human = build_answer_prompt(ctx, select_products(ctx), "en")
    assert "Reply in English." in system and "{{" not in system
    assert human.startswith("Audience: a doctor or pharmacist.")
    assert "steps card shows the details" in human
    assert "No exact match" not in human and "out of stock" not in human
    assert human.rstrip().endswith("MEANING (English): how do I use it")
    assert "customer_service" not in human      # the unfilled {{CS_CONTACT}} is dropped


def test_unknown_persona_is_a_customer_and_arabic_is_named():
    ctx = make_ctx(persona="unknown", language="ar", names=["Vacation Vitamin C Serum"])
    system, human = build_answer_prompt(ctx, select_products(ctx), "ar")
    assert "Egyptian Arabic, in Arabic script" in system and human.startswith("Audience: a customer.")


def test_other_intent_is_out_of_scope_without_retrieval_else_a_product_question():
    assert intent_block_key(make_ctx(intent="other", retrieval=None)) == "out_of_scope"
    assert intent_block_key(make_ctx(intent="other", names=["Capixy Hair Serum"])) == "product_info"


def test_no_match_block_names_the_blocking_keys():
    ctx = make_ctx("sunscreen roll on", intent="find_products", route="products_only",
                   filters=MetadataFilters(product_type=["sunscreen"], product_form=["roll-on"]))
    assert ctx.mode == "no_match" and not ctx.retrieval.products
    blocks = situation_blocks(ctx, [])
    assert "type = sunscreen" in blocks[0] and "form = roll-on" in blocks[0] and "Do not name products" in blocks[0]


def test_unresolved_excluded_and_out_of_stock_blocks():
    ctx = make_ctx("x", intent="product_info", retrieval=RetrievalResult(
        products=[], k=10, unresolved_names=["Cerave Foaming Cleanser"]))
    assert any("Could not find: Cerave Foaming Cleanser" in b for b in situation_blocks(ctx, []))

    ctx = make_ctx("does it have menthol", intent="safety", names=["Capixy Intense Dry Foam"],
                   filters=MetadataFilters(name_en=["Capixy Intense Dry Foam"],
                                           ingredients=IngredientFilter(exclude=["Menthol"])))
    products = select_products(ctx)
    assert products[0]["match"].get("conflict") == "contains_excluded"
    assert any("contains Menthol, which the user wants to avoid" in b for b in situation_blocks(ctx, products))

    ctx = make_ctx("price", intent="price_offer", names=["Soralone Anti-Dandruff Shampoo"])
    assert any("currently out of stock" in b for b in situation_blocks(ctx, select_products(ctx)))


def test_store_facts_drop_placeholders(tmp_path):
    path = tmp_path / "facts.json"
    path.write_text(json.dumps({"site": "e-majestic.com", "customer_service": "{{CS_CONTACT}}"}), encoding="utf-8")
    load_store_facts.cache_clear()
    try:
        assert load_store_facts(str(path)) == {"site": "e-majestic.com"}
    finally:
        load_store_facts.cache_clear()


# ---------------------------------------------------------------- cards and disclaimer


def test_how_to_use_card_uses_catalog_steps():
    ctx = make_ctx(names=["Capixy Intense Dry Foam"], language="ar")
    [card] = build_cards(ctx, select_products(ctx), "ar")
    assert card["type"] == "how_to_use" and card["handle"] == DRY_FOAM
    assert card["steps"] == get_index().products[DRY_FOAM]["how_to_use"]
    assert card["name"] == get_index().products[DRY_FOAM]["name_ar"]


def test_safety_flags():
    assert safety_flag(record("pregnastep-1-dietary-supplement"), "pregnancy", PREGNANCY_RE) == "suitable"
    assert safety_flag(record("movelex-advance"), "pregnancy", PREGNANCY_RE) == "warning"
    assert safety_flag(record(VIT_C), "pregnancy", PREGNANCY_RE) == "not_listed"
    ctx = make_ctx(intent="safety", names=["Methytral Nipple Cream"])
    [card] = build_cards(ctx, select_products(ctx), "en")
    assert card["type"] == "safety" and card["flags"] == {"pregnancy": "not_listed", "breastfeeding": "suitable"}
    ctx = make_ctx(intent="safety", names=["Movelex Ultra"])
    [card] = build_cards(ctx, select_products(ctx), "en")
    assert card["flags"] == {"pregnancy": "warning", "breastfeeding": "warning"}    # "pregnant\\lactating"


def test_compare_card_and_price_offer_has_none():
    ctx = make_ctx(intent="compare", names=["Capixy Intense Dry Foam", "Capixy Hair Serum"])
    [card] = build_cards(ctx, select_products(ctx), "en")
    assert card["type"] == "compare" and len(card["columns"]) == 2 == len(card["names"])
    assert {r["key"] for r in card["rows"]} >= {"hero_ingredient", "price", "size"}
    assert all(len(r["values"]) == 2 for r in card["rows"])
    ctx = make_ctx(intent="price_offer", names=["Capixy Intense Dry Foam"])
    assert build_cards(ctx, select_products(ctx), "en") == []


def test_disclaimer_rule():
    ctx = make_ctx(intent="safety", persona="customer", names=["Vacation Vitamin C Serum"], language="ar")
    assert disclaimer_for(ctx, select_products(ctx), "ar") == "المعلومات للتوعية ومش بديلة عن استشارة الطبيب."
    # Doctor, safety, a face serum: not a medical group -> no disclaimer.
    ctx = make_ctx(intent="safety", persona="doctor", names=["Vacation Vitamin C Serum"])
    assert disclaimer_for(ctx, select_products(ctx), "en") is None
    # Any persona, how-to-use of a supplement or a joint product -> disclaimer.
    ctx = make_ctx(intent="how_to_use", persona="sales_trainee", names=["Movelex Cream 50gm"])
    assert disclaimer_for(ctx, select_products(ctx), "en").startswith("This information is for awareness")
    # Price of a supplement -> no disclaimer.
    ctx = make_ctx(intent="price_offer", persona="customer", names=["Pregnastep 1"])
    assert disclaimer_for(ctx, select_products(ctx), "en") is None


# ---------------------------------------------------------------- guardrails


def test_price_check_finds_amounts_and_percentages():
    assert amounts_in("Now 403 EGP instead of 575 ج.م, a 30% discount") == [403.0, 575.0, 30.0]
    assert amounts_in("سعره ٤٠٣ جنيه وخصم ٣٠٪") == [403.0, 30.0]
    assert amounts_in("Use it 2 times a day for 30 days") == []
    allowed = ['{"price":403.0,"compare_at_price":575.0,"discount_percent":30}']
    assert ungrounded_amounts("403 EGP, was 575 EGP (30% off)", allowed) == []
    assert ungrounded_amounts("only 350 EGP, 40% off", allowed) == [350.0, 40.0]


def test_ingredient_check():
    products = [record(VIT_C)]
    assert unknown_ingredients("Vitamin C and Hyaluronic Acid brighten skin.", products, ["x"]) == []
    assert unknown_ingredients("It also has Retinol.", products, ["x"]) == ["Retinol"]
    assert unknown_ingredients("It has no Retinol.", products, ["does it have retinol?"]) == []


def test_language_check():
    assert language_ok("سيروم Vacation Vitamin C مناسب للبشرة الدهنية", "ar")
    assert not language_ok("This serum suits oily skin very well.", "ar")
    assert language_ok("This serum suits oily skin.", "en") and not language_ok("السيروم ده مناسب جدا للبشرة", "en")
    # An English coaching reply with the persona's suggested Arabic line is still English.
    assert language_ok("Three selling points for the serum, then tell the customer: السيروم ده ممتاز", "en")


def test_length_cut_keeps_complete_sentences():
    text = "One two three. Four five six seven. Eight nine ten eleven twelve."
    assert cut_to_limit(text, 20) == text
    assert cut_to_limit(text, 9) == "One two three. Four five six seven."
    assert cut_to_limit("a b c d e f g h", 4) == "a b c d…"


# ---------------------------------------------------------------- templates and sales card


def test_fallback_templates_are_never_empty():
    ctx = make_ctx(language="ar", names=["Capixy Intense Dry Foam"])
    text = template_answer(ctx, select_products(ctx), "ar")
    assert text.startswith("طريقة استخدام كابيكسي") and "Shake well" in text
    ctx = make_ctx(intent="price_offer", names=["Capixy Intense Dry Foam"], language="ar")
    assert template_answer(ctx, select_products(ctx), "ar").startswith("كابيكسي إنتنس دراي فوم 120 مل سعره 403 ج.م بدل 575")
    ctx = make_ctx(intent="find_products", route="products_only", retrieval=RetrievalResult(products=[], k=10))
    assert template_answer(ctx, [], "en").startswith("I couldn't find")
    for intent in ("greeting", "out_of_scope", "other", "compare", "sales_training"):
        for lang in ("ar", "arabizi", "en"):
            assert template_answer(make_ctx(intent=intent, retrieval=None), [], lang).strip()


def test_sales_card_validation():
    quiz = {"card_type": "quiz", "objection": None, "quiz": [
        {"q": "Hero?", "options": ["a", "b", "c", "d"], "answer_index": 2, "explain": "c"},
        {"q": "Bad", "options": ["a", "b"], "answer_index": 0, "explain": ""},
        {"q": "Bad index", "options": ["a", "b", "c", "d"], "answer_index": 7, "explain": ""}]}
    assert validate_card(quiz) == {"type": "quiz", "items": [
        {"q": "Hero?", "options": ["a", "b", "c", "d"], "answer_index": 2, "explain": "c"}]}
    objection = {"card_type": "objection", "quiz": [], "objection": {
        "objection": "Too expensive", "talking_points": ["Acknowledge", "Reframe", "Close"], "suggested_reply": "..."}}
    assert validate_card(objection)["type"] == "objection"
    assert validate_card({"card_type": "objection", "quiz": [], "objection": None}) is None
    assert validate_card({"card_type": "quiz", "quiz": [], "objection": None}) is None


# ---------------------------------------------------------------- the event stream


@pytest.fixture
def fake_llm(monkeypatch):
    """R1 streams `state['chunks']` (or raises state['error'] at state['error_at']);
    the language retry returns state['retry']; R2 returns state['card']."""
    state = {"chunks": ["Shake well, ", "then apply to the scalp."], "error": None, "error_at": 0,
             "retry": None, "card": None, "card_delay": 0.0, "r1_messages": []}

    async def stream_answer(messages, persona, info=None):
        state["r1_messages"].append(messages)
        for i, c in enumerate(state["chunks"]):
            if state["error"] is not None and i == state["error_at"]:
                raise state["error"]
            yield c
        if state["error"] is not None and state["error_at"] >= len(state["chunks"]):
            raise state["error"]

    async def complete_answer(messages, persona):
        return state["retry"]

    async def sales_card(ctx, products, reply_lang, deadline_s=None):
        await asyncio.sleep(state["card_delay"])
        return state["card"]

    monkeypatch.setattr(agent, "stream_answer", stream_answer)
    monkeypatch.setattr(agent, "complete_answer", complete_answer)
    monkeypatch.setattr(agent, "sales_card", sales_card)
    return state


def events_of(ctx):
    async def main():
        return [ev async for ev in agent.respond_stream(ctx)]
    return run(main())


def test_event_order_for_a_how_to_use_answer(fake_llm):
    ctx = make_ctx("how do I use Movelex Gel", names=["Movelex Gel"])
    evs = events_of(ctx)
    names = [e["event"] for e in evs]
    assert names == ["status", "card", "message", "message", "message", "done"]
    assert evs[1]["data"]["type"] == "how_to_use"
    assert [e["data"].get("delta") for e in evs[2:4]] == fake_llm["chunks"]
    assert evs[4]["data"]["disclaimer"] is True                         # joint product -> disclaimer
    assert evs[-1]["data"]["partial"] is False and evs[-1]["data"]["source"] == "llm"
    out = agent.collect(evs)
    assert out["text"] == "Shake well, then apply to the scalp." and out["disclaimer"]


def test_sales_training_gets_the_r2_card_after_the_text(fake_llm):
    fake_llm["card"] = {"type": "objection", "objection": "Too expensive", "talking_points": ["a"],
                        "suggested_reply": "b"}
    fake_llm["chunks"] = ["Point one."]
    ctx = make_ctx("the customer says Capixy Dry Foam is expensive", intent="sales_training",
                   persona="sales_trainee", names=["Capixy Intense Dry Foam"])
    evs = events_of(ctx)
    assert [e["event"] for e in evs] == ["status", "message", "card", "done"]
    assert evs[2]["data"]["type"] == "objection"


def test_a_slow_sales_card_is_skipped(fake_llm, monkeypatch):
    monkeypatch.setattr(settings, "RESPONDER_SALES_CARD_GRACE_S", 0.05)
    fake_llm["card"] = {"type": "quiz", "items": []}
    fake_llm["card_delay"] = 1.0
    ctx = make_ctx("quiz me on Capixy Dry Foam", intent="sales_training", persona="sales_trainee",
                   names=["Capixy Intense Dry Foam"])
    evs = events_of(ctx)
    assert "card" not in [e["event"] for e in evs] and "sales_card_timeout" in evs[-1]["data"]["notes"]


def test_r1_failure_sends_the_template(fake_llm):
    fake_llm["error"] = AllRoutesFailed(["zai error: 401"], timed_out=False)
    ctx = make_ctx(intent="price_offer", names=["Capixy Intense Dry Foam"])
    evs = events_of(ctx)
    msg = [e for e in evs if e["event"] == "message"]
    assert msg[0]["data"]["replace"] is True and "403 EGP" in msg[0]["data"]["text"]
    assert evs[-1]["data"]["source"] == "template"


def test_a_broken_stream_ends_cleanly_with_partial(fake_llm):
    fake_llm["chunks"] = ["Shake well, ", "then", " never finished"]
    fake_llm["error"] = StreamBroken("zai:glm-5.3-flash", "timeout after 5.00s")
    fake_llm["error_at"] = 2
    evs = events_of(make_ctx(names=["Capixy Intense Dry Foam"]))
    assert agent.collect(evs)["text"] == "Shake well, then"
    assert evs[-1]["event"] == "done" and evs[-1]["data"]["partial"] is True
    assert not any(e["data"].get("replace") for e in evs if e["event"] == "message")


def test_ungrounded_price_swaps_in_the_template(fake_llm):
    fake_llm["chunks"] = ["It costs 350 EGP now."]
    ctx = make_ctx(intent="price_offer", names=["Capixy Intense Dry Foam"])
    evs = events_of(ctx)
    replaced = [e for e in evs if e["event"] == "message" and e["data"].get("replace")]
    assert replaced and "403 EGP" in replaced[0]["data"]["text"]
    assert "ungrounded_amounts=[350.0]" in evs[-1]["data"]["notes"]


def test_grounded_price_is_kept(fake_llm):
    fake_llm["chunks"] = ["It costs 403 EGP instead of 575 EGP, 30% off."]
    evs = events_of(make_ctx(intent="price_offer", names=["Capixy Intense Dry Foam"]))
    assert not any(e["data"].get("replace") for e in evs if e["event"] == "message")


def test_long_answers_are_cut_at_a_sentence(fake_llm, monkeypatch):
    monkeypatch.setitem(settings.RESPONDER_WORD_LIMITS, "customer", 4)    # x1.5 -> 6 words
    fake_llm["chunks"] = ["Shake the foam well. ", "Apply it to a dry scalp ", "every single day please."]
    evs = events_of(make_ctx(names=["Capixy Intense Dry Foam"]))
    assert agent.collect(evs)["text"] == "Shake the foam well."
    assert "cut_to_length" in evs[-1]["data"]["notes"]


def test_wrong_language_is_regenerated_once(fake_llm):
    fake_llm["chunks"] = ["Shake it well and apply to the scalp."]
    fake_llm["retry"] = "## **رجّي العبوة** كويس وحطيها على فروة الراس."
    evs = events_of(make_ctx(language="ar", names=["Capixy Intense Dry Foam"]))
    assert agent.collect(evs)["text"] == "رجّي العبوة كويس وحطيها على فروة الراس."   # cleaned like the stream
    assert "language_retry" in evs[-1]["data"]["notes"]


def test_respond_joins_text_and_disclaimer(fake_llm):
    text = run(agent.respond(make_ctx(intent="safety", names=["Vacation Vitamin C Serum"])))
    assert text.startswith("Shake well") and text.endswith("not a substitute for medical advice.")


def test_prompt_carries_history_and_greetings_send_no_products(fake_llm):
    ctx = make_ctx("hi", intent="greeting", retrieval=None,
                   history=[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi!"}])
    evs = events_of(ctx)
    human = fake_llm["r1_messages"][0][1].content
    assert "U: hello\nA: hi!" in human and "PRODUCT DATA:\n(none)" in human
    assert [e["event"] for e in evs] == ["status", "message", "message", "done"]


# ---------------------------------------------------------------- FallBack.astream


class Chunk:
    def __init__(self, content):
        self.content = content


def fake_models(monkeypatch, plans):
    """plans[model] = list of chunk texts, or an Exception to raise at that position."""
    def get_cached_model(router, model, **kw):
        plan = plans[model]

        class Model:
            async def astream(self, messages):
                for item in plan:
                    if isinstance(item, Exception):
                        raise item
                    if item == "HANG":
                        await asyncio.sleep(10)
                    yield Chunk(item)
        return Model()
    monkeypatch.setattr(fallback_module.client_llm, "get_cached_model", get_cached_model)


def collect_stream(routes, **kw):
    info = StreamInfo()

    async def main():
        return [c async for c in fallback_client.astream([], routes, info=info, **kw)], info
    return run(main())


def test_astream_moves_on_when_a_route_fails_before_its_first_token(monkeypatch):
    fake_models(monkeypatch, {"a": ["", RuntimeError("429")], "b": ["", "Hel", "lo"]})
    chunks, info = collect_stream(["zai:a", "zai:b"])
    assert chunks == ["Hel", "lo"] and info.route == "zai:b" and info.attempt == 1
    assert "429" in info.errors[0]


def test_astream_first_token_timeout(monkeypatch):
    fake_models(monkeypatch, {"a": ["HANG"], "b": ["ok"]})
    chunks, info = collect_stream(["zai:a", "zai:b"], first_token_timeouts=[0.05, 1.0])
    assert chunks == ["ok"] and "timeout" in info.errors[0]


def test_astream_raises_stream_broken_after_text(monkeypatch):
    fake_models(monkeypatch, {"a": ["Hel", RuntimeError("reset")], "b": ["never"]})
    seen = []

    async def main():
        with pytest.raises(StreamBroken):
            async for c in fallback_client.astream([], ["zai:a", "zai:b"]):
                seen.append(c)
    run(main())
    assert seen == ["Hel"]


def test_astream_all_routes_failed(monkeypatch):
    fake_models(monkeypatch, {"a": [RuntimeError("401")], "b": [""]})
    with pytest.raises(AllRoutesFailed):
        collect_stream(["zai:a", "zai:b"])


def test_an_unresolved_name_with_no_filters_sends_no_products():
    # Retrieval fills the list from the whole pool when the only name is not found.
    ctx = make_ctx("Do you have Bioderma Sensibio H2O?", intent="product_info", names=["Bioderma Sensibio H2O"])
    assert ctx.retrieval.products and ctx.retrieval.unresolved_names == ["Bioderma Sensibio H2O"]
    assert ctx.mode == "no_match" and select_products(ctx) == []
    blocks = situation_blocks(ctx, [])
    assert any("Could not find: Bioderma Sensibio H2O" in b for b in blocks)


def test_markdown_bold_is_stripped_across_chunks(fake_llm):
    fake_llm["chunks"] = ["**Sell", "ing point*", "*: it works.\n", "### Next\nSay this."]
    evs = events_of(make_ctx(intent="product_info", names=["Capixy Intense Dry Foam"]))
    deltas = "".join(e["data"]["delta"] for e in evs if e["event"] == "message" and "delta" in e["data"]
                     and not e["data"].get("disclaimer"))
    assert "*" not in deltas
    assert agent.collect(evs)["text"] == "Selling point: it works.\nNext\nSay this."


def test_plain_text_filter_keeps_single_asterisks_and_flushes():
    md = agent.PlainTextFilter()
    assert md.feed("a *") == "a " and md.feed("b") == "*b" and md.flush() == ""
    assert md.feed("x_") == "x" and md.flush() == "_"


def test_emoji_are_stripped(fake_llm):
    fake_llm["chunks"] = ["Two options \U0001F447 ", "\u26a0\ufe0f both are out of stock."]
    text = agent.collect(events_of(make_ctx(intent="compare", names=["Movelex Gel", "Movelex Nano Spray"])))["text"]
    assert " ".join(text.split()) == "Two options both are out of stock."


def test_promotions_are_spelled_out():
    rec = record(VIT_C)
    assert product_fields(rec, ("promotion",)) == {"promotion": "buy 2 get 1 free"}


def test_store_facts_only_for_store_questions():
    ctx = make_ctx("how do I use it", names=["Capixy Intense Dry Foam"])
    assert "STORE FACTS: (none)" in build_answer_prompt(ctx, select_products(ctx), "en")[1]
    ctx = make_ctx("is delivery free?", intent="other", retrieval=None)
    assert '"free_shipping_threshold_egp":999' in build_answer_prompt(ctx, [], "en")[1]
