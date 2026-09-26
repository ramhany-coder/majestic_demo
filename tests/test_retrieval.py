"""Retrieval over the real catalog (no LLM)."""

from agents.retrieval.retrieval import retrieve, text_search
from models.filter_extractor import IngredientFilter, MetadataFilters, UnmatchedTerms


def handles(res):
    return [p["handle"] for p in res.products]


def test_hair_serum_without_silicone():
    f = MetadataFilters(product_type=["hair serum"], ingredients=IngredientFilter(exclude=["Silicone"]))
    res = retrieve(f, "I need a hair serum without silicone.")
    assert res.mode == "filters" and res.relaxed == []
    assert set(handles(res)) == {"capixy-hair-serum", "capixy-anti-dandruff-serum-spray-120ml"}


def test_and_across_keys_or_within():
    f = MetadataFilters(product_type=["sunscreen"], suitable_for=["oily skin"])
    res = retrieve(f, "sunscreen for oily skin", top_k=10)
    assert res.products and all(p["product_type"] == "sunscreen" for p in res.products)
    assert all("oily skin" in p["suitable_for"] for p in res.products)


def test_relaxes_in_order_until_something_matches():
    # No sunscreen is a roll-on: product_form is dropped first.
    f = MetadataFilters(product_type=["sunscreen"], product_form=["roll-on"])
    res = retrieve(f, "sunscreen roll on")
    assert res.relaxed == ["product_form"] and res.products
    assert all(p["product_type"] == "sunscreen" for p in res.products)


def test_exclude_is_never_relaxed():
    f = MetadataFilters(hero_ingredient=["Retinol"], ingredients=IngredientFilter(exclude=["Retinol"]))
    res = retrieve(f, "retinol serum without retinol")
    assert res.products == []


def test_matched_handles_restrict_in_order():
    f = MetadataFilters(matched_handles=["vacation-vitamin-c-10-30-ml"], suitable_for=["pregnancy"])
    res = retrieve(f, "Is Vacation Vitamin C Serum 10% safe during pregnancy?")
    assert res.mode == "handles" and handles(res) == ["vacation-vitamin-c-10-30-ml"]


def test_empty_filters_use_text_search():
    res = retrieve(MetadataFilters(), "hair serum")
    assert res.mode == "text" and res.products
    assert res.products[0]["product_type"] == "hair serum"
    assert retrieve(None, "").products == []


def test_text_search_handles_arabic_names():
    assert text_search("كابيكسي سيروم الرموش")[0]["handle"] == "capixy-lashes-treatment-serum-10ml"


def test_unmatched_exclude_penalizes_mentions():
    f = MetadataFilters(product_type=["deodorant"], unmatched=UnmatchedTerms(exclude=["aluminum"]))
    res = retrieve(f, "deodorant without aluminum", top_k=30)
    assert res.products and res.mode == "filters"


def test_top_k_and_cards():
    res = retrieve(MetadataFilters(product_type=["deodorant"]), "deodorant", top_k=3)
    assert len(res.products) == 3
    card = res.cards()[0]
    assert {"handle", "name", "name_ar", "price", "url", "image"} <= set(card)
