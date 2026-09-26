import pytest

from agents.filter_extractor.rule_based import extract_rule_based as rb


def test_ingredients_include_exclude_and_unmatched():
    out = rb("ingredients", "patient allergic to salicilic, needs a retinl serum with hyalronic and no alchohol")
    assert out["include"] == ["Retinol", "Hyaluronic Acid"]
    assert out["exclude"] == ["Salicylic Acid"]
    assert out["unmatched_exclude"] == ["alcohol"]


def test_negation_flips_only_the_next_ingredient():
    out = rb("ingredients", "no retinol, needs vitamin c")
    assert out["exclude"] == ["Retinol"] and out["include"] == ["Vitamin C"]


def test_negation_runs_along_and_or_chains():
    out = rb("ingredients", "without retinol or salicylic acid")
    assert out["exclude"] == ["Retinol", "Salicylic Acid"]


def test_free_suffix_excludes():
    out = rb("ingredients", "alcohol-free toner with niacinamide")
    assert out["unmatched_exclude"] == ["alcohol"]
    assert out["include"] == ["Niacinamide (Vitamin B3)"]


def test_negation_marker_is_not_swallowed_by_a_fuzzy_ngram():
    out = rb("ingredients", "oily skin, no salicylic acid, leave-on")
    assert out["exclude"] == ["Salicylic Acid"] and out["include"] == []


def test_brand_word_is_not_an_ingredient_and_vice_versa():
    assert rb("ingredients", "capixy tonic")["include"] == []
    assert rb("brand", "does capixyl help")["brand"] == []


@pytest.mark.parametrize("query,expected", [
    ("do you have anything from vacasion or soralon", ["Vacation", "Soralone"]),
    ("sebio control toner", ["Vacation"]),
    ("hi", []),
])
def test_brand(query, expected):
    assert rb("brand", query)["brand"] == expected


def test_forms_rejected_and_phrasal():
    assert rb("product_form", "sunscreen in sprey or gel, not a heavy creem")["product_form"] == ["spray", "gel"]
    assert rb("product_form", "want something i dont need to rinse")["product_form"] == ["leave-in cream"]
    assert rb("product_form", "no salicylic acid, leave-on")["product_form"] == ["leave-in cream"]


def test_product_type_synonyms_and_plain_serum():
    assert rb("product_type", "need a sun blok spray")["product_type"] == ["sunscreen"]
    assert rb("product_type", "vitamin c serum")["product_type"] == ["face serum"]
    assert rb("product_type", "a serum please")["product_type"] == []


def test_concerns_and_suitable_for():
    assert rb("concerns", "i have pimpels, big poors and old acne marks")["concerns"] == [
        "acne", "enlarged pores", "post-acne marks"]
    assert rb("suitable_for", "im pregnent and my skin is sensitve and combo")["suitable_for"] == [
        "pregnancy", "sensitive skin", "combination skin"]


def test_hero_needs_a_product_word():
    assert rb("hero_ingredient", "a urea cream for elbows")["hero_ingredient"] == ["Urea"]
    assert rb("hero_ingredient", "does it contain urea")["hero_ingredient"] == []


def test_names_in_mention_order_with_arabic():
    out = rb("names", "whats the diffrence between capixi dray foom and the vials")
    assert out["name_en"] == ["Capixy Intense Dry Foam", "Capixy Anti Hair Loss Vials"]
    assert all(out["name_ar"]) and "مل" not in "".join(out["name_ar"])


def test_category_from_words_and_implied_types():
    assert rb("category", "need somthing for my hair falling and a good deodrant")["category"] == [
        "Hair Care", "Personal Care"]
