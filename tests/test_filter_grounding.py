from agents.filter_extractor.merge import merge_outputs


def m(outputs, query, context=None):
    return merge_outputs(outputs, query=query, context=context or [])


def test_invented_name_without_brand_in_query_is_dropped():
    f = m({"names": {"name_en": ["Capixy Anti Hair Loss Shampoo"], "name_ar": ["x"]}},
          "anti hairloss shampoo for men")
    assert f.name_en == [] and f.name_ar == [] and f.matched_handles == []


def test_brand_only_name_is_dropped():
    f = m({"names": {"name_en": ["Capixy"], "name_ar": ["كابيكسي"]}, "brand": {"brand": ["Capixy"]}},
          "any offers or bundles on capixy?")
    assert f.matched_handles == [] and f.brand == ["Capixy"]


def test_name_grounded_by_context():
    f = m({"names": {"name_en": ["Capixy Intense Tonic Spray"], "name_ar": []}},
          "how often should i use it?", ["Capixy Intense Tonic Spray 125ml"])
    assert f.matched_handles == ["capixy-intense-tonic-spray-125ml"]


def test_name_grounded_by_brand_typo_or_product_line():
    f = m({"names": {"name_en": ["Capixy Intense Dry Foam"], "name_ar": []}}, "capixi dray foom")
    assert f.matched_handles == ["capixy-dry-foam-120-ml"]
    f = m({"names": {"name_en": ["Sebio-Control"], "name_ar": []}}, "sebio control products for acne")
    assert len(f.matched_handles) == 5


def test_brand_word_read_as_ingredient_is_dropped():
    f = m({"ingredients": {"include": ["Capixyl"], "exclude": []}}, "capixy tonic vs vials")
    assert f.ingredients.include == []
    f = m({"ingredients": {"include": ["Capixyl"], "exclude": []}}, "something with capixyl")
    assert f.ingredients.include == ["Capixyl"]


def test_recommended_type_is_dropped_but_asked_type_kept():
    f = m({"product_type": {"product_type": ["hair tonic", "deodorant"]}},
          "need somthing for my hair falling and a good deodrant")
    assert f.product_type == ["deodorant"]
    f = m({"product_type": {"product_type": ["face serum", "sunscreen"]}}, "vitamn c seerum and a sun blok")
    assert f.product_type == ["face serum", "sunscreen"]


def test_type_via_alias_is_kept():
    f = m({"product_type": {"product_type": ["cleanser"]}}, "micellar water to remove makeup")
    assert f.product_type == ["cleanser"]


def test_assumed_area_is_dropped_but_stated_area_kept():
    f = m({"suitable_for": {"suitable_for": ["body", "sensitive skin"]}}, "burn spray for sensitive skin")
    assert f.suitable_for == ["sensitive skin"]
    f = m({"suitable_for": {"suitable_for": ["eye area"]}}, "eye creme for wrinkles under the eyes")
    assert f.suitable_for == ["eye area"]


def test_brand_not_mentioned_is_dropped_unless_a_product_implies_it():
    f = m({"brand": {"brand": ["Capixy"]}}, "androgenetic alopecia with redensyl or capixyl")
    assert f.brand == []
    f = m({"brand": {"brand": ["Vacation"]}}, "sebio control toner")
    assert f.brand == ["Vacation"]


def test_grounding_drop_does_not_cascade_through_hierarchy():
    f = m({"product_type": {"product_type": ["eye cream", "lip balm"]},
           "product_group": {"product_group": ["Eye & Lash Care", "Lip Care"]}},
          "a cream for dark circles and something for chapped lips")
    assert f.product_group == ["Eye & Lash Care", "Lip Care"]


def test_every_drop_is_noted():
    f = m({"suitable_for": {"suitable_for": ["scalp"]}}, "hair mask for frizzy hair")
    assert any("suitable_for: dropped" in n for n in f.meta.notes)


def test_name_matcher_prefers_rare_words():
    f = merge_outputs({"names": {"name_en": ["Capixy Lash Serum"], "name_ar": []}})
    assert f.matched_handles == ["capixy-lashes-treatment-serum-10ml"]
    f = merge_outputs({"names": {"name_en": ["Soralone Urea 15"], "name_ar": []}})
    assert f.matched_handles == ["soralone-urea-15-cream-gel-60ml"]
    f = merge_outputs({"names": {"name_en": ["Capixy Intense Tonic Spray"], "name_ar": []}})
    assert f.matched_handles == ["capixy-intense-tonic-spray-125ml"]
