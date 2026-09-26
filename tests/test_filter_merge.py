from agents.filter_extractor.merge import merge_outputs


def test_hierarchy_drops_contradicting_group_and_category():
    f = merge_outputs({
        "product_type": {"product_type": ["sunscreen"]},
        "product_group": {"product_group": ["Sun Care", "Hair & Scalp Care"]},
        "category": {"category": ["Skin Care", "Personal Care"]},
    })
    assert f.product_group == ["Sun Care"]
    assert f.category == ["Skin Care"]
    assert any("Hair & Scalp Care" in n for n in f.meta.notes)


def test_hierarchy_never_adds_parents():
    f = merge_outputs({"product_type": {"product_type": ["shampoo"]}})
    assert f.product_group == [] and f.category == []


def test_group_constrains_category_when_no_type():
    f = merge_outputs({
        "product_group": {"product_group": ["Lip Care"]},
        "category": {"category": ["Hair Care", "Skin Care"]},
    })
    assert f.category == ["Skin Care"]


def test_type_spanning_several_groups_keeps_all_of_them():
    # "bundle" appears under three categories in the catalog.
    f = merge_outputs({
        "product_type": {"product_type": ["bundle"]},
        "category": {"category": ["Hair Care", "Health & Support"]},
    })
    assert f.category == ["Hair Care", "Health & Support"]


def test_hero_adds_mapped_ingredients_unless_excluded():
    f = merge_outputs({
        "hero_ingredient": {"hero_ingredient": ["Niacinamide + Zinc PCA"]},
        "ingredients": {"include": ["Hyaluronic Acid"], "exclude": ["Zinc PCA"]},
    })
    assert f.ingredients.include == ["Hyaluronic Acid", "Niacinamide (Vitamin B3)"]
    assert f.ingredients.exclude == ["Zinc PCA"]


def test_exclude_wins_over_include():
    f = merge_outputs({"ingredients": {"include": ["Retinol", "Vitamin C"], "exclude": ["Retinol"],
                                       "unmatched_include": ["alcohol"], "unmatched_exclude": ["alcohol"]}})
    assert f.ingredients.include == ["Vitamin C"]
    assert f.ingredients.exclude == ["Retinol"]
    assert f.unmatched.include == [] and f.unmatched.exclude == ["alcohol"]


def test_names_match_handles_and_add_brand():
    f = merge_outputs({
        "names": {"name_en": ["Capixy Intense Dry Foam", "Capixy Anti Hair Loss Vials"], "name_ar": []},
        "brand": {"brand": []},
    })
    assert f.matched_handles == ["capixy-dry-foam-120-ml", "capixy-anti-hair-loss-vials-70ml"]
    assert f.brand == ["Capixy"]


def test_name_match_prefers_the_exact_product_over_near_misses():
    f = merge_outputs({"names": {"name_en": ["Capixy Hair Serum"], "name_ar": []}})
    assert f.matched_handles == ["capixy-hair-serum"]


def test_product_line_name_matches_every_product_in_the_line():
    f = merge_outputs({"names": {"name_en": ["Sebio-Control"], "name_ar": []}})
    assert len(f.matched_handles) == 5
    assert f.brand == ["Vacation"]


def test_bundles_only_when_asked_for():
    single = merge_outputs({"names": {"name_en": ["Capixy Anti Hair Loss Vials"], "name_ar": []}})
    assert single.matched_handles == ["capixy-anti-hair-loss-vials-70ml"]
    bundle = merge_outputs({"names": {"name_en": ["3 x Capixy Anti Hair Loss Vials bundle"], "name_ar": []}})
    assert "3-capixy-anti-hair-vials-70ml" in bundle.matched_handles


def test_arabic_name_matches_despite_letter_variants():
    # The catalog spells Vacation both ڤاكيشن and ڈاكيشن; either should match.
    f = merge_outputs({"names": {"name_en": [], "name_ar": ["ڤاكيشن سيروم ريتينول"]}})
    assert "vacation-retinol-serum-1-30-ml" in f.matched_handles


def test_dedupe_keeps_first_mention_order():
    f = merge_outputs({"concerns": {"concerns": ["acne", "dryness", "acne"], "unmatched": []}})
    assert f.concerns == ["acne", "dryness"]


def test_empty_outputs_give_empty_filters():
    assert merge_outputs({}).is_empty()
