"""Retrieval agent (no LLM): exact filters, relaxation, bundle rule, name
hits, fusion and the output contract. Logic tests run on a small synthetic
catalog, where every expected set can be worked out by hand; the last tests
use the real catalog."""

import statistics
import time

import pytest

from agents.retrieval.agent import resolve_k, retrieve
from agents.retrieval.filters import RELAX_ORDER, apply_filters, form_restates_type, form_variants
from agents.retrieval.fusion import mentions, says_free_of
from agents.retrieval.embedding_registry import embedding_text
from agents.retrieval.index_builder import build_index, get_index
from agents.retrieval.semantic import SemanticScores
from config import settings
from models.filter_extractor import IngredientFilter, MetadataFilters, UnmatchedTerms
from models.retrieval import RetrievalResult


def prod(handle, name, brand="Alpha", product_type="serum", product_form="serum", concerns=(), suitable_for=(),
         ingredients=(), hero=None, price=100.0, available=True, kind="single", group="Face Care",
         contains=(), description="", best_seller=False):
    return {
        "handle": handle, "name": name, "name_ar": None, "brand": brand, "category": "Skin Care",
        "product_group": group, "product_type": product_type, "product_form": product_form,
        "concerns": list(concerns), "suitable_for": list(suitable_for), "ingredients_canonical": list(ingredients),
        "hero_ingredient": {"name": hero} if hero else None, "price": price, "compare_at_price": None,
        "promotion": None, "available": available, "product_kind": kind, "contains_product_types": list(contains),
        "bundle_components": [], "description": description, "best_seller": best_seller,
        "url": f"https://example.com/{handle}", "images": [{"src": f"https://img/{handle}.png"}],
    }


PRODUCTS = [
    prod("a", "Alpha Niacinamide Serum", concerns=["acne", "excess sebum"], suitable_for=["oily skin"],
         ingredients=["Niacinamide", "Zinc PCA"], price=100,
         description="A silicone-free serum for oily skin. Free from parabens, alcohol and fragrance."),
    prod("b", "Alpha Niacinamide Spray", product_form="spray", concerns=["acne"], suitable_for=["dry skin"],
         ingredients=["Niacinamide", "Silicone"], price=90, description="Contains alcohol for a quick dry finish."),
    prod("c", "Beta Urea Cream", brand="Beta", product_type="cream", product_form="cream", concerns=["dryness"],
         suitable_for=["dry skin"], ingredients=["Urea"], hero="Urea", price=80, description="Rich urea cream."),
    prod("d", "Beta Sun Gel", brand="Beta", product_type="sunscreen", product_form="gel",
         concerns=["sun protection", "acne"], suitable_for=["oily skin"], ingredients=["Zinc PCA"], price=120,
         available=False, description="Light sun gel."),
    prod("x", "Alpha Serum Bundle", product_type="bundle", product_form="kit", group="Bundles & Offers",
         kind="bundle", contains=["serum"], ingredients=["Niacinamide"], price=150, description="Two serums."),
]


@pytest.fixture(scope="module")
def ix():
    return build_index({"products": PRODUCTS}, digest="test")


def candidates(ix, intent="find_products", **filters):
    return apply_filters(ix, MetadataFilters(**filters), intent)


def ok_sem(**scores):
    return SemanticScores(dict(scores), status="ok")


@pytest.fixture
def relax(monkeypatch):
    """The plan's relaxation (RETRIEVAL_RELAX_FILTERS); the default is strict."""
    monkeypatch.setattr(settings, "RETRIEVAL_RELAX_FILTERS", True)


# --- exact filters -------------------------------------------------------------

def test_or_within_a_key(ix):
    assert candidates(ix, product_type=["serum", "cream"]).candidates == {"a", "b", "c"}


def test_and_across_keys(ix):
    out = candidates(ix, product_type=["serum"], suitable_for=["oily skin"])
    assert out.candidates == {"a"} and out.relaxed_keys == []
    assert out.applied == {"product_type": ["serum"], "suitable_for": ["oily skin"]}


def test_include_needs_all_ingredients(ix):
    out = candidates(ix, ingredients=IngredientFilter(include=["Niacinamide", "Zinc PCA"]))
    assert out.candidates == {"a"}                  # b has Niacinamide only, d has Zinc PCA only


def test_exclude_removes_any_match(ix):
    assert candidates(ix, product_type=["serum"], ingredients=IngredientFilter(exclude=["Silicone"])).candidates == {"a"}
    out = candidates(ix, ingredients=IngredientFilter(exclude=["Silicone", "Urea"]))
    assert out.candidates == {"a", "d"} and out.excluded == {"b", "c"}   # no active keys: the pool, minus bundle x
    assert out.applied == {"ingredients.exclude": ["Silicone", "Urea"]}


def test_no_filters_is_the_whole_pool_without_bundles(ix):
    assert apply_filters(ix, None, "find_products").candidates == {"a", "b", "c", "d"}
    assert apply_filters(ix, MetadataFilters(), "find_products").fallback is None


def test_unknown_value_matches_nothing(ix, relax):
    out = candidates(ix, product_type=["not a type"])
    assert out.relaxed_keys == ["product_type"] and out.fallback == "semantic_only"


# --- strict (default): every requested key must match ---------------------------------

def test_strict_by_default_returns_nothing_when_no_product_matches_every_key(ix):
    assert settings.RETRIEVAL_RELAX_FILTERS is False
    out = candidates(ix, product_type=["cream"], product_form=["spray"])
    assert out.candidates == set() and out.relaxed_keys == [] and out.fallback is None
    assert out.applied == {"product_type": ["cream"], "product_form": ["spray"]}
    assert out.near_miss_keys == ["product_type", "product_form"]   # each alone would give b or c


def test_strict_keeps_every_candidate_that_matches_all_keys(ix):
    res = retrieve(MetadataFilters(product_type=["serum"], concerns=["acne"]), ok_sem(a=0.2, b=0.9),
                   intent="find_products", index=ix)
    assert res.handles == ["b", "a"] and res.relaxed_keys == [] and res.meta["near_miss_keys"] == []


def test_strict_empty_result_still_shows_name_hits(ix):
    f = MetadataFilters(name_en=["Beta Urea Cream"], product_type=["cream"], product_form=["spray"])
    res = retrieve(f, ok_sem(), intent="find_products", index=ix)
    assert res.handles == ["c"] and res.total_candidates == 0
    assert res.products[0].match["conflict"] == "filter_mismatch"


def test_near_miss_names_only_the_keys_that_block():
    idx = get_index()
    f = MetadataFilters(product_type=["moisturizer"], product_form=["cream"], concerns=["sensitivity"],
                        suitable_for=["sensitive skin"], product_group=["Face Care"], category=["Skin Care"])
    res = retrieve(f, ok_sem(), intent="find_products")
    assert res.products == [] and res.meta["near_miss_keys"] == ["product_type", "suitable_for"]


# --- product_form variants -------------------------------------------------------------

def test_form_variants_add_forms_containing_the_word():
    forms = ["cream", "cream gel", "tinted cream", "leave-in cream", "gel", "spray"]
    assert form_variants(forms, ["cream"]) == ["cream", "cream gel", "tinted cream", "leave-in cream"]
    assert form_variants(forms, ["gel"]) == ["gel", "cream gel"]
    assert form_variants(forms, ["spray"]) == ["spray"]


def test_cream_matches_cream_gels_for_dry_skin():
    f = MetadataFilters(category=["Skin Care"], product_group=["Face Care"], product_form=["cream"],
                        concerns=["dryness"], suitable_for=["dry skin"])
    res = retrieve(f, ok_sem(), intent="find_products")
    assert set(res.handles) == {"soralone-urea-15-cream-gel-60ml", "soralone-hydra-cream-gel-100ml-copy"}


def test_form_variants_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(settings, "RETRIEVAL_FORM_VARIANTS", False)
    f = MetadataFilters(product_group=["Face Care"], product_form=["cream"], suitable_for=["dry skin"])
    assert not {"soralone-urea-15-cream-gel-60ml"} & set(retrieve(f, ok_sem(), intent="find_products").handles)


@pytest.mark.parametrize("requested,restates", [
    ({"product_type": ["body lotion"], "product_form": ["lotion"]}, True),
    ({"product_type": ["anti-aging cream"], "product_form": ["cream"]}, True),
    ({"product_type": ["sunscreen"], "product_form": ["spray"]}, False),
    ({"product_type": ["body lotion", "sunscreen"], "product_form": ["lotion", "spray"]}, False),
    ({"product_form": ["cream"]}, False),
])
def test_form_restates_type(requested, restates):
    assert form_restates_type(requested) is restates


def test_form_that_restates_the_type_is_not_applied():
    # The Body Milk is typed "body lotion" with form "milk".
    f = MetadataFilters(product_type=["body lotion"], product_form=["lotion"], suitable_for=["dry skin"])
    res = retrieve(f, ok_sem(), intent="find_products")
    assert res.handles == ["vacation-body-milk"] and res.meta["implied_keys"] == ["product_form"]
    assert "product_form" not in res.applied_filters
    assert res.products[0].match["filters_matched"] == ["product_type", "suitable_for"]


# --- relaxation ----------------------------------------------------------------

def test_relax_order_is_the_plan_order():
    assert RELAX_ORDER == ("product_form", "suitable_for", "concerns", "product_group", "category",
                           "hero_ingredient", "ingredients.include", "product_type", "brand")


def test_relaxes_one_key_at_a_time_and_stops_at_first_non_empty(ix, relax):
    out = candidates(ix, product_type=["cream"], product_form=["spray"], suitable_for=["oily skin"],
                     concerns=["acne"])
    assert out.relaxed_keys == ["product_form", "suitable_for", "concerns"] and out.candidates == {"c"}
    assert out.applied == {"product_type": ["cream"]} and out.fallback is None


def test_product_type_is_relaxed_before_brand(ix, relax):
    out = candidates(ix, brand=["Alpha"], product_type=["sunscreen"])
    assert out.relaxed_keys == ["product_type"] and out.candidates == {"a", "b"}


def test_exclude_is_never_relaxed(ix, relax):
    out = candidates(ix, hero_ingredient=["Urea"], ingredients=IngredientFilter(exclude=["Urea"]))
    assert out.relaxed_keys == ["hero_ingredient"] and out.fallback == "semantic_only"
    assert "c" not in out.candidates and out.candidates == {"a", "b", "d"}
    assert out.applied == {"ingredients.exclude": ["Urea"]}


# --- bundle rule -------------------------------------------------------------------

@pytest.mark.parametrize("intent,filters,allowed", [
    ("find_products", {"product_type": ["serum"]}, False),
    ("price_offer", {"product_type": ["serum"]}, True),                 # indexed under contains_product_types
    ("find_products", {"product_type": ["bundle"]}, True),
    ("find_products", {"product_group": ["Bundles & Offers"]}, True),
])
def test_bundle_rule(ix, intent, filters, allowed):
    out = candidates(ix, intent=intent, **filters)
    assert ("x" in out.candidates) is allowed and out.bundles_allowed is allowed


def test_bundle_rule_can_be_turned_off(ix, monkeypatch):
    monkeypatch.setattr(settings, "RETRIEVAL_BUNDLE_RULE", False)
    assert "x" in candidates(ix, product_type=["serum"]).candidates


def test_named_bundle_allows_bundles_and_they_rank_lower(ix):
    res = retrieve(MetadataFilters(name_en=["Alpha Serum Bundle"], product_type=["serum"]), ok_sem(a=0.5, b=0.5, x=0.5),
                   intent="find_products", index=ix)
    assert res.name_hits[0] == "x" and res.meta["bundles_allowed"]
    res = retrieve(MetadataFilters(product_type=["serum"]), ok_sem(a=0.5, b=0.5, x=0.5), intent="price_offer", index=ix)
    assert res.handles[-1] == "x" and "bundle" in res.products[-1].match["boosts"]


# --- "all skin types" ----------------------------------------------------------------

@pytest.fixture(scope="module")
def ix_all():
    extra = prod("e", "Alpha Gentle Serum", suitable_for=["all skin types"], price=70, description="For every skin.")
    return build_index({"products": PRODUCTS + [extra]}, digest="test-all-skin")


def test_all_skin_types_matches_a_specific_skin_type(ix_all):
    out = candidates(ix_all, product_type=["serum"], suitable_for=["oily skin"])
    assert out.candidates == {"a", "e"} and out.relaxed_keys == []


def test_exact_skin_type_label_ranks_first(ix_all):
    res = retrieve(MetadataFilters(product_type=["serum"], suitable_for=["oily skin"]), ok_sem(a=0.5, e=0.52),
                   intent="find_products", index=ix_all)
    assert res.handles == ["a", "e"] and "all_skin_types" in res.products[1].match["boosts"]
    assert res.products[1].match["filters_matched"] == ["product_type", "suitable_for"]


def test_asking_for_all_skin_types_stays_exact(ix_all):
    out = candidates(ix_all, suitable_for=["all skin types"])
    assert out.candidates == {"e"}
    res = retrieve(MetadataFilters(suitable_for=["all skin types"]), ok_sem(), intent="find_products", index=ix_all)
    assert "all_skin_types" not in res.products[0].match["boosts"]


def test_all_skin_types_match_can_be_turned_off(ix_all, monkeypatch):
    monkeypatch.setattr(settings, "RETRIEVAL_ALL_SKIN_TYPES_MATCH", False)
    assert candidates(ix_all, product_type=["serum"], suitable_for=["oily skin"]).candidates == {"a"}
    out = candidates(ix_all, product_type=["serum"], suitable_for=["sensitive skin"])
    assert out.candidates == set() and out.near_miss_keys == ["suitable_for"]


# --- name hits ---------------------------------------------------------------------

def test_name_hits_come_first_even_when_they_fail_the_filters(ix):
    f = MetadataFilters(name_en=["alpha niacinamid spray"], suitable_for=["oily skin"])
    res = retrieve(f, ok_sem(a=0.9, d=0.8), intent="find_products", index=ix)
    assert res.handles[0] == "b" and res.name_hits == ["b"]
    assert res.products[0].match["conflict"] == "filter_mismatch" and res.products[0].match["source"] == "name"
    assert res.handles[1:] == ["a", "d"]              # then C, filled for a find intent (d is out of stock)


def test_named_product_with_excluded_ingredient_is_kept_and_flagged(ix):
    f = MetadataFilters(name_en=["Alpha Niacinamide Spray"], ingredients=IngredientFilter(exclude=["Silicone"]))
    res = retrieve(f, ok_sem(), intent="product_info", index=ix)
    assert res.handles == ["b"] and res.products[0].match["conflict"] == "contains_excluded"


@pytest.mark.parametrize("intent,filled", [("find_products", True), ("refine_products", True),
                                           ("product_info", False), ("compare", False), ("how_to_use", False),
                                           ("safety", False), ("price_offer", False)])
def test_only_find_intents_fill_after_name_hits(ix, intent, filled):
    res = retrieve(MetadataFilters(name_en=["Beta Urea Cream"]), ok_sem(), intent=intent, index=ix)
    assert res.handles[0] == "c" and (len(res.handles) > 1) is filled


def test_unresolved_name_carries_on_with_the_filters(ix):
    res = retrieve(MetadataFilters(name_en=["Zeta Hair Tonic"], product_type=["cream"]), ok_sem(),
                   intent="product_info", index=ix)
    assert res.unresolved_names == ["Zeta Hair Tonic"] and res.name_hits == [] and res.handles == ["c"]


def test_arabic_form_resolves_a_name(ix):
    res = retrieve(MetadataFilters(name_en=["Gamma Thing"], name_ar=["alpha niacinamide serum"]), ok_sem(),
                   intent="product_info", index=ix)
    assert res.name_hits == ["a"] and res.unresolved_names == []


# --- fusion ------------------------------------------------------------------------

def test_out_of_stock_always_ranks_below_in_stock(ix):
    res = retrieve(MetadataFilters(concerns=["acne"]), ok_sem(a=0.2, b=0.1, d=0.99), index=ix)
    assert res.handles == ["a", "b", "d"] and res.products[-1].available is False


def test_ties_break_on_lower_price(ix):
    res = retrieve(MetadataFilters(brand=["Alpha"]), ok_sem(a=0.5, b=0.5), index=ix)
    assert res.handles == ["b", "a"]                    # b is cheaper (90 < 100)


def test_extra_concern_and_unmatched_terms(ix):
    f = MetadataFilters(concerns=["acne", "excess sebum"], unmatched=UnmatchedTerms(exclude=["alcohol"]))
    res = retrieve(f, ok_sem(a=0.5, b=0.5), index=ix)
    boosts = {p.handle: p.match["boosts"] for p in res.products}
    assert "extra_concerns:1" in boosts["a"] and "free_of:alcohol" in boosts["a"]
    assert "mentions:alcohol" in boosts["b"]
    w = settings.RETRIEVAL_FUSION_WEIGHTS
    score = {p.handle: p.score for p in res.products}
    assert score["a"] == pytest.approx(0.5 + w["available"] + w["extra_concern"] + w["exclude_free"])
    assert score["b"] == pytest.approx(0.5 + w["available"] + w["unmatched_exclude"])


def test_free_of_patterns():
    desc = ("free from ammonia, aluminum, and alcohol. it contains no silicone, paraben, sodium, or sulfate. "
            "a sulfate-free wash, fragrance free. no matter the weather it stays fresh.")
    for term in ("aluminum", "alcohol", "silicones", "parabens", "sulfate", "fragrance"):
        assert says_free_of(desc, term), term
    assert not says_free_of("contains alcohol for a quick dry finish.", "alcohol")
    assert mentions("contains alcohol.", "alcohol") and not mentions("alcoholic", "alcohol")


# --- output contract, k, semantic failure --------------------------------------------

def test_k_defaults_to_10_and_caps_at_20():
    assert resolve_k(None) == 10 and resolve_k(0) == 10 and resolve_k(3) == 3 and resolve_k(50) == 20
    assert len(retrieve(None, ok_sem()).products) == 10
    assert len(retrieve(None, ok_sem(), k=50).products) == 20
    assert len(retrieve(MetadataFilters(product_type=["deodorant"]), ok_sem(), k=3).products) == 3


def test_semantic_failure_sets_no_semantic_and_still_ranks(ix):
    res = retrieve(MetadataFilters(concerns=["acne"]), SemanticScores(status="failed", error="boom"), index=ix)
    assert res.fallback == "no_semantic" and res.meta["semantic"] == "failed" and res.handles == ["b", "a", "d"]
    assert retrieve(MetadataFilters(), None, index=ix).fallback == "no_semantic"     # search didn't run


def test_both_fallbacks_are_recorded(ix, relax):
    f = MetadataFilters(hero_ingredient=["Urea"], ingredients=IngredientFilter(exclude=["Urea"]))
    res = retrieve(f, SemanticScores(status="timeout"), index=ix)
    assert res.fallback == "no_semantic" and res.meta["fallbacks"] == ["no_semantic", "semantic_only"]


def test_output_contract(ix, relax):
    f = MetadataFilters(product_type=["cream"], product_form=["spray"])
    res = retrieve(f, ok_sem(c=0.4), intent="find_products", k=5, index=ix)
    assert isinstance(res, RetrievalResult) and res.k == 5 and res.total_candidates == 1
    assert res.relaxed_keys == ["product_form"] and res.meta["relaxed_values"] == {"product_form": ["spray"]}
    assert res.applied_filters == {"product_type": ["cream"]}
    p = res.products[0]
    assert (p.handle, p.name, p.brand, p.product_type, p.price, p.image) == (
        "c", "Beta Urea Cream", "Beta", "cream", 80.0, "https://img/c.png")
    assert set(p.match) >= {"name_score", "sem_score", "filters_matched", "boosts", "conflict"}
    assert p.match["filters_matched"] == ["product_type"]
    assert set(res.meta["latency_ms"]) == {"names", "filters", "semantic", "fusion", "total"}


def test_embedding_text_falls_back_to_bundle_components():
    p = prod("y", "Duo Kit", kind="bundle")
    p["bundle_components"] = [{"name": "Serum A"}, {"name": "Cream B"}]
    assert embedding_text(p) == "Duo Kit | Serum A | Cream B"
    assert embedding_text(prod("z", "Z", description="  Real   text. ")) == "Real text."


# --- real catalog ---------------------------------------------------------------------

def test_real_catalog_pool_and_indexes():
    idx = get_index()
    assert len(idx.order) == 108 and len(idx.bundles) == 14
    assert "3-capixy-anti-hair-vials-70ml" in idx.inverted["product_type"]["hair vials"]   # via contains
    assert idx.inverted["hero_ingredient"]["Urea"] == {"soralone-urea-5-hand-cream-60ml", "soralone-urea-15-cream-gel-60ml"}


def test_real_hair_serum_without_silicone():
    f = MetadataFilters(product_type=["hair serum"], ingredients=IngredientFilter(exclude=["Silicone"]))
    res = retrieve(f, ok_sem(), intent="find_products")
    assert set(res.handles) == {"capixy-hair-serum", "capixy-anti-dandruff-serum-spray-120ml"}
    assert res.relaxed_keys == [] and res.total_candidates == 2


def test_real_sunscreen_spray_for_oily_skin_is_the_all_skin_types_spray():
    f = MetadataFilters(product_type=["sunscreen"], product_form=["spray"], suitable_for=["oily skin"])
    res = retrieve(f, ok_sem(), intent="find_products")
    assert res.relaxed_keys == [] and res.handles == ["vacation-sunscreen-lotion-spray-200ml"]


def test_real_sunscreen_spray_for_oily_skin_relaxes_the_form_with_exact_labels(monkeypatch, relax):
    # The plan's behavior: no spray is labeled oily skin, so product_form is relaxed.
    monkeypatch.setattr(settings, "RETRIEVAL_ALL_SKIN_TYPES_MATCH", False)
    f = MetadataFilters(product_type=["sunscreen"], product_form=["spray"], suitable_for=["oily skin"])
    res = retrieve(f, ok_sem(), intent="find_products")
    assert res.relaxed_keys == ["product_form"] and res.products
    assert all(p.product_type == "sunscreen" for p in res.products)


def test_filters_names_and_fusion_are_fast():
    f = MetadataFilters(name_en=["Capixy Intense Dry Foam"], product_type=["shampoo"], concerns=["hair loss"],
                        unmatched=UnmatchedTerms(exclude=["sulfate"]))
    sem = ok_sem(**{h: 0.3 for h in get_index().order})
    retrieve(f, sem, intent="find_products")
    runs = []
    for _ in range(30):
        t = time.perf_counter()
        retrieve(f, sem, intent="find_products")
        runs.append((time.perf_counter() - t) * 1000)
    assert statistics.median(runs) < 20      # target <= 10 ms, reported by scripts/eval_retrieval.py
