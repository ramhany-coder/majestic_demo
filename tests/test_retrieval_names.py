"""Name step: normalization, the bigram tokenizer, the Dice check, and the
generated typo test (targets: top-1 >= 95%, top-3 >= 99%)."""

import pytest

from agents.retrieval.index_builder import get_index
from config import settings
from agents.retrieval.name_search import (
    bigrams, dice, is_arabic, match_name, normalize_name_ar, normalize_name_en, search_names,
)
from scripts.name_typos import REAL_CASES, TARGET_TOP1, TARGET_TOP3, drop_brand, evaluate, generate, real_cases


def test_bigrams_with_boundary_markers():
    assert bigrams("capixy foam") == ["^c", "ca", "ap", "pi", "ix", "xy", "y$", "^f", "fo", "oa", "am", "m$"]
    assert bigrams("3") == ["^3", "3$"]
    assert bigrams("") == []


def test_arabic_bigrams():
    assert bigrams("كابيكسي فوم") == ["^ك", "كا", "اب", "بي", "يك", "كس", "سي", "ي$", "^ف", "فو", "وم", "م$"]


@pytest.mark.parametrize("raw,norm", [
    ("Capixy Intense Dry Foam 120 ml", "capixy intense dry foam"),
    ("Movelex Cream 120gm", "movelex cream"),
    ("Capixy Hair Oil Elixir 30 ML", "capixy hair oil elixir"),
    ("Vacation Piña Colada Whitening Deodorant 50 ml", "vacation pina colada whitening deodorant"),
    ("Vacation Niacinamide Serum 30 ml ( 10% + Zinc PCA )", "vacation niacinamide serum 10 zinc pca"),
    ("Capixy Intense Tonic Spray 125ml+1000 Laser Pulses Card", "capixy intense tonic spray 1000 laser pulses card"),
    ("Methytral®️ Nano™ Spray", "methytral nano spray"),
    ("Pirlome 60 caps", "pirlome"),
    ("Soralone Urea 15% Cream Gel 60ml", "soralone urea 15 cream gel"),
])
def test_english_normalization(raw, norm):
    assert normalize_name_en(raw) == norm


@pytest.mark.parametrize("raw,norm", [
    ("ڈاكيشن سيروم ڈيتامين سي 10% – 30 مل", "فاكيشن سيروم فيتامين سي 10"),
    ("ڤاكيشن سيروم", "فاكيشن سيروم"),
    ("موڄلكس أدڄانس", "موفلكس ادفانس"),
    ("3 × كابيكسي أمبولات ضد تساقط الشعر 70 مل", "3 كابيكسي امبولات ضد تساقط الشعر"),
    ("موڤلكس كريم 120 جم", "موفلكس كريم"),
    ("ڤاكيشن واقي شمس ملوّن 50 مل", "فاكيشن واقي شمس ملون"),          # shadda dropped; ملون is not a size
    ("إن شيب – عدة الليزر", "ان شيب عده الليزر"),
    ("مرطّب شفاه – SPF 30+", "مرطب شفاه spf 30"),
    ("فيمي ٩ غسول", "فيمي 9 غسول"),                                      # Arabic-Indic digit
    ("كـــريم آمن ى", "كريم امن ي"),                                      # tatweel, alef madda, alef maqsura
])
def test_arabic_normalization(raw, norm):
    assert normalize_name_ar(raw) == norm


def test_dice_and_script_detection():
    a, b = frozenset(bigrams("capixy")), frozenset(bigrams("capixi"))
    assert dice(a, a) == 1.0 and 0 < dice(a, b) < 1 and dice(a, frozenset()) == 0.0
    assert is_arabic("كابكسي") and not is_arabic("capixy")


def test_both_acceptance_checks_apply():
    idx = get_index()
    hits, _ = match_name("zzzz qqqq", idx.names_en)
    assert hits == []                                                   # nothing clears the Dice floor
    hits, _ = match_name("capixy intense dry foam", idx.names_en)
    assert hits[0].handle == "capixy-dry-foam-120-ml" and hits[0].score == 1.0
    assert all(h.score >= 0.8 and h.dice >= settings.RETRIEVAL_NAME_DICE_MIN for h in hits)


def test_a_product_line_matches_its_variants():
    hits, _ = match_name("Vacation whitening deodorant", get_index().names_en)
    handles = [h.handle for h in hits]
    assert len(handles) == 7 and all("whitening-deodorant" in h for h in handles)


def test_search_names_pairs_english_and_arabic():
    idx = get_index()
    res = search_names(["Capixy Intense Dry Foam", "Zzqx Wibble"], ["كابيكسي إنتنس دراي فوم"],
                       idx.names_en, idx.names_ar)
    assert res.handles[0] == "capixy-dry-foam-120-ml" and res.unresolved == ["Zzqx Wibble"]


def test_dice_floor_is_what_rejects_long_weak_matches():
    # "Nonexistent Thing" shares common bigrams with a long deodorant name:
    # Dice 0.35 exactly, so the plan's floor accepts it and the default 0.45
    # does not (ARCHITECTURE_NOTES.md section 10, "Dice threshold sweep").
    idx = get_index()
    assert match_name("Nonexistent Thing", idx.names_en, dice_min=0.35)[0]
    assert match_name("Nonexistent Thing", idx.names_en, dice_min=0.45)[0] == []


@pytest.mark.parametrize("query", list(REAL_CASES))
def test_real_misspellings(query):
    top = real_cases()[query]
    assert top and top[0] in REAL_CASES[query]


def test_soralon_cica_finds_the_cica_line():
    top = real_cases()["soralon cica"]
    assert set(top) == REAL_CASES["soralon cica"]


def test_drop_brand():
    assert drop_brand("capixy intense dry foam", ["capixy"]) == "intense dry foam"
    assert drop_brand("in shape laser kit", ["in shape", "inshape"]) == "laser kit"


def test_generated_typos_meet_the_targets():
    cases = generate(seed=13)
    assert len(cases) > 1000
    assert {c.kind for c in cases} == {"delete", "swap", "replace", "two_edits", "no_brand"}
    stats = evaluate(cases)["stats"]["all"]
    assert stats["top1"] / stats["n"] >= TARGET_TOP1
    assert stats["top3"] / stats["n"] >= TARGET_TOP3


def test_generator_is_deterministic():
    assert [c.query for c in generate(seed=5)] == [c.query for c in generate(seed=5)]
