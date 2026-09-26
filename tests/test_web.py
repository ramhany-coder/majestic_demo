"""The Jamila design system's rules, checked in code: token contrast, generated
files up to date, the card allow-list shared by Python and JS, and the voice
rules on copy the widget speaks on its own."""

import json
import re

from agents.orchestrator.orchestrator import PRODUCTS_INTRO
from agents.prequal.small_talk import REPLIES
from api.widget import CARD_FIELDS
from scripts.build_web import BUNDLE_JS, TOKENS_JSON, check_tokens, contrast, stale_files

BUNDLE = BUNDLE_JS.read_text(encoding="utf-8")
TOKENS = json.loads(TOKENS_JSON.read_text(encoding="utf-8"))
CATALOG = json.loads((TOKENS_JSON.parent.parent / "data" / "majestic_catalog.json").read_text(encoding="utf-8"))
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿️]")


def _block(start: str, end: str) -> str:
    i = BUNDLE.index(start)
    return BUNDLE[i:BUNDLE.index(end, i)]


def _string_literals(js: str):
    return [a or b for a, b in re.findall(r"'((?:[^'\\\n]|\\.)*)'|\"((?:[^\"\\\n]|\\.)*)\"", js)]


def test_every_text_token_clears_its_floor_in_both_themes():
    assert check_tokens(TOKENS) == []


def test_brand_is_a_fill_that_fails_as_text_on_white():
    brand = TOKENS["color"]["brand"]
    assert brand["role"] == "fill" and "on" not in brand
    assert round(contrast(brand["light"], "#ffffff"), 2) == 2.95


def test_generated_files_are_up_to_date():
    assert stale_files() == [], "run: python -m scripts.build_web"


def test_arabic_type_runs_one_line_step_looser():
    types = TOKENS["type"]
    for name, latin in types.items():
        if name.endswith("-ar"):
            continue
        ar = types[name + "-ar"]
        assert ar["size"] == latin["size"] and ar["line"] == latin["line"] + 4, name
        assert (latin["family"], ar["family"]) == ("latin", "arabic")


def test_card_allow_list_is_the_same_in_python_and_js():
    js = _block("const CARD_FIELDS = Object.freeze([", "]);")
    assert tuple(_string_literals(js)) == CARD_FIELDS
    for hidden in ("description", "key_ingredients", "how_to_use", "warnings", "tags", "collections", "sku"):
        assert hidden not in CARD_FIELDS


def test_widget_copy_has_no_exclamation_marks_or_emoji():
    copy = _string_literals(_block("const STRINGS = {", "// Catalogue vocabulary"))
    assert copy, "STRINGS block not found"
    assert [s for s in copy if "!" in s or EMOJI.search(s)] == []
    assert not EMOJI.search(BUNDLE)


def test_templated_server_replies_follow_the_voice_rules():
    texts = list(PRODUCTS_INTRO.values()) + [t for kinds in REPLIES.values() for t in kinds.values()]
    for text in texts:
        assert "!" not in text and not EMOJI.search(text), text
        assert not re.search(r"\bليكي\b|\bليك\b", text), text


def test_health_copy_carries_the_disclaimer_verbatim():
    assert "المعلومات للتوعية ومش بديلة عن استشارة الطبيب" in BUNDLE


def test_every_catalogue_value_on_a_card_has_an_arabic_label():
    vocab = _block("const VOCAB = {", "suitConcerns")
    concerns = {v for p in CATALOG["products"] for v in p["concerns"]} - {"pregnancy"}
    suitable = {v for p in CATALOG["products"] for v in p["suitable_for"]} | {"pregnancy"}
    labelled = set(re.findall(r"'([^']+)': ", vocab))
    assert sorted(concerns - labelled) == [] and sorted(suitable - labelled) == []


def test_only_the_send_arrow_and_chevron_flip():
    assert "const FLIP = new Set(['send', 'chevron']);" in BUNDLE


def test_the_console_shows_products_through_the_widgets_cards():
    console = (BUNDLE_JS.parent.parent / "console.js").read_text(encoding="utf-8")
    assert "Jamila.ProductResults(" in console and "ProductResults: ProductResults," in BUNDLE
