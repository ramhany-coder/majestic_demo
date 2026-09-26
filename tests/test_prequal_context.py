"""History trimming, last_products formatting and the session store."""

from agents.prequal.session_context import (
    SessionContext, SessionStore, format_history, format_last_products, trim_history,
)


def _history(n):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(n)]


def test_keeps_last_six_messages_oldest_first():
    out = trim_history(_history(10))
    assert [m["content"] for m in out] == ["m4", "m5", "m6", "m7", "m8", "m9"]


def test_user_cut_to_300_and_assistant_to_150():
    out = trim_history([{"role": "user", "content": "u" * 500}, {"role": "assistant", "content": "a" * 500}])
    assert len(out[0]["content"]) == 300 and out[0]["content"].endswith("…")
    assert len(out[1]["content"]) == 150 and out[1]["content"].endswith("…")


def test_short_messages_are_not_cut_and_newlines_collapse():
    out = trim_history([{"role": "user", "content": "line one\n\nline   two"}])
    assert out[0]["content"] == "line one line two"


def test_format_history_lines_and_empty():
    assert format_history([]) == "(none)"
    text = format_history([{"role": "user", "content": "عايزة سيروم للشعر"},
                           {"role": "assistant", "content": "دي المنتجات"}])
    assert text == "U: عايزة سيروم للشعر\nA: دي المنتجات"


def test_format_last_products_numbered_max_five():
    names = [f"P{i}" for i in range(1, 8)]
    assert format_last_products(names) == "1) P1 2) P2 3) P3 4) P4 5) P5"
    assert format_last_products([]) == "none"


def test_add_turn_replaces_last_products_only_when_a_list_was_shown():
    ctx = SessionContext()
    ctx.add_turn("hair serum", "here", ["Capixy Hair Serum 120ml", "Capixy Anti- Dandruff Serum Spray 120ml"])
    ctx.add_turn("thanks", "you're welcome", None)          # no product list this turn
    assert ctx.last_products[1] == "Capixy Anti- Dandruff Serum Spray 120ml"
    ctx.add_turn("sunscreen without zinc", "nothing found", [])
    assert ctx.last_products == []
    assert [m["role"] for m in ctx.history] == ["user", "assistant"] * 3


def test_store_returns_copies():
    store = SessionStore()
    ctx = store.get("s1")
    ctx.add_turn("hi", "hello")
    assert store.get("s1").history == []          # not saved yet
    store.save("s1", ctx)
    ctx.add_turn("again", "hello again")
    assert len(store.get("s1").history) == 2      # later edits don't leak
    assert store.get("other").history == []
