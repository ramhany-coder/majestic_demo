"""Greetings / thanks / goodbyes end the graph in prequal."""

import asyncio

import pytest

from agents.prequal.small_talk import (KIND_ACK, KIND_BYE, KIND_GREETING, KIND_THANKS, is_small_talk,
                                       small_talk_kind, small_talk_language, small_talk_reply)


@pytest.mark.parametrize("text", [
    "hi", "Hello!", "hey there", "good morning", "thanks a lot!", "thank you doctor", "bye", "ok",
    "السلام عليكم", "وعليكم السلام", "أهلاً", "صباح الخير", "شكراً يا دكتور 🙏", "مع السلامة",
    "ahlan, ezayak?", "shokran", "salamo 3aleko", "🙏👍", "!!",
])
def test_small_talk(text):
    assert is_small_talk(text)


@pytest.mark.parametrize("text", [
    "hi, do you have sunscreen?", "thanks, any cheaper one?", "عايزة سيروم للشعر",
    "السلام عليكم عندكم كريم لليدين؟", "ahlan, 3ayez sun block", "", "   ", "3",
])
def test_not_small_talk(text):
    assert not is_small_talk(text)


@pytest.mark.parametrize("text,kind", [
    ("hi", KIND_GREETING), ("السلام عليكم", KIND_GREETING), ("thanks a lot", KIND_THANKS),
    ("شكرا", KIND_THANKS), ("thanks, bye", KIND_BYE), ("مع السلامة", KIND_BYE), ("ok", KIND_ACK),
    ("🙏", KIND_ACK), ("hey, how is your day going my friend", KIND_GREETING),
])
def test_kind(text, kind):
    assert small_talk_kind(text) == kind


@pytest.mark.parametrize("text,lang", [("hi", "en"), ("السلام عليكم", "ar"), ("ahlan", "arabizi"),
                                       ("shokran ya doctor", "arabizi")])
def test_language(text, lang):
    assert small_talk_language(text) == lang


def test_reply_follows_language_and_kind():
    assert small_talk_reply("hi").startswith("Hello")
    assert small_talk_reply("شكرا").startswith("العفو")
    assert small_talk_reply("thanks", "mixed").startswith("العفو")
    assert small_talk_reply("ahlan").startswith("Ahlan")
