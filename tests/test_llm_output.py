"""Tests for the Sinhala-only output rule: replies contain no Latin characters, and the guard logs a warning when they do."""

import logging
import re

import pytest

from src.llm import api_key_available, find_latin, reply_text, warn_if_latin

SINHALA_RE = re.compile(r"[඀-෿]")

PROMPTS = [
    "ඔයාට කොහොමද?",                                   # Sinhala
    "හෙට කොළඹ කාලගුණය කොහොමද?",                        # Sinhala
    "Can you help me send an email to my boss?",        # English
    "මට phone එකට WhatsApp app එක download කරන්න ඕනේ.",  # mixed
    "Google Maps වලින් Kandy යන්නේ කොහොමද?",            # mixed
]


# --- Guard (offline) -------------------------------------------------------

def test_find_latin_detects_english_words():
    assert find_latin("මට app එක ඕනේ") == ["app"]


def test_find_latin_ignores_sinhala_digits_and_punctuation():
    assert find_latin("ශ්‍රී ලංකාව 2026 දී ලස්සනයි! ඇප් එක, ඉමේල්.") == []


def test_find_latin_detects_accented_latin():
    assert find_latin("කැෆේ café") == ["café"]


def test_warn_if_latin_logs_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="src.llm"):
        assert warn_if_latin("ඔබේ email එක බලන්න") is True
    assert "email" in caplog.text


def test_warn_if_latin_silent_for_sinhala(caplog):
    with caplog.at_level(logging.WARNING, logger="src.llm"):
        assert warn_if_latin("ඔබේ ඉමේල් එක බලන්න") is False
    assert caplog.text == ""


# --- Live LLM (needs an API key) -------------------------------------------

@pytest.mark.skipif(not api_key_available(), reason="no API key for the selected LLM provider")
@pytest.mark.parametrize("prompt", PROMPTS)
def test_reply_is_sinhala_only(prompt):
    reply = reply_text([], prompt)
    assert SINHALA_RE.search(reply), f"no Sinhala script in reply: {reply!r}"
    assert find_latin(reply) == [], f"Latin letters in reply: {reply!r}"
