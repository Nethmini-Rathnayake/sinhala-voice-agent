"""Tests for ``src.chunker``: sentence boundaries, Sinhala ``।``, newline handling and end-of-stream flush."""

from collections.abc import Iterator

import pytest

from src.chunker import sentences


def chars(text: str) -> Iterator[str]:
    """Stream text one character at a time (worst-case token splitting)."""
    yield from text


def test_single_sentence_with_full_stop():
    assert list(sentences(["ආයුබෝවන්."])) == ["ආයුබෝවන්."]


@pytest.mark.parametrize("mark", [".", "?", "!", "।"])
def test_each_terminator_splits(mark):
    text = f"ඔයාට කොහොමද{mark} මට හොඳයි."
    assert list(sentences([text])) == [f"ඔයාට කොහොමද{mark}", "මට හොඳයි."]


def test_newline_splits_without_punctuation():
    assert list(sentences(["පළමු පේළිය\nදෙවන පේළිය"])) == ["පළමු පේළිය", "දෙවන පේළිය"]


def test_multiple_sentences_in_one_token():
    token = "හෙට වැස්ස තියෙනවා. කුඩයක් ගෙනියන්න! ඔබට තවත් උදව් ඕනෙද?"
    assert list(sentences([token])) == [
        "හෙට වැස්ස තියෙනවා.",
        "කුඩයක් ගෙනියන්න!",
        "ඔබට තවත් උදව් ඕනෙද?",
    ]


def test_character_by_character_stream():
    text = "ඇප් එක විවෘත කරන්න. ඊට පස්සේ ඉමේල් එක බලන්න।"
    assert list(sentences(chars(text))) == [
        "ඇප් එක විවෘත කරන්න.",
        "ඊට පස්සේ ඉමේල් එක බලන්න।",
    ]


def test_remainder_flushed_at_end():
    assert list(sentences(["ස්තූතියි. ", "නැවත හමු", "වෙමු"])) == [
        "ස්තූතියි.",
        "නැවත හමුවෙමු",
    ]


def test_punctuation_run_split_across_tokens_stays_together():
    tokens = ["ඇත්තද", "?", "!", " ඔව්", ".", ".", ". හරි."]
    assert list(sentences(tokens)) == ["ඇත්තද?!", "ඔව්...", "හරි."]


def test_terminator_at_token_end_emits_when_next_token_starts():
    tokens = ["මම හොඳින්.", " ඔබට", " කොහොමද?"]
    stream = sentences(tokens)
    assert next(stream) == "මම හොඳින්."
    assert next(stream) == "ඔබට කොහොමද?"
    with pytest.raises(StopIteration):
        next(stream)


def test_sentence_is_yielded_before_stream_ends():
    consumed = []

    def tokens():
        for t in ["පළමු වාක්‍යය.", " දෙවන", " වාක්‍යය."]:
            consumed.append(t)
            yield t

    stream = sentences(tokens())
    assert next(stream) == "පළමු වාක්‍යය."
    assert consumed == ["පළමු වාක්‍යය.", " දෙවන"]


def test_decimal_number_not_split():
    assert list(sentences(chars("මිල රුපියල් 3.5 යි. හරි."))) == ["මිල රුපියල් 3.5 යි.", "හරි."]


def test_closing_quote_stays_with_sentence():
    assert list(sentences(['ඔහු කිව්වා "හරි." ඊට පස්සේ ගියා.'])) == [
        'ඔහු කිව්වා "හරි."',
        "ඊට පස්සේ ගියා.",
    ]


def test_zero_width_joiner_preserved():
    # ශ්‍රී uses ZWJ (U+200D); it must survive chunking unchanged for TTS.
    text = "ශ්‍රී ලංකාව ලස්සනයි."
    assert list(sentences(chars(text))) == [text]


def test_punctuation_only_and_blank_fragments_skipped():
    assert list(sentences(["\n\n", "... ", "හරි.", "\n", "  "])) == ["හරි."]


def test_empty_stream():
    assert list(sentences([])) == []


def test_custom_terminators():
    assert list(sentences(["එක; දෙක; තුන"], terminators=[";"])) == ["එක;", "දෙක;", "තුන"]
