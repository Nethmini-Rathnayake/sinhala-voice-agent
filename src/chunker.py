"""Sentence chunker for streamed LLM output.

Exposes ``sentences(token_iter) -> Iterator[str]``. Emits a sentence as soon
as it ends in ``.``, ``?``, ``!``, ``।`` or a newline, and flushes any
remainder when the token stream ends. Pure Python, no model dependencies.

Boundary rule: a run of punctuation terminators (optionally followed by
closing quotes/brackets) ends a sentence once whitespace follows it; a newline
ends a sentence immediately. Waiting for the whitespace keeps runs such as
``...`` or ``?!`` intact when they are split across tokens, and avoids
splitting numbers like ``3.5``. Fragments with no letters or digits (e.g. a
stray ``...``) are skipped, since there is nothing for TTS to speak.

CLI: ``python -m src.chunker "text to split"``
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Iterator

DEFAULT_TERMINATORS: tuple[str, ...] = (".", "?", "!", "।", "\n")

_CLOSERS = "\"'”’)]"


def _boundary_pattern(terminators: Iterable[str]) -> re.Pattern[str]:
    """Build the regex that matches the end of a sentence."""
    terms = list(terminators)
    punct = "".join(re.escape(t) for t in terms if t != "\n")
    alternatives = []
    if punct:
        alternatives.append(f"[{punct}]+[{re.escape(_CLOSERS)}]*(?=\\s)")
    if "\n" in terms:
        alternatives.append("\n")
    if not alternatives:
        raise ValueError("at least one terminator is required")
    return re.compile("|".join(alternatives))


def _speakable(text: str) -> bool:
    """True if ``text`` contains at least one letter or digit."""
    return any(ch.isalnum() for ch in text)


def sentences(
    token_iter: Iterable[str],
    terminators: Iterable[str] = DEFAULT_TERMINATORS,
) -> Iterator[str]:
    """Yield complete, stripped sentences from a stream of text tokens."""
    boundary = _boundary_pattern(terminators)
    buffer = ""
    for token in token_iter:
        buffer += token
        while (match := boundary.search(buffer)) is not None:
            sentence = buffer[: match.end()].strip()
            buffer = buffer[match.end():]
            if _speakable(sentence):
                yield sentence
    remainder = buffer.strip()
    if _speakable(remainder):
        yield remainder


def _fake_stream(text: str, size: int = 3) -> Iterator[str]:
    """Split text into fixed-size tokens to imitate LLM streaming."""
    for i in range(0, len(text), size):
        yield text[i : i + size]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('usage: python -m src.chunker "text to split"')
    for i, s in enumerate(sentences(_fake_stream(sys.argv[1])), 1):
        print(f"{i}: {s}")
