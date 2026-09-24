"""Streaming LLM client (OpenAI-compatible API).

Exposes ``stream_reply(history, user_text) -> Iterator[str]`` yielding tokens.
Provider (Gemini by default, or Groq) comes from ``config.yaml`` / the
``LLM_PROVIDER`` env var; API keys come only from environment variables.
The system prompt forces short replies (1-3 sentences) entirely in Sinhala
script, and a guard logs a warning if Latin characters appear in the output.

CLI: ``python -m src.llm "user text"``
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Iterator, Sequence
from functools import lru_cache
from typing import Any

from openai import OpenAI

from src.config import load_config, load_env

logger = logging.getLogger(__name__)

Message = dict[str, str]  # {"role": "user" | "assistant", "content": str}

SYSTEM_PROMPT = """\
You are a friendly, helpful voice assistant for Sinhala speakers in Sri Lanka.
Every reply is read aloud by a Sinhala-only text-to-speech engine that cannot \
pronounce Latin letters, so these rules are strict:

1. Reply ONLY in Sinhala script (සිංහල අකුරු). Never use English/Latin letters, \
not even for names, brands, abbreviations or technical terms.
2. Write English words phonetically in Sinhala script, the way Sri Lankans say them. \
Examples: app → ඇප් එක, email → ඉමේල්, WiFi → වයිෆයි, Google → ගූගල්, \
phone → ෆෝන් එක, OK → ඕකේ, Colombo → කොළඹ.
3. Write numbers as Sinhala words (e.g. 25 → විසිපහ).
4. Keep every reply to 1–3 short, natural spoken sentences.
5. Plain sentences only: no markdown, lists, emojis, URLs or special symbols.
6. The user may speak Sinhala, English, or a mix. Always reply in Sinhala.
"""

# Basic Latin letters plus Latin-1 Supplement / Latin Extended-A/B letters.
_LATIN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ɏ]+")


def find_latin(text: str) -> list[str]:
    """Return every run of Latin letters in ``text`` (empty if none)."""
    return _LATIN_RE.findall(text)


def warn_if_latin(text: str) -> bool:
    """Log a warning if ``text`` contains Latin letters. Returns True if it did.

    The text is never modified: dropping words silently would hide the problem.
    """
    latin = find_latin(text)
    if latin:
        logger.warning("LLM reply contains Latin characters %s: %r", latin, text)
    return bool(latin)


def _provider_settings() -> tuple[str, dict[str, Any]]:
    """Return (provider name, provider config), honouring LLM_PROVIDER."""
    llm_cfg = load_config()["llm"]
    name = os.environ.get("LLM_PROVIDER") or llm_cfg["provider"]
    try:
        return name, llm_cfg["providers"][name]
    except KeyError:
        known = ", ".join(llm_cfg["providers"])
        raise ValueError(f"Unknown LLM provider {name!r}; expected one of: {known}") from None


def api_key_available() -> bool:
    """True if the API key for the selected provider is set."""
    load_env()
    _, settings = _provider_settings()
    return bool(os.environ.get(settings["api_key_env"]))


@lru_cache(maxsize=None)
def _client(base_url: str, api_key_env: str, timeout_s: float) -> OpenAI:
    """Build (and cache) an OpenAI-compatible client for one provider."""
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set (add it to .env or Colab secrets)")
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=1)


def build_messages(history: Sequence[Message], user_text: str) -> list[Message]:
    """System prompt + the last ``history_turns`` turns + the new user message."""
    max_messages = 2 * load_config()["llm"]["history_turns"]
    recent = list(history)[-max_messages:] if max_messages > 0 else []
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *({"role": m["role"], "content": m["content"]} for m in recent),
        {"role": "user", "content": user_text},
    ]


def stream_reply(history: Sequence[Message], user_text: str) -> Iterator[str]:
    """Stream the assistant's reply to ``user_text`` token by token.

    ``history`` is a list of ``{"role", "content"}`` dicts from earlier turns.
    After the stream ends, the full reply is checked for Latin characters.
    """
    load_env()
    llm_cfg = load_config()["llm"]
    _, settings = _provider_settings()
    client = _client(settings["base_url"], settings["api_key_env"], llm_cfg["timeout_s"])

    kwargs: dict[str, Any] = {}
    if settings.get("reasoning_effort"):
        kwargs["reasoning_effort"] = settings["reasoning_effort"]

    stream = client.chat.completions.create(
        model=settings["model"],
        messages=build_messages(history, user_text),
        max_tokens=llm_cfg["max_tokens"],
        temperature=llm_cfg["temperature"],
        stream=True,
        **kwargs,
    )

    parts: list[str] = []
    for chunk in stream:
        if not chunk.choices:
            continue
        token = chunk.choices[0].delta.content
        if token:
            parts.append(token)
            yield token
    warn_if_latin("".join(parts))


def reply_text(history: Sequence[Message], user_text: str) -> str:
    """Convenience wrapper: return the whole streamed reply as one string."""
    return "".join(stream_reply(history, user_text))


if __name__ == "__main__":
    from src.timing import ms_since, now

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) < 2:
        sys.exit('usage: python -m src.llm "ඔයාට කොහොමද?"')

    provider, settings = _provider_settings()
    print(f"[{provider} / {settings['model']}]", file=sys.stderr)
    start = now()
    first_token_ms: float | None = None
    for tok in stream_reply([], sys.argv[1]):
        if first_token_ms is None:
            first_token_ms = ms_since(start)
        print(tok, end="", flush=True)
    print()
    print(f"[first token {first_token_ms or 0:.0f} ms, total {ms_since(start):.0f} ms]", file=sys.stderr)
