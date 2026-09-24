"""Orchestrates one conversational turn.

Runs VAD -> STT -> LLM (streaming) -> chunker -> TTS and records per-stage
timings: ``vad_ms, stt_ms, llm_first_token_ms, llm_total_ms,
tts_first_sentence_ms, total_to_first_audio_ms``. Every turn is appended to
the latency log from ``config.yaml``.

This is Phase 1: the whole LLM reply is collected before synthesis, so the
user hears the reply only once it is complete. The timings still show where
the time goes, which is what Phase 2 (streaming per sentence) will improve.

CLI: ``python -m src.pipeline path/to/file.wav`` (writes reply.wav)
"""

from __future__ import annotations

import logging
import sys

import numpy as np

from src import llm, stt, tts, vad
from src.audio import target_sample_rate
from src.chunker import sentences
from src.config import load_config
from src.timing import log_turn, ms_since, now, timed

logger = logging.getLogger(__name__)

Message = dict[str, str]
Turn = tuple[np.ndarray, str, str, dict[str, float]]


def reply_sample_rate() -> int:
    """Sample rate of the audio returned by ``run_turn`` (the TTS model's rate)."""
    return tts.sample_rate()


def _finish(
    timings: dict[str, float],
    transcript: str,
    reply_text: str,
    audio: np.ndarray,
    **extra: object,
) -> Turn:
    """Append the turn to the latency log and return the four-tuple."""
    log_turn(
        timings,
        path=load_config()["logging"]["latency_log_path"],
        transcript=transcript,
        reply=reply_text,
        reply_audio_s=round(len(audio) / tts.sample_rate(), 2) if len(audio) else 0.0,
        **extra,
    )
    return audio, transcript, reply_text, timings


def run_turn(audio: np.ndarray, sr: int, history: list[Message] | None = None) -> Turn:
    """Run one turn: speech in, spoken Sinhala reply out.

    ``history`` is appended to in place with this turn's user and assistant
    messages, so the same list can be passed to the next call.

    Returns ``(reply_audio, transcript, reply_text, timings)``. ``reply_audio``
    is mono float32 at ``reply_sample_rate()``, and is empty if the user was
    silent, unintelligible, or the reply could not be spoken.
    """
    history = [] if history is None else history
    timings: dict[str, float] = {}
    silence = np.zeros(0, dtype=np.float32)
    turn_start = now()

    with timed(timings, "vad_ms"):
        speech = vad.trim(audio, sr)
    if speech is None:
        logger.info("No speech detected; nothing to do")
        return _finish(timings, "", "", silence, skipped="no_speech")

    with timed(timings, "stt_ms"):
        transcript = stt.transcribe(speech, target_sample_rate())
    if not transcript:
        logger.info("Empty transcript; nothing to do")
        return _finish(timings, "", "", silence, skipped="empty_transcript")

    tokens: list[str] = []
    first_token_ms: float | None = None
    llm_start = now()
    for token in llm.stream_reply(history, transcript):
        if first_token_ms is None:
            first_token_ms = ms_since(llm_start)
        tokens.append(token)
    timings["llm_first_token_ms"] = first_token_ms if first_token_ms is not None else ms_since(llm_start)
    timings["llm_total_ms"] = ms_since(llm_start)
    reply_text = "".join(tokens)

    history.append({"role": "user", "content": transcript})
    history.append({"role": "assistant", "content": reply_text})

    if not reply_text.strip():
        logger.warning("Empty LLM reply")
        return _finish(timings, transcript, reply_text, silence, skipped="empty_reply")

    tts_start = now()
    chunks: list[np.ndarray] = []
    for sentence in sentences(tokens):
        _, wav = tts.synthesize(sentence)
        if not len(wav):
            continue
        if not chunks:  # first audible sentence
            timings["tts_first_sentence_ms"] = ms_since(tts_start)
            timings["total_to_first_audio_ms"] = ms_since(turn_start)
        chunks.append(wav)

    if not chunks:
        logger.warning("Nothing in the reply could be synthesized: %r", reply_text)
        return _finish(timings, transcript, reply_text, silence, skipped="no_audio")

    reply_audio = np.concatenate(chunks)
    return _finish(timings, transcript, reply_text, reply_audio, sentences=len(chunks))


if __name__ == "__main__":
    import soundfile as sf

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "TTS"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.pipeline path/to/file.wav")

    wav_in, file_sr = sf.read(sys.argv[1], dtype="float32")
    reply_audio, transcript, reply_text, timings = run_turn(wav_in, file_sr)

    print(f"user:  {transcript}")
    print(f"agent: {reply_text}")
    if len(reply_audio):
        sf.write("reply.wav", reply_audio, reply_sample_rate())
        print(f"saved reply.wav ({len(reply_audio) / reply_sample_rate():.2f} s)")
    for key, value in timings.items():
        print(f"  {key:<26} {value:8.0f} ms")
