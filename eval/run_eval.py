"""Batch evaluation over ``eval/utterances/``.

Runs the pipeline on every recorded test WAV (Sinhala, English, mixed) and
writes a CSV of transcripts, replies and per-stage latencies. A WAV may have a
sidecar ``.txt`` holding the reference transcript; where one exists, word error
rate is computed with jiwer.

Each utterance runs with empty history, so results do not depend on order.

CLI: ``python -m eval.run_eval [--utterances DIR] [--out CSV]``
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import soundfile as sf

from src.config import REPO_ROOT
from src.pipeline import reply_sample_rate, run_turn

logger = logging.getLogger(__name__)

TIMING_FIELDS = [
    "vad_ms",
    "stt_ms",
    "llm_first_token_ms",
    "llm_total_ms",
    "tts_first_sentence_ms",
    "total_to_first_audio_ms",
]
FIELDNAMES = ["filename", "reference", "transcript", "wer", "cer", "reply", "reply_audio_s", "skipped", *TIMING_FIELDS]

DEFAULT_UTTERANCES = REPO_ROOT / "eval" / "utterances"
DEFAULT_OUT = REPO_ROOT / "eval" / "results.csv"

_PUNCTUATION = re.compile(r"[.,!?;:।\"'()\[\]]")


def normalize(text: str) -> str:
    """Lowercase, drop punctuation and collapse whitespace, for fair WER."""
    return " ".join(_PUNCTUATION.sub(" ", text.lower()).split())


def error_rates(reference: str, hypothesis: str) -> tuple[float, float] | None:
    """(WER, CER) between reference and hypothesis, or None if jiwer is missing.

    CER matters more than WER for Sinhala: writing splits compound words
    inconsistently (``ඔය වගේ`` vs ``ඔයවගේ``), so a single spacing difference
    can score two whole words wrong while the characters are nearly right.
    """
    try:
        import jiwer
    except ImportError:
        logger.warning("jiwer is not installed; skipping WER/CER (pip install jiwer)")
        return None
    ref, hyp = normalize(reference), normalize(hypothesis)
    return jiwer.wer(ref, hyp), jiwer.cer(ref, hyp)


def reference_for(wav_path: Path) -> str:
    """Read the sidecar ``.txt`` transcript, or "" if there is none."""
    sidecar = wav_path.with_suffix(".txt")
    return sidecar.read_text(encoding="utf-8").strip() if sidecar.exists() else ""


def evaluate(wav_path: Path) -> dict[str, object]:
    """Run one utterance through the pipeline and collect a CSV row."""
    audio, sr = sf.read(wav_path, dtype="float32")
    reply_audio, transcript, reply_text, timings = run_turn(audio, sr, [])

    reference = reference_for(wav_path)
    rates = error_rates(reference, transcript) if reference else None

    row: dict[str, object] = {
        "filename": wav_path.name,
        "reference": reference,
        "transcript": transcript,
        "wer": "" if rates is None else round(rates[0], 3),
        "cer": "" if rates is None else round(rates[1], 3),
        "reply": reply_text,
        "reply_audio_s": round(len(reply_audio) / reply_sample_rate(), 2) if len(reply_audio) else 0.0,
        "skipped": "" if transcript and len(reply_audio) else "yes",
    }
    row.update({field: round(timings[field], 1) if field in timings else "" for field in TIMING_FIELDS})
    return row


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile."""
    return float(np.percentile(values, pct))


def summarize(rows: list[dict[str, object]]) -> None:
    """Print mean latency per stage, WER, and P50/P95 time to first audio."""
    print(f"\n{len(rows)} utterance(s)")

    wers = [float(r["wer"]) for r in rows if r["wer"] != ""]
    cers = [float(r["cer"]) for r in rows if r["cer"] != ""]
    if wers:
        print(f"  over {len(wers)} utterance(s) with references:")
        print(f"    WER (mean) {mean(wers):.3f}")
        print(f"    CER (mean) {mean(cers):.3f}   <- the meaningful one for Sinhala")
    else:
        print("  WER/CER: no reference transcripts found (add a .txt beside each .wav)")

    print("  mean latency:")
    for field in TIMING_FIELDS:
        values = [float(r[field]) for r in rows if r[field] != ""]
        if values:
            print(f"    {field:<26} {mean(values):8.0f} ms  (n={len(values)})")

    first_audio = [float(r["total_to_first_audio_ms"]) for r in rows if r["total_to_first_audio_ms"] != ""]
    if first_audio:
        print("  total_to_first_audio_ms:")
        print(f"    P50 {percentile(first_audio, 50):8.0f} ms")
        print(f"    P95 {percentile(first_audio, 95):8.0f} ms")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch-run the pipeline over recorded utterances.")
    parser.add_argument("--utterances", type=Path, default=DEFAULT_UTTERANCES, help="directory of .wav files")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="CSV to write")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "TTS"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    wavs = sorted(args.utterances.glob("*.wav"))
    if not wavs:
        print(f"No .wav files in {args.utterances}", file=sys.stderr)
        return 1

    rows: list[dict[str, object]] = []
    for index, wav_path in enumerate(wavs, 1):
        print(f"[{index}/{len(wavs)}] {wav_path.name}", file=sys.stderr)
        try:
            rows.append(evaluate(wav_path))
        except Exception as exc:  # one bad file should not end the run
            logger.error("Failed on %s: %s: %s", wav_path.name, type(exc).__name__, exc)

    if not rows:
        print("Every utterance failed; no results written", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    summarize(rows)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
