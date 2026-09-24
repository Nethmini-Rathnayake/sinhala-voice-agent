"""Silero VAD wrapper.

Exposes ``trim(audio, sr) -> np.ndarray | None``: converts audio to mono
16 kHz, keeps only the detected speech segments (concatenated) and returns
``None`` if no speech is found. Later used for end-of-turn detection and
barge-in (Phase 3). Thresholds come from ``config.yaml``.

The model is loaded once with ``silero_vad.load_silero_vad()`` (the pip
method in the Silero README). The package must be installed with
``pip install --no-deps silero-vad==6.2.2`` because its metadata caps
torchaudio below the version that matches our torch; see requirements.txt.

CLI: ``python -m src.vad path/to/file.wav``
"""

from __future__ import annotations

import sys
import threading
from typing import Any

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from src.audio import target_sample_rate, to_mono_16k
from src.config import load_config

_model: Any = None
_lock = threading.Lock()


def _load() -> Any:
    """Load the Silero model once (thread-safe); weights ship inside the package."""
    global _model
    with _lock:
        if _model is None:
            _model = load_silero_vad()
    return _model


def load() -> None:
    """Load the VAD model now instead of on first use (warm-up)."""
    _load()


def speech_segments(audio_16k: np.ndarray) -> list[tuple[int, int]]:
    """Return ``(start, end)`` sample indices of speech in 16 kHz mono audio."""
    if audio_16k.size == 0:
        return []
    cfg = load_config()["vad"]
    model = _load()
    with _lock:  # the model is stateful; one call at a time
        stamps = get_speech_timestamps(
            torch.from_numpy(audio_16k),
            model,
            sampling_rate=target_sample_rate(),
            threshold=cfg["threshold"],
            min_speech_duration_ms=cfg["min_speech_duration_ms"],
            min_silence_duration_ms=cfg["min_silence_duration_ms"],
            speech_pad_ms=cfg["speech_pad_ms"],
        )
    return [(s["start"], s["end"]) for s in stamps]


def trim(audio: np.ndarray, sr: int) -> np.ndarray | None:
    """Return only the speech in ``audio`` as mono float32 at 16 kHz, or None if silent."""
    audio_16k = to_mono_16k(audio, sr)
    segments = speech_segments(audio_16k)
    if not segments:
        return None
    return np.concatenate([audio_16k[start:end] for start, end in segments])


if __name__ == "__main__":
    import soundfile as sf

    from src.timing import ms_since, now

    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.vad path/to/file.wav")

    wav, file_sr = sf.read(sys.argv[1], dtype="float32")
    _load()
    t0 = now()
    audio_16k = to_mono_16k(wav, file_sr)
    segments = speech_segments(audio_16k)
    vad_ms = ms_since(t0)

    sr16 = target_sample_rate()
    total_s = len(audio_16k) / sr16
    speech_s = sum(end - start for start, end in segments) / sr16
    share = 100 * speech_s / total_s if total_s else 0.0
    print(f"speech {speech_s:.2f} s / total {total_s:.2f} s ({share:.0f}%), {len(segments)} segment(s)")
    for start, end in segments:
        print(f"  {start / sr16:6.2f} – {end / sr16:6.2f} s")
    print(f"[vad {vad_ms:.0f} ms]", file=sys.stderr)
