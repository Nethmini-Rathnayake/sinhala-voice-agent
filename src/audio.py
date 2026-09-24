"""Audio helpers shared by pipeline stages.

``to_mono_16k(audio, sample_rate)`` converts any mono/stereo float audio to
the pipeline convention: mono float32 at ``audio.sample_rate`` (16 kHz).

CLI: ``python -m src.audio path/to/file.wav`` prints format before/after conversion.
"""

from __future__ import annotations

import sys

import librosa
import numpy as np

from src.config import load_config


def target_sample_rate() -> int:
    """Pipeline-internal sample rate from config (16 kHz)."""
    return load_config()["audio"]["sample_rate"]


def to_mono_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert audio to mono float32 at the pipeline sample rate."""
    target_sr = target_sample_rate()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        # Channels are the shorter axis: (samples, ch) from soundfile, (ch, samples) from librosa.
        audio = audio.mean(axis=int(np.argmin(audio.shape)))
    if sample_rate != target_sr and audio.size:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
    return audio.astype(np.float32, copy=False)


if __name__ == "__main__":
    import soundfile as sf

    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.audio path/to/file.wav")
    wav, sr = sf.read(sys.argv[1], dtype="float32")
    out = to_mono_16k(wav, sr)
    print(f"in:  {sr} Hz, shape {wav.shape}, {len(wav) / sr:.2f} s")
    print(f"out: {target_sample_rate()} Hz, shape {out.shape}, {len(out) / target_sample_rate():.2f} s")
