"""SinhalaVITS text-to-speech wrapper (dialoglk/SinhalaVITS-TTS-F1).

Exposes ``synthesize(sentence) -> (sample_rate, np.ndarray)`` returning mono
float32 audio at the model's rate (22050 Hz).

The model is a Coqui TTS VITS checkpoint trained on *romanized* Sinhala, so
text goes through the repo's own ``romanizer.py`` before synthesis. Files are
downloaded from Hugging Face at a pinned revision (config.yaml) and the
``Synthesizer`` is built once on first use, on CUDA if available. It runs in
fp32: Coqui's Synthesizer has no fp16 path.

Characters outside the model's vocabulary (digits, Latin capitals, ``।``) are
dropped by Coqui; a warning names them so the loss is never silent.

CLI: ``python -m src.tts "ආයුබෝවන්, මම ඔබට උදව් කරන්නම්."`` (writes out.wav)
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
from huggingface_hub import hf_hub_download
from TTS.utils.synthesizer import PAD_SILENCE_SAMPLES, Synthesizer

from src.config import hf_cache_dir, load_config, select_device

logger = logging.getLogger(__name__)

_synth: Synthesizer | None = None
_romanize: Callable[[str], str] | None = None
_vocab: frozenset[str] = frozenset()
_lock = threading.Lock()


def _download(filename: str) -> Path:
    """Fetch one file from the TTS model repo (pinned revision) into the HF cache."""
    cfg = load_config()["tts"]
    return Path(
        hf_hub_download(cfg["model_id"], filename, revision=cfg["revision"], cache_dir=hf_cache_dir())
    )


def _import_file(path: Path, name: str) -> ModuleType:
    """Import a Python file by path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load() -> tuple[Synthesizer, Callable[[str], str]]:
    """Download files and build the Synthesizer and romanizer once (thread-safe)."""
    global _synth, _romanize, _vocab
    with _lock:
        if _synth is None:
            cfg = load_config()["tts"]
            device = select_device()
            logger.info("Loading TTS model %s on %s", cfg["model_id"], device)
            romanizer = _import_file(_download(cfg["romanizer_file"]), "sinhala_vits_romanizer")
            synth = Synthesizer(
                tts_checkpoint=str(_download(cfg["checkpoint_file"])),
                tts_config_path=str(_download(cfg["config_file"])),
                use_cuda=device == "cuda",
            )
            _vocab = frozenset(synth.tts_model.tokenizer.characters.vocab)
            _romanize = romanizer.sinhala_to_roman
            _synth = synth
    return _synth, _romanize


def load() -> None:
    """Download files and build the Synthesizer now instead of on first use (warm-up)."""
    _load()


def sample_rate() -> int:
    """Output sample rate of the loaded model (22050 Hz for SinhalaVITS)."""
    synth, _ = _load()
    return int(synth.output_sample_rate)


def romanize(sentence: str) -> str:
    """Sinhala script -> the romanized form the model was trained on."""
    _, to_roman = _load()
    return to_roman(sentence)


def unsupported_chars(text: str) -> list[str]:
    """Characters in (romanized) ``text`` the model cannot pronounce, in order of appearance."""
    _load()
    return list(dict.fromkeys(ch for ch in text if ch not in _vocab))


def synthesize(sentence: str) -> tuple[int, np.ndarray]:
    """Speak one Sinhala sentence. Returns (sample_rate, mono float32 audio).

    Returns an empty array if nothing in the sentence is pronounceable.
    """
    synth, to_roman = _load()
    sr = int(synth.output_sample_rate)
    roman = to_roman(sentence.strip())

    dropped = unsupported_chars(roman)
    if dropped:
        logger.warning("TTS will drop unsupported characters %s from %r", dropped, sentence)
    if not any(ch in _vocab and ch.isalpha() for ch in roman):
        logger.warning("Nothing pronounceable in %r; returning silence", sentence)
        return sr, np.zeros(0, dtype=np.float32)

    # split_sentences=False: the chunker already splits, and Coqui's splitter is English-only.
    wav = np.asarray(synth.tts(roman, split_sentences=False), dtype=np.float32)
    # Coqui appends a fixed ~0.45 s of silence per call; replace it with our own gap.
    wav = wav[:-PAD_SILENCE_SAMPLES] if len(wav) > PAD_SILENCE_SAMPLES else wav
    gap = np.zeros(int(sr * load_config()["tts"]["trailing_silence_ms"] / 1000), dtype=np.float32)
    return sr, np.concatenate([wav, gap])


if __name__ == "__main__":
    import soundfile as sf

    from src.timing import ms_since, now

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("TTS").setLevel(logging.WARNING)  # Coqui logs its full audio config at INFO
    if len(sys.argv) < 2:
        sys.exit('usage: python -m src.tts "ආයුබෝවන්, මම ඔබට උදව් කරන්නම්."')

    text = sys.argv[1]
    t0 = now()
    _load()
    load_ms = ms_since(t0)
    print(f"romanized: {romanize(text)}", file=sys.stderr)
    t1 = now()
    sr, audio = synthesize(text)
    synth_ms = ms_since(t1)
    sf.write("out.wav", audio, sr)
    duration_s = len(audio) / sr
    rtf = synth_ms / 1000 / duration_s if duration_s else float("nan")
    print(
        f"[saved out.wav | {duration_s:.2f} s audio @ {sr} Hz | load {load_ms:.0f} ms"
        f" | synthesis {synth_ms:.0f} ms | RTF {rtf:.2f}]"
    )
