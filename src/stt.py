"""Whisper (Sinhala fine-tune) speech-to-text wrapper.

Exposes ``transcribe(audio, sample_rate) -> str``. Audio is mono float32 at
any sample rate; it is resampled to 16 kHz before Whisper. Language is forced
to Sinhala and task to transcribe. The model is loaded once on first use; if
the primary model in ``config.yaml`` fails to load, the fallbacks are tried in
order. fp16 is used on GPU only.

CLI: ``python -m src.stt path/to/file.wav``
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any

import numpy as np
import torch
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)

from src.audio import to_mono_16k
from src.config import hf_cache_dir, load_config, select_device

logger = logging.getLogger(__name__)

WHISPER_WINDOW_S = 30  # fixed by the Whisper architecture
WHISPER_MAX_NEW_TOKENS = 440  # 448-token decoder limit minus the prompt tokens

_asr: AutomaticSpeechRecognitionPipeline | None = None
_asr_model_id: str | None = None
_lock = threading.Lock()


def _build_pipeline(model_id: str, device: str, dtype: torch.dtype) -> AutomaticSpeechRecognitionPipeline:
    """Load one Whisper checkpoint and wrap it in an ASR pipeline."""
    cache_dir = hf_cache_dir()
    processor = WhisperProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir, dtype=dtype)
    # These checkpoints ship old-style forced_decoder_ids with the language slot
    # left open; clear them so the language/task passed at generate time apply.
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        dtype=dtype,
    )


def get_asr() -> AutomaticSpeechRecognitionPipeline:
    """Return the shared ASR pipeline, loading it (with fallbacks) on first call."""
    global _asr, _asr_model_id
    if _asr is not None:
        return _asr
    with _lock:
        if _asr is not None:
            return _asr
        cfg = load_config()
        device = select_device()
        dtype = torch.float16 if device == "cuda" and cfg["device"]["fp16_on_gpu"] else torch.float32
        candidates = [cfg["stt"]["model_id"], *cfg["stt"].get("fallback_model_ids", [])]
        for model_id in candidates:
            try:
                logger.info("Loading STT model %s on %s (%s)", model_id, device, dtype)
                _asr = _build_pipeline(model_id, device, dtype)
                _asr_model_id = model_id
                return _asr
            except Exception as exc:  # any load failure: try the next model
                logger.error("Failed to load STT model %s: %s: %s", model_id, type(exc).__name__, exc)
        raise RuntimeError(f"Could not load any STT model: {candidates}")


def loaded_model_id() -> str | None:
    """ID of the STT model currently loaded, or None if not loaded yet."""
    return _asr_model_id


def max_new_tokens(duration_s: float) -> int:
    """Token budget for one Whisper window of ``duration_s`` seconds."""
    per_s = load_config()["stt"]["max_new_tokens_per_s"]
    window_s = min(duration_s, WHISPER_WINDOW_S)
    return min(WHISPER_MAX_NEW_TOKENS, 10 + int(per_s * window_s))


def transcribe(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """Transcribe speech to Sinhala text. Returns "" for empty audio."""
    audio = to_mono_16k(audio, sample_rate)
    if audio.size == 0:
        return ""
    cfg = load_config()
    sr = cfg["audio"]["sample_rate"]
    asr = get_asr()
    result: dict[str, Any] = asr(
        {"raw": audio, "sampling_rate": sr},
        generate_kwargs={
            "language": cfg["stt"]["language"],
            "task": cfg["stt"]["task"],
            "max_new_tokens": max_new_tokens(len(audio) / sr),
        },
        # Whisper's own long-form mode handles audio beyond its 30 s window.
        return_timestamps=len(audio) > WHISPER_WINDOW_S * sr,
    )
    # A token cap can cut a multi-byte Sinhala character in half, leaving U+FFFD.
    return result["text"].replace("\ufffd", "").strip()


if __name__ == "__main__":
    import soundfile as sf

    from src.timing import ms_since, now

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.stt path/to/file.wav")

    wav, sr = sf.read(sys.argv[1], dtype="float32")
    t0 = now()
    get_asr()
    load_ms = ms_since(t0)
    t1 = now()
    text = transcribe(wav, sr)
    stt_ms = ms_since(t1)
    print(text)
    print(
        f"[{loaded_model_id()} | audio {len(wav) / sr:.2f} s | load {load_ms:.0f} ms | transcribe {stt_ms:.0f} ms]",
        file=sys.stderr,
    )
