"""Gradio UI for the voice agent.

Microphone input via ``gr.Audio`` and audio output of the spoken reply, plus
the transcript, the reply text and this turn's per-stage latencies.
Conversation history lives in ``gr.State``, so each browser session has its
own; "Reset conversation" clears it.

Launched locally, or from Colab with ``COLAB=1`` set for a public share link.

CLI: ``python -m src.app``
"""

from __future__ import annotations

import logging
import os

import gradio as gr
import numpy as np

from src.pipeline import Message, reply_sample_rate, run_turn

logger = logging.getLogger(__name__)

TIMING_HEADERS = ["stage", "ms"]
INTRO = "සිංහලෙන් කතා කරන්න — Speak Sinhala, and the agent replies in Sinhala."


def _to_float32(audio: np.ndarray) -> np.ndarray:
    """Gradio hands back int16 (or float) samples; the pipeline wants float32 in [-1, 1]."""
    audio = np.asarray(audio)
    if np.issubdtype(audio.dtype, np.integer):
        max_value = float(np.iinfo(audio.dtype).max)
        return audio.astype(np.float32) / max_value
    return audio.astype(np.float32, copy=False)


def _timing_rows(timings: dict[str, float]) -> list[list[str]]:
    """Per-stage latencies as table rows."""
    return [[key, f"{value:.0f}"] for key, value in timings.items()]


def respond(
    recording: tuple[int, np.ndarray] | None,
    history: list[Message],
) -> tuple[tuple[int, np.ndarray] | None, str, str, list[list[str]], list[Message]]:
    """Run one turn for the recorded audio and update the UI."""
    if recording is None:
        return None, "", "Record something first.", [], history

    sr, samples = recording
    reply_audio, transcript, reply_text, timings = run_turn(_to_float32(samples), sr, history)
    audio_out = (reply_sample_rate(), reply_audio) if len(reply_audio) else None
    if not transcript:
        reply_text = reply_text or "No speech detected — try again."
    return audio_out, transcript, reply_text, _timing_rows(timings), history


def reset() -> tuple[None, None, str, str, list[list[str]], list[Message]]:
    """Clear the conversation and every output."""
    return None, None, "", "", [], []


def build_ui() -> gr.Blocks:
    """Assemble the Blocks app."""
    with gr.Blocks(title="Sinhala Voice Agent") as demo:
        gr.Markdown(f"# Sinhala Voice Agent\n{INTRO}")
        history = gr.State([])

        with gr.Row():
            with gr.Column():
                mic = gr.Audio(sources=["microphone"], type="numpy", label="Your turn")
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("Reset conversation")
            with gr.Column():
                reply_audio = gr.Audio(label="Reply", autoplay=True, interactive=False)
                transcript_box = gr.Textbox(label="Transcript (what you said)", interactive=False)
                reply_box = gr.Textbox(label="Reply (Sinhala)", interactive=False)
                timings_table = gr.Dataframe(
                    headers=TIMING_HEADERS,
                    col_count=(2, "fixed"),
                    label="Latency, this turn",
                    interactive=False,
                )

        outputs = [reply_audio, transcript_box, reply_box, timings_table, history]
        send.click(respond, inputs=[mic, history], outputs=outputs)
        clear.click(reset, outputs=[mic, *outputs])
    return demo


def main() -> None:
    """Launch the app; COLAB=1 asks Gradio for a public share link."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "TTS"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    share = os.environ.get("COLAB") == "1"
    build_ui().launch(share=share, server_name="0.0.0.0" if share else "127.0.0.1")


if __name__ == "__main__":
    main()
