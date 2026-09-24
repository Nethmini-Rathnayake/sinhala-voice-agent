# CLAUDE.md — Sinhala Voice Agent (Prototype)

## Goal
Build a low-cost, low-latency voice agent that understands spoken Sinhala (with English words mixed in) and replies in spoken Sinhala. This is a **prototype**: correctness and measurable latency matter more than polish.

## Pipeline
```
Mic → VAD → Whisper (Sinhala fine-tune) → LLM (streaming) → Sentence chunker → SinhalaVITS → Speaker
```

| Stage | Tech | Notes |
|---|---|---|
| Mic | Gradio `Audio` component (microphone source) | Browser handles capture; works through Colab share link |
| VAD | Silero VAD | Trims silence; later used for end-of-turn detection and barge-in |
| STT | Hugging Face Whisper fine-tune for Sinhala | Primary: `Subhaka/whisper-small-Sinhala-Fine_Tune`. Fallbacks: `Ransaka/whisper-tiny-sinhala-20k`, `AqeelShafy7/Whisper-Sinhala_Audio_to_Text`. Force language = Sinhala, task = transcribe |
| LLM | OpenAI-compatible client, provider chosen by config | Default: Gemini Flash-Lite via Google's OpenAI-compatible endpoint (free tier). Alternative: Groq (free tier). Must stream |
| Chunker | Pure Python | Emit a sentence as soon as it ends in `.`, `?`, `!`, `।` or newline; flush remainder at stream end |
| TTS | SinhalaVITS (`dialoglk/SinhalaVITS-TTS-F1` on Hugging Face) | **Read the model card and repo files before writing inference code — do not guess the API.** Non-commercial licence: prototype use only |
| Speaker | Gradio audio output | Phase 1: play concatenated reply. Phase 2: stream sentence-by-sentence |

## Runtime environments
- **Development:** local machine with Claude Code. Code must run on CPU (slow is OK).
- **Testing:** Google Colab free tier (T4 GPU). The repo is cloned into Colab and launched from `notebooks/colab_demo.ipynb` with `share=True`.
- Detect device automatically (`cuda` if available, else `cpu`). Use fp16 on GPU only.
- Colab free tier disconnects and resets: never assume persistent disk; cache model downloads in `/content/hf_cache`.

## Language rules (important)
- The LLM system prompt must force replies **entirely in Sinhala script**. English loanwords are written phonetically in Sinhala script (e.g. "ඇප් එක", "ඉමේල්"). No Latin characters in output, because SinhalaVITS reads only Sinhala.
- Add a post-processing guard: if Latin characters still appear, log a warning (do not silently drop text).
- Keep replies short (1–3 sentences) — this is a voice agent, not a chat bot.

## Project structure
```
voice-agent/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .env.example            # GEMINI_API_KEY=, GROQ_API_KEY=, LLM_PROVIDER=gemini
├── config.yaml             # model IDs, VAD thresholds, LLM provider/model, max tokens
├── src/
│   ├── vad.py              # Silero VAD wrapper: trim(audio) -> speech-only audio
│   ├── stt.py              # Whisper wrapper: transcribe(audio_16k) -> str
│   ├── llm.py              # stream_reply(history, user_text) -> iterator[str tokens]
│   ├── chunker.py          # sentences(token_iter) -> iterator[str sentences]
│   ├── tts.py              # SinhalaVITS wrapper: synthesize(sentence) -> (sr, np.ndarray)
│   ├── pipeline.py         # orchestrates one turn, records per-stage timings
│   ├── timing.py           # simple timer/context manager + JSON latency log
│   └── app.py              # Gradio UI
├── notebooks/
│   └── colab_demo.ipynb    # clone repo, install, set secrets, launch app
├── eval/
│   ├── utterances/         # recorded test WAVs (Sinhala, English, mixed)
│   └── run_eval.py         # batch-run pipeline, output CSV of transcripts + latencies
└── tests/
    ├── test_chunker.py
    └── test_llm_output.py  # checks Sinhala-only output rule
```

## Coding conventions
- Python 3.10+, type hints, small single-purpose functions.
- Each module independently testable from the command line (`python -m src.stt path/to/file.wav`).
- All model IDs, thresholds and provider names live in `config.yaml`, never hard-coded.
- API keys only from environment variables (locally `.env`, in Colab `google.colab.userdata`). Never print or commit keys. `.env` is in `.gitignore`.
- Audio convention inside the pipeline: mono float32 numpy arrays; resample to 16 kHz before VAD/STT.
- Log per-stage latency for every turn: `vad_ms, stt_ms, llm_first_token_ms, llm_total_ms, tts_first_sentence_ms, total_to_first_audio_ms`.

## Build phases
1. **Phase 1 — turn-based:** user records → stop → pipeline runs → full reply plays. Goal: everything works end-to-end.
2. **Phase 2 — pipelined:** stream LLM tokens → chunker → TTS per sentence → audio streamed to the user while later sentences are still generating. Goal: reduce time-to-first-audio.
3. **Phase 3 — real-time (optional):** continuous mic streaming, VAD end-of-turn detection, barge-in (stop playback when user speaks). Consider Gradio's real-time audio tooling (e.g. FastRTC) — verify current docs first.

## Definition of done (prototype)
- End-to-end turn works on Colab T4 via share link.
- Latency log produced for every turn; eval script runs over `eval/utterances/`.
- README explains setup in under 10 steps.

## Things to avoid
- Do not run the LLM locally on the Colab T4 (VRAM needed for Whisper + TTS; Sinhala quality of small open LLMs is weak).
- Do not add a language router or English TTS — one Sinhala voice only.
- Do not invent model APIs: when unsure, inspect the Hugging Face model card/files or print the object's methods.
