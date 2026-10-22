# Sinhala Voice Agent (prototype)

A low-latency voice agent that listens to spoken Sinhala — including English words
mixed in, as people actually speak — and replies out loud in Sinhala.

```
Mic → Silero VAD → Whisper (Sinhala fine-tune) → LLM (streaming) → sentence chunker → SinhalaVITS → speaker
```

Every turn is timed stage by stage and appended to `logs/latency.jsonl`, because the
point of the prototype is to find out where the latency actually goes.

## Quick start on Colab (recommended)

The GPU path. Everything below runs in the browser.

1. Open <https://colab.research.google.com> → **File → Open notebook → GitHub** →
   enter `Nethmini-Rathnayake/sinhala-voice-agent` → pick `notebooks/colab_demo.ipynb`.
2. **Runtime → Change runtime type → T4 GPU**.
3. Get an API key from <https://aistudio.google.com/apikey> → **Create API key**.
   Make a new one: since September 2026 the Gemini API rejects older "standard" keys.
4. Add it as a secret: the **🔑 key icon** in the left icon strip → **+ Add new secret** →
   name it `GEMINI_API_KEY`, paste the value, enable **Notebook access**.
   (No key icon? Run the cell anyway — it prompts you, and nothing is saved to the notebook.)
5. Run the cells in order. They check the GPU, clone this repo, install dependencies,
   set the model cache, read your key, warm the models, and launch the app.
6. The last cell prints a `*.gradio.live` link. Open it, allow microphone access, speak
   Sinhala, press **Send**.

First run downloads ~2 GB of models and takes several minutes. Colab resets on disconnect,
so this repeats each session.

## Local setup

Runs on CPU. Slow but complete — useful for development, not for judging latency.

```bash
git clone https://github.com/Nethmini-Rathnayake/sinhala-voice-agent.git
cd sinhala-voice-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps silero-vad==6.2.2     # see "Dependency notes" below
cp .env.example .env                         # then add your GEMINI_API_KEY
python -m src.app                            # http://127.0.0.1:7860
```

On macOS you also need FFmpeg 8 — see "Dependency notes".

## Running the parts on their own

Each module works from the command line, which is the fastest way to find out which
stage is misbehaving.

```bash
python -m src.vad      recording.wav        # speech vs total duration, per segment
python -m src.stt      recording.wav        # transcript + time taken
python -m src.llm      "ඔයාට කොහොමද?"        # streamed reply + time to first token
python -m src.tts      "ආයුබෝවන්."           # writes out.wav
python -m src.pipeline recording.wav        # whole turn, writes reply.wav
python -m src.timing   logs/latency.jsonl   # mean latency per stage across turns
```

## Evaluation

Put recordings in `eval/utterances/`. A `.wav` may have a sidecar `.txt` holding the
reference transcript; where one exists, word error rate is computed with jiwer.

```bash
python -m eval.run_eval
```

Writes `eval/results.csv` (filename, reference, transcript, WER, reply, all six timings)
and prints mean latency per stage plus P50/P95 time to first audio.

## Configuration

Everything tunable lives in `config.yaml` — model IDs, VAD thresholds, LLM provider and
model, token limits. No model name or threshold is hard-coded.

To switch the LLM from Gemini to Groq, set `LLM_PROVIDER=groq` and supply `GROQ_API_KEY`.

API keys come only from the environment: `.env` locally (git-ignored), Colab Secrets in
the notebook. They are never printed or committed.

## Replies are Sinhala-only, by design

SinhalaVITS can pronounce Sinhala script and nothing else. So the LLM is instructed to
reply entirely in Sinhala script, writing English words phonetically — app → ඇප් එක,
email → ඉමේල්, WiFi → වයිෆයි — and numbers as words. If Latin characters survive anyway,
`src/llm.py` logs a warning rather than silently dropping text, and `src/tts.py` warns
about any character it cannot pronounce. Silent loss would be the worse failure.

## Measured latency

One turn on an Apple M1 (CPU, fp32):

| Stage | Time |
|---|---|
| `vad_ms` | 191 ms |
| `stt_ms` | 22,006 ms |
| `llm_first_token_ms` | 1,854 ms |
| `llm_total_ms` | 2,298 ms |
| `tts_first_sentence_ms` | 3,546 ms |
| `total_to_first_audio_ms` | 28,042 ms |

Whisper dominates at 79% of the turn. A T4 with fp16 should cut that sharply, and Phase 2
(synthesising each sentence while the LLM is still generating) attacks time-to-first-audio
directly. Neither number is measured yet.

## Dependency notes

Three installs need care. All three are the result of actual breakages, not caution.

- **`silero-vad` must be installed with `--no-deps`.** Its metadata caps `torchaudio<2.10`,
  which drags torch backwards and, on Colab, replaces the CUDA build. The parts needing
  torchaudio are unused here.
- **Gradio is pinned below 6.** Gradio 6 requires `huggingface-hub>=1.0`, which
  `transformers<5` refuses — and `transformers<5` is required by coqui-tts. Installing
  Gradio 6 breaks speech-to-text outright.
- **macOS needs `ffmpeg@8`.** transformers imports torchcodec whenever it is installed, and
  torchcodec 0.11 supports FFmpeg 4–8 while Homebrew's default is 9:
  ```bash
  brew install ffmpeg@8
  export DYLD_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg@8/lib
  ```
  Colab's FFmpeg already works.

If model downloads stall at 0 B, set `HF_HUB_DISABLE_XET=1`. An `HF_TOKEN` makes them
faster and less rate-limited.

## Layout

```
src/
  vad.py        Silero VAD: trim(audio, sr) -> speech only, or None
  stt.py        Whisper: transcribe(audio, sr) -> Sinhala text
  llm.py        stream_reply(history, text) -> token iterator
  chunker.py    sentences(tokens) -> sentence iterator
  tts.py        synthesize(sentence) -> (22050, float32 audio)
  pipeline.py   run_turn(audio, sr, history) -> (audio, transcript, reply, timings)
  timing.py     stage timer + JSON latency log
  audio.py      mono float32 @ 16 kHz conversion
  config.py     config.yaml + .env loading, device selection
  app.py        Gradio UI
notebooks/colab_demo.ipynb
eval/run_eval.py, eval/utterances/
tests/
```

## Status

Phase 1 (turn-based) works end to end: speak, wait, hear the reply. Phase 2 (streaming
audio sentence by sentence) and Phase 3 (continuous mic, barge-in) are not built.

Models used: [Subhaka/whisper-small-Sinhala-Fine_Tune](https://huggingface.co/Subhaka/whisper-small-Sinhala-Fine_Tune)
and [dialoglk/SinhalaVITS-TTS-F1](https://huggingface.co/dialoglk/SinhalaVITS-TTS-F1)
(Dialog Axiata and the Dialog–UoM Research Lab; its model card states MPL-2.0). Check both
licences before any use beyond prototyping. This repo contains no model weights.
