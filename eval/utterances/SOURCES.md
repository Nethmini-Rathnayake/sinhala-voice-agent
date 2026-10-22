# Test utterances

`slr52_*.wav` with their `.txt` references are 10 clips from the
**Large Sinhala ASR training data set** (OpenSLR [SLR52](https://www.openslr.org/52/)),
collected by Google and the University of Moratuwa, licensed **CC BY-SA 4.0**.
Fetched from the Hugging Face mirror [openslr/openslr](https://huggingface.co/datasets/openslr/openslr).

These are read speech: clean, formal, no English mixed in. They give a comparable
baseline WER, but they do not represent the code-switched Sinhala-English speech this
agent is built for. Record your own clips for that.

Note: SLR52 transcripts contain known errors (mis-transcriptions, raw digits, some
English-only rows). Clips with digits were skipped here, because the TTS cannot
pronounce them.
