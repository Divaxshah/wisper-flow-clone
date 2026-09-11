# Wisper Flow clone — local ASR tester

## Setup

```bash
uv sync
# NeMo's git metadata conflicts with uv lock, so install it into the venv separately
uv pip install "nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git"
sudo apt-get install -y libsndfile1
```

First run will download ~5-6GB of model weights to `~/.cache/huggingface` —
do this on a decent connection, and expect the NeMo import itself to be
slow the first time (it's a big, dependency-heavy package).

Do not run `uv sync` after the NeMo pip install, or uv will remove it.

## Run

```bash
uv run --no-sync wisper-flow-clone
```

Open the local URL Gradio prints (usually http://127.0.0.1:7860). Models load
**before** that URL is served, so the first Transcribe click does not wait on
checkpoint restore.

The UI has two tabs:

- **Canary-Qwen 2.5B** — local ASR + cleanup from the same checkpoint (English-only).
- **Nemotron 3.5 ASR** — multilingual cache-aware streaming ASR
  (`nvidia/nemotron-3.5-asr-streaming-0.6b`), then optional cleanup via OpenRouter
  (OpenAI SDK). Use the Latency dropdown to pick chunk size (80ms–1.12s).

For the Nemotron tab, set an OpenRouter key before you tick cleanup:

```bash
export OPENROUTER_API_KEY=sk-or-...
# optional; defaults to openai/gpt-4o-mini
export OPENROUTER_MODEL=openai/gpt-4o-mini
```

You can also paste the key into the tab. A project-root `.env` is loaded on
startup (`OPENROUTER_API_KEY=...`); that file is gitignored.

## What to actually test tonight

1. A clean, short (5-10s) English sentence, spoken normally. Sanity check —
   should transcribe well and the cleanup pass shouldn't change much.
2. The same sentence but rambled with "um"s and a mid-sentence correction
   ("let's meet at 3pm, actually no, 2pm"). See if the cleanup pass catches
   the correction the way Rambler/Wispr Flow do.
3. A Hinglish sentence. On the Canary tab this will likely go badly (English-only).
   Use the Nemotron tab with `auto` or `hi-IN` for Hindi / mixed speech.
4. Something with your actual technical vocabulary (fine-tuning terms,
   product names). See what it mangles — that's your future hotword/
   fine-tuning list.

## Device

The app picks CUDA automatically when `torch.cuda.is_available()` is true.
Canary uses bf16/fp16 on GPU. Nemotron stays float32 on GPU — its streaming
prompt path is not mixed-precision safe (you'll get `float != BFloat16` otherwise).
On CPU both models use fp32. The Gradio timing line reports which device ran.

Hide the GPU (force CPU) with `CUDA_VISIBLE_DEVICES=""`.

## Known rough edges

- CPU inference is slow. A 10s clip might take 20-40s for ASR alone, more
  for the cleanup pass. CUDA is much faster when a GPU is visible to PyTorch.
- If model loading fails with a CUDA OOM, you need more VRAM — bf16/fp16
  weights want ~6GB+ headroom. On CPU, fp32 wants ~10GB+ system RAM.
- Canary audio over 40s gets truncated (the model wasn't trained past that
  length). Nemotron clips are capped at 180s to avoid blowing GPU memory.
- Both models load at process start. Loading Canary + Nemotron on one GPU can
  OOM; if that happens, only the model that fitted will be available.