# Canary-Qwen-2.5B local tester

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

Open the local URL Gradio prints (usually http://127.0.0.1:7860).

## What to actually test tonight

1. A clean, short (5-10s) English sentence, spoken normally. Sanity check —
   should transcribe well and the cleanup pass shouldn't change much.
2. The same sentence but rambled with "um"s and a mid-sentence correction
   ("let's meet at 3pm, actually no, 2pm"). See if the cleanup pass catches
   the correction the way Rambler/Wispr Flow do.
3. A Hinglish sentence. Expect this to go badly — the model card says
   English-only, this just confirms it firsthand and tells you how badly.
4. Something with your actual technical vocabulary (fine-tuning terms,
   product names). See what it mangles — that's your future hotword/
   fine-tuning list.

## Device

The app picks CUDA automatically when `torch.cuda.is_available()` is true
(bf16 if the GPU supports it, otherwise fp16). Otherwise it stays on CPU
in fp32. The Gradio timing line reports which device actually ran.

Hide the GPU (force CPU) with `CUDA_VISIBLE_DEVICES=""`.

## Known rough edges

- CPU inference is slow. A 10s clip might take 20-40s for ASR alone, more
  for the cleanup pass. CUDA is much faster when a GPU is visible to PyTorch.
- If model loading fails with a CUDA OOM, you need more VRAM — bf16/fp16
  weights want ~6GB+ headroom. On CPU, fp32 wants ~10GB+ system RAM.
- Audio over 40s gets truncated in this script (the model wasn't trained
  past that length) — you'll see it silently cut in `_load_and_resample`.