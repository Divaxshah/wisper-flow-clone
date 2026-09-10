"""
Canary-Qwen-2.5B tester — CUDA when available, otherwise CPU.

What this does:
  1. Loads nvidia/canary-qwen-2.5b via NeMo's SALM class.
  2. Gradio UI: record/upload audio -> raw ASR transcript.
  3. Optional second pass: feeds the transcript back into the SAME model's
     LLM mode (adapter disabled) to clean it up / reformat it. This mimics
     the "step 2" cleanup layer we talked about, using nothing but this
     one checkpoint.

Known constraints (from the model card, not a bug in this script):
  - English only. Hinglish / Gujarati will likely transcribe badly or
    hallucinate English-sounding words. That's expected — test it, don't
    debug it.
  - Max training audio length was 40s. Feed it short clips (5-20s) for
    now. Longer clips may still run but accuracy isn't guaranteed.
  - CPU inference is slow (tens of seconds per clip). CUDA is used
    automatically when torch.cuda.is_available() is true.

Usage:
    pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git" gradio soundfile librosa
    python app.py
"""

import tempfile
import time

import gradio as gr
import soundfile as sf
import librosa
import numpy as np
import re

MODEL_ID = "nvidia/canary-qwen-2.5b"
TARGET_SR = 16000
MAX_AUDIO_SECONDS = 40  # hard model limit per the card

_model = None
_load_error = None
_runtime = None  # {"device": torch.device, "label": str}


def _select_runtime():
    """Prefer CUDA (bf16/fp16) when a GPU is actually usable; otherwise CPU fp32."""
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.is_bf16_supported():
            dtype, dtype_name = torch.bfloat16, "bfloat16"
        else:
            dtype, dtype_name = torch.float16, "float16"
        gpu_name = torch.cuda.get_device_name(0)
        label = f"cuda ({gpu_name}, {dtype_name})"
        return device, dtype, label

    return torch.device("cpu"), torch.float32, "cpu (float32)"


def get_model():
    """Lazy-load so the Gradio UI shows up immediately, model loads on first use."""
    global _model, _load_error, _runtime
    if _model is not None:
        return _model
    if _load_error is not None:
        raise RuntimeError(_load_error)
    try:
        from nemo.collections.speechlm2.models import SALM

        device, dtype, label = _select_runtime()
        print(f"Loading {MODEL_ID} on {label}... this can take a few minutes the first time "
              f"(downloading ~5GB of weights + NeMo import overhead).")
        model = SALM.from_pretrained(MODEL_ID)
        # NVIDIA's recommended GPU path for this checkpoint is .bfloat16().eval().to(cuda).
        model = model.to(dtype=dtype).to(device).eval()
        _model = model
        _runtime = {"device": device, "label": label}
        print(f"Model loaded on {label}.")
        return _model
    except Exception as e:
        _load_error = (
            f"Failed to load model: {e}\n\n"
            "Common causes:\n"
            "- NeMo not installed correctly (needs the git/trunk version, see docstring)\n"
            "- Missing system deps (libsndfile) for audio loading\n"
            "- CUDA OOM (2.5B in bf16/fp16 wants ~6GB+ VRAM; the script falls back to CPU only if CUDA is unavailable)\n"
            "- Not enough RAM (2.5B params in fp32 needs ~10GB+ system RAM on CPU)"
        )
        raise RuntimeError(_load_error)


def _load_and_resample(audio_path: str) -> str:
    """Ensure input is 16kHz mono wav, since that's what the model expects.
    Gradio's mic/upload can hand us other sample rates/channels."""
    audio, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    duration = len(audio) / TARGET_SR
    if duration > MAX_AUDIO_SECONDS:
        audio = audio[: TARGET_SR * MAX_AUDIO_SECONDS]
        duration = MAX_AUDIO_SECONDS

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, TARGET_SR, subtype="PCM_16")
    return tmp.name, duration

def _extract_reply(decoded: str) -> str:
    """model.tokenizer.ids_to_text() decodes the WHOLE sequence, chat template
    scaffolding included — role markers ('user' / 'assistant') and, since the
    underlying LLM is Qwen3, its <think>...</think> reasoning block. We only
    want the actual reply text. Strip both."""
    text = decoded
 
    # Drop everything up to and including the last "assistant" role marker,
    # if present (that's where the actual reply starts).
    marker = re.split(r"\bassistant\b", text)
    if len(marker) > 1:
        text = marker[-1]
 
    # Drop <think>...</think> blocks (Qwen3 reasoning trace), even if
    # /no-think didn't fully suppress it.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
 
    return text.strip()


def transcribe(audio_path, run_cleanup):
    if audio_path is None:
        return "No audio received.", "", ""

    try:
        model = get_model()
    except RuntimeError as e:
        return str(e), "", ""

    t0 = time.time()
    wav_path, duration = _load_and_resample(audio_path)

    # --- ASR mode ---
    answer_ids = model.generate(
        prompts=[
            [{
                "role": "user",
                "content": f"Transcribe the following: {model.audio_locator_tag}",
                "audio": [wav_path],
            }]
        ],
        max_new_tokens=256,
    )
    raw_transcript = _extract_reply(model.tokenizer.ids_to_text(answer_ids[0].cpu()).strip())
    t1 = time.time()

    device_label = _runtime["label"] if _runtime else "unknown"
    timing = (
        f"Audio duration: {duration:.1f}s | ASR pass: {t1 - t0:.1f}s | device: {device_label}"
    )

    cleaned = ""
    
    if run_cleanup and raw_transcript:
        with model.llm.disable_adapter():
            cleanup_ids = model.generate(
                prompts=[[
                    {"role": "user", "content": "Clean up this raw speech transcript. Remove filler "
                        "words, fix self-corrections, and drop abandoned thoughts entirely — if the "
                        "speaker starts a sentence, discards it, and restates their point, keep only "
                        "the final resolved version.\n\n"
                        "Transcript: Um so I wanted to, uh, talk about the the project timeline."},
                    {"role": "assistant", "content": "I wanted to talk about the project timeline."},

                    {"role": "user", "content": "Transcript: Let's meet at 3pm, actually no, let's "
                        "make it 2pm instead."},
                    {"role": "assistant", "content": "Let's meet at 2pm."},

                    {"role": "user", "content": "Transcript: So my initial point was, um, well actually, that's not important. What I wanted to mention is that I'm preparing for a coding interview."},
                    {"role": "assistant", "content": "I wanted to mention that I'm preparing for a coding interview."},
                   
                    {"role": "user", "content": f"Transcript: {raw_transcript} /no-think"},
                ]],
                max_new_tokens=512,
            )
        cleaned = _extract_reply(model.tokenizer.ids_to_text(cleanup_ids[0].cpu()))
        # print(f"Cleaned transcript: {cleaned}")
        t2 = time.time()
        timing += f" | Cleanup pass: {t2 - t1:.1f}s"

    return timing, raw_transcript, cleaned


with gr.Blocks(title="Canary-Qwen-2.5B Tester") as demo:
    gr.Markdown(
        "# Canary-Qwen-2.5B — local test\n"
        "Record or upload a short clip (under 40s). English only — this will "
        "misbehave on Hinglish/other languages, and that's expected on this model. "
        "Uses CUDA automatically when PyTorch can see a GPU; otherwise CPU "
        "(CPU is slow — tens of seconds per clip)."
    )

    with gr.Row():
        audio_in = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Your audio",
        )

    cleanup_toggle = gr.Checkbox(
        value=True,
        label="Also run LLM cleanup pass (fillers/self-corrections/formatting)",
    )

    submit_btn = gr.Button("Transcribe", variant="primary")

    timing_out = gr.Textbox(label="Timing")
    raw_out = gr.Textbox(label="Raw ASR transcript", lines=4)
    cleaned_out = gr.Textbox(label="Cleaned transcript (LLM mode)", lines=4)

    submit_btn.click(
        fn=transcribe,
        inputs=[audio_in, cleanup_toggle],
        outputs=[timing_out, raw_out, cleaned_out],
    )

def main() -> None:
    demo.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()