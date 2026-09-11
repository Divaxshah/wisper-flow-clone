"""
Local ASR tester with two pipelines:

  1. Canary-Qwen-2.5B (NeMo SALM) — ASR + same-checkpoint LLM cleanup.
  2. Nemotron 3.5 ASR streaming 0.6B — multilingual cache-aware streaming ASR,
     then OpenRouter (OpenAI SDK) for transcript cleanup.

CUDA is used automatically when torch.cuda.is_available() is true.
Models are loaded before Gradio serves the UI.
"""

import os
import re
import tempfile
import threading
import time
from pathlib import Path

import dotenv
import gradio as gr
import librosa
import soundfile as sf
import torch

# python-dotenv looks at cwd by default; also pick up a file next to this module.
for _env_file in (
    Path(__file__).resolve().parent / ".env",
    Path(__file__).resolve().parents[2] / ".env",
    Path.cwd() / ".env",
):
    dotenv.load_dotenv(_env_file)

# NeMo RNNT/numba tries to init CUDA graphs even on CPU. Disable that when
# there is no GPU, before NeMo is imported.
if not torch.cuda.is_available():
    os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")

CANARY_MODEL_ID = "nvidia/canary-qwen-2.5b"
NEMOTRON_MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
TARGET_SR = 16000
CANARY_MAX_AUDIO_SECONDS = 40  # hard model limit per the Canary card
NEMOTRON_MAX_AUDIO_SECONDS = 180
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

NEMOTRON_LANG_CHOICES = [
    ("Auto-detect", "auto"),
    ("Arabic (ar-AR)", "ar-AR"),
    ("English (en-GB)", "en-GB"),
    ("English (en-US)", "en-US"),
    ("French (fr-CA)", "fr-CA"),
    ("French (fr-FR)", "fr-FR"),
    ("German (de-DE)", "de-DE"),
    ("Gujarati (gu-IN)", "gu-IN"),
    ("Hindi (hi-IN)", "hi-IN"),
    ("Italian (it-IT)", "it-IT"),
    ("Japanese (ja-JP)", "ja-JP"),
    ("Korean (ko-KR)", "ko-KR"),
    ("Mandarin (zh-CN)", "zh-CN"),
    ("Polish (pl-PL)", "pl-PL"),
    ("Portuguese (pt-BR)", "pt-BR"),
    ("Portuguese (pt-PT)", "pt-PT"),
    ("Russian (ru-RU)", "ru-RU"),
    ("Spanish (es-ES)", "es-ES"),
    ("Spanish (es-US)", "es-US"),
    ("Turkish (tr-TR)", "tr-TR"),
    ("Ukrainian (uk-UA)", "uk-UA"),
    ("Vietnamese (vi-VN)", "vi-VN"),
]
NEMOTRON_LANG_CODES = {value for _, value in NEMOTRON_LANG_CHOICES}

# Cache-aware streaming lookahead. Values are [left, right] in 80ms frames.
NEMOTRON_CHUNK_PROFILES = {
    "Lowest latency - 80 ms": [56, 0],
    "Fast - 160 ms": [56, 1],
    "Balanced - 320 ms": [56, 3],
    "Accurate - 560 ms": [56, 6],
    "Most accurate - 1.12 s": [56, 13],
}
NEMOTRON_DEFAULT_CHUNK = "Most accurate - 1.12 s"

CLEANUP_FEW_SHOT = [
    (
        "Um so I wanted to, uh, talk about the the project timeline.",
        "I wanted to talk about the project timeline.",
    ),
    (
        "Let's meet at 3pm, actually no, let's make it 2pm instead.",
        "Let's meet at 2pm.",
    ),
    (
        "So my initial point was, um, well actually, that's not important. "
        "What I wanted to mention is that I'm preparing for a coding interview.",
        "I wanted to mention that I'm preparing for a coding interview.",
    ),
]

CLEANUP_INSTRUCTIONS = (
    "Clean up this raw speech transcript. Remove filler words, fix self-corrections, "
    "and drop abandoned thoughts entirely — if the speaker starts a sentence, discards it, "
    "and restates their point, keep only the final resolved version."
)

_canary_model = None
_canary_error = None
_canary_runtime = None

_nemotron_model = None
_nemotron_error = None
_nemotron_runtime = None
_nemotron_lock = threading.RLock()


def _select_runtime():
    """Prefer CUDA (bf16/fp16) when a GPU is actually usable; otherwise CPU fp32."""
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


def get_canary_model():
    global _canary_model, _canary_error, _canary_runtime
    if _canary_model is not None:
        return _canary_model
    if _canary_error is not None:
        raise RuntimeError(_canary_error)
    try:
        from nemo.collections.speechlm2.models import SALM

        device, dtype, label = _select_runtime()
        print(
            f"Loading {CANARY_MODEL_ID} on {label}... this can take a few minutes the first time "
            f"(downloading ~5GB of weights + NeMo import overhead)."
        )
        model = SALM.from_pretrained(CANARY_MODEL_ID)
        model = model.to(dtype=dtype).to(device).eval()
        _canary_model = model
        _canary_runtime = {"device": device, "label": label}
        print(f"Canary loaded on {label}.")
        return _canary_model
    except Exception as e:
        _canary_error = (
            f"Failed to load Canary: {e}\n\n"
            "Common causes:\n"
            "- NeMo not installed correctly (needs the git/trunk version, see README)\n"
            "- Missing system deps (libsndfile) for audio loading\n"
            "- CUDA OOM (2.5B in bf16/fp16 wants ~6GB+ VRAM)\n"
            "- Not enough RAM (2.5B params in fp32 needs ~10GB+ system RAM on CPU)"
        )
        raise RuntimeError(_canary_error)


def get_nemotron_model():
    global _nemotron_model, _nemotron_error, _nemotron_runtime
    if _nemotron_model is not None:
        return _nemotron_model
    if _nemotron_error is not None:
        raise RuntimeError(_nemotron_error)
    try:
        import nemo.collections.asr as nemo_asr
        from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig

        device, _dtype, _label = _select_runtime()
        # Cache-aware streaming + language-ID prompt fusion is float32 in NeMo.
        # Casting this checkpoint to bf16/fp16 makes preprocessor/prompt matmuls
        # see float vs BFloat16. NVIDIA's Space keeps the model in fp32.
        dtype = torch.float32
        if device.type == "cuda":
            label = f"cuda ({torch.cuda.get_device_name(0)}, float32)"
        else:
            label = "cpu (float32)"
        print(
            f"Loading {NEMOTRON_MODEL_ID} on {label}... first run downloads the checkpoint."
        )
        model = nemo_asr.models.ASRModel.from_pretrained(NEMOTRON_MODEL_ID)
        # Same as NVIDIA's Space: disable CUDA-graph RNNT decoding (breaks on CPU).
        if hasattr(model, "change_decoding_strategy") and hasattr(model, "joint"):
            decoding_cfg = RNNTDecodingConfig(fused_batch_size=-1)
            decoding_cfg.greedy.use_cuda_graph_decoder = False
            model.change_decoding_strategy(decoding_cfg)
        model = model.to(dtype=dtype).to(device).eval()
        _nemotron_model = model
        _nemotron_runtime = {"device": device, "label": label}
        print(f"Nemotron ASR loaded on {label}.")
        return _nemotron_model
    except Exception as e:
        _nemotron_error = (
            f"Failed to load Nemotron ASR: {e}\n\n"
            "Common causes:\n"
            "- NeMo not installed correctly (needs the git/trunk version, see README)\n"
            "- Missing system deps (libsndfile) for audio loading\n"
            "- CUDA OOM (0.6B wants ~2GB+ VRAM; loading both models on one GPU needs more)"
        )
        raise RuntimeError(_nemotron_error)


def load_models_at_startup() -> None:
    """Load both checkpoints before Gradio binds the port so Transcribe is ready."""
    print("Loading models before Gradio starts. The UI will open when they are ready.")
    try:
        get_nemotron_model()
    except Exception as e:
        print(f"Nemotron failed to load at startup: {e}")
    try:
        get_canary_model()
    except Exception as e:
        print(f"Canary failed to load at startup: {e}")
    print("Startup loading finished.")


def _load_and_resample(audio_path: str, max_seconds: int | None) -> tuple[str, float]:
    """Ensure input is 16kHz mono wav, since that's what the models expect."""
    audio, _sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    duration = len(audio) / TARGET_SR
    if max_seconds is not None and duration > max_seconds:
        audio = audio[: TARGET_SR * max_seconds]
        duration = float(max_seconds)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, TARGET_SR, subtype="PCM_16")
    return tmp.name, duration


def _extract_reply(decoded: str) -> str:
    """Strip chat-template scaffolding and Qwen3 <think> blocks from Canary output."""
    text = decoded

    marker = re.split(r"\bassistant\b", text)
    if len(marker) > 1:
        text = marker[-1]

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    return text.strip()


def _split_nemotron_lang_tag(text: str) -> tuple[str, str]:
    """Nemotron auto-detect appends a locale tag like <en-US> after terminal punctuation."""
    match = re.search(r"\s*<([a-z]{2}(?:-[A-Za-z]{2})?)>\s*$", text)
    if not match:
        return text.strip(), ""
    return text[: match.start()].strip(), match.group(1)


def _extract_transcriptions(hypotheses) -> list[str]:
    from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

    hypotheses = list(hypotheses)
    if not hypotheses:
        return [""]
    if isinstance(hypotheses[0], Hypothesis):
        return [hyp.text or "" for hyp in hypotheses]
    return [str(hyp) for hyp in hypotheses]


def _drop_extra_pre_encoded(asr_model, step_num: int, pad_and_drop_preencoded: bool) -> int:
    if step_num == 0 and not pad_and_drop_preencoded:
        return 0
    return asr_model.encoder.streaming_cfg.drop_extra_pre_encoded


def _configure_nemotron(asr_model, target_lang: str, chunk_profile: str) -> None:
    if target_lang not in NEMOTRON_LANG_CODES:
        raise ValueError(f"Unsupported language code: {target_lang}")
    att_context_size = NEMOTRON_CHUNK_PROFILES[chunk_profile]
    if hasattr(asr_model.encoder, "set_default_att_context_size"):
        asr_model.encoder.set_default_att_context_size(att_context_size=att_context_size)
    if hasattr(asr_model, "set_inference_prompt"):
        asr_model.set_inference_prompt(target_lang)
    if hasattr(asr_model, "decoding") and hasattr(asr_model.decoding, "set_strip_lang_tags"):
        # Keep the tag so the UI can show detected language; we strip for display.
        asr_model.decoding.set_strip_lang_tags(False, lang_tag_pattern=None)


def _stream_transcribe_file(asr_model, wav_path: str) -> str:
    """Cache-aware streaming inference used by NVIDIA's Nemotron Space.

    `model.transcribe()` drops the language prompt (prompt key becomes None).
    `set_inference_prompt` + `conformer_stream_step` is the supported path.
    """
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    model_device = next(asr_model.parameters()).device
    # Must match the streaming preprocessor and prompt kernel (both fp32 in NeMo).
    stream_dtype = torch.float32
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=False,
        pad_and_drop_preencoded=False,
    )
    streaming_buffer.append_audio_file(wav_path, stream_id=-1)

    cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
        batch_size=1
    )

    def _move_cache(value):
        if torch.is_tensor(value):
            if value.is_floating_point():
                return value.to(device=model_device, dtype=stream_dtype)
            return value.to(device=model_device)
        if isinstance(value, (list, tuple)):
            return type(value)(_move_cache(item) for item in value)
        return value

    cache_last_channel = _move_cache(cache_last_channel)
    cache_last_time = _move_cache(cache_last_time)
    cache_last_channel_len = _move_cache(cache_last_channel_len)
    previous_hypotheses = None
    previous_pred_out = None
    text = ""

    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer):
        with torch.inference_mode():
            chunk_audio = chunk_audio.to(device=model_device, dtype=stream_dtype)
            chunk_lengths = chunk_lengths.to(device=model_device)
            (
                previous_pred_out,
                transcribed_texts,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
                previous_hypotheses,
            ) = asr_model.conformer_stream_step(
                processed_signal=chunk_audio,
                processed_signal_length=chunk_lengths,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=streaming_buffer.is_buffer_empty(),
                previous_hypotheses=previous_hypotheses,
                previous_pred_out=previous_pred_out,
                drop_extra_pre_encoded=_drop_extra_pre_encoded(asr_model, step_num, False),
                return_transcription=True,
            )
        text = _extract_transcriptions(transcribed_texts)[0]

    return text


def transcribe_canary(audio_path, run_cleanup):
    if audio_path is None:
        return "No audio received.", "", ""

    try:
        model = get_canary_model()
    except RuntimeError as e:
        return str(e), "", ""

    t0 = time.time()
    wav_path, duration = _load_and_resample(audio_path, CANARY_MAX_AUDIO_SECONDS)

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

    device_label = _canary_runtime["label"] if _canary_runtime else "unknown"
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
        t2 = time.time()
        timing += f" | Cleanup pass: {t2 - t1:.1f}s"

    return timing, raw_transcript, cleaned


def _openrouter_cleanup(raw_transcript: str, api_key: str, model_id: str) -> str:
    from openai import OpenAI

    key = (api_key or "").strip() or os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "No OpenRouter API key. Set OPENROUTER_API_KEY or paste a key in the tab."
        )

    chosen_model = (model_id or "").strip() or DEFAULT_OPENROUTER_MODEL
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)

    messages = [
        {
            "role": "system",
            "content": (
                f"{CLEANUP_INSTRUCTIONS} Return only the cleaned transcript, nothing else."
            ),
        }
    ]
    for src, dst in CLEANUP_FEW_SHOT:
        messages.append({"role": "user", "content": f"Transcript: {src}"})
        messages.append({"role": "assistant", "content": dst})
    messages.append({"role": "user", "content": f"Transcript: {raw_transcript}"})

    response = client.chat.completions.create(
        model=chosen_model,
        messages=messages,
        extra_headers={
            "HTTP-Referer": "http://localhost:7860",
            "X-Title": "Wisper Flow Clone",
        },
    )
    return (response.choices[0].message.content or "").strip()


def transcribe_nemotron(audio_path, target_lang, chunk_profile, run_cleanup, api_key, openrouter_model):
    if audio_path is None:
        return "No audio received.", "", "", ""

    try:
        model = get_nemotron_model()
    except RuntimeError as e:
        return str(e), "", "", ""

    lang = target_lang or "auto"
    profile = chunk_profile or NEMOTRON_DEFAULT_CHUNK
    if profile not in NEMOTRON_CHUNK_PROFILES:
        return f"Unsupported chunk profile: {profile}", "", "", ""

    t0 = time.time()
    wav_path, duration = _load_and_resample(audio_path, NEMOTRON_MAX_AUDIO_SECONDS)

    try:
        with _nemotron_lock:
            _configure_nemotron(model, lang, profile)
            raw = _stream_transcribe_file(model, wav_path)
    except Exception as e:
        return f"Nemotron ASR failed: {e}", "", "", ""

    raw_transcript, detected_lang = _split_nemotron_lang_tag(raw)
    t1 = time.time()

    device_label = _nemotron_runtime["label"] if _nemotron_runtime else "unknown"
    lang_note = detected_lang or lang
    timing = (
        f"Audio duration: {duration:.1f}s | ASR pass: {t1 - t0:.1f}s | "
        f"device: {device_label} | lang: {lang_note} | {profile}"
    )

    cleaned = ""
    if run_cleanup and raw_transcript:
        try:
            cleaned = _openrouter_cleanup(raw_transcript, api_key, openrouter_model)
            t2 = time.time()
            timing += f" | Cleanup pass: {t2 - t1:.1f}s"
        except Exception as e:
            cleaned = f"OpenRouter cleanup failed: {e}"

    return timing, raw_transcript, detected_lang, cleaned


with gr.Blocks(title="Wisper Flow clone tester") as demo:
    gr.Markdown("# Wisper Flow clone — local ASR tester")

    with gr.Tabs():
        with gr.Tab("Canary-Qwen 2.5B"):
            gr.Markdown(
                "Record or upload a short clip (under 40s). English only — this will "
                "misbehave on Hinglish/other languages, and that's expected on this model. "
                "Cleanup uses the same checkpoint's LLM mode. "
                "Uses CUDA automatically when PyTorch can see a GPU; otherwise CPU "
                "(CPU is slow — tens of seconds per clip)."
            )

            canary_audio = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Your audio",
            )
            canary_cleanup = gr.Checkbox(
                value=True,
                label="Also run LLM cleanup pass (fillers/self-corrections/formatting)",
            )
            canary_btn = gr.Button("Transcribe", variant="primary")
            canary_timing = gr.Textbox(label="Timing")
            canary_raw = gr.Textbox(label="Raw ASR transcript", lines=4)
            canary_cleaned = gr.Textbox(label="Cleaned transcript (LLM mode)", lines=4)

            canary_btn.click(
                fn=transcribe_canary,
                inputs=[canary_audio, canary_cleanup],
                outputs=[canary_timing, canary_raw, canary_cleaned],
            )

        with gr.Tab("Nemotron 3.5 ASR"):
            env_key_set = bool(os.environ.get("OPENROUTER_API_KEY", "").strip())
            key_hint = (
                "OPENROUTER_API_KEY is already set in the environment."
                if env_key_set
                else "Paste an OpenRouter key below, or export OPENROUTER_API_KEY before launching."
            )
            gr.Markdown(
                "Multilingual cache-aware streaming ASR (`nvidia/nemotron-3.5-asr-streaming-0.6b`). "
                "Pick a language (or auto-detect) and a latency/accuracy chunk size. "
                "Optional cleanup goes through OpenRouter. "
                f"{key_hint}"
            )

            nemotron_audio = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Your audio",
            )
            with gr.Row():
                nemotron_lang = gr.Dropdown(
                    choices=NEMOTRON_LANG_CHOICES,
                    value="auto",
                    label="Language",
                    filterable=True,
                )
                nemotron_chunk = gr.Dropdown(
                    choices=list(NEMOTRON_CHUNK_PROFILES.keys()),
                    value=NEMOTRON_DEFAULT_CHUNK,
                    label="Latency",
                )
                nemotron_cleanup = gr.Checkbox(
                    value=True,
                    label="Clean up via OpenRouter",
                )
            with gr.Row():
                openrouter_key = gr.Textbox(
                    type="password",
                    label="OpenRouter API key",
                    placeholder="Leave blank to use OPENROUTER_API_KEY",
                )
                openrouter_model = gr.Textbox(
                    value=DEFAULT_OPENROUTER_MODEL,
                    label="OpenRouter model",
                    placeholder="openai/gpt-4o-mini",
                )
            nemotron_btn = gr.Button("Transcribe", variant="primary")
            nemotron_timing = gr.Textbox(label="Timing")
            nemotron_raw = gr.Textbox(label="Raw ASR transcript", lines=4)
            nemotron_lang_out = gr.Textbox(label="Detected language tag")
            nemotron_cleaned = gr.Textbox(label="Cleaned transcript (OpenRouter)", lines=4)

            nemotron_btn.click(
                fn=transcribe_nemotron,
                inputs=[
                    nemotron_audio,
                    nemotron_lang,
                    nemotron_chunk,
                    nemotron_cleanup,
                    openrouter_key,
                    openrouter_model,
                ],
                outputs=[
                    nemotron_timing,
                    nemotron_raw,
                    nemotron_lang_out,
                    nemotron_cleaned,
                ],
            )


def main() -> None:
    load_models_at_startup()
    demo.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
