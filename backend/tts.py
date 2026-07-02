"""Real narration via Qwen3-TTS — manages the three sub-models.

  * "custom"  CustomVoice  - fixed preset speakers   generate_custom_voice()
  * "design"  VoiceDesign  - voice from free text     generate_voice_design()
  * "base"    Base         - clone a reference clip   generate_voice_clone()

Only ONE sub-model is resident at a time (each is ~4.5 GB FP16 and the card has
8 GB), so switching kinds unloads the previous model and empties the CUDA
cache. Callers should batch work by kind — render all clone clips, then all
design clips — to avoid load thrashing (see generation._narrate_lines).

Turing-safe config (RTX 2080 / sm75):
  * dtype=float16  (no bfloat16 on Turing)
  * attn_implementation="sdpa"  (FlashAttention 2 needs Ampere+)
  * device auto-picks GPU 1 when two cards are present, so it doesn't contend
    with Ollama's LLM on GPU 0.

If torch / qwen-tts aren't installed (or a load fails), `available()` returns
False and callers fall back to a silent placeholder clip — the app keeps
working either way. All synthesis is serialized behind one lock (one GPU).
"""
import gc
import threading

import config

_lock = threading.RLock()
_model = None
_kind = None            # which sub-model is resident: 'custom' | 'design' | 'base'
_sr = None              # sample rate, known after the first successful synth
_load_error = None      # last import/load failure, for /api/tts/status
_imports_ok = None      # cached result of the cheap torch+qwen_tts import probe
_device = None

_MODEL_IDS = {
    "custom": lambda: config.TTS_MODEL,
    "design": lambda: config.TTS_MODEL_DESIGN,
    "base": lambda: config.TTS_MODEL_BASE,
}


def _pick_device(torch) -> str:
    if config.TTS_DEVICE:
        return config.TTS_DEVICE
    try:
        n = torch.cuda.device_count()
    except Exception:
        n = 0
    if n >= 2:
        return "cuda:1"   # keep GPU 0 for Ollama
    if n == 1:
        return "cuda:0"
    return "cpu"


def _check_imports() -> bool:
    """Cheap availability probe — does NOT download or load any model."""
    global _imports_ok, _load_error
    if _imports_ok is None:
        try:
            import torch  # noqa: F401
            import qwen_tts  # noqa: F401
            _imports_ok = True
        except Exception as e:  # noqa: BLE001 - degrade gracefully
            _imports_ok = False
            _load_error = e
    return _imports_ok


def available() -> bool:
    if config.TTS_DISABLED:
        return False
    return _check_imports()


def sample_rate():
    return _sr


def load_error():
    return str(_load_error) if _load_error else None


def status() -> dict:
    """Health snapshot for /api/tts/status."""
    return {
        "disabled": config.TTS_DISABLED,
        "available": available(),
        "loaded": _kind,
        "device": _device,
        "sample_rate": _sr,
        "error": load_error(),
    }


def _ensure(kind: str):
    """Load the requested sub-model, unloading any other one first (8 GB card)."""
    global _model, _kind, _load_error, _device
    with _lock:
        if _kind == kind and _model is not None:
            return _model
        unload()
        import torch
        from qwen_tts import Qwen3TTSModel

        _device = _pick_device(torch)
        dtype = torch.float16 if _device.startswith("cuda") else torch.float32
        try:
            _model = Qwen3TTSModel.from_pretrained(
                _MODEL_IDS[kind](),
                device_map=_device,
                dtype=dtype,
                attn_implementation="sdpa",  # Turing-safe (no flash_attention_2)
            )
            _kind = kind
            print(f"[tts] loaded {_MODEL_IDS[kind]()} on {_device} ({dtype})", flush=True)
        except Exception as e:  # noqa: BLE001 - callers fall back to placeholder
            _load_error = e
            _model, _kind = None, None
            raise
        return _model


def unload() -> None:
    """Free the resident model + VRAM (called between kinds and after a render)."""
    global _model, _kind
    with _lock:
        if _model is None:
            return
        _model, _kind = None, None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cache cleanup is best-effort
            pass


def _write(wavs, sr: int, out_path: str) -> int:
    global _sr
    import soundfile as sf
    _sr = sr
    sf.write(out_path, wavs[0], sr, subtype="PCM_16")
    return sr


def synthesize_custom(text: str, speaker: str, instruct: str, out_path: str,
                      language: str = None) -> int:
    """Preset-speaker narration (CustomVoice). Returns the sample rate."""
    with _lock:
        model = _ensure("custom")
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language or config.TTS_LANGUAGE,
            speaker=speaker or config.NARRATOR_SPEAKER,
            instruct=instruct or "",
        )
        return _write(wavs, sr, out_path)


def synthesize_design(text: str, instruct: str, out_path: str,
                      language: str = None) -> int:
    """Voice from a free-text description (VoiceDesign) — best emotion,
    not reproducible between calls. Returns the sample rate."""
    with _lock:
        model = _ensure("design")
        wavs, sr = model.generate_voice_design(
            text=text,
            instruct=instruct or "",
            language=language or config.TTS_LANGUAGE,
        )
        return _write(wavs, sr, out_path)


def synthesize_clone(text: str, ref_audio: str, ref_text: str, out_path: str,
                     language: str = None) -> int:
    """Clone a reference clip's voice (Base model) — reproducible, no emotion
    knob (copies the reference's delivery). Returns the sample rate."""
    with _lock:
        model = _ensure("base")
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language or config.TTS_LANGUAGE,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        return _write(wavs, sr, out_path)
