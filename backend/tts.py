"""Real narration via Qwen3-TTS — lazy singleton loader.

Loads the 1.7B CustomVoice model ONCE, on first use, with a Turing-safe config:
  * dtype=float16  (RTX 2080 / Turing has no bfloat16)
  * attn_implementation="sdpa"  (FlashAttention 2 needs Ampere+; not on Turing)
  * device auto-picks GPU 1 when two cards are present, so it doesn't contend
    with Ollama's LLM on GPU 0.

If torch / qwen-tts / CUDA aren't available (or loading fails), `available()`
returns False and callers fall back to a silent placeholder clip — the app keeps
working either way.
"""
import threading

import config

_lock = threading.Lock()
_model = None
_sr = None              # sample rate, known after the first successful synth
_load_error = None
_loaded = False         # we attempted a load (success or failure)
_device = None


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


def _load() -> None:
    global _model, _load_error, _loaded, _device
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        try:
            import torch
            from qwen_tts import Qwen3TTSModel

            _device = _pick_device(torch)
            dtype = torch.float16 if _device.startswith("cuda") else torch.float32
            _model = Qwen3TTSModel.from_pretrained(
                config.TTS_MODEL,
                device_map=_device,
                dtype=dtype,
                attn_implementation="sdpa",  # Turing-safe (no flash_attention_2)
            )
            print(f"[tts] loaded {config.TTS_MODEL} on {_device} ({dtype})", flush=True)
        except Exception as e:  # noqa: BLE001 - degrade gracefully to placeholder
            _load_error = e
            _model = None
            print(f"[tts] unavailable, will use silent placeholder: {e}", flush=True)
        finally:
            _loaded = True


def available() -> bool:
    if config.TTS_DISABLED:
        return False
    _load()
    return _model is not None


def sample_rate():
    return _sr


def load_error():
    return str(_load_error) if _load_error else None


def synthesize(text: str, speaker: str, instruct: str, out_path: str,
               language: str = None) -> int:
    """Generate one narration clip as a 16-bit PCM WAV. Returns the sample rate.

    Raises if the model is unavailable or generation fails (callers handle it).
    """
    _load()
    if _model is None:
        raise RuntimeError(load_error() or "TTS model unavailable")
    import soundfile as sf

    global _sr
    wavs, sr = _model.generate_custom_voice(
        text=text,
        language=language or config.TTS_LANGUAGE,
        speaker=speaker or config.NARRATOR_SPEAKER,
        instruct=instruct or "",
    )
    _sr = sr
    sf.write(out_path, wavs[0], sr, subtype="PCM_16")
    return sr
