# AI Storyteller — local MVP

Personalized bedtime stories for kids, written by a small crew of AI agents and
read aloud with consistent per-character AI voices.

## What works today

- **GUI** (vanilla HTML/JS): enter the child, story details, a **genre**
  (Fluffy / Fantasy / Sci-Fi / Suspense / Horror) and a **content-rating ceiling**
  (G / PG / PG-13 / R), pick a cast, press **Build Tonight's Story**, watch a
  **readiness bar**, then **play** + read the script. (Genre sets style, rating
  sets the intensity ceiling — e.g. Horror+G is charming-spooky, Horror+PG-13 is
  genuinely eerie.)
- **Writers' room** (local, via Ollama): **Writer → QA/QC (iterates) → Manager
  (final say on age-rating + quality) → 🎭 Casting**.
- **Casting / Character Designer agent**: detects significant new characters the
  story introduced (not already in the roster), designs each (persona,
  motivations, backstory, voice description), maps the voice to a Qwen preset,
  and **persists** them — so they're reused with the **same voice** in every
  future story.
- **Playwright script**: the approved story is split into ordered lines, each with
  a **character, emotion, and resolved voice** (saved characters keep the same
  voice every story).
- **Per-line narration + concatenation**: each line becomes a clip, clips are
  joined in order into one audio file.
- **Library**: past stories with summaries (fed back in for continuity).

- **Real narration** via local **Qwen3-TTS 1.7B** (after one-time setup, below):
  each line is voiced with its character's speaker + emotion, then clips are
  concatenated. Without the model installed, narration falls back to silent clips
  so the app still runs.

## Requirements

- **Python 3.12** (`py` launcher) — installs `fastapi`, `uvicorn`, `requests`.
- **Ollama** running locally with a model. Default: `qwen2.5:7b-instruct`
  (already installed on this machine). Override with `OLLAMA_MODEL`.

## Run

```bat
run.bat
```

Then open <http://localhost:8000>. (Or manually: `py -m pip install -r requirements.txt` then `py backend\main.py`.)

## Config (env vars)

| Var | Default | Purpose |
|-----|---------|---------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | story-writing model |
| `STORY_TARGET_WORDS` | `1500` | ~5 min read |
| `MAX_QA_ROUNDS` | `2` | writer↔QA revision rounds |
| `NARRATOR_SPEAKER` | `Aiden` | default narration voice |

## Layout

```
backend/
  config.py      paths, model, preset speakers
  db.py          SQLite schema + data access
  generation.py  writers' room + scriptify + per-line narration (job runner)
  main.py        FastAPI API + static GUI server
frontend/        index.html, styles.css, app.js
data/            sqlite db + audio (gitignored, created on first run)
```

## Data model (key tables)

- `subjects` — the child (name, age).
- `characters` — reusable cast + **voice** fields (`voice_mode` preset|clone,
  `speaker`, `instruct`, `ref_audio_path`, `ref_text`) so a character sounds the
  same every appearance.
- `stories` — params, full text, summary, status/progress, final audio.
- `story_lines` — the **playwright script**: per line `character`, `emotion`,
  `text`, resolved `speaker` + `voice_instruct`, per-line `audio_path`.

## Narration setup (Qwen3-TTS 1.7B)

One-time install of PyTorch (CUDA), `qwen-tts`, and `soundfile`:

```bat
setup-tts.bat
```

The first story generated afterward downloads the ~4.5 GB model (cached). It loads
**FP16** on **GPU 1** automatically (`backend/tts.py`), leaving GPU 0 for Ollama.

Turing notes (RTX 2080 = sm75): we use `dtype=float16` (no BF16) and
`attn_implementation="sdpa"` (FlashAttention 2 needs Ampere+); do **not** install
flash-attn. Override placement with `TTS_DEVICE=cuda:0`; force the silent
placeholder with `TTS_DISABLED=1`.

Voices: characters map to Qwen's 9 presets. For a warm *female English* narrator,
use voice **cloning** (no English-female preset exists) — the `clone` voice fields
are already in the schema (future wiring).
