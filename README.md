# AI Storyteller — local MVP

Personalized bedtime stories for kids, written by a small crew of AI agents and
(eventually) read aloud with consistent per-character AI voices.

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
- **Narration = the validated recipe** (PLAN.md §1b, kid-approved 2026-06-27):
  - **Narrator**: consecutive narrator lines are merged into smooth clips and
    voiced by **cloning** the gravelly reference clip
    (`data/vd_british_gravelly.wav`, tracked in git) with the **Base** model —
    reproducible, every story.
  - **Characters**: each line is voiced with **VoiceDesign** from the
    character's `voice_description` + the line's emotion — best emotion, slight
    timbre drift accepted.
  - **Explicit presets** (characters you created and pinned to a preset voice)
    use **CustomVoice**. Auto-assigned presets are English-only (Aiden/Ryan).
  - Clips render batched by sub-model (only one fits in 8 GB VRAM at a time),
    then concatenate in story order with ~0.28 s gaps. Without the models
    installed, narration falls back to silent clips (lines marked `silent`,
    not `done`) so the app still runs.
- **Library**: past stories with summaries (fed back in for continuity); errored
  stories show a **Retry** button.
- **Job queue**: one background worker builds stories FIFO (the LLM and TTS GPU
  can't take concurrent builds). On startup, never-started stories are requeued
  and interrupted ones are marked as errors (retryable).

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
| `NARRATOR_SPEAKER` | `Aiden` | preset fallback narrator voice |
| `NARRATOR_REF_AUDIO` | `data/vd_british_gravelly.wav` | narrator clone reference clip |
| `NARRATOR_REF_TEXT` | (its transcript) | reference transcript for cloning |
| `NARRATOR_MERGE_CHARS` | `380` | soft cap when merging narrator lines into one clip |
| `TTS_GAP_SECONDS` | `0.28` | silence between clips in the final audio |
| `TTS_MODEL` / `TTS_MODEL_DESIGN` / `TTS_MODEL_BASE` | Qwen3-TTS 1.7B trio | CustomVoice / VoiceDesign / Base model IDs |

## Layout

```
backend/
  config.py      paths, models, preset speakers, narration recipe knobs
  db.py          SQLite schema + data access
  generation.py  writers' room + scriptify + recipe renderer (queued job runner)
  main.py        FastAPI API + static GUI server
  tts.py         Qwen3-TTS manager (CustomVoice / VoiceDesign / Base, swapped in VRAM)
frontend/        index.html, styles.css, app.js
tests/           unit tests — run: py -m unittest discover -s tests
data/            sqlite db + audio (gitignored except the narrator reference clip)
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

The first story generated afterward downloads the models it needs (~4.5 GB each,
cached): **Base** (narrator clone), **VoiceDesign** (characters), and
**CustomVoice** (explicit presets). They load **FP16** on **GPU 1** automatically
(`backend/tts.py`), one at a time (8 GB card), leaving GPU 0 for Ollama.

Turing notes (RTX 2080 = sm75): we use `dtype=float16` (no BF16) and
`attn_implementation="sdpa"` (FlashAttention 2 needs Ampere+); do **not** install
flash-attn. Override placement with `TTS_DEVICE=cuda:0`; force the silent
placeholder with `TTS_DISABLED=1`. Check `/api/tts/status` to see why narration
is (or isn't) real audio.
