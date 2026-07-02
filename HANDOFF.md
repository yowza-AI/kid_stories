# HANDOFF — change log & gotchas

Terse trail of behavior changes: what changed, why, gotchas. Newest first.

## 2026-07-02 — Productize the validated voice recipe + reliability pass

**What changed**

- **Winning narration recipe is now in the product** (was only prose in PLAN.md §1b
  + lost scratchpad scripts):
  - `backend/tts.py` manages all three Qwen3-TTS sub-models (CustomVoice /
    VoiceDesign / Base), swapping them sequentially in VRAM (one fits on the
    8 GB card). All synthesis serialized behind one lock.
  - `backend/generation.py` renders: narrator runs merged into ~380-char clips →
    **Base clone** of `data/vd_british_gravelly.wav`; character lines solo →
    **VoiceDesign** (`voice_description` + per-line emotion); explicit
    user-pinned presets → CustomVoice. Clips render batched by sub-model
    (avoids reload thrashing), concat in story order with 0.28 s gaps.
  - The narrator reference clip is now **tracked in git** (`git add -f`, it's
    460 KB) — it was double-gitignored and irreplaceable.
- **Child hero gets a designed voice** (bright warm child, age-aware) instead of
  a hash-picked preset; casting ignores the hero so nothing else covered them.
- **Auto-assigned presets are English-only** (Aiden/Ryan). Non-English presets
  reading English sound accented — the failure the kids panned. User-selected
  presets are unrestricted.
- **Single-worker job queue** (`generation.start_story_job`): POST /api/stories
  enqueues; one worker builds FIFO. The LLM + TTS GPU can't take concurrent builds.
- **Startup recovery** (`generation.recover_interrupted`, called from FastAPI
  lifespan): `queued` stories are requeued; mid-flight ones are marked error
  ("Interrupted by a server restart — press Retry"). Previously they spun forever.
- **Retry**: `POST /api/stories/{id}/retry` (only from `error` state) + a Retry
  button on errored library items.
- **Silent fallbacks are visible**: lines whose clip fell back to silence get
  `status='silent'` (not `'done'`), so failed lines are findable/re-renderable.
- **Concat is sample-rate-safe**: `_concat_wavs` converts every clip to the first
  clip's rate/format (stdlib `audioop.ratecv`) and inserts gaps. Previously mixed
  22.05 k silence + 24 k TTS clips produced speed-shifted audio.
- **`/api/tts/status`** endpoint + live note in the GUI (why narration is/isn't real).
- **Schema**: `story_lines.voice_mode` ('clone'|'design'|'preset') added with the
  lightweight `_ensure_columns` migration; line status can now be `silent`.
- **Tests**: `tests/test_generation.py`, 25 unit tests over scriptify/attribution/
  merge-groups/concat/voice-resolution. Run: `py -m unittest discover -s tests`.
- **requirements.txt pinned** (fastapi 0.138.1, uvicorn 0.49.0, requests 2.32.5);
  deprecated `@app.on_event` replaced with lifespan.

**Gotchas**

- `audioop` is deprecated stdlib (removed in Python 3.13). Fine on the pinned
  Py 3.12; swap for numpy/soundfile if the runtime moves.
- VoiceDesign is NOT seed-reproducible — character timbre drifts slightly between
  lines. Accepted trade-off (emotion > consistency, per the A/B/C bake-off).
- A clone-mode line whose reference clip goes missing at render time falls back
  to CustomVoice mid-batch, which loads a second model (slow, not wrong).
- Merged narrator lines share one clip file; `story_lines.audio_path` is
  therefore not unique per line.
- Retry re-runs the WHOLE pipeline (new text too, since the draft may be the
  failure); it does not resume from the failed stage.
