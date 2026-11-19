# HANDOFF — change log & gotchas

Terse trail of behavior changes: what changed, why, gotchas. Newest first.

## 2026-08-07 — Prosody: gap length now depends on what the boundary means

**What changed**

- **Three gap tiers instead of one flat 0.28 s** (`config.TTS_GAP_TAG` 0.12 /
  `TTS_GAP_SECONDS` 0.28 / `TTS_GAP_PARA` 0.5), chosen per join by
  `generation._gaps_for_groups`:
  - **TAG** — a character line followed by its own trailing speech tag. That is
    ONE sentence in two voices and must not sound like two utterances.
  - **LINE** — the validated 0.28 s, between lines inside one paragraph.
  - **PARA** — across a paragraph break: one thought ending, the next starting.
- **Paragraph structure now survives `_scriptify`.** Lines carry `para` (source
  paragraph index) and `is_tag`. Nothing downstream could recover this — the
  splitter used to flatten the prose and the boundary was gone for good.
- **`_merge_render_groups` no longer merges narrator lines across a paragraph
  break** — merging swallowed the boundary into one clip and made the pause
  unreachable.
- **`_concat_wavs(paths, out, gaps=[...])`** takes a per-boundary list; the flat
  `gap_seconds` remains the fallback. Gap indices travel with the clip, so a
  clip that failed to render cannot shift every later gap by one.
- **Schema**: `story_lines.para`, `story_lines.is_tag` via `_ensure_columns`.
- **Tests**: 9 new (34 total). `py -m unittest discover -s tests`.

**Gotchas**

- Stories written before this migrate to `para=0` on every line, so they get no
  PARA gaps and render exactly as they did. Only re-scriptified stories improve.
- `_scriptify` still DROPS bare speech tags of <=6 words, but a tag carrying an
  action beat is longer than that, survives, and becomes a real narrator clip.
  So "richer tags" and "tags must not interrupt a speech" are the same rule:
  the richer the tag, the worse a mid-speech one sounds.
- The tight TAG gap only applies when a CHARACTER spoke before the tag. A tag
  after narration is ordinary narration and gets the normal gap.

## 2026-08-07 — Speaker attribution rewritten; mid-speech tags reflowed

Found while building the gap tiers: `_scriptify` mis-cast most dialogue. On a
real chapter Phillip got 25 of 29 spoken lines, Nessa 1, and Ilsabet Quorn was
split into two characters cast as two different voices in one scene. Silent in
the text, ruinous in audio — the whole cast in the hero's voice.

**What changed** (`generation.py`)

- **Pronoun tags resolve.** `"Vane," she said` now attributes to the last person
  named in narration, so the invigilator keeps her own line.
- **Forename and surname unify** (`_alias_map`): Ilsabet / Quorn are one person.
  A surname shared by two characters (Phillip Vane, Tobin Vane) is deliberately
  left unmapped — folding it would put a father's lines in his son's voice.
- **Untagged exchanges alternate** instead of sticking to `last_speaker`, using
  everyone *present in the scene*, not only those who have already spoken — a
  character can be named, addressed, and then answer.
- **Direct address is not speech.** "Nessa —" is Phillip talking to Nessa, and
  "It is not praise, Mr Vane" cannot be Vane. Checked at a line's start and
  after any comma.
- **Only people become speakers** (`_cast_names`): a name qualifies via the
  story's cast, a speech verb, a title, or being the subject of an action
  sentence. Titles are stripped, never optional — an optional group backtracks,
  so "Mr Vane" parsed as a person called Mr. Place names built like people
  names ("the Tally Floor goes silent") no longer join the cast.
- **`_reflow_tags`** lifts a speech tag out from between two halves of one
  speech and rejoins the speech, including when the tag was dropped for being
  short. Arc-wide this cut character clips 321 -> 260: fewer renders, less GPU,
  and no sentence split across two non-reproducible VoiceDesign calls.
- `_scriptify(draft, known_names=...)` — `_run_job` passes the known cast.
- **Tests**: 15 new (49 total).

**Verification**

Session 1 of the arc, checked line by line against the prose: 15/15 spoken lines
correct, from 2/13 before. Across all seven sessions only 3 lines out of ~260
still resolve to scenery rather than a person.

**Still wrong**

- Nicknames are not unified: `Odalric "Odd" Pell` yields both "Odd" and
  "Odalric" (and once "Pell") — three voices for one boy. Needs an alias field
  on `characters`, or the nickname form parsed out of the prose.
- Chapters are scriptified independently, so an alias only holds if the full
  name appears in *that* chapter. In the app `known_names` covers this; scanning
  a loose file does not.
- Attribution is verified line-by-line for session 1 only. The other six are
  much improved and spot-checked, not audited.

## 2026-08-07 — First multi-night arc, written outside the app (PLAN §3 / §8 dry run)

**What changed**

- **`content/arcs/phillip-dragonhall/` — a 7-session arc, text only.** *The Ledger of
  Ninefold*: Phillip Vane, 15, high fantasy, PG-13. ~15.5k words of prose plus a
  Story Bible and a chapter map. Voices deliberately deferred.
- **Built by a Claude multi-agent workflow, NOT by `backend/generation.py`.** Stages:
  Story Bible → chapter map → 7 chapter writers in parallel → hostile continuity
  audit → revision. The app's Ollama pipeline was not involved at any point.
- **`content/` is a new TRACKED directory.** Written deliverables must not go in
  `data/` — it is gitignored (see the lost-scratchpad entry below; same failure mode).
- **PLAN §3's architecture is validated in practice.** The Story Bible + a chapter map
  carrying an explicit per-session **end-state** is what let 7 writers run in
  PARALLEL and still chain: session N's end-state is session N+1's opening
  condition, which does the job a sequential running summary would. This is the
  design to port into the app for the multi-night feature (PLAN §8).
- **The audit stage earns its keep**: 23 defects across 7 chapters (2 blocking,
  10 major, 11 minor) — parallel writers really do contradict each other. All
  blocking/major fixed by revisers; the mechanical minors fixed by hand after.

**Gotchas**

- The arc is written to the conventions `_scriptify` expects (named speakers,
  ~⅔ narration, no headings/markdown inside the prose), so the voice pass should
  consume it without a rewrite. Untested — no TTS has been run against it.
- **The app cannot produce this today.** There is no Claude/frontier backend in
  `backend/generation.py` (`config.OLLAMA_MODEL` is the only text backend), and no
  schema for series/chapters — `stories` has no arc or sequence column. This arc
  lives only as markdown on disk; it is not in the DB.
- A reviser told to fix a seam by editing a NEIGHBOURING chapter can't — each
  reviser is scoped to its own file. The ch3 seam was closed from the ch3 side
  instead. Fine here, but cross-file fixes need a different stage shape.
- Three audit findings were left unapplied on purpose because they add new
  material rather than fix a contradiction (Odd's sponsorship payoff, a daylight
  "people, filed" scene before the Ledger Houses, an S6 callback to Odd's
  "third of next month"). They are authorial calls, not defects.

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
