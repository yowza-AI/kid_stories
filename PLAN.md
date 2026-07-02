# AI Storyteller — Voice Stack & Plan

**Last updated:** 2026-06-27

A personalized bedtime-story app: a frontier LLM writes the story, AI voices narrate it
with emotion, an optional music bed plays underneath, and parents buy it on subscription.
This doc captures the **voice decisions** (Zonos2 + Qwen3-TTS-with-emotion) and the
**overall plan** so we don't forget what we landed on.

---

## 1. Voice / TTS — what we landed on

### 1a. Zonos2 (Zyphra) — the PRODUCTION voice (later, not now)
- **Highest quality voice we evaluated** (emotion + voice cloning in one model). This is the
  target for the premium voice tier.
- **Will NOT run on our hardware.** Needs **Ampere or newer** (RTX 30-series+). Our RTX 2080
  SUPERs are Turing (sm75) and lack the Mamba/bf16 kernels Zonos2 requires — it can't run
  *at any VRAM* on these cards. The model itself is light (~6–10 GB), so the blocker is the
  **architecture, not memory**.
- **Online HF Space** (`multimodalart/ZONOS2`) is **blocked for anonymous API use**: each
  `/generate` call reserves ~315 s of Zero-GPU time, above the anonymous cap (and `accurate_mode`
  / `max_seconds` don't lower it). To use the Space you need an **HF PRO token** (set `HF_TOKEN`)
  or to **duplicate the Space** into your own account. Viable manually at low (concierge) volume;
  not for scale.
- **Decision:** Zonos2 is the **production** voice, run on an **Ampere+ GPU** (own card or cloud).
  Treat it as "best we've heard *so far*" — re-A/B against current models when we provision
  production GPUs (this space moves fast).

### 1b. Qwen3-TTS with emotion — the voice that WORKS NOW (validated)
Runs on the 2080s today (fp16 + `attn_implementation="sdpa"`, NO bf16/flash-attn on Turing).
Qwen3-TTS ships **three sub-models**, each with different powers:

| Sub-model | Power | Reproducible? | Emotion? |
|---|---|---|---|
| **CustomVoice** | 9 fixed preset speakers; `instruct` sets emotion per line | ✅ yes | ✅ via `instruct` |
| **VoiceDesign** | free-text "describe the voice + emotion" → speech | ❌ no (drifts per call) | ✅✅ best emotion |
| **Base** | clone a voice from a reference clip (+ ref text) | ✅ yes | ❌ none (copies the reference's delivery) |

**THE WINNING RECIPE (validated 2026-06-27 — Steve's kids got genuinely creeped out by it):**

- **Narrator = gravelly voice cloned via the Base model.**
  - Reference clip: `data/vd_british_gravelly.wav` (made once with VoiceDesign).
  - `ref_text` = "And here, beneath the silver moon, a tiny dragon settles down, gently, into a peaceful sleep."
  - Steady, ominous, gravelly tone. **No per-line emotion** (clone copies the reference's
    delivery) — and that's *fine* for narration; the calm/ominous tone fits.
- **Characters = per-line VoiceDesign with a rich emotion description.**
  - e.g. *"A young girl about ten, her voice high and trembling, breathless and terrified."*
  - This is what finally **sold the emotion**. In an A/B/C test (preset+weak instruct vs.
    preset+strong instruct vs. VoiceDesign) the user picked **VoiceDesign ("C") across the
    board** — describing the character + emotion in free text beats nudging a fixed preset.
  - **Trade-off accepted:** VoiceDesign isn't perfectly reproducible, so a character's voice
    can drift slightly between their lines. For *expressive* characters, **emotion > consistency**.

**Rule of thumb:**
- Need **per-line emotion** (expressive characters) → **VoiceDesign per line**.
- Need **consistency** (a recurring narrator/character) → **design once, then Base-clone** that clip.
  (The clone is reproducible but flatter — no emotion knob.)

**Render mechanics** — **productized 2026-07-02**: this recipe now lives in the
app (`backend/tts.py` sub-model manager + `backend/generation.py` renderer; the
original `scratchpad/render_frontier_v2.py` was lost — scratchpads are ephemeral,
which is why the recipe was moved into tracked code). The reference clip
`data/vd_british_gravelly.wav` is now tracked in git. Steps as implemented:
1. Build the line list (`role = N` narrator, else character + emotion).
2. Merge consecutive **narrator** lines into clips (soft cap ~380 chars) for smooth narration;
   each **character** line stays solo (preserves its per-line emotion instruct).
3. **Pass A** — load **Base**, render all narrator clips via `generate_voice_clone(...)`.
4. **Pass B** — unload Base, load **VoiceDesign**, render all character clips via
   `generate_voice_design(text, language, instruct)`.
5. Concatenate in order with ~0.28 s gaps between clips.
- Two models loaded **sequentially** on one GPU (cuda:1) to stay within 8 GB.
- Speed on the 2080: ~30 min for a ~10-min story; the **clone narrator is the slow part**
  (~3–4× the preset model). Fine for a generate-ahead-of-bedtime queue.

**Known voice caveats:**
- Only **Aiden** and **Ryan** are native-English CustomVoice presets (both male). Female/other
  presets are non-English-native → accented English. (VoiceDesign sidesteps this — describe the voice.)
- Character voice **drift** between lines (the VoiceDesign trade-off).
- The old `scratchpad/` test scripts (`render_frontier_v2.py`, `character_emotion_test.py`,
  `render_clone.py`, `zonos_test.py`) are GONE (ephemeral scratchpad). Their findings are
  captured above; the render recipe itself is now tracked code (see HANDOFF.md 2026-07-02).

### 1c. Other voice models tried (for the record)
- **OmniVoice** (Apache): real British accent + runs great on the 2080 (~2 GB, ~2 s/clip) but
  **no emotion** → flat. Good fallback for a calm British narrator.
- **Kokoro** (Apache): nice British named voices, reproducible, **no emotion/design**.
- **Conclusion:** across open models, **British accent and emotion control are mutually exclusive** —
  except via design/clone tricks. Qwen VoiceDesign gives the best *emotion*; Zonos2 (Ampere) gives
  the best *everything* in one model.

---

## 2. Music — what we landed on
- **Don't generate music per story.** Background beds are low-variety; a subtle orchestral bed
  doesn't need to be unique per story.
- **Plan: a curated library of mood beds** (cozy / adventurous / spooky / calm-wind-down),
  reused, mixed as a **subtle bed at ~10% of voice RMS** (the level the user A/B-approved).
- **Source the library from licensed stock** (Epidemic Sound / Artlist / Soundstripe, ~$10–30/mo):
  clean commercial license + good quality + **no AI-music copyright cloud** (there's active
  litigation over Suno/Udio training data — avoid that risk for a kids' product).
- Open-weight music gen is "meh" right now: **MusicGen** = non-commercial (CC-BY-NC, prototype only);
  **ACE-Step** = Apache + commercial OK but only-okay quality and needs Ampere. Use ACE-Step on a
  production GPU only if we want zero-marginal music at scale. Best *generative* quality lives in
  closed APIs (Suno / Udio / ElevenLabs Music / Google Lyria) — use those to build the library once,
  not per story, and verify commercial terms.

---

## 3. Story-generation architecture (the plan)
Hierarchical plan → cheap execution, with swappable LLM backends:

```
FRONTIER PLANNER  → Story Bible (characters, world facts, voices, the arc's payoff)
                  → Chapter map (per night: theme, goal, tension, gentle ending)
                  → Scene beats (2–4 per chapter: what happens, who, the emotional turn)
        │
        ▼
WRITER (tier depends on product tier)  writes each scene
        │  ← gets: scene beats + Story Bible + running summary/recap (NOT the whole story)
        ▼
QA → Manager (rating ceiling) → Casting → deterministic scriptify → TTS (§1b) → music bed (§2)
```

- **The two pieces people skip (build them):** a **Story Bible** + a **running summary** fed into
  every scene-write call. This is what keeps a smaller writer coherent across scenes and across nights.
- **Tune the writer model to the product tier** (premium = frontier/mid writer; budget = local 7B).
  Don't reflexively drop to the 7B — **text is the cheap part of the stack.**
- **Swappable LLM backend** behind one flag: frontier API ↔ self-hosted 32B reasoner ↔ local 7B.
- This is a natural evolution of the existing `Planner → Writer → QA → Manager → Casting` pipeline
  (`backend/generation.py`). Scriptify is already deterministic regex (`_scriptify`).

---

## 4. Cost & economics (the reality)
- **Per story (frontier text + Zonos2 voice + music, marginal compute):** ~$0.50
  (text $0.15–0.50, TTS GPU $0.10–0.40, music ~$0.03). → ~**$15/mo for a DAILY user**.
- **Text is cheap even at frontier (~$0.05/story); the GPU (voice) is the cost driver.**
  → Two moats: **writing is the nearly-free moat; voice is the expensive moat.**
- **Self-hosting GPU (TTS / music / LLM) only beats APIs at HIGH utilization.** The batched
  **generate-ahead-of-bedtime queue + spot instances + scale-to-zero** is the design that makes
  self-hosting pay. An idle dedicated GPU is worse than the API.
- **DeepSeek-R1 self-hosted** needs a full 8×80 GB node (~$10–72k/mo) — absurd here. Use **R1 via
  API** (pennies) or self-host a **32B reasoner** (QwQ-32B / Qwen3-32B-thinking) on one GPU.

---

## 5. Positioning (the plan)
- **Do NOT race to the $4.99 commodity tier.** Competitors hit $4.99 with cheap components
  (small/templated LLM, commodity TTS, no music) + low average usage + the app-store cut leaving
  only ~$3.50–4.25 net. A solo builder can't out-scale incumbents on cost.
- **Win on quality = premium tier ($9.99–14.99):** genuinely good frontier writing + emotional/
  cloned premium voice + music + multi-night arcs + human-reviewed safety. Moat = **trust/safety +
  voice quality**, not price.
- **Validation (2026-06-27):** Steve's kids ask non-stop for new stories. His 10-year-old called the
  local-7B story "dumb… sounds terrible" (writing AND voice), but got **creeped out** (engaged) by the
  frontier story + emotional voices. Quality proven on **both axes** — and the axes are
  **multiplicative**: a great story in a bad voice still fails, and vice-versa. Must clear both.
  → Kids = a free, brutally honest QA oracle. Rule: **if it doesn't pass the 10-year-old, it doesn't ship.**

---

## 6. Go-to-market ladder (no skipping rungs)
1. **Concierge / Wizard-of-Oz MVP** — human (Steve) in the loop:
   - App collects a **request packet** (names, subject, themes, rating, genre, length, night#).
   - Steve pastes packet + an **instructions file** into the **flat-fee ChatGPT/Claude web UI**
     (manual = near-$0 text, ToS-clean, doubles as the human safety review = a trust feature).
   - Runs the voice pipeline (§1b) → publishes audio + text to the app DB.
   - Build a thin **operator console**: queue → one-click **Build-Prompt** → **paste-back** script →
     **Render & Publish**. (It's the automation scaffold, not throwaway.)
   - Set a submission cutoff/SLA ("order by 3 pm for tonight").
2. **Graduate to an API worker** when manual hurts (~>15 orders/day): batch API (~50% off) +
   prompt caching on the fixed instructions file → pennies/story, fully automated, stable, ToS-clean.

**REJECTED: browser-automation / multi-account farm against the consumer UI.** Explicit ToS
violation, actively banned, existential supply risk for a paid kids' product, brittle, and a false
economy (risks 100% of the business to save ~3% on the cheapest component). **The API is the
legitimate version of "automate it and walk away."**

---

## 7. Hardware path
- **Now (2080s, Turing):** can't run Zonos2 or big LLMs (no bf16/Mamba). Runs the Qwen3-TTS stack (§1b)
  and a small local LLM only.
- **Cheapest unblock for dev:** a **used RTX 3090 (24 GB, ~$800)** — runs Zonos2 + a 32B reasoner
  (4-bit) + TTS on one card. Highest-leverage hardware move.
- **Serious local "do-everything" box (many AI projects, incl. training):**
  **RTX PRO 6000 Blackwell (96 GB GDDR7, ~$7.5–9k)** — ~4× compute and ~6.5× memory-bandwidth of a
  **DGX Spark**, and even better perf-per-dollar; a real workstation card (active-cooled, display out).
  - **DGX Spark (128 GB unified, ~$3–4k)** only wins on memory-capacity-per-dollar; ~273 GB/s
    bandwidth makes it slow for inference and unsuitable for training. Buy it to *fit* big models
    cheaply, not to run them fast. Sweet spot = MoE inference.
  - Decision driver: **training/fine-tuning/gen → PRO 6000; light inference only → 24 GB card or Spark.**
- **Production voice:** Zonos2 on a **cloud Ampere GPU** (g6/g5, ~$0.80–1.00/hr), batched + spot +
  scale-to-zero.

---

## 8. Open / next
- Build the **concierge MVP**: request packet format + instructions file + operator console.
- Wire **swappable planner/writer backends** (frontier API ↔ self-hosted reasoner ↔ local 7B).
- Build the **unit-economics model** (usage distribution, tier mix, GPU utilization, app-store cut).
- Buy a **24 GB Ampere card** to unblock Zonos2/reasoner locally (or the PRO 6000 if going all-in).
- Stand up the **music bed library** (license stock, tag by mood, mix at ~10% RMS).
- Build the **multi-night (up to 14) feature** on the hierarchical-plan architecture (§3).
