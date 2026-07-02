"""Story generation pipeline (runs in a background thread).

Writers' room (all local via Ollama), then narration:

  1. WRITER   - creative draft: fun, adventurous, imaginative.
  2. QA/QC    - checks continuity with past stories + internal cohesion;
                iterates with the writer up to MAX_QA_ROUNDS.
  3. MANAGER  - final say: age-appropriateness (no mature themes, profanity,
                violence, scares) and overall quality. Can send back once.
  4. SCRIPTIFY- converts the approved story into a playwright script:
                an ordered list of {character, emotion, line} (incl. Narrator).
  5. NARRATE  - the validated recipe (PLAN.md §1b, kid-approved 2026-06-27):
                * Narrator lines are merged into smooth clips and voiced by
                  CLONING the gravelly reference clip (Base model) — steady,
                  reproducible, every story.
                * Character lines stay solo and are voiced per line with
                  VoiceDesign (voice_description + emotion as free text) —
                  emotion beats timbre consistency for expressive characters.
                * Explicit user-picked presets (and fallbacks) use CustomVoice.
                Clips render batched by sub-model (one fits in VRAM at a time),
                then concatenate in story order with short gaps.

Jobs run on a SINGLE background worker (queue) — the LLM and the TTS GPU can't
take concurrent builds. If TTS is unavailable, clips fall back to silence and
the line is marked 'silent' (not 'done') so failures are visible and re-renderable.
"""
import json
import os
import queue
import re
import threading
import wave

# stdlib audio resampling; deprecated upstream (removed in Python 3.13) but the
# project pins Python 3.12. Replace with numpy/soundfile if we move past 3.12.
import audioop

import requests

import config
import db
import tts

# --------------------------- agent system prompts ------------------------- #
PLANNER_SYS = (
    "You are a story architect for children's bedtime stories. Design a "
    "COMPELLING, original plot — imaginative and specific, never generic or "
    "predictable. Output a tight outline containing: a one-line LOGLINE; the "
    "SETTING/world; the HERO plus 1-3 supporting characters (each with a clear "
    "want); the central PROBLEM or quest with real stakes; 4-6 STORY BEATS "
    "(setup, inciting incident, rising action, a turning point or clever twist, "
    "resolution); and a calm, reassuring WIND-DOWN to sleep. Weave in the "
    "child's interests meaningfully. Fit the requested genre and stay within "
    "the content rating. Output ONLY the outline."
)
WRITER_SYS = (
    "You are a wildly imaginative children's bedtime-story author. Make the "
    "story FUN, vivid, and creative in the requested GENRE and within the "
    "requested content RATING. Give it a REAL PLOT: a gentle problem or quest, "
    "a little rising action, and a satisfying resolution — then wind the ending "
    "down to a calm, comforting, sleepy close suitable for bedtime. The child "
    "is always the hero, and the story should feel cohesive from beginning to "
    "end. Tell it PRIMARILY through narration — roughly two-thirds narration — "
    "and use character dialogue sparingly (about one-third), only for key "
    "moments. Because the story is HEARD, not seen, the narration must paint "
    "RICH, VIVID descriptions of every setting — sights, sounds, smells, "
    "textures, light, and mood — so the child can picture the world in their "
    "imagination. Spend real narration on scene and atmosphere, not only action "
    "and dialogue. Use simple, sensory language. Output ONLY the flowing story "
    "prose to be read aloud — NO title, logline, headings, chapter or scene "
    "labels, markdown, asterisks, or section markers of any kind, and no notes "
    "or preamble."
)
QA_SYS = (
    "You are a meticulous story editor. Check the draft for (1) COHESION & "
    "CONTINUITY — internal consistency (plot, character behavior, names) and "
    "consistency with the summaries of past stories provided — and (2) whether "
    "it is COMPELLING: clear stakes, a satisfying arc, vivid and specific (not "
    "generic or flat). Respond with STRICT JSON only: "
    '{"cohesive": true|false, "feedback": "concrete, actionable notes"}. '
    "Set cohesive=false if it is inconsistent OR bland/generic; otherwise "
    "cohesive=true with feedback=\"\"."
)
MANAGER_SYS = (
    "You are the executive editor with FINAL SAY over a children's bedtime "
    "story. Enforce two things: (1) the story stays WITHIN the specified maximum "
    "content rating — it must NOT exceed it; and (2) high quality and clarity. "
    "Respond with STRICT JSON only: "
    '{"approved": true|false, "feedback": "what must change, if anything"}.'
)
SCRIPTIFY_SYS = (
    "You convert a children's story into a playwright-style script for audio "
    "narration. Output a JSON object {\"lines\": [ ... ]} where each item is "
    '{"character": "Narrator"|<character name>, "emotion": <one or two words>, '
    '"text": <the words to speak>}.\n'
    "RULES — follow exactly:\n"
    "1. DEFAULT every line to \"Narrator\". Narration, description, action, "
    "scene-setting, and speech tags (he said, she asked) are ALWAYS the "
    "Narrator — even when the sentence is about a character.\n"
    "2. Make a CHARACTER line ONLY for the exact words a character says aloud. "
    "Its text must contain ONLY those spoken words — no description, no action, "
    "no 'said X', and no quotation marks. Use the speaker's name.\n"
    "3. A source sentence that mixes narration with a quote becomes MULTIPLE "
    "lines: put the narration in Narrator line(s) and each spoken quote in its "
    "own character line. NEVER put narration and speech in the same line.\n"
    "4. Attribute each quote to the character who actually says it.\n"
    "5. Drop fragments that are only a speech tag (like 'said Leo'); omit any "
    "stray headings or symbols.\n"
    "Keep the story's wording and order; split into natural sentence-level "
    "lines. Give each line a fitting emotion."
)
CASTING_SYS = (
    "You are a casting director and character designer for children's bedtime "
    "stories. Given a finished story and the characters that ALREADY exist, "
    "identify any SIGNIFICANT new characters that appear in the story but are "
    "NOT already in that list. Ignore the child hero and the Narrator, and "
    "ignore trivial one-off mentions. Design each new character; everything "
    "must be gentle and age-appropriate. Respond with a JSON object of the form "
    '{"characters": [ ... ]} where each item is '
    '{"name": <as used in the story>, "gender": "male"|"female"|"unspecified", '
    '"persona": <one warm sentence>, "motivations": <what they want>, '
    '"backstory": <1-2 sentences>, "voice_description": <timbre words, e.g. '
    '"gruff, low, kindly older male" or "bright, playful young female">}. '
    "Use an empty characters array if there are no new characters."
)


# --- JSON schemas that force array-shaped responses (Ollama structured output) ---
_LINE_ITEM = {
    "type": "object",
    "properties": {
        "character": {"type": "string"},
        "emotion": {"type": "string"},
        "text": {"type": "string"},
    },
    "required": ["character", "emotion", "text"],
}
SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {"lines": {"type": "array", "items": _LINE_ITEM}},
    "required": ["lines"],
}
_CHAR_ITEM = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "gender": {"type": "string"},
        "persona": {"type": "string"},
        "motivations": {"type": "string"},
        "backstory": {"type": "string"},
        "voice_description": {"type": "string"},
    },
    "required": ["name"],
}
CAST_SCHEMA = {
    "type": "object",
    "properties": {"characters": {"type": "array", "items": _CHAR_ITEM}},
    "required": ["characters"],
}


# ------------------------------- Ollama I/O ------------------------------- #
def _ollama_chat(messages: list, temperature: float = 0.8, num_predict: int = 3500,
                 json_mode: bool = False, schema: dict = None) -> str:
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    if schema is not None:
        payload["format"] = schema     # JSON schema → forces exact structure (e.g. an array)
    elif json_mode:
        payload["format"] = "json"     # valid JSON (good for single-object responses)
    resp = requests.post(f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=900)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def _json_obj(raw: str, default: dict) -> dict:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    for c in (raw, m.group(0) if m else None):
        if not c:
            continue
        try:
            data = json.loads(c)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return default


def _json_arr(raw: str):
    """Return a JSON array from `raw`, tolerating object-wrapped arrays (json mode)."""
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    for c in (raw, m.group(0) if m else None):
        if not c:
            continue
        try:
            data = json.loads(c)
        except Exception:
            continue
        if isinstance(data, list):
            return data
        if isinstance(data, dict):  # e.g. {"lines": [...]}
            for v in data.values():
                if isinstance(v, list):
                    return v
    return None


# ----------------------------- prompt context ----------------------------- #
def _story_context(subject, story, characters, prior_summaries) -> str:
    name = (subject or {}).get("name") or "the child"
    age = (subject or {}).get("age")
    rating = story.get("rating") or "G"
    guide = config.RATING_GUIDE.get(rating, config.RATING_GUIDE["G"])
    genre = story.get("genre") or "Fluffy"
    gguide = config.GENRE_GUIDE.get(genre, config.GENRE_GUIDE["Fluffy"])
    parts = [
        f"Write a bedtime story for {name}, who is the hero." +
        (f" {name} is {age} years old." if age else ""),
        f"Target length: about {config.STORY_TARGET_WORDS} words (~5 minutes read aloud).",
        f"Genre: {genre} — {gguide}",
        f"Maximum content rating: {rating} — {guide} Keep all content within this rating.",
    ]
    if story.get("location"):
        parts.append(f"Setting: {story['location']}.")
    if story.get("themes"):
        parts.append("Themes: " + ", ".join(story["themes"]) + ".")
    if story.get("interests"):
        parts.append(f"{name} loves: " + ", ".join(story["interests"]) + ".")
    if story.get("extra_notes"):
        parts.append("Parent's notes: " + story["extra_notes"])
    if characters:
        lines = ["Recurring characters (keep them true to their persona):"]
        for ch in characters:
            lines.append(f"- {ch['name']}: {ch.get('persona') or 'a friendly companion'}")
        parts.append("\n".join(lines))
    if prior_summaries:
        lines = ["Recent past stories (for continuity; do not repeat):"]
        for s in prior_summaries:
            lines.append(f"- {s.get('title') or 'Untitled'}: {s.get('summary')}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


# ------------------------------- the agents ------------------------------- #
def _planner(context: str) -> str:
    return _ollama_chat(
        [{"role": "system", "content": PLANNER_SYS}, {"role": "user", "content": context}],
        temperature=0.85, num_predict=1200,
    )


def _writer(context: str, outline: str = None, prev_draft: str = None, feedback: str = None) -> str:
    user = context
    if outline:
        user = context + "\n\nWrite the full story following this outline:\n" + outline
    if prev_draft and feedback:
        user = (
            user
            + "\n\nHere is your previous draft:\n" + prev_draft
            + "\n\nRevise it to address this feedback while keeping it compelling "
              "and faithful to the outline:\n" + feedback
        )
    return _ollama_chat(
        [{"role": "system", "content": WRITER_SYS}, {"role": "user", "content": user}],
        temperature=0.9,
    )


def _qa(draft: str, prior_summaries: list) -> dict:
    prior = "\n".join(
        f"- {s.get('title')}: {s.get('summary')}" for s in prior_summaries
    ) or "(none)"
    raw = _ollama_chat(
        [
            {"role": "system", "content": QA_SYS},
            {"role": "user", "content":
                f"PAST STORY SUMMARIES:\n{prior}\n\nDRAFT:\n{draft}"},
        ],
        temperature=0.2, num_predict=600, json_mode=True,
    )
    return _json_obj(raw, {"cohesive": True, "feedback": ""})


def _manager(draft: str, rating: str = "G") -> dict:
    guide = config.RATING_GUIDE.get(rating, config.RATING_GUIDE["G"])
    raw = _ollama_chat(
        [
            {"role": "system", "content": MANAGER_SYS},
            {"role": "user", "content":
                f"MAXIMUM CONTENT RATING: {rating} — {guide}\n\nSTORY:\n{draft}"},
        ],
        temperature=0.2, num_predict=600, json_mode=True,
    )
    return _json_obj(raw, {"approved": True, "feedback": ""})


# --- Deterministic scriptifier (regex split) — reliable where the 7B was not ---
_SPEECH_VERBS = (r"said|asked|replied|whispered|shouted|exclaimed|cried|called|added|"
                 r"murmured|answered|declared|sighed|announced|yelled|gasped|laughed|"
                 r"grinned|chuckled|spoke|begged|warned|agreed|continued|responded|"
                 r"explained|told|repeated|mused|insisted|offered|wondered|greeted|"
                 r"squealed|teased|admitted|hummed|sang")
_TAG_ONLY = re.compile(
    rf"^[\s,;:\-]*([A-Z][a-zA-Z]+\s+)?(?:{_SPEECH_VERBS})\b", re.IGNORECASE)


def _norm_quotes(s: str) -> str:
    return (s.replace("“", '"').replace("”", '"')
             .replace("‘", "'").replace("’", "'"))


def _emotion_for(text: str, is_narrator: bool) -> str:
    t = text.lower()
    if any(w in t for w in ("sleep", "drift", "slumber", "yawn", "goodnight",
                            "good night", "dream", "rest")):
        return "sleepy"
    if "!" in text:
        return "excited"
    if "?" in text:
        return "curious"
    return "warm" if is_narrator else "gentle"


def _split_sentences(s: str) -> list:
    s = re.sub(r"(?m)^\s*#{1,6}.*$", "", s)         # drop markdown headings
    s = re.sub(r"(?m)^\s*[-*_=~]{2,}\s*$", "", s)   # drop separator lines (---, ***)
    s = re.sub(r"[*_`>]+", "", s).strip()
    if not s:
        return []
    out = []
    for p in re.split(r"(?<=[.!?])\s+", s):
        p = p.strip().strip("-").strip()
        if p and re.search(r"[A-Za-z0-9]", p):  # skip pure punctuation/separators
            out.append(p)
    return out


# Capitalized words that are not character names (avoid attributing quotes to them).
_NON_NAMES = {
    "He", "She", "They", "It", "We", "You", "I", "His", "Her", "Their", "The",
    "A", "An", "And", "But", "Then", "Now", "With", "As", "When", "Finally",
    "Suddenly", "Just", "So", "There", "Here", "After", "Before", "Soon",
}


def _valid_name(name):
    return name if (name and name not in _NON_NAMES) else None


def _attr_speaker(after: str, before: str):
    vb = f"(?i:{_SPEECH_VERBS})"  # verb may be any case; names must be capitalized
    # "said Name" or "Name said" immediately after the quote
    for pat in (rf'^[\s,;:\-"\']*{vb}\s+([A-Z][a-zA-Z]+)',
                rf'^[\s,;:\-"\']*([A-Z][a-zA-Z]+)\s+{vb}'):
        m = re.match(pat, after)
        if m and _valid_name(m.group(1)):
            return m.group(1)
    # "Name said," just before the quote
    m = re.search(rf'([A-Z][a-zA-Z]+)\s+{vb}[\s,:\-]*$', before)
    if m and _valid_name(m.group(1)):
        return m.group(1)
    # fallback: the actor named in the immediately preceding narration
    # (handles action beats like "Ezra nodded eagerly. <quote>")
    names = [n for n in re.findall(r"[A-Z][a-zA-Z]+", before) if _valid_name(n)]
    if names:
        return names[-1]
    return None


def _scriptify(draft: str) -> list:
    """Split story prose into ordered lines: quoted speech -> character, rest -> Narrator."""
    text = _norm_quotes(draft)
    tokens = re.split(r'("[^"]*")', text)  # keep quoted spans as their own tokens
    lines = []
    last_speaker = None
    for i, tok in enumerate(tokens):
        if not tok:
            continue
        if len(tok) >= 2 and tok[0] == '"' and tok[-1] == '"':  # a spoken quote
            spoken = tok[1:-1].strip()
            if not spoken:
                continue
            after = tokens[i + 1] if i + 1 < len(tokens) else ""
            before = tokens[i - 1] if i >= 1 else ""
            speaker = _attr_speaker(after, before) or last_speaker or "Narrator"
            if speaker != "Narrator":
                last_speaker = speaker
            lines.append({"character": speaker,
                          "emotion": _emotion_for(spoken, speaker == "Narrator"),
                          "text": spoken})
        else:  # narration
            for sent in _split_sentences(tok):
                if _TAG_ONLY.match(sent) and len(sent.split()) <= 6:
                    continue  # drop bare speech-tag fragments ("said Leo, smiling.")
                lines.append({"character": "Narrator",
                              "emotion": _emotion_for(sent, True), "text": sent})
    return lines or [{"character": "Narrator", "emotion": "warm", "text": draft.strip()}]


def _character_designer(draft: str, existing_names: list) -> list:
    """Casting Director: find + design NEW characters introduced by the story."""
    roster = ", ".join(existing_names) if existing_names else "(none)"
    raw = _ollama_chat(
        [
            {"role": "system", "content": CASTING_SYS},
            {"role": "user", "content": f"EXISTING CHARACTERS: {roster}\n\nSTORY:\n{draft}"},
        ],
        temperature=0.5, num_predict=1500, schema=CAST_SCHEMA,
    )
    arr = _json_arr(raw)
    if not isinstance(arr, list):
        return []
    return [x for x in arr if isinstance(x, dict) and (x.get("name") or "").strip()]


def _summarize(draft: str, fallback_title: str) -> tuple:
    raw = _ollama_chat(
        [
            {"role": "system", "content": "You summarize children's stories for a library."},
            {"role": "user", "content":
                'Respond with STRICT JSON {"title": "...", "summary": "..."}. '
                "Title short & whimsical; summary 2-3 sentences for continuity.\n\n"
                + draft},
        ],
        temperature=0.3, num_predict=400, json_mode=True,
    )
    d = _json_obj(raw, {})
    title = (d.get("title") or fallback_title).strip()
    summary = (d.get("summary") or "").strip()
    return title, summary


# --------------------------- voice resolution ----------------------------- #
def _auto_speaker(name: str) -> str:
    """Deterministic preset for an unsaved character (same name -> same voice).

    English presets only — a non-English preset reading English sounds strongly
    accented (the failure the kids panned).
    """
    names = config.ENGLISH_PRESETS
    h = sum(ord(c) for c in (name or "x").lower())
    return names[h % len(names)]


def _pick_speaker_for_voice(voice_description: str, gender: str) -> str:
    """Map a natural-language voice description to the best-fit Qwen preset.

    Only Aiden/Ryan are native-English presets; a designed female voice falls
    back to a non-English preset until voice cloning is wired (Phase C).
    """
    d = (voice_description or "").lower()
    g = (gender or "").lower()
    female = ("female" in g or "woman" in g or "girl" in g
              or any(w in d for w in ["female", "woman", "girl", " she ", "motherly"]))
    if female:
        if any(w in d for w in ["bright", "edgy", "sharp", "shrill"]):
            return "Vivian"
        if "playful" in d:
            return "Ono_Anna"
        if any(w in d for w in ["rich", "emotional"]):
            return "Sohee"
        return "Serena"  # warm, gentle default female
    if any(w in d for w in ["dynamic", "energetic", "strong", "lively", "bold", "booming"]):
        return "Ryan"
    if any(w in d for w in ["gruff", "low", "mellow", "old", "deep", "seasoned", "raspy"]):
        return "Uncle_Fu"
    return "Aiden"  # clear, sunny English male default


def _voice_instruct(emotion: str) -> str:
    """Instruct for CustomVoice preset lines (mild emotion nudge)."""
    emotion = (emotion or "warm").strip()
    return f"Speak in a {emotion}, gentle and soothing bedtime tone."


def _design_instruct(voice_description: str, emotion: str) -> str:
    """Instruct for VoiceDesign lines — describing the character AND the
    emotion in free text is what sold the emotion in the A/B/C bake-off."""
    desc = (voice_description or "a warm, friendly character voice").strip().rstrip(".")
    emotion = (emotion or "warm").strip()
    return f"{desc}, speaking in a {emotion} voice, vivid and expressive."


def _narrator_clone_ready() -> bool:
    return bool(config.NARRATOR_REF_AUDIO) and os.path.exists(config.NARRATOR_REF_AUDIO)


def _line_voice(character_name: str, emotion: str, saved_by_name: dict) -> dict:
    """Resolve how one script line is voiced (the validated recipe, PLAN §1b).

    Narrator          -> clone of the gravelly reference clip (reproducible);
                         preset fallback if the clip is missing.
    Saved char, clone -> clone of the character's own reference clip.
    Saved char with a voice_description -> per-line VoiceDesign (best emotion).
    Saved char with an explicit preset  -> CustomVoice (honors the user's pick).
    Unknown char      -> deterministic English preset.
    """
    key = (character_name or "").strip().lower()
    if key in ("", "narrator", "the narrator"):
        mode = "clone" if _narrator_clone_ready() else "preset"
        return {"character_id": None, "voice_mode": mode,
                "speaker": config.NARRATOR_SPEAKER,
                "voice_instruct": _voice_instruct(emotion)}
    ch = saved_by_name.get(key)
    if ch:
        if ch.get("voice_mode") == "clone" and ch.get("ref_audio_path"):
            return {"character_id": ch.get("id"), "voice_mode": "clone",
                    "speaker": ch.get("speaker") or config.NARRATOR_SPEAKER,
                    "voice_instruct": _voice_instruct(emotion)}
        if ch.get("voice_description"):
            return {"character_id": ch.get("id"), "voice_mode": "design",
                    "speaker": ch.get("speaker"),  # kept for fallback only
                    "voice_instruct": _design_instruct(ch["voice_description"], emotion)}
        return {"character_id": ch.get("id"), "voice_mode": "preset",
                "speaker": ch.get("speaker") or _auto_speaker(character_name),
                "voice_instruct": _voice_instruct(emotion)}
    return {"character_id": None, "voice_mode": "preset",
            "speaker": _auto_speaker(character_name),
            "voice_instruct": _voice_instruct(emotion)}


# ------------------------------ rendering --------------------------------- #
def _write_silence(path: str, seconds: float, rate: int = 24000) -> None:
    frames = max(1, int(seconds * rate))
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * frames)


def _merge_render_groups(stored: list, max_chars: int = None) -> list:
    """Group script lines into render clips.

    Consecutive NARRATOR lines merge into one clip (soft char cap) for smooth
    narration; each character line stays solo so it keeps its own per-line
    emotion instruct. Returns groups in story order.
    """
    if max_chars is None:
        max_chars = config.NARRATOR_MERGE_CHARS
    groups = []
    cur = None  # the open narrator group, if the previous line was narration
    for ln in stored:
        text = (ln.get("text") or "").strip()
        if not text:
            continue
        is_narr = (ln.get("character_name") or "").strip().lower() in (
            "", "narrator", "the narrator")
        if is_narr and cur is not None and len(cur["text"]) + 1 + len(text) <= max_chars:
            cur["text"] += " " + text
            cur["line_ids"].append(ln["id"])
            continue
        g = {
            "seq": ln["seq"],
            "line_ids": [ln["id"]],
            "text": text,
            "voice_mode": ln.get("voice_mode") or "preset",
            "speaker": ln.get("speaker"),
            "voice_instruct": ln.get("voice_instruct"),
            "character_id": ln.get("character_id"),
            "emotion": ln.get("emotion"),
        }
        groups.append(g)
        cur = g if is_narr else None
    return groups


def _clone_ref(group: dict, chars_by_id: dict) -> tuple:
    """Reference clip + transcript for a clone-mode group."""
    ch = chars_by_id.get(group.get("character_id"))
    if ch and ch.get("ref_audio_path"):
        return ch["ref_audio_path"], ch.get("ref_text") or ""
    return config.NARRATOR_REF_AUDIO, config.NARRATOR_REF_TEXT


def _render_group(group: dict, chars_by_id: dict, out_path: str) -> str:
    """Render one clip; returns its line status: 'done' or 'silent'.

    'silent' means the clip is a placeholder (TTS off or synth failed) — kept
    distinct from 'done' so failed lines are visible and re-renderable.
    """
    text = group["text"]
    if text and tts.available():
        try:
            mode = group.get("voice_mode") or "preset"
            if mode == "clone":
                ref_audio, ref_text = _clone_ref(group, chars_by_id)
                if ref_audio and os.path.exists(ref_audio):
                    tts.synthesize_clone(text, ref_audio, ref_text, out_path)
                    return "done"
                mode = "preset"  # reference clip went missing
            if mode == "design":
                tts.synthesize_design(text, group.get("voice_instruct"), out_path)
                return "done"
            tts.synthesize_custom(
                text, group.get("speaker") or config.NARRATOR_SPEAKER,
                group.get("voice_instruct"), out_path)
            return "done"
        except Exception as e:  # noqa: BLE001 - one bad clip shouldn't kill the story
            print(f"[tts] clip synth failed, using silence: {e}", flush=True)
    # Fallback: silent clip sized to read time (match model sample rate if known).
    words = len(text.split())
    seconds = max(0.6, words * 0.38) + 0.3
    _write_silence(out_path, seconds, rate=tts.sample_rate() or 24000)
    return "silent"


def _concat_wavs(paths: list, out_path: str, gap_seconds: float = None) -> None:
    """Join clips in story order with a short gap between them.

    Clips can arrive with different sample rates (e.g. silent fallbacks written
    before the model's rate is known vs. real 24 kHz TTS clips) — everything is
    converted to the first clip's rate/format so the output never plays
    speed-shifted.
    """
    if gap_seconds is None:
        gap_seconds = config.TTS_GAP_SECONDS
    clips = []
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        with wave.open(p, "rb") as w:
            clips.append((w.getframerate(), w.getnchannels(), w.getsampwidth(),
                          w.readframes(w.getnframes())))
    if not clips:
        _write_silence(out_path, 1.0)
        return
    rate = clips[0][0]
    gap = b"\x00\x00" * int(gap_seconds * rate)
    data = bytearray()
    for i, (r, ch, sw, frames) in enumerate(clips):
        if ch == 2:
            frames = audioop.tomono(frames, sw, 0.5, 0.5)
        if sw != 2:
            frames = audioop.lin2lin(frames, sw, 2)
        if r != rate:
            frames, _ = audioop.ratecv(frames, 2, 1, r, rate, None)
        if i:
            data += gap
        data += frames
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(data))


# ------------------------------- job runner ------------------------------- #
def _run_job(story_id: int) -> None:
    try:
        story = db.get_story(story_id)
        if not story:
            return
        subject = story.get("subject")
        characters = story.get("characters") or []  # explicitly selected cast (writer context)
        rating = story.get("rating") or "G"
        prior = (
            db.recent_summaries_for_subject(story["subject_id"])
            if story.get("subject_id") else []
        )
        context = _story_context(subject, story, characters, prior)

        # 1) Plan a compelling plot (story architect)
        db.update_story_progress(story_id, "planning", "Planning the plot…", 5)
        outline = _planner(context)

        # 2) Writer draft from the outline
        db.update_story_progress(story_id, "writing", "Writer: drafting from the outline…", 12)
        draft = _writer(context, outline=outline)
        if not draft:
            raise RuntimeError("Writer returned empty text.")

        # 3) QA/QC <-> Writer iteration (cohesion + compelling, faithful to outline)
        for i in range(config.MAX_QA_ROUNDS):
            db.update_story_progress(
                story_id, "reviewing", f"QA: reviewing story (round {i + 1})…",
                20 + i * 8,
            )
            review = _qa(draft, prior)
            if review.get("cohesive", True) or not (review.get("feedback") or "").strip():
                break
            db.update_story_progress(
                story_id, "writing", "Writer: revising from QA notes…", 24 + i * 8
            )
            draft = _writer(context, outline=outline, prev_draft=draft, feedback=review["feedback"])

        # 4) Manager: final say (one revision pass if rejected)
        db.update_story_progress(story_id, "reviewing", "Manager: final review…", 42)
        verdict = _manager(draft, rating)
        if not verdict.get("approved", True) and (verdict.get("feedback") or "").strip():
            db.update_story_progress(
                story_id, "writing", "Writer: applying manager's final notes…", 46
            )
            draft = _writer(
                context, outline=outline, prev_draft=draft,
                feedback="MANAGER (final authority) requires: " + verdict["feedback"],
            )

        # 5) Casting: design any NEW characters the story introduced, then store them
        db.update_story_progress(story_id, "casting", "Casting & designing new characters…", 50)
        global_chars = db.list_characters()
        saved_by_name = {(c["name"] or "").strip().lower(): c for c in global_chars}
        for nc in _character_designer(draft, [c["name"] for c in global_chars]):
            cname = (nc.get("name") or "").strip()
            if not cname or cname.lower() in saved_by_name:
                continue
            created = db.create_character({
                "name": cname,
                "persona": nc.get("persona"),
                "motivations": nc.get("motivations"),
                "backstory": nc.get("backstory"),
                "voice_mode": "preset",
                "speaker": _pick_speaker_for_voice(nc.get("voice_description"), nc.get("gender")),
                "voice_description": nc.get("voice_description"),
                "language": "English",
                "auto_generated": 1,
            })
            if created:
                saved_by_name[cname.lower()] = created

        # The child hero speaks too: give them a designed voice (never a
        # random preset) unless a saved character already shares the name.
        if subject and subject.get("name"):
            skey = subject["name"].strip().lower()
            if skey and skey not in saved_by_name:
                age = subject.get("age")
                saved_by_name[skey] = {
                    "id": None,
                    "voice_description": (
                        f"a bright, warm child's voice, about {age} years old"
                        if age else "a bright, warm child's voice"),
                }

        # 6) Scriptify into a playwright script
        db.update_story_progress(story_id, "scripting", "Formatting the script…", 56)
        raw_lines = _scriptify(draft)

        # Resolve a voice for every line and persist the script.
        lines = []
        for ln in raw_lines:
            if not isinstance(ln, dict):
                continue
            cname = (ln.get("character") or "Narrator").strip()
            text = (ln.get("text") or "").strip()
            if not text:
                continue
            emotion = (ln.get("emotion") or "warm").strip()
            voice = _line_voice(cname, emotion, saved_by_name)
            lines.append({
                "character_name": cname,
                "emotion": emotion,
                "text": text,
                "status": "pending",
                **voice,
            })
        db.replace_story_lines(story_id, lines)
        # Link every character that actually appears (selected, reused, or new).
        resolved_ids = {ln["character_id"] for ln in lines if ln.get("character_id")}
        if resolved_ids:
            db.link_story_characters(story_id, list(resolved_ids))

        title, summary = _summarize(
            draft, f"A Story for {subject['name']}" if subject else "A Bedtime Story"
        )

        # 7) Narrate: merge narrator runs into clips, render batched by
        #    sub-model (only one fits in VRAM at a time), concat in story order.
        db.update_story_progress(story_id, "narrating", "Recording narration…", 60)
        clip_dir = os.path.join(config.AUDIO_DIR, f"story_{story_id}")
        os.makedirs(clip_dir, exist_ok=True)
        stored = db.get_story_lines(story_id)
        chars_by_id = {c["id"]: c for c in db.list_characters()}
        groups = _merge_render_groups(stored)
        for g in groups:
            g["clip_path"] = os.path.join(clip_dir, f"clip_{g['seq']:04d}.wav")
        model_order = {"clone": 0, "design": 1, "preset": 2}
        total = len(groups) or 1
        render_order = sorted(
            groups, key=lambda g: (model_order.get(g["voice_mode"], 3), g["seq"]))
        for n, g in enumerate(render_order):
            line_status = _render_group(g, chars_by_id, g["clip_path"])
            for lid in g["line_ids"]:
                db.set_line_audio(lid, g["clip_path"], line_status)
            db.update_story_progress(
                story_id, "narrating",
                f"Voicing clip {n + 1} of {total}…",
                60 + int(32 * (n + 1) / total),
            )
        tts.unload()  # free VRAM between stories

        final_audio = os.path.join(config.AUDIO_DIR, f"story_{story_id}.wav")
        _concat_wavs([g["clip_path"] for g in groups], final_audio)

        # 8) Done
        db.set_story_result(story_id, title, draft, summary, final_audio)

    except requests.exceptions.RequestException as e:
        db.set_story_error(
            story_id,
            f"Could not reach Ollama at {config.OLLAMA_URL} (model "
            f"'{config.OLLAMA_MODEL}'). Is Ollama running and the model pulled? {e}",
        )
    except Exception as e:  # noqa: BLE001 - surface any failure to the UI
        import traceback
        tb = traceback.format_exc().replace("\n", " / ")
        db.set_story_error(story_id, f"{type(e).__name__}: {e} | {tb}")


# One background worker renders jobs sequentially — the local LLM and the TTS
# GPU can't take concurrent builds (this is the generate-ahead queue, locally).
_job_queue: "queue.Queue[int]" = queue.Queue()
_worker_lock = threading.Lock()
_worker = None


def _worker_loop() -> None:
    while True:
        story_id = _job_queue.get()
        try:
            _run_job(story_id)  # catches its own exceptions -> story error state
        finally:
            _job_queue.task_done()


def start_story_job(story_id: int) -> None:
    """Enqueue generation and return immediately (single worker, FIFO)."""
    global _worker
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_worker_loop, daemon=True)
            _worker.start()
    _job_queue.put(story_id)


def recover_interrupted() -> None:
    """Startup sweep: requeue stories that never started; fail the ones that
    were mid-flight when the server died (their partial state is unusable)."""
    for s in db.get_unfinished_stories():
        if s["status"] == "queued":
            start_story_job(s["id"])
        else:
            db.set_story_error(
                s["id"], "Interrupted by a server restart — press Retry to rebuild.")
