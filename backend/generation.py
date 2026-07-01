"""Story generation pipeline (runs in a background thread).

Writers' room (all local via Ollama), then narration:

  1. WRITER   - creative draft: fun, adventurous, imaginative.
  2. QA/QC    - checks continuity with past stories + internal cohesion;
                iterates with the writer up to MAX_QA_ROUNDS.
  3. MANAGER  - final say: age-appropriateness (no mature themes, profanity,
                violence, scares) and overall quality. Can send back once.
  4. SCRIPTIFY- converts the approved story into a playwright script:
                an ordered list of {character, emotion, line} (incl. Narrator).
  5. NARRATE  - per line: resolve the character's voice + an emotion-driven
                "instruct", synthesize a clip, then concatenate clips in order.

Story TEXT is fully real (local LLM). Story AUDIO is currently a PLACEHOLDER:
each line becomes a short silent clip and clips are concatenated, so the whole
per-line pipeline (voice resolution, ordering, concatenation) is exercised. Real
narration swaps `_synthesize_line` for a Qwen3-TTS call in the next phase — the
resolved `speaker` + `voice_instruct` per line are exactly what Qwen needs.
"""
import json
import os
import re
import threading
import wave

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
    """Deterministic preset for an unsaved character (same name -> same voice)."""
    names = [s["name"] for s in config.PRESET_SPEAKERS]
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


def _resolve_voice(character_name: str, saved_by_name: dict) -> tuple:
    """Return (character_id, speaker, instruct_base) for a script character."""
    key = (character_name or "").strip().lower()
    ch = saved_by_name.get(key)
    if ch:
        return ch.get("id"), (ch.get("speaker") or config.NARRATOR_SPEAKER)
    if key in ("narrator", "the narrator", ""):
        return None, config.NARRATOR_SPEAKER
    return None, _auto_speaker(character_name)


def _voice_instruct(emotion: str) -> str:
    emotion = (emotion or "warm").strip()
    return f"Speak in a {emotion}, gentle and soothing bedtime tone."


# ------------------------------ TTS (stub) -------------------------------- #
def _write_silence(path: str, seconds: float, rate: int = 22050) -> None:
    frames = max(1, int(seconds * rate))
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * frames)


def _synthesize_line(line: dict, out_path: str) -> None:
    """Narrate one line with Qwen3-TTS; fall back to a silent clip if TTS is off."""
    text = (line.get("text") or "").strip()
    if text and tts.available():
        try:
            tts.synthesize(
                text=text,
                speaker=line.get("speaker"),
                instruct=line.get("voice_instruct"),
                out_path=out_path,
                language=config.TTS_LANGUAGE,
            )
            return
        except Exception as e:  # noqa: BLE001 - one bad line shouldn't kill the story
            print(f"[tts] line synth failed, using silence: {e}", flush=True)
    # Fallback: silent clip sized to read time (match model sample rate if known).
    words = len(text.split())
    seconds = max(0.6, words * 0.38) + 0.3
    _write_silence(out_path, seconds, rate=tts.sample_rate() or 22050)


def _concat_wavs(paths: list, out_path: str) -> None:
    params = None
    data = b""
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        with wave.open(p, "rb") as w:
            params = w.getparams()
            data += w.readframes(w.getnframes())
    if params is None:
        _write_silence(out_path, 1.0)
        return
    with wave.open(out_path, "wb") as w:
        w.setparams(params)
        w.writeframes(data)


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

        # 4) Casting: design any NEW characters the story introduced, then store them
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

        # 5) Scriptify into a playwright script
        db.update_story_progress(story_id, "scripting", "Formatting the script…", 56)
        raw_lines = _scriptify(draft)

        # Resolve a voice for every line and persist the script.
        lines = []
        for ln in raw_lines:
            if not isinstance(ln, dict):
                continue
            cname = (ln.get("character") or "Narrator").strip()
            cid, speaker = _resolve_voice(cname, saved_by_name)
            lines.append({
                "character_name": cname,
                "character_id": cid,
                "emotion": (ln.get("emotion") or "warm").strip(),
                "text": (ln.get("text") or "").strip(),
                "speaker": speaker,
                "voice_instruct": _voice_instruct(ln.get("emotion")),
                "status": "pending",
            })
        lines = [ln for ln in lines if ln["text"]]
        db.replace_story_lines(story_id, lines)
        # Link every character that actually appears (selected, reused, or new).
        resolved_ids = {ln["character_id"] for ln in lines if ln.get("character_id")}
        if resolved_ids:
            db.link_story_characters(story_id, list(resolved_ids))

        title, summary = _summarize(
            draft, f"A Story for {subject['name']}" if subject else "A Bedtime Story"
        )

        # 6) Narrate each line, then concatenate in order.
        db.update_story_progress(story_id, "narrating", "Recording narration…", 60)
        clip_dir = os.path.join(config.AUDIO_DIR, f"story_{story_id}")
        os.makedirs(clip_dir, exist_ok=True)
        stored = db.get_story_lines(story_id)
        clip_paths = []
        total = len(stored) or 1
        for idx, ln in enumerate(stored):
            clip_path = os.path.join(clip_dir, f"line_{ln['seq']:04d}.wav")
            _synthesize_line(ln, clip_path)
            db.set_line_audio(ln["id"], clip_path, "done")
            clip_paths.append(clip_path)
            db.update_story_progress(
                story_id, "narrating",
                f"Voicing line {idx + 1} of {total}…",
                60 + int(32 * (idx + 1) / total),
            )

        final_audio = os.path.join(config.AUDIO_DIR, f"story_{story_id}.wav")
        _concat_wavs(clip_paths, final_audio)

        # 7) Done
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


def start_story_job(story_id: int) -> None:
    """Kick off generation in a daemon thread and return immediately."""
    threading.Thread(target=_run_job, args=(story_id,), daemon=True).start()
