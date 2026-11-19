"""SQLite data layer for the AI Storyteller MVP.

Schema overview
---------------
subjects          : the child a story is told for (name, age)
characters        : reusable cast members + their pinned VOICE info so the
                    same character sounds the same in every story
stories           : a generated story: params, full text, summary, audio,
                    and generation status/progress (drives the readiness bar)
story_characters  : which characters appeared in which story (M:N)

Each call opens its own connection so the background generation thread and the
request handlers never share a connection (SQLite-safe for this MVP scale).
"""
import json
import os
import sqlite3
from datetime import datetime, timezone

import config


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _ensure_columns(conn, table: str, columns: dict) -> None:
    """Add any missing columns to an existing table (lightweight migration)."""
    existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def init_db() -> None:
    os.makedirs(config.AUDIO_DIR, exist_ok=True)
    with _connect() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS subjects (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                age         INTEGER,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS characters (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT NOT NULL,
                persona         TEXT,            -- personality fed to the story LLM
                description     TEXT,            -- looks / role
                motivations     TEXT,            -- what drives them
                backstory       TEXT,            -- brief history
                -- VOICE: how this character is narrated, reproducibly.
                voice_mode      TEXT NOT NULL DEFAULT 'preset',  -- 'preset' | 'clone'
                speaker         TEXT,            -- preset speaker name (preset mode)
                instruct        TEXT,            -- optional voice style instruction
                voice_description TEXT,          -- natural-language voice (gruff, shrill, male...)
                ref_audio_path  TEXT,            -- reference clip (clone mode)
                ref_text        TEXT,            -- transcript of reference clip
                language        TEXT DEFAULT 'English',
                auto_generated  INTEGER NOT NULL DEFAULT 0,  -- 1 = made by the Character Designer agent
                created_at      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS stories (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id  INTEGER REFERENCES subjects(id),
                title       TEXT,
                location    TEXT,
                themes      TEXT,                -- JSON array
                interests   TEXT,                -- JSON array
                extra_notes TEXT,
                rating      TEXT NOT NULL DEFAULT 'G',  -- max content rating
                genre       TEXT NOT NULL DEFAULT 'Fluffy',  -- story genre/style
                full_text   TEXT,
                summary     TEXT,
                status      TEXT NOT NULL DEFAULT 'queued',  -- queued|writing|summarizing|narrating|ready|error
                stage       TEXT,                -- human-readable stage label
                progress    INTEGER NOT NULL DEFAULT 0,      -- 0..100
                error       TEXT,
                audio_path  TEXT,
                created_at  TEXT NOT NULL,
                ready_at    TEXT
            );

            CREATE TABLE IF NOT EXISTS story_characters (
                story_id     INTEGER NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
                character_id INTEGER NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
                PRIMARY KEY (story_id, character_id)
            );

            -- Playwright script: one row per spoken line (narration or dialogue).
            -- This is what gets clipped per-line and fed to Qwen3-TTS.
            CREATE TABLE IF NOT EXISTS story_lines (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                story_id       INTEGER NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
                seq            INTEGER NOT NULL,            -- order within the story
                character_name TEXT NOT NULL,              -- 'Narrator' or a character name
                character_id   INTEGER REFERENCES characters(id),  -- resolved saved char, if any
                emotion        TEXT,                       -- delivery emotion for this line
                text           TEXT NOT NULL,              -- the spoken line
                voice_mode     TEXT NOT NULL DEFAULT 'preset',  -- 'clone'|'design'|'preset'
                speaker        TEXT,                       -- resolved Qwen preset speaker
                voice_instruct TEXT,                       -- instruct string fed to Qwen TTS
                audio_path     TEXT,                       -- per-line clip (merged narrator
                                                           -- lines share one clip)
                status         TEXT NOT NULL DEFAULT 'pending', -- pending|done|silent
                -- Prosody: which paragraph of the source prose this line came from,
                -- and whether it is a speech tag belonging to the line before it.
                -- Both drive the length of the silence inserted before the clip.
                para           INTEGER NOT NULL DEFAULT 0,
                is_tag         INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_story_lines_story
                ON story_lines (story_id, seq);
            """
        )
        # Migrate older DBs that predate the richer character fields.
        _ensure_columns(c, "characters", {
            "motivations": "motivations TEXT",
            "backstory": "backstory TEXT",
            "voice_description": "voice_description TEXT",
            "auto_generated": "auto_generated INTEGER NOT NULL DEFAULT 0",
        })
        _ensure_columns(c, "stories", {
            "rating": "rating TEXT NOT NULL DEFAULT 'G'",
            "genre": "genre TEXT NOT NULL DEFAULT 'Fluffy'",
        })
        _ensure_columns(c, "story_lines", {
            "voice_mode": "voice_mode TEXT NOT NULL DEFAULT 'preset'",
            "para": "para INTEGER NOT NULL DEFAULT 0",
            "is_tag": "is_tag INTEGER NOT NULL DEFAULT 0",
        })


# ------------------------------- subjects --------------------------------- #
def get_or_create_subject(name: str, age) -> int:
    name = (name or "").strip()
    with _connect() as c:
        row = c.execute(
            "SELECT id FROM subjects WHERE name = ? COLLATE NOCASE", (name,)
        ).fetchone()
        if row:
            if age is not None:
                c.execute("UPDATE subjects SET age = ? WHERE id = ?", (age, row["id"]))
            return row["id"]
        cur = c.execute(
            "INSERT INTO subjects (name, age, created_at) VALUES (?, ?, ?)",
            (name, age, _now()),
        )
        return cur.lastrowid


def list_subjects() -> list:
    with _connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM subjects ORDER BY name")]


# ------------------------------ characters -------------------------------- #
def create_character(data: dict) -> dict:
    with _connect() as c:
        cur = c.execute(
            """INSERT INTO characters
               (name, persona, description, motivations, backstory, voice_mode,
                speaker, instruct, voice_description, ref_audio_path, ref_text,
                language, auto_generated, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                (data.get("name") or "").strip(),
                data.get("persona"),
                data.get("description"),
                data.get("motivations"),
                data.get("backstory"),
                data.get("voice_mode") or "preset",
                data.get("speaker"),
                data.get("instruct"),
                data.get("voice_description"),
                data.get("ref_audio_path"),
                data.get("ref_text"),
                data.get("language") or "English",
                1 if data.get("auto_generated") else 0,
                _now(),
            ),
        )
        # Read back on the SAME connection — a separate connection wouldn't see
        # this row until the enclosing transaction commits.
        row = c.execute(
            "SELECT * FROM characters WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
        return dict(row) if row else None


def get_character(cid: int):
    with _connect() as c:
        row = c.execute("SELECT * FROM characters WHERE id = ?", (cid,)).fetchone()
        return dict(row) if row else None


def list_characters() -> list:
    with _connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM characters ORDER BY name")]


# -------------------------------- stories --------------------------------- #
def create_story(subject_id: int, params: dict) -> int:
    with _connect() as c:
        cur = c.execute(
            """INSERT INTO stories
               (subject_id, location, themes, interests, extra_notes, rating, genre,
                status, stage, progress, created_at)
               VALUES (?,?,?,?,?,?,?, 'queued', 'Queued', 0, ?)""",
            (
                subject_id,
                params.get("location"),
                json.dumps(params.get("themes") or []),
                json.dumps(params.get("interests") or []),
                params.get("extra_notes"),
                params.get("rating") or "G",
                params.get("genre") or "Fluffy",
                _now(),
            ),
        )
        return cur.lastrowid


def link_story_characters(story_id: int, character_ids: list) -> None:
    if not character_ids:
        return
    with _connect() as c:
        c.executemany(
            "INSERT OR IGNORE INTO story_characters (story_id, character_id) VALUES (?, ?)",
            [(story_id, cid) for cid in character_ids],
        )


def update_story_progress(story_id: int, status: str, stage: str, progress: int) -> None:
    with _connect() as c:
        c.execute(
            "UPDATE stories SET status = ?, stage = ?, progress = ? WHERE id = ?",
            (status, stage, progress, story_id),
        )


def set_story_result(story_id: int, title: str, full_text: str, summary: str,
                     audio_path: str) -> None:
    with _connect() as c:
        c.execute(
            """UPDATE stories
               SET title = ?, full_text = ?, summary = ?, audio_path = ?,
                   status = 'ready', stage = 'Ready', progress = 100, ready_at = ?
               WHERE id = ?""",
            (title, full_text, summary, audio_path, _now(), story_id),
        )


def set_story_error(story_id: int, message: str) -> None:
    with _connect() as c:
        c.execute(
            "UPDATE stories SET status = 'error', stage = 'Error', error = ? WHERE id = ?",
            (message, story_id),
        )


def get_unfinished_stories() -> list:
    """Stories that never reached a terminal state (for startup recovery)."""
    with _connect() as c:
        rows = c.execute(
            "SELECT id, status FROM stories WHERE status NOT IN ('ready', 'error')"
        ).fetchall()
        return [dict(r) for r in rows]


def reset_story_for_retry(story_id: int) -> None:
    """Put an errored story back in the queue state so it can be rebuilt."""
    with _connect() as c:
        c.execute(
            """UPDATE stories
               SET status = 'queued', stage = 'Queued', progress = 0, error = NULL
               WHERE id = ?""",
            (story_id,),
        )


def _story_to_dict(row) -> dict:
    d = dict(row)
    d["themes"] = json.loads(d.get("themes") or "[]")
    d["interests"] = json.loads(d.get("interests") or "[]")
    return d


def get_story(story_id: int):
    with _connect() as c:
        row = c.execute("SELECT * FROM stories WHERE id = ?", (story_id,)).fetchone()
        if not row:
            return None
        d = _story_to_dict(row)
        subj = c.execute(
            "SELECT name, age FROM subjects WHERE id = ?", (d["subject_id"],)
        ).fetchone()
        d["subject"] = dict(subj) if subj else None
        chars = c.execute(
            """SELECT ch.* FROM characters ch
               JOIN story_characters sc ON sc.character_id = ch.id
               WHERE sc.story_id = ?""",
            (story_id,),
        ).fetchall()
        d["characters"] = [dict(r) for r in chars]
        lines = c.execute(
            "SELECT * FROM story_lines WHERE story_id = ? ORDER BY seq", (story_id,)
        ).fetchall()
        d["lines"] = [dict(r) for r in lines]
        return d


def list_stories() -> list:
    with _connect() as c:
        rows = c.execute(
            """SELECT s.id, s.title, s.status, s.stage, s.progress, s.summary,
                      s.created_at, s.ready_at, su.name AS subject_name
               FROM stories s LEFT JOIN subjects su ON su.id = s.subject_id
               ORDER BY s.id DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


def recent_summaries_for_subject(subject_id: int, limit: int = 3) -> list:
    """Past story summaries for continuity, newest first."""
    with _connect() as c:
        rows = c.execute(
            """SELECT title, summary FROM stories
               WHERE subject_id = ? AND status = 'ready' AND summary IS NOT NULL
               ORDER BY id DESC LIMIT ?""",
            (subject_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


# ------------------------------ story lines ------------------------------- #
def replace_story_lines(story_id: int, lines: list) -> None:
    """Store the playwright script for a story (replaces any existing lines)."""
    with _connect() as c:
        c.execute("DELETE FROM story_lines WHERE story_id = ?", (story_id,))
        c.executemany(
            """INSERT INTO story_lines
               (story_id, seq, character_name, character_id, emotion, text,
                voice_mode, speaker, voice_instruct, audio_path, status,
                para, is_tag)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    story_id, i,
                    ln.get("character_name") or "Narrator",
                    ln.get("character_id"),
                    ln.get("emotion"),
                    ln.get("text") or "",
                    ln.get("voice_mode") or "preset",
                    ln.get("speaker"),
                    ln.get("voice_instruct"),
                    ln.get("audio_path"),
                    ln.get("status") or "pending",
                    int(ln.get("para") or 0),
                    1 if ln.get("is_tag") else 0,
                )
                for i, ln in enumerate(lines)
            ],
        )


def get_story_lines(story_id: int) -> list:
    with _connect() as c:
        return [
            dict(r)
            for r in c.execute(
                "SELECT * FROM story_lines WHERE story_id = ? ORDER BY seq", (story_id,)
            )
        ]


def set_line_audio(line_id: int, audio_path: str, status: str = "done") -> None:
    with _connect() as c:
        c.execute(
            "UPDATE story_lines SET audio_path = ?, status = ? WHERE id = ?",
            (audio_path, status, line_id),
        )
