"""AI Storyteller — local MVP API + static GUI server.

Run:  py backend/main.py     (or use run.bat)
Then open http://localhost:8000
"""
import os
import sys

# Make sibling modules importable no matter the working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import db
import generation

app = FastAPI(title="AI Storyteller (local MVP)")


@app.on_event("startup")
def _startup() -> None:
    db.init_db()


# ------------------------------- schemas ---------------------------------- #
class CharacterIn(BaseModel):
    name: str
    persona: Optional[str] = None
    description: Optional[str] = None
    voice_mode: str = "preset"
    speaker: Optional[str] = None
    instruct: Optional[str] = None
    language: str = "English"


class StoryIn(BaseModel):
    subject_name: str
    subject_age: Optional[int] = None
    location: Optional[str] = None
    themes: List[str] = []
    interests: List[str] = []
    extra_notes: Optional[str] = None
    rating: str = "G"
    genre: str = "Fluffy"
    character_ids: List[int] = []


# ------------------------------- API -------------------------------------- #
@app.get("/api/voices")
def get_voices():
    return config.PRESET_SPEAKERS


@app.get("/api/characters")
def get_characters():
    return db.list_characters()


@app.post("/api/characters")
def post_character(payload: CharacterIn):
    return db.create_character(payload.model_dump())


@app.get("/api/subjects")
def get_subjects():
    return db.list_subjects()


@app.post("/api/stories")
def post_story(payload: StoryIn):
    if not payload.subject_name.strip():
        raise HTTPException(400, "subject_name is required")
    subject_id = db.get_or_create_subject(payload.subject_name, payload.subject_age)
    rating = payload.rating if payload.rating in config.RATINGS else "G"
    genre = payload.genre if payload.genre in config.GENRES else "Fluffy"
    story_id = db.create_story(subject_id, {
        "location": payload.location,
        "themes": payload.themes,
        "interests": payload.interests,
        "extra_notes": payload.extra_notes,
        "rating": rating,
        "genre": genre,
    })
    db.link_story_characters(story_id, payload.character_ids)
    generation.start_story_job(story_id)
    return {"id": story_id, "status": "queued"}


@app.get("/api/stories")
def get_stories():
    return db.list_stories()


@app.get("/api/stories/{story_id}")
def get_story(story_id: int):
    story = db.get_story(story_id)
    if not story:
        raise HTTPException(404, "story not found")
    return story


@app.get("/api/stories/{story_id}/status")
def get_status(story_id: int):
    story = db.get_story(story_id)
    if not story:
        raise HTTPException(404, "story not found")
    return {
        "id": story_id,
        "status": story["status"],
        "stage": story["stage"],
        "progress": story["progress"],
        "error": story["error"],
    }


@app.get("/api/audio/{story_id}")
def get_audio(story_id: int):
    story = db.get_story(story_id)
    if not story or not story.get("audio_path") or not os.path.exists(story["audio_path"]):
        raise HTTPException(404, "audio not ready")
    return FileResponse(story["audio_path"], media_type="audio/wav")


# --------------------------- static GUI ----------------------------------- #
@app.get("/")
def index():
    return FileResponse(os.path.join(config.FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=config.FRONTEND_DIR), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
