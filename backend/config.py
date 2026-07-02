"""Central configuration and constants for the AI Storyteller MVP."""
import os

# --- Paths (all derived from this file's location, so CWD doesn't matter) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_DIR = os.path.join(BASE_DIR, "data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
DB_PATH = os.path.join(DATA_DIR, "storyteller.db")

# --- Local story-text generation via Ollama ---
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")

# --- Story shape ---
STORY_TARGET_WORDS = int(os.environ.get("STORY_TARGET_WORDS", "1500"))

# --- Story genre / style (Writer uses it to set tone) ---
GENRES = ["Fluffy", "Fantasy", "Science-Fiction", "Suspense", "Horror"]
GENRE_GUIDE = {
    "Fluffy": "Light, cozy, low-stakes and comforting — warm everyday adventures "
              "and gentle fun. The sweetest, easiest bedtime fit.",
    "Fantasy": "Magic, mythical creatures, and enchanted worlds; wonder and a "
               "gentle quest.",
    "Science-Fiction": "Friendly future worlds, space, robots, and clever gadgets; "
                       "curiosity and discovery.",
    "Suspense": "A gentle mystery or puzzle with mild, playful tension that always "
                "resolves safely and reassuringly.",
    "Horror": "A playfully spooky, slightly eerie atmosphere (friendly ghosts, "
              "silly monsters). Stay within the chosen rating — at G/PG keep it "
              "charming-spooky, never genuinely frightening.",
}

# --- Content rating ceiling (Writer targets it, Manager enforces it) ---
RATINGS = ["G", "PG", "PG-13", "R"]
RATING_GUIDE = {
    "G": "General Audiences: wholesome and gentle; no violence, frightening "
         "content, romance, or any bad language. Ideal for young children at bedtime.",
    "PG": "Parental Guidance: mild peril or light rude humor is okay, but nothing "
          "scary, mature, or crude. Still firmly kid-friendly.",
    "PG-13": "Some adventure tension or mild thematic elements suitable for older "
             "kids/teens. No strong language, graphic content, or explicit themes.",
    "R": "Mature themes permitted for adult listeners. Avoid gratuitous content; "
         "prioritize story quality.",
}

# --- Writers' room (multi-agent) ---
# Max writer<->QA revision rounds before the manager's final review.
MAX_QA_ROUNDS = int(os.environ.get("MAX_QA_ROUNDS", "2"))
# Default voice for narration lines (only Aiden/Ryan are native English presets).
NARRATOR_SPEAKER = os.environ.get("NARRATOR_SPEAKER", "Aiden")

# --- Qwen3-TTS narration ---
# Three sub-models, each with different powers (see PLAN.md §1b):
#   CustomVoice  - 9 fixed preset speakers (reproducible, mild emotion via instruct)
#   VoiceDesign  - free-text voice+emotion description (best emotion, NOT reproducible)
#   Base         - voice clone from a reference clip (reproducible, no emotion knob)
# Only one fits in VRAM at a time on an 8 GB card; tts.py swaps them sequentially.
TTS_MODEL = os.environ.get("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
TTS_MODEL_DESIGN = os.environ.get("TTS_MODEL_DESIGN", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
TTS_MODEL_BASE = os.environ.get("TTS_MODEL_BASE", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
# "" = auto-pick: GPU 1 if there are 2+ cards (keeps GPU 0 free for Ollama),
# else GPU 0, else CPU. Override with e.g. TTS_DEVICE=cuda:0.
TTS_DEVICE = os.environ.get("TTS_DEVICE", "")
TTS_LANGUAGE = os.environ.get("TTS_LANGUAGE", "English")
# Set TTS_DISABLED=1 to force the silent placeholder (e.g. before the model is set up).
TTS_DISABLED = os.environ.get("TTS_DISABLED", "") == "1"

# --- The validated narration recipe (PLAN.md §1b, kid-approved 2026-06-27) ---
# Narrator = gravelly voice CLONED via the Base model from this reference clip.
# The clip was made once with VoiceDesign; the clone reproduces it exactly every
# story. If the clip is missing, narration falls back to the CustomVoice preset.
NARRATOR_REF_AUDIO = os.environ.get(
    "NARRATOR_REF_AUDIO", os.path.join(DATA_DIR, "vd_british_gravelly.wav"))
NARRATOR_REF_TEXT = os.environ.get(
    "NARRATOR_REF_TEXT",
    "And here, beneath the silver moon, a tiny dragon settles down, gently, "
    "into a peaceful sleep.")
# Merge consecutive narrator lines into one clip up to this soft char cap
# (smooth narration, fewer clips); character lines stay solo (per-line emotion).
NARRATOR_MERGE_CHARS = int(os.environ.get("NARRATOR_MERGE_CHARS", "380"))
# Silence inserted between clips when concatenating the final audio.
TTS_GAP_SECONDS = float(os.environ.get("TTS_GAP_SECONDS", "0.28"))

# --- Qwen3-TTS preset speakers (verified from the CustomVoice model card) ---
# A character pinned to a fixed preset speaker sounds the SAME every appearance
# (reproducible). NOTE: the only native-English presets are Aiden and Ryan; a
# warm English *female* narrator will need voice cloning (schema supports it).
PRESET_SPEAKERS = [
    {"name": "Aiden", "native": "English", "gender": "male",
     "desc": "Sunny American male, clear midrange", "recommended": True},
    {"name": "Ryan", "native": "English", "gender": "male",
     "desc": "Dynamic male, strong rhythmic drive", "recommended": False},
    {"name": "Serena", "native": "Chinese", "gender": "female",
     "desc": "Warm, gentle young female", "recommended": False},
    {"name": "Vivian", "native": "Chinese", "gender": "female",
     "desc": "Bright, slightly edgy young female", "recommended": False},
    {"name": "Uncle_Fu", "native": "Chinese", "gender": "male",
     "desc": "Seasoned male, low mellow timbre", "recommended": False},
    {"name": "Dylan", "native": "Chinese (Beijing)", "gender": "male",
     "desc": "Youthful Beijing male, clear timbre", "recommended": False},
    {"name": "Eric", "native": "Chinese (Sichuan)", "gender": "male",
     "desc": "Lively Chengdu male, husky brightness", "recommended": False},
    {"name": "Ono_Anna", "native": "Japanese", "gender": "female",
     "desc": "Playful Japanese female, light timbre", "recommended": False},
    {"name": "Sohee", "native": "Korean", "gender": "female",
     "desc": "Warm Korean female, rich emotion", "recommended": False},
]

# Presets that speak unaccented English. Auto-assigned voices must come from
# this list — a non-English preset reading English sounds strongly accented
# (the exact failure the kids panned). Others remain user-selectable.
ENGLISH_PRESETS = [s["name"] for s in PRESET_SPEAKERS if s["native"] == "English"]
