"""Unit tests for the deterministic story pipeline (no LLM, no TTS, no DB).

Run from the project root:  py -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

import config
import generation as g


class TestScriptify(unittest.TestCase):
    def test_plain_prose_is_narrator(self):
        lines = g._scriptify("The moon rose over the hills. Everything was calm.")
        self.assertEqual([ln["character"] for ln in lines], ["Narrator", "Narrator"])

    def test_mixed_sentence_splits_into_multiple_lines(self):
        draft = 'Leo looked up. "Hello there," said Leo. The wind sighed.'
        lines = g._scriptify(draft)
        self.assertEqual(
            [(ln["character"], ln["text"]) for ln in lines],
            [("Narrator", "Leo looked up."),
             ("Leo", "Hello there,"),
             ("Narrator", "The wind sighed.")])

    def test_name_said_before_quote(self):
        lines = g._scriptify('Maya said, "Let us go."')
        spoken = [ln for ln in lines if ln["character"] != "Narrator"]
        self.assertEqual(spoken[0]["character"], "Maya")

    def test_action_beat_attribution(self):
        lines = g._scriptify('Ezra nodded eagerly. "Yes!"')
        spoken = [ln for ln in lines if ln["character"] != "Narrator"]
        self.assertEqual(spoken[0]["character"], "Ezra")
        self.assertEqual(spoken[0]["emotion"], "excited")

    def test_last_speaker_fallback(self):
        lines = g._scriptify('Mira smiled. "Hello." A pause. "Goodbye."')
        spoken = [ln for ln in lines if ln["character"] != "Narrator"]
        self.assertEqual([ln["character"] for ln in spoken], ["Mira", "Mira"])

    def test_pronouns_never_become_speakers(self):
        lines = g._scriptify('She grinned. "Ready?"')
        spoken = [ln for ln in lines if ln["character"] != "Narrator"]
        # "She" is filtered; with no other name the quote stays with the Narrator
        self.assertEqual(spoken, [])

    def test_markdown_and_separators_stripped(self):
        draft = "# Chapter One\n\nThe stars blinked awake.\n\n---\n\n**The** end came softly."
        lines = g._scriptify(draft)
        texts = " ".join(ln["text"] for ln in lines)
        self.assertNotIn("#", texts)
        self.assertNotIn("---", texts)
        self.assertNotIn("*", texts)

    def test_quoteless_story_yields_lines(self):
        lines = g._scriptify("Hello world.")
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["character"], "Narrator")


class TestEmotion(unittest.TestCase):
    def test_keywords_and_punctuation(self):
        self.assertEqual(g._emotion_for("Time to sleep now.", True), "sleepy")
        self.assertEqual(g._emotion_for("Look out!", False), "excited")
        self.assertEqual(g._emotion_for("What is that?", False), "curious")
        self.assertEqual(g._emotion_for("The hills were green.", True), "warm")
        self.assertEqual(g._emotion_for("The hills were green.", False), "gentle")


class TestVoiceResolution(unittest.TestCase):
    def test_auto_speaker_is_english_only_and_deterministic(self):
        names = ["Pip", "Grandmother Willow", "Zork", "Captain Pebble", "Луна"]
        for n in names:
            self.assertIn(g._auto_speaker(n), config.ENGLISH_PRESETS)
            self.assertEqual(g._auto_speaker(n), g._auto_speaker(n))

    def test_pick_speaker_for_voice(self):
        self.assertEqual(g._pick_speaker_for_voice("warm, gentle woman", ""), "Serena")
        self.assertEqual(g._pick_speaker_for_voice("gruff, low, kindly older male", "male"), "Uncle_Fu")
        self.assertEqual(g._pick_speaker_for_voice("dynamic and bold", "male"), "Ryan")
        self.assertEqual(g._pick_speaker_for_voice("", ""), "Aiden")

    def test_narrator_clones_when_reference_exists(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            ref = f.name
        self.addCleanup(os.unlink, ref)
        old = config.NARRATOR_REF_AUDIO
        self.addCleanup(setattr, config, "NARRATOR_REF_AUDIO", old)
        config.NARRATOR_REF_AUDIO = ref
        v = g._line_voice("Narrator", "warm", {})
        self.assertEqual(v["voice_mode"], "clone")

    def test_narrator_falls_back_to_preset_without_reference(self):
        old = config.NARRATOR_REF_AUDIO
        self.addCleanup(setattr, config, "NARRATOR_REF_AUDIO", old)
        config.NARRATOR_REF_AUDIO = os.path.join(tempfile.gettempdir(), "nope_missing.wav")
        v = g._line_voice("Narrator", "warm", {})
        self.assertEqual(v["voice_mode"], "preset")
        self.assertEqual(v["speaker"], config.NARRATOR_SPEAKER)

    def test_saved_character_with_description_uses_design(self):
        saved = {"pip": {"id": 7, "voice_mode": "preset", "speaker": "Aiden",
                         "voice_description": "bright, playful young female"}}
        v = g._line_voice("Pip", "terrified", saved)
        self.assertEqual(v["voice_mode"], "design")
        self.assertEqual(v["character_id"], 7)
        self.assertIn("bright, playful young female", v["voice_instruct"])
        self.assertIn("terrified", v["voice_instruct"])

    def test_saved_character_with_clone_ref_uses_clone(self):
        saved = {"gran": {"id": 3, "voice_mode": "clone",
                          "ref_audio_path": "x.wav", "ref_text": "hi"}}
        v = g._line_voice("Gran", "warm", saved)
        self.assertEqual(v["voice_mode"], "clone")

    def test_saved_character_with_explicit_preset_is_honored(self):
        saved = {"captain pebble": {"id": 4, "voice_mode": "preset", "speaker": "Ryan"}}
        v = g._line_voice("Captain Pebble", "excited", saved)
        self.assertEqual(v["voice_mode"], "preset")
        self.assertEqual(v["speaker"], "Ryan")

    def test_unknown_character_gets_english_preset(self):
        v = g._line_voice("Mysterious Fox", "curious", {})
        self.assertEqual(v["voice_mode"], "preset")
        self.assertIn(v["speaker"], config.ENGLISH_PRESETS)


def _line(id_, seq, who, text, mode="preset"):
    return {"id": id_, "seq": seq, "character_name": who, "text": text,
            "voice_mode": mode, "speaker": None, "voice_instruct": None,
            "character_id": None, "emotion": "warm"}


class TestMergeRenderGroups(unittest.TestCase):
    def test_consecutive_narrator_lines_merge_and_characters_stay_solo(self):
        stored = [
            _line(1, 0, "Narrator", "The night was still.", "clone"),
            _line(2, 1, "Narrator", "Stars filled the sky.", "clone"),
            _line(3, 2, "Pip", "Look up there!", "design"),
            _line(4, 3, "Narrator", "And so they did.", "clone"),
        ]
        groups = g._merge_render_groups(stored, max_chars=380)
        self.assertEqual(len(groups), 3)
        self.assertEqual(groups[0]["line_ids"], [1, 2])
        self.assertEqual(groups[0]["text"], "The night was still. Stars filled the sky.")
        self.assertEqual(groups[1]["line_ids"], [3])
        self.assertEqual(groups[2]["line_ids"], [4])

    def test_soft_char_cap_starts_a_new_group(self):
        stored = [
            _line(1, 0, "Narrator", "a" * 30),
            _line(2, 1, "Narrator", "b" * 30),
        ]
        groups = g._merge_render_groups(stored, max_chars=40)
        self.assertEqual(len(groups), 2)

    def test_character_line_breaks_the_narrator_run(self):
        stored = [
            _line(1, 0, "Narrator", "One."),
            _line(2, 1, "Pip", "Two!"),
            _line(3, 2, "Narrator", "Three."),
            _line(4, 3, "Narrator", "Four."),
        ]
        groups = g._merge_render_groups(stored, max_chars=380)
        self.assertEqual([grp["line_ids"] for grp in groups], [[1], [2], [3, 4]])


class TestConcatWavs(unittest.TestCase):
    def _write_wav(self, path, seconds, rate):
        with wave.open(path, "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(b"\x00\x00" * int(seconds * rate))

    def test_mixed_sample_rates_are_unified(self):
        tmp = tempfile.mkdtemp()
        a = os.path.join(tmp, "a.wav")
        b = os.path.join(tmp, "b.wav")
        out = os.path.join(tmp, "out.wav")
        self._write_wav(a, 1.0, 22050)
        self._write_wav(b, 1.0, 24000)   # different rate: must be resampled
        g._concat_wavs([a, b], out, gap_seconds=0.28)
        with wave.open(out, "rb") as w:
            self.assertEqual(w.getframerate(), 22050)
            duration = w.getnframes() / w.getframerate()
        # 1 s + 0.28 s gap + 1 s (resampled), small tolerance for rounding
        self.assertAlmostEqual(duration, 2.28, delta=0.05)

    def test_missing_paths_yield_placeholder(self):
        tmp = tempfile.mkdtemp()
        out = os.path.join(tmp, "out.wav")
        g._concat_wavs([os.path.join(tmp, "missing.wav")], out)
        self.assertTrue(os.path.exists(out))


class TestInstructBuilders(unittest.TestCase):
    def test_design_instruct_combines_description_and_emotion(self):
        s = g._design_instruct("a gruff old badger", "grumpy")
        self.assertIn("a gruff old badger", s)
        self.assertIn("grumpy", s)

    def test_design_instruct_defaults(self):
        s = g._design_instruct(None, None)
        self.assertIn("warm", s)

    def test_voice_instruct(self):
        self.assertIn("sleepy", g._voice_instruct("sleepy"))


if __name__ == "__main__":
    unittest.main()
