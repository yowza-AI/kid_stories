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


class TestSpeakerAttribution(unittest.TestCase):
    """Who says what. Getting this wrong is inaudible in the text and casts the
    whole scene in one voice in the audio."""

    def _spoken(self, draft, **kw):
        return [(l["character"], l["text"])
                for l in g._scriptify(draft, **kw) if l["character"] != "Narrator"]

    def test_pronoun_tag_resolves_to_last_named_actor(self):
        draft = ('Auditor Crake went down the list and did not look up.\n\n'
                 '"Vane. Chalk your sigil and pay," she said.')
        self.assertEqual(self._spoken(draft), [("Crake", "Vane. Chalk your sigil and pay,")])

    def test_forename_and_surname_are_one_character(self):
        draft = ('Bursar Ilsabet Quorn stood at the rail.\n\n'
                 '"Bond thirty-three," said Ilsabet.\n\n'
                 '"It is a record," Quorn added.')
        self.assertEqual({c for c, _ in self._spoken(draft)}, {"Ilsabet"})

    def test_untagged_exchange_alternates_between_paragraphs(self):
        # both speakers have to be established first — nothing can alternate
        # onto a character the text has never mentioned
        draft = ('"Sixty is honest," Phillip said.\n\n'
                 'Nessa put her hands flat on the desk.\n\n'
                 '"You were."\n\n"I wasn\'t going to."\n\n"You were."')
        self.assertEqual([c for c, _ in self._spoken(draft)],
                         ["Phillip", "Nessa", "Phillip", "Nessa"])

    def test_speaker_continues_through_own_action_beat(self):
        # same paragraph: the narration between the quotes is her, still speaking
        draft = ('Nessa got up.\n\n'
                 '"Don\'t." She was already gathering her chalk. "Don\'t gantry me."')
        self.assertEqual([c for c, _ in self._spoken(draft)], ["Nessa", "Nessa"])

    def test_person_addressed_by_name_is_not_the_speaker(self):
        draft = ('"Sixty is honest," Phillip said.\n\n'
                 'Nessa got up off her knees.\n\n"Nessa —"\n\n"Don\'t."')
        self.assertEqual([c for c, _ in self._spoken(draft)][1:], ["Phillip", "Nessa"])

    def test_vocative_after_a_comma_also_counts(self):
        draft = ('Bursar Ilsabet Quorn watched from the rail.\n\n'
                 '"Thank you, Bursar," Phillip Vane said.\n\n'
                 '"It is not praise, Mr Vane. It is a record."')
        self.assertEqual(self._spoken(draft)[1][0], "Ilsabet")

    def test_place_names_never_become_speakers(self):
        # "the Tally Floor goes silent" has the shape of an action beat
        draft = ('The Tally Floor goes silent when somebody spends.\n\n'
                 '"It is iron, and it is cold."')
        self.assertNotIn("Floor", [c for c, _ in self._spoken(draft)])

    def test_action_beat_still_names_the_speaker(self):
        # the old behaviour must survive the people-filter
        self.assertEqual(self._spoken('Ezra nodded eagerly. "Yes!"'), [("Ezra", "Yes!")])

    def test_known_cast_seeds_attribution(self):
        draft = 'Marrable turned the key.\n\n"Nine galleries."'
        self.assertEqual(self._spoken(draft, known_names=["Marrable"])[0][0], "Marrable")

    def test_titles_are_not_names(self):
        self.assertEqual(g._strip_titles("Bursar Ilsabet Quorn and Mr Vane"),
                         "Ilsabet Quorn and Vane")
        self.assertNotIn("Mr", g._cast_names('"Yes," said Mr Vane.'))

    def test_shared_surname_stays_ambiguous(self):
        # Phillip Vane and Tobin Vane both claim "Vane" — mapping it would put
        # the father's lines in the son's voice
        alias = g._alias_map('Phillip Vane rowed. Tobin Vane said, "Get the rope in."')
        self.assertNotIn("Vane", alias)


class TestMidSpeechTagReflow(unittest.TestCase):
    """A tag between two halves of one speech is put back where it belongs.
    Needed for prose we did not write — the concierge flow pastes in text that
    follows none of our conventions."""

    def test_interrupted_speech_is_rejoined(self):
        lines = g._scriptify(
            'Nessa Bellhook waited.\n\n'
            '"Hundredweight," Nessa said, and did not look at the iron. '
            '"Off the boards."')
        self.assertEqual(
            [(l["character"], l["text"]) for l in lines if l["text"].strip()][-2:],
            [("Nessa", "Hundredweight. Off the boards."),
             ("Narrator", "Nessa said, and did not look at the iron.")])

    def test_continuing_clause_keeps_its_comma(self):
        # the second half starts lowercase: it is the same clause, not a new one
        self.assertEqual(
            g._join_speech("The winch handle at Lock Four,", "in November."),
            "The winch handle at Lock Four, in November.")

    def test_join_promotes_a_tag_comma_to_a_stop(self):
        self.assertEqual(g._join_speech("Hundredweight,", "Off the boards."),
                         "Hundredweight. Off the boards.")

    def test_different_speakers_are_never_joined(self):
        lines = g._scriptify(
            'Hew Marrable turned the key.\n\n'
            '"Nine galleries," Marrable said, steering him back. "Don\'t touch."')
        spoken = [l for l in lines if l["character"] != "Narrator"]
        self.assertEqual(len(spoken), 1)          # same speaker -> one clip
        self.assertIn("Nine galleries", spoken[0]["text"])


class TestParagraphsAndGaps(unittest.TestCase):
    """Prosody: paragraph structure must survive scriptify, and the gap between
    two clips must reflect what the boundary means."""

    DRAFT = (
        '"Hundredweight. Count of sixty," Nessa said, and did not once look at '
        'the iron while she said it.\n\n'
        '"Sixty\'s honest," Phillip said.\n\n'
        'The nibs stopped. Three hundred of them, all at once.'
    )

    def test_paragraph_index_survives_scriptify(self):
        lines = g._scriptify(self.DRAFT)
        self.assertEqual(sorted({l["para"] for l in lines}), [0, 1, 2])

    def test_loaded_tag_is_kept_and_marked(self):
        # >6 words, so it survives the bare-tag filter and becomes a real clip
        lines = g._scriptify(self.DRAFT)
        tags = [l for l in lines if l["is_tag"]]
        self.assertEqual(len(tags), 1)
        self.assertIn("did not once look at the iron", tags[0]["text"])
        self.assertEqual(tags[0]["character"], "Narrator")

    def test_bare_tag_is_still_dropped(self):
        lines = g._scriptify('"Sixty\'s honest," Phillip said.')
        self.assertEqual([l["character"] for l in lines], ["Phillip"])

    def test_narration_after_narration_is_not_a_tag(self):
        # a tag only belongs to the line before it if that line was speech
        lines = g._scriptify("He knelt on the boards. Crake said nothing at all for a while.")
        self.assertFalse(any(l["is_tag"] for l in lines))

    def test_narrator_lines_do_not_merge_across_paragraphs(self):
        stored = [
            {"id": 1, "seq": 0, "character_name": "Narrator", "text": "One.", "para": 0},
            {"id": 2, "seq": 1, "character_name": "Narrator", "text": "Two.", "para": 0},
            {"id": 3, "seq": 2, "character_name": "Narrator", "text": "Three.", "para": 1},
        ]
        groups = g._merge_render_groups(stored, max_chars=400)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["text"], "One. Two.")
        self.assertEqual(groups[1]["text"], "Three.")

    def _groups(self):
        stored = [
            {"id": 1, "seq": 0, "character_name": "Nessa", "text": "Count of sixty.",
             "para": 0, "is_tag": 0},
            {"id": 2, "seq": 1, "character_name": "Narrator", "para": 0, "is_tag": 1,
             "text": "Nessa said, and did not once look at the iron."},
            {"id": 3, "seq": 2, "character_name": "Narrator", "para": 0, "is_tag": 0,
             "text": "The nibs stopped."},
            {"id": 4, "seq": 3, "character_name": "Narrator", "para": 1, "is_tag": 0,
             "text": "They called him first."},
        ]
        return g._merge_render_groups(stored, max_chars=20)

    def test_three_gap_tiers(self):
        gaps = g._gaps_for_groups(self._groups())
        self.assertEqual(gaps, [config.TTS_GAP_TAG,       # speech -> its own tag
                                config.TTS_GAP_SECONDS,   # tag -> next line, same para
                                config.TTS_GAP_PARA])     # across a paragraph break
        self.assertLess(config.TTS_GAP_TAG, config.TTS_GAP_SECONDS)
        self.assertLess(config.TTS_GAP_SECONDS, config.TTS_GAP_PARA)

    def test_tag_after_narration_gets_an_ordinary_gap(self):
        # is_tag only tightens the gap when a CHARACTER was speaking before it
        groups = [{"para": 0, "is_tag": False, "is_narrator": True},
                  {"para": 0, "is_tag": True, "is_narrator": True}]
        self.assertEqual(g._gaps_for_groups(groups), [config.TTS_GAP_SECONDS])

    def test_concat_uses_per_boundary_gaps(self):
        tmp = tempfile.mkdtemp()
        paths = []
        for i in range(3):
            p = os.path.join(tmp, f"c{i}.wav")
            g._write_silence(p, 0.5)
            paths.append(p)
        out = os.path.join(tmp, "out.wav")
        g._concat_wavs(paths, out, gaps=[0.1, 0.9])
        with wave.open(out, "rb") as w:
            duration = w.getnframes() / w.getframerate()
        self.assertAlmostEqual(duration, 1.5 + 0.1 + 0.9, delta=0.05)

    def test_missing_clip_does_not_shift_remaining_gaps(self):
        # clip 1 never rendered; the join before clip 2 must still use gaps[1]
        tmp = tempfile.mkdtemp()
        a = os.path.join(tmp, "a.wav"); g._write_silence(a, 0.5)
        c = os.path.join(tmp, "c.wav"); g._write_silence(c, 0.5)
        out = os.path.join(tmp, "out.wav")
        g._concat_wavs([a, os.path.join(tmp, "gone.wav"), c], out, gaps=[0.1, 0.9])
        with wave.open(out, "rb") as w:
            duration = w.getnframes() / w.getframerate()
        self.assertAlmostEqual(duration, 1.0 + 0.9, delta=0.05)


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
