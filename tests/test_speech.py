# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Autonomous checks for pronunciation/acoustic/speech.py.

Two layers:
  1. Fast unit tests over the pure logic (scoring, DTW alignment, helpers).
     They never load the Wav2Vec2 model, so they run offline and quickly.
  2. An optional end-to-end runner (``python tests/test_speech.py user.wav
     [ref.wav]``) that exercises ``analyze`` on real audio. This downloads the
     ~1.2 GB model and needs espeak-ng installed, so it is gated behind the CLI
     and is not run by the unit tests.

Run unit tests:   python -m unittest tests.test_speech
End-to-end check: python tests/test_speech.py path/to/user.wav [path/to/ref.wav]
"""

import sys
import unittest
from typing import Optional

import numpy as np

from pronunciation.acoustic import speech


class TestPureLogic(unittest.TestCase):
    """Tests that need no model weights."""

    def test_score_is_perfect_for_zero_distances(self):
        # Self-comparison (the Test button): zero acoustic distance, no errors.
        self.assertEqual(speech.compute_pronunciation_score(0, 0, 0), 100.0)

    def test_score_is_clamped_to_zero_for_terrible_attempt(self):
        score = speech.compute_pronunciation_score(2.0, 1.0, 1.0)
        self.assertEqual(score, 0.0)

    def test_score_stays_within_bounds(self):
        score = speech.compute_pronunciation_score(0.3, 0.2, 0.1)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)

    def test_score_decreases_as_distance_grows(self):
        # Explicit floor/ceiling keep the test independent of calibration.json.
        better = speech.compute_pronunciation_score(
            0.25, 0.05, 0.02, acoustic_bad=0.6, acoustic_good=0.2)
        worse = speech.compute_pronunciation_score(
            0.50, 0.50, 0.40, acoustic_bad=0.6, acoustic_good=0.2)
        self.assertGreater(better, worse)

    def test_acoustic_floor_yields_full_acoustic_component(self):
        # At the floor the acoustic component is 100 -> its 40% weight survives
        # even when the other components are zeroed out.
        score = speech.compute_pronunciation_score(
            0.2, 1.0, 1.0, acoustic_bad=0.6, acoustic_good=0.2)
        self.assertEqual(score, 40.0)

    def test_acoustic_ceiling_adapts_to_baseline(self):
        low = speech.acoustic_bad_for(0.3, acoustic_good=0.2)
        high = speech.acoustic_bad_for(0.8, acoustic_good=0.2)
        self.assertGreater(high, low)
        # The ceiling never collapses onto the floor.
        degenerate = speech.acoustic_bad_for(0.0, acoustic_good=0.2)
        self.assertGreaterEqual(degenerate, 0.2 + speech.ACOUSTIC_MIN_SPAN)

    def test_score_is_length_invariant(self):
        # The same error *rates* must give the same score regardless of the
        # phrase length they came from (the original bug: absolute distances).
        short = speech.compute_pronunciation_score(
            0.3, 2 / 10, 3 / 15, acoustic_bad=0.6, acoustic_good=0.2)
        long = speech.compute_pronunciation_score(
            0.3, 8 / 40, 12 / 60, acoustic_bad=0.6, acoustic_good=0.2)
        self.assertEqual(short, long)

    def test_random_pair_baseline_handles_empty_embeddings(self):
        # Near-empty audio yields zero embedding frames; the baseline must fall
        # back to the fixed ceiling instead of crashing on integers(0, 0).
        empty = np.zeros((0, 4), dtype=np.float32)
        frames = np.ones((5, 4), dtype=np.float32)
        self.assertEqual(speech._random_pair_baseline(empty, frames),
                         speech.ACOUSTIC_BAD_DEFAULT)
        self.assertEqual(speech._random_pair_baseline(frames, empty),
                         speech.ACOUSTIC_BAD_DEFAULT)

    def test_clean_transcription_normalises_text(self):
        self.assertEqual(speech.clean_transcription("  Hello, WORLD!! "), "hello world")

    def test_prepare_waveform_downmixes_and_resamples(self):
        # 2 channels of 24 kHz audio -> mono 16 kHz.
        stereo_24k = np.ones((2, 24_000), dtype=np.float32)
        out = speech._prepare_waveform(stereo_24k, orig_sr=24_000)
        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.dtype, np.float32)
        # 1 s at 24 kHz -> ~1 s at 16 kHz.
        self.assertAlmostEqual(len(out), 16_000, delta=50)

    def test_result_has_spec_fields(self):
        result = speech.PronunciationResult(
            score=88.0, word_errors=[], prosody={"f0": [], "energy": []},
            transcription="hello world")
        self.assertTrue(hasattr(result, "score"))
        self.assertTrue(hasattr(result, "word_errors"))
        self.assertTrue(hasattr(result, "prosody"))
        self.assertTrue(hasattr(result, "transcription"))

    # --- word_level_diff: target-vs-ASR alignment --------------------------
    def test_word_diff_empty_on_exact_match(self):
        # Case/punctuation are normalised away, so this counts as a clean match.
        self.assertEqual(
            speech.word_level_diff("i like the time", "I like the time."), [])

    def test_word_diff_reports_substitution(self):
        self.assertEqual(
            speech.word_level_diff("i like the times", "i like the time"),
            [{"expected": "time", "heard": "times"}])

    def test_word_diff_reports_deletion(self):
        # The target word "the" is dropped from the attempt -> heard side empty.
        self.assertEqual(
            speech.word_level_diff("i like time", "i like the time"),
            [{"expected": "the", "heard": ""}])

    def test_word_diff_reports_insertion(self):
        # An extra recognised word -> expected side empty.
        self.assertEqual(
            speech.word_level_diff("i like the time", "i like time"),
            [{"expected": "", "heard": "the"}])

    # --- heard_word_tags: per-word correctness for the ASR line ------------
    def test_heard_word_tags_marks_matches_and_mismatches(self):
        # "hullo" is wrong; the rest match the target word for word.
        tags = speech.heard_word_tags(
            "hullo i am at a train station", "hello i am at a train station")
        self.assertEqual([t["word"] for t in tags],
                         ["hullo", "i", "am", "at", "a", "train", "station"])
        self.assertEqual([t["correct"] for t in tags],
                         [False, True, True, True, True, True, True])

    def test_heard_word_tags_all_correct_on_exact_match(self):
        tags = speech.heard_word_tags("i am here", "I am here.")
        self.assertTrue(all(t["correct"] for t in tags))

    # --- reference_word_tags: engine-neutral target-phrase highlighting --------
    def test_reference_word_tags_preserves_tokens_and_flags_errors(self):
        # Original case/punctuation kept for display; "world" flagged from the
        # (normalised) error list.
        tags = speech.reference_word_tags("Hello, world!", ["world"])
        self.assertEqual([t["word"] for t in tags], ["Hello,", "world!"])
        self.assertEqual([t["correct"] for t in tags], [True, False])

    def test_reference_word_tags_all_correct_when_no_errors(self):
        tags = speech.reference_word_tags("i am here", [])
        self.assertTrue(tags and all(t["correct"] for t in tags))


# Note: the prosody-visualisation helpers (to_semitones, resample_series) live in
# mimora/prosody_utils.py (tests: tests/test_prosody_utils.py). The prosody
# *extraction* (extract_f0/energy, interpolate_f0) moved to mimora/prosody.py
# (tests: tests/test_prosody.py) — this engine no longer owns prosody.


def _run_end_to_end(user_path: str, reference_path: Optional[str]) -> None:
    """Run the full analyze() pipeline on real WAV files (loads the model)."""
    import soundfile as sf

    user_audio, user_sr = sf.read(user_path, dtype="float32")
    if reference_path:
        reference_audio, reference_sr = sf.read(reference_path, dtype="float32")
    else:
        # No reference supplied: reuse the user's own audio. A faithful repetition
        # of itself should score high — a quick sanity check of the pipeline.
        print("No reference WAV given; using the user audio as its own reference.")
        reference_audio, reference_sr = user_audio, user_sr

    expected_text = input("Expected text for this recording: ").strip()

    print("Loading Wav2Vec2 (first run downloads ~1.2 GB)...")
    speech.load_models()

    result = speech.analyze(
        user_audio=user_audio,
        expected_text=expected_text,
        reference_audio=reference_audio,
        user_sr=user_sr,
        reference_sr=reference_sr,
    )

    print(f"\nScore:         {result.score} (passed={result.passed})")
    print(f"Transcription: {result.transcription!r}")
    print(f"Problem words: {result.words_with_errors}")
    print(f"Acoustic DTW:  {result.acoustic_distance} total, "
          f"{result.acoustic_per_step:.4f}/step (baseline {result.acoustic_baseline:.4f})")
    print(result.feedback)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        _run_end_to_end(sys.argv[1], sys.argv[2] if len(sys.argv) >= 3 else None)
    else:
        unittest.main()
