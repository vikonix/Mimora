# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Autonomous checks for pronunciation/phoneme/speech.py.

Two layers (mirrors tests/test_speech.py for the acoustic engine):
  1. Fast unit tests over the pure scoring logic (tokenization, inventory fold,
     edit alignment, per-phone distance capping, the [good, bad] score map, recall,
     0-5 bucketization and the per-word highlight). They never load the wav2vec2
     phoneme model, so they run offline and quickly.
  2. An optional end-to-end runner (``python tests/test_phoneme.py user.wav
     [ref.wav]``) that exercises ``analyze`` on real audio. That path downloads the
     ~1.2 GB recognizer and needs espeak-ng + panphon installed, so it is gated
     behind the CLI and is not run by the unit tests.

The articulatory substitution cost (``_substitution_cost``) is backed by panphon's
feature table; only its ``a == b -> 0.0`` short-circuit is exercised directly. Every
test of a *higher-level* algorithm (alignment, recall, scoring, word mapping) stubs
the cost with a deterministic ``0.0 if equal else 1.0`` so the logic is verified
without depending on panphon's data being installed.

Run unit tests:   python -m unittest tests.test_phoneme
End-to-end check: python tests/test_phoneme.py path/to/user.wav [path/to/ref.wav]
"""

import sys
import unittest
from typing import Optional
from unittest import mock

import numpy as np

from pronunciation.phoneme import speech


def _binary_cost(a: str, b: str) -> float:
    """Deterministic stand-in for the panphon-backed substitution cost.

    Identical phones cost 0; any mismatch costs the full 1.0. This makes the
    feature-weighted edit distance behave like a plain token Levenshtein, so the
    higher-level algorithms can be asserted exactly without panphon installed.
    """
    return 0.0 if a == b else 1.0


def _patch_cost():
    """Context manager swapping the substitution cost for the binary stub."""
    return mock.patch.object(speech, "_substitution_cost", _binary_cost)


class TestTokenizationAndFold(unittest.TestCase):
    """Phone splitting, diacritic stripping and the inventory fold (no deps)."""

    def test_tokenize_splits_on_whitespace(self):
        self.assertEqual(speech._tokenize_ipa("h ə l oʊ"), ["h", "ə", "l", "oʊ"])

    def test_tokenize_falls_back_to_per_character(self):
        # No separators -> one symbol per (non-space) character.
        self.assertEqual(speech._tokenize_ipa("kæt"), ["k", "æ", "t"])

    def test_normalize_folds_rhotic_and_vowels(self):
        # ɹ -> r, æ -> a, oʊ -> o : recognizer and espeak conventions are unified.
        self.assertEqual(speech._normalize_phones(["ɹ", "æ", "oʊ"]), ["r", "a", "o"])

    def test_normalize_strips_length_mark(self):
        # The length mark (U+02D0) is emitted by one side only; it must be dropped
        # so "iː" and "i" align instead of looking like a substitution.
        self.assertEqual(speech._normalize_phones(["iː"]), ["i"])

    def test_normalize_drops_empty_tokens(self):
        # A token that is nothing but a stripped diacritic disappears entirely.
        self.assertEqual(speech._normalize_phones(["ː", "a"]), ["a"])


class TestSubstitutionCost(unittest.TestCase):
    """Only the panphon-free short-circuit of the real cost is exercised here."""

    def test_identical_phones_cost_zero(self):
        # a == b returns 0.0 before any panphon lookup, so this needs no data files.
        self.assertEqual(speech._substitution_cost("a", "a"), 0.0)


class TestEditAlignment(unittest.TestCase):
    """Feature-weighted edit distance over phone-token lists (cost stubbed)."""

    def test_identical_sequences_align_with_zero_distance(self):
        with _patch_cost():
            pairs, distance = speech._edit_alignment(["a", "b", "c"], ["a", "b", "c"])
        self.assertEqual(distance, 0.0)
        self.assertEqual(pairs, [("a", "a"), ("b", "b"), ("c", "c")])

    def test_deletion_emits_empty_spoken_side(self):
        # Reference "b" never spoken -> ("b", "") pair, distance 1.0.
        with _patch_cost():
            pairs, distance = speech._edit_alignment(["a", "b", "c"], ["a", "c"])
        self.assertEqual(distance, 1.0)
        self.assertIn(("b", ""), pairs)

    def test_insertion_emits_empty_reference_side(self):
        # Extra spoken "x" -> ("", "x") pair, distance 1.0.
        with _patch_cost():
            pairs, distance = speech._edit_alignment(["a", "b"], ["a", "x", "b"])
        self.assertEqual(distance, 1.0)
        self.assertIn(("", "x"), pairs)

    def test_substitution_counts_one_pair(self):
        with _patch_cost():
            pairs, distance = speech._edit_alignment(["a", "b"], ["a", "z"])
        self.assertEqual(distance, 1.0)
        self.assertEqual(pairs, [("a", "a"), ("b", "z")])


class TestCappedPerPhoneDistance(unittest.TestCase):
    """Insertions are capped per reference phone; real errors pass through."""

    def test_no_insertions_is_plain_mean(self):
        pairs = [("a", "a"), ("b", "z")]   # one substitution, no insertions
        self.assertEqual(
            speech._capped_per_phone_distance(pairs, distance=1.0, n_reference=2), 0.5)

    def test_insertions_capped_per_reference_phone(self):
        # 10 insertions (distance 10) but the cap is 0.25 * 4 = 1.0, so the
        # per-phone distance can rise to at most 1.0 / 4 = 0.25 from insertions.
        pairs = [("", "x")] * 10
        with mock.patch.object(speech, "INSERTION_CAP_PER_PHONE", 0.25):
            out = speech._capped_per_phone_distance(pairs, distance=10.0, n_reference=4)
        self.assertAlmostEqual(out, 0.25)

    def test_zero_reference_returns_zero(self):
        self.assertEqual(
            speech._capped_per_phone_distance([], distance=0.0, n_reference=0), 0.0)

    def test_disabled_cap_is_plain_mean(self):
        pairs = [("", "x"), ("", "y")]
        with mock.patch.object(speech, "INSERTION_CAP_PER_PHONE", 0.0):
            out = speech._capped_per_phone_distance(pairs, distance=2.0, n_reference=2)
        self.assertEqual(out, 1.0)


class TestScoreFromDistance(unittest.TestCase):
    """Mapping a per-phone distance onto 0-100 against the [good, bad] window."""

    def test_distance_at_good_scores_100(self):
        self.assertEqual(speech._score_from_distance(0.10, bad=0.50, good=0.10), 100.0)

    def test_distance_at_bad_scores_0(self):
        self.assertEqual(speech._score_from_distance(0.50, bad=0.50, good=0.10), 0.0)

    def test_midpoint_scores_50(self):
        self.assertEqual(speech._score_from_distance(0.30, bad=0.50, good=0.10), 50.0)

    def test_monotonic_decreasing_in_distance(self):
        better = speech._score_from_distance(0.20, bad=0.50, good=0.10)
        worse = speech._score_from_distance(0.40, bad=0.50, good=0.10)
        self.assertGreater(better, worse)

    def test_clamped_within_bounds(self):
        # Distance below good clamps to 100; far above bad clamps to 0.
        self.assertEqual(speech._score_from_distance(0.0, bad=0.5, good=0.1), 100.0)
        self.assertEqual(speech._score_from_distance(5.0, bad=0.5, good=0.1), 0.0)

    def test_min_span_guards_degenerate_window(self):
        # good == bad would divide by ~0; BAD_MIN_SPAN keeps the map finite.
        out = speech._score_from_distance(0.10, bad=0.10, good=0.10)
        self.assertEqual(out, 100.0)


class TestWidenBadForLength(unittest.TestCase):
    """Short phrases widen the noisy ``bad`` anchor toward a ceiling."""

    def test_disabled_returns_observed(self):
        with mock.patch.object(speech, "BAD_SHRINK_PHONES", 0):
            self.assertEqual(speech._widen_bad_for_length(0.2, n_reference=3), 0.2)

    def test_short_phrase_widens_more_than_long(self):
        with mock.patch.object(speech, "BAD_SHRINK_PHONES", 12), \
             mock.patch.object(speech, "BAD_CEILING", 0.40):
            short = speech._widen_bad_for_length(0.2, n_reference=2)
            long = speech._widen_bad_for_length(0.2, n_reference=100)
        self.assertGreater(short, long)

    def test_never_lowers_below_observed(self):
        # max(observed, ceiling) guarantees widening only ever raises ``bad``.
        with mock.patch.object(speech, "BAD_SHRINK_PHONES", 12), \
             mock.patch.object(speech, "BAD_CEILING", 0.40):
            self.assertGreaterEqual(speech._widen_bad_for_length(0.2, 2), 0.2)
            # An already-high observed bad is left at/above itself, not pulled down.
            self.assertGreaterEqual(speech._widen_bad_for_length(0.8, 2), 0.8)


class TestPhonemeRecall(unittest.TestCase):
    """Fraction of reference phones actually produced, read off the alignment."""

    def test_all_recalled(self):
        pairs = [("a", "a"), ("b", "b")]
        with _patch_cost():
            self.assertEqual(speech._phoneme_recall(pairs), 1.0)

    def test_insertions_are_ignored(self):
        # ("", "x") has no reference phone, so it neither helps nor hurts recall.
        pairs = [("a", "a"), ("", "x"), ("b", "b")]
        with _patch_cost():
            self.assertEqual(speech._phoneme_recall(pairs), 1.0)

    def test_deletion_lowers_recall(self):
        # "b" was deleted -> 1 of 2 reference phones recalled.
        pairs = [("a", "a"), ("b", "")]
        with _patch_cost():
            self.assertEqual(speech._phoneme_recall(pairs), 0.5)

    def test_empty_alignment_is_zero(self):
        self.assertEqual(speech._phoneme_recall([]), 0.0)


class TestScoreToBucket(unittest.TestCase):
    """Coarse 0-5 grade: how many ascending cutpoints a score clears (task §4)."""

    def test_no_cutpoints_returns_minus_one(self):
        with mock.patch.object(speech, "BUCKET_CUTPOINTS", []):
            self.assertEqual(speech._score_to_bucket(80.0), -1)

    def test_counts_cleared_cutpoints(self):
        with mock.patch.object(speech, "BUCKET_CUTPOINTS", [10.0, 20.0, 30.0]):
            self.assertEqual(speech._score_to_bucket(25.0), 2)

    def test_score_on_cutpoint_clears_it(self):
        # The comparison is ``>=``, so a score sitting exactly on a cutpoint counts.
        with mock.patch.object(speech, "BUCKET_CUTPOINTS", [10.0, 20.0, 30.0]):
            self.assertEqual(speech._score_to_bucket(30.0), 3)

    def test_below_all_cutpoints_is_zero(self):
        with mock.patch.object(speech, "BUCKET_CUTPOINTS", [10.0, 20.0, 30.0]):
            self.assertEqual(speech._score_to_bucket(5.0), 0)


class TestBucketToPercent(unittest.TestCase):
    """User-facing percent for a bucket: the midpoint of its [lo, hi] band."""

    def test_midpoint_of_band(self):
        with mock.patch.object(speech, "BUCKET_TO_PERCENT", {"4": [90, 95]}):
            self.assertEqual(speech._bucket_to_percent(4, fallback=0.0), 92.5)

    def test_missing_band_falls_back(self):
        with mock.patch.object(speech, "BUCKET_TO_PERCENT", {"4": [90, 95]}):
            self.assertEqual(speech._bucket_to_percent(9, fallback=77.0), 77.0)


class TestGradeForScore(unittest.TestCase):
    """0-5 grade with a +/- shade from the score's third of its bucket's range."""

    def test_no_cutpoints_is_ungraded(self):
        with mock.patch.object(speech, "BUCKET_CUTPOINTS", []):
            self.assertEqual(speech._grade_for_score(2, 25.0), ("", -1.0))

    def test_negative_bucket_is_ungraded(self):
        with mock.patch.object(speech, "BUCKET_CUTPOINTS", [10.0, 20.0, 30.0]):
            self.assertEqual(speech._grade_for_score(-1, 25.0), ("", -1.0))

    def test_middle_third_is_plain(self):
        # Bucket 2 spans [20, 30); 25 sits in the middle third.
        with mock.patch.object(speech, "BUCKET_CUTPOINTS", [10.0, 20.0, 30.0]):
            self.assertEqual(speech._grade_for_score(2, 25.0), ("2", 2.0))

    def test_lower_third_is_minus(self):
        with mock.patch.object(speech, "BUCKET_CUTPOINTS", [10.0, 20.0, 30.0]):
            self.assertEqual(speech._grade_for_score(2, 21.0), ("2-", 1.67))

    def test_upper_third_is_plus(self):
        with mock.patch.object(speech, "BUCKET_CUTPOINTS", [10.0, 20.0, 30.0]):
            self.assertEqual(speech._grade_for_score(2, 29.0), ("2+", 2.33))

    def test_top_bucket_extends_to_hundred(self):
        # Top bucket spans [30, 100]; 40 is still its lower third, 100 clamps to "+".
        with mock.patch.object(speech, "BUCKET_CUTPOINTS", [10.0, 20.0, 30.0]):
            self.assertEqual(speech._grade_for_score(3, 40.0)[0], "3-")
            self.assertEqual(speech._grade_for_score(3, 100.0)[0], "3+")

    def test_bottom_bucket_starts_at_zero(self):
        # Bucket 0 spans [0, 10); 5 is its middle third.
        with mock.patch.object(speech, "BUCKET_CUTPOINTS", [10.0, 20.0, 30.0]):
            self.assertEqual(speech._grade_for_score(0, 5.0), ("0", 0.0))


class TestWordLevel(unittest.TestCase):
    """Three-level per-word highlight cutoffs over the [good, bad] window."""

    def setUp(self):
        self._patchers = [
            mock.patch.object(speech, "WORD_GOOD_FRAC", 0.33),
            mock.patch.object(speech, "WORD_BAD_FRAC", 0.66),
        ]
        for p in self._patchers:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._patchers])

    def test_low_distance_is_good(self):
        self.assertEqual(speech._word_level(0.10, good=0.10, bad=0.50), "good")

    def test_high_distance_is_bad(self):
        self.assertEqual(speech._word_level(0.50, good=0.10, bad=0.50), "bad")

    def test_middle_distance_is_ok(self):
        self.assertEqual(speech._word_level(0.30, good=0.10, bad=0.50), "ok")


class TestReferenceWordTags(unittest.TestCase):
    """One {"word", "level", "correct"} per target token, display text preserved."""

    def test_levels_map_to_tags_and_correct_flag(self):
        tags = speech._reference_word_tags(
            ["Hello,", "world!"], ["good", "bad"])
        self.assertEqual([t["word"] for t in tags], ["Hello,", "world!"])
        self.assertEqual([t["level"] for t in tags], ["good", "bad"])
        # Only a "bad" word is not "correct".
        self.assertEqual([t["correct"] for t in tags], [True, False])

    def test_missing_levels_default_to_good(self):
        tags = speech._reference_word_tags(["a", "b", "c"], ["bad"])
        self.assertEqual([t["level"] for t in tags], ["bad", "good", "good"])


class TestWordRecall(unittest.TestCase):
    """Phone errors attributed back to whole words for the GUI highlight (§6)."""

    def test_clean_match_recalls_every_word(self):
        groups = [["a", "b"], ["c"]]
        pairs = [("a", "a"), ("b", "b"), ("c", "c")]
        with _patch_cost():
            recalled, heard, word_dist = speech._word_recall(groups, pairs)
        self.assertEqual(recalled, [2, 1])
        self.assertEqual(heard, [["a", "b"], ["c"]])
        self.assertEqual(word_dist, [0.0, 0.0])

    def test_deletion_raises_that_words_distance(self):
        # "b" (in word 0) was deleted: it contributes the maximum 1.0, so word 0's
        # mean distance is (0 + 1) / 2 = 0.5; word 1 stays clean.
        groups = [["a", "b"], ["c"]]
        pairs = [("a", "a"), ("b", ""), ("c", "c")]
        with _patch_cost():
            recalled, heard, word_dist = speech._word_recall(groups, pairs)
        self.assertEqual(recalled, [1, 1])
        self.assertEqual(word_dist, [0.5, 0.0])


class TestAlignAndScore(unittest.TestCase):
    """End-to-end of the pure scoring path (alignment + two-axis blend)."""

    def test_perfect_read_scores_100_with_full_recall(self):
        # good=0.0 pins the phoneme axis independently of the calibrated anchor;
        # the binary cost makes a self-comparison a flawless read.
        with _patch_cost(), \
             mock.patch.object(speech, "WEIGHT_PHONEME", 0.7), \
             mock.patch.object(speech, "WEIGHT_WORD", 0.3):
            result = speech.align_and_score(["a", "b", "c"], ["a", "b", "c"], good=0.0)
        self.assertEqual(result.recall, 1.0)
        self.assertEqual(result.per_phone_distance, 0.0)
        self.assertEqual(result.score, 100.0)

    def test_empty_reference_raises(self):
        with self.assertRaises(ValueError):
            speech.align_and_score([], ["a"])


# Note: reference_phonemes / reference_word_phonemes need espeak-ng, and analyze()
# needs the wav2vec2 recognizer + panphon, so they are exercised only by the
# optional end-to-end runner below, never by the offline unit tests.


def _run_end_to_end(user_path: str, reference_path: Optional[str]) -> None:
    """Run the full analyze() pipeline on real WAV files (loads the model)."""
    import soundfile as sf

    user_audio, user_sr = sf.read(user_path, dtype="float32")
    if reference_path:
        reference_audio, reference_sr = sf.read(reference_path, dtype="float32")
        is_reference = False
    else:
        # No reference supplied: reuse the user's own audio as its own reference
        # (the Test-button self-test). A faithful read should score near the top.
        print("No reference WAV given; using the user audio as its own reference.")
        reference_audio, reference_sr = user_audio, user_sr
        is_reference = True

    expected_text = input("Expected text for this recording: ").strip()

    print("Loading wav2vec2 phoneme recognizer (first run downloads ~1.2 GB)...")
    speech.load_models()

    result = speech.analyze(
        user_audio=user_audio,
        expected_text=expected_text,
        reference_audio=reference_audio,
        user_sr=user_sr,
        reference_sr=reference_sr,
        is_reference=is_reference,
    )

    print(f"\nScore:         {result.score} -> bucket {result.bucket}/5 "
          f"({result.user_percent:.0f}%, passed={result.passed})")
    print(f"Reference IPA: {' '.join(result.expected_phonemes)}")
    print(f"Heard IPA:     {result.transcription!r}")
    print(f"Problem words: {result.words_with_errors}")
    print(f"Per-phone:     {result.per_phone_distance:.4f} "
          f"(good={result.good_anchor:.3f} bad={result.bad_baseline:.3f}) "
          f"recall={result.recall:.2f}")
    print(result.feedback)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        _run_end_to_end(sys.argv[1], sys.argv[2] if len(sys.argv) >= 3 else None)
    else:
        unittest.main()
