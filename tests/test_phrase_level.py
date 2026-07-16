# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Unit tests for the phrase proficiency-level helpers in mimora/llm.py.

The validator (_fits_level / _min_zipf) and the focus-word frequency filter
are pure functions around wordfreq's zipf_frequency. wordfreq is replaced by
a deterministic fake here, so the tests are fast, offline, and green whether
or not the package is installed. Run from the project root with:

    python -m unittest tests.test_phrase_level
"""

import unittest
from unittest import mock

from mimora import llm

# Deterministic stand-in for wordfreq.zipf_frequency: listed words answer
# their table value, everything else counts as unknown/rare (0.0) - the same
# convention wordfreq itself uses.
_FAKE_ZIPF_TABLE = {
    "the": 6.5, "cat": 5.5, "eats": 4.9, "fish": 5.0, "today": 5.2,
    "ubiquitous": 2.8,
}


def _fake_zipf(word: str, lang: str) -> float:
    return _FAKE_ZIPF_TABLE.get(word, 0.0)


# A strict level (word range 3..5, common vocabulary only) and an open one
# (no vocabulary floor), mirroring the shape of profile phrase_gen["levels"].
_STRICT = {"vocab_hint": "v", "grammar_hint": "g",
           "words": (3, 5), "min_zipf": 4.5}
_OPEN = {"vocab_hint": "v", "grammar_hint": "g",
         "words": (3, 10), "min_zipf": None}


class _FakeZipfTestBase(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(llm, "_zipf_fn", _fake_zipf)
        patcher.start()
        self.addCleanup(patcher.stop)


class FitsLevelTests(_FakeZipfTestBase):
    def test_common_words_in_range_pass(self):
        self.assertTrue(
            llm._fits_level("The cat eats fish.", _STRICT, "en", False))

    def test_word_count_below_range_fails(self):
        self.assertFalse(llm._fits_level("The cat.", _STRICT, "en", False))

    def test_word_count_above_range_fails(self):
        self.assertFalse(llm._fits_level(
            "The cat eats fish today the cat.", _STRICT, "en", False))

    def test_rare_word_fails_the_vocabulary_floor(self):
        self.assertFalse(llm._fits_level(
            "The ubiquitous cat eats.", _STRICT, "en", False))

    def test_no_floor_accepts_rare_words(self):
        self.assertTrue(llm._fits_level(
            "The ubiquitous cat eats fish today.", _OPEN, "en", False))

    def test_fragment_skips_the_word_count(self):
        # 2 words is below the full-sentence range, but fragments have their
        # own fixed 2-4 word length - only the vocabulary floor applies.
        self.assertTrue(llm._fits_level("the cat", _STRICT, "en", True))

    def test_fragment_still_faces_the_vocabulary_floor(self):
        self.assertFalse(
            llm._fits_level("ubiquitous cat", _STRICT, "en", True))

    def test_mid_phrase_capitalized_word_is_treated_as_a_name(self):
        # "Zorblax" is unknown (zipf 0.0) but capitalized mid-phrase: a proper
        # noun from the source text must not fail the vocabulary check.
        self.assertTrue(llm._fits_level(
            "The cat eats Zorblax.", _STRICT, "en", False))

    def test_missing_wordfreq_disables_the_floor_only(self):
        # _zipf_fn is False after a failed import: the vocabulary check is
        # skipped (prompt hints still steer the level), the word count stays.
        with mock.patch.object(llm, "_zipf_fn", False):
            self.assertTrue(llm._fits_level(
                "ubiquitous cat eats", _STRICT, "en", False))
            self.assertFalse(
                llm._fits_level("The cat.", _STRICT, "en", False))


class MinZipfTests(_FakeZipfTestBase):
    def test_returns_the_lowest_frequency(self):
        self.assertEqual(llm._min_zipf("The cat eats.", "en"), 4.9)

    def test_short_words_are_skipped(self):
        # "a" (article) must not drag the minimum to "unknown word".
        self.assertEqual(llm._min_zipf("a cat", "en"), 5.5)

    def test_nothing_checkable_returns_none(self):
        self.assertIsNone(llm._min_zipf("a of", "en"))

    def test_unavailable_wordfreq_returns_none(self):
        with mock.patch.object(llm, "_zipf_fn", False):
            self.assertIsNone(llm._min_zipf("The cat eats.", "en"))


class FocusWordFilterTests(_FakeZipfTestBase):
    def test_floor_excludes_rare_words(self):
        # "fish" (5.0) passes the 4.5 floor, "ubiquitous" (2.8) does not -
        # the pick must always be the frequent word.
        for _ in range(20):
            self.assertEqual(
                llm.LLMManager._pick_focus_word(
                    "fish ubiquitous", 4.5, "en"),
                "fish")

    def test_empty_filter_falls_back_to_the_unfiltered_pool(self):
        self.assertEqual(
            llm.LLMManager._pick_focus_word("ubiquitous", 4.5, "en"),
            "ubiquitous")

    def test_no_floor_keeps_the_historical_behavior(self):
        self.assertIn(
            llm.LLMManager._pick_focus_word("fish ubiquitous"),
            {"fish", "ubiquitous"})

    def test_accented_words_survive_the_tokenizer(self):
        # An ASCII-only tokenizer used to mangle accented Spanish words
        # ("práctica" -> "ctica"); the picker must see the whole word.
        self.assertEqual(
            llm.LLMManager._pick_focus_word("práctica"), "práctica")

    def test_profile_stopwords_are_never_picked(self):
        with mock.patch.object(llm.config, "PHRASE_GEN_STOPWORDS",
                               frozenset({"cuando"})):
            for _ in range(20):
                self.assertEqual(
                    llm.LLMManager._pick_focus_word("cuando fish"), "fish")


if __name__ == "__main__":
    unittest.main()
