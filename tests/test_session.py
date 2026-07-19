# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Unit tests for mimora/session.py (pure, no tkinter / no ML stack).

Run: python -m unittest tests.test_session
"""

import unittest

from mimora.session import SessionState


class TestRecordTake(unittest.TestCase):
    """Session tally: distinct phrases, mean of per-phrase bests, attempts."""

    def test_blank_phrase_records_nothing(self):
        state = SessionState()
        self.assertIsNone(state.record_take("", 80.0))
        self.assertIsNone(state.record_take("   ", 80.0))

    def test_repeats_track_the_best_not_the_mean(self):
        # Two attempts at one phrase: count stays 1, the average is the
        # phrase's best score - a low retry must not drag it down.
        state = SessionState()
        state.record_take("hello there", 60.0)
        count, average, maximum, attempts = state.record_take(
            "hello there", 80.0)
        self.assertEqual(count, 1)
        self.assertEqual(average, 80.0)
        self.assertEqual(maximum, 100.0)
        self.assertEqual(attempts, [60.0, 80.0])
        # A worse retry keeps the best.
        _, average, _, attempts = state.record_take("hello there", 50.0)
        self.assertEqual(average, 80.0)
        self.assertEqual(attempts, [60.0, 80.0, 50.0])

    def test_average_is_mean_of_phrase_bests(self):
        # Bests 80 and 90 average to 85 regardless of attempt counts.
        state = SessionState()
        state.record_take("hello there", 60.0)
        state.record_take("hello there", 80.0)
        count, average, _, attempts = state.record_take("good morning", 90.0)
        self.assertEqual(count, 2)
        self.assertEqual(average, 85.0)
        # The attempt list follows the current (new) phrase.
        self.assertEqual(attempts, [90.0])

    def test_returning_to_a_phrase_restarts_attempts_keeps_best(self):
        state = SessionState()
        state.record_take("hello there", 80.0)
        state.record_take("good morning", 90.0)
        # Back to the first phrase: the dot column restarts, the session
        # average still remembers its 80 best (mean of 80 and 90 = 85).
        _, average, _, attempts = state.record_take("hello there", 70.0)
        self.assertEqual(attempts, [70.0])
        self.assertEqual(average, 85.0)

    def test_attempts_are_a_fresh_copy(self):
        # Mutating the returned list must not corrupt the internal tally.
        state = SessionState()
        _, _, _, attempts = state.record_take("a phrase", 60.0)
        attempts.append(99.0)
        _, _, _, attempts = state.record_take("a phrase", 70.0)
        self.assertEqual(attempts, [60.0, 70.0])

    def test_raw_take_uses_percent_scale(self):
        state = SessionState()
        _, average, maximum, _ = state.record_take("a phrase", 82.4)
        self.assertEqual(average, 82.4)
        self.assertEqual(maximum, 100.0)

    def test_graded_take_uses_grade_scale(self):
        # The 0-5 grade axis: the average stays unrounded, the scale says 5.
        state = SessionState()
        state.record_take("a phrase", 4.0, graded=True)
        _, average, maximum, _ = state.record_take("a phrase", 3.5, graded=True)
        self.assertEqual(average, 4.0)
        self.assertEqual(maximum, 5.0)


class TestHistoryTrend(unittest.TestCase):
    """Trend arrows: compared on the displayed mark, per phrase."""

    @staticmethod
    def attempt(phrase, score, score_text=None):
        record = {"kind": "attempt", "phrase": phrase, "score": score}
        if score_text is not None:
            record["score_text"] = score_text
        return record

    def test_first_attempt_has_no_trend(self):
        state = SessionState()
        history = state.add_history_entry(self.attempt("hello", 70.0))
        self.assertIsNone(history[-1]["trend"])

    def test_numeric_up_down_same(self):
        state = SessionState()
        state.add_history_entry(self.attempt("hello", 70.0))
        self.assertEqual(
            state.add_history_entry(self.attempt("hello", 80.0))[-1]["trend"],
            "up")
        self.assertEqual(
            state.add_history_entry(self.attempt("hello", 60.0))[-1]["trend"],
            "down")
        self.assertEqual(
            state.add_history_entry(self.attempt("hello", 60.0))[-1]["trend"],
            "same")

    def test_sub_point_difference_reads_as_same(self):
        # 82.4 vs 81.6 both display as 82, so the arrow must not flap.
        state = SessionState()
        state.add_history_entry(self.attempt("hello", 81.6))
        history = state.add_history_entry(self.attempt("hello", 82.4))
        self.assertEqual(history[-1]["trend"], "same")

    def test_trend_is_per_phrase(self):
        # An attempt of another phrase must not become the comparison base.
        state = SessionState()
        state.add_history_entry(self.attempt("hello", 90.0))
        history = state.add_history_entry(self.attempt("goodbye", 50.0))
        self.assertIsNone(history[-1]["trend"])

    def test_grade_texts_compare_by_text_then_score(self):
        # Equal chips read "same"; different chips take the direction from
        # the numeric score behind them ("4-" < "4" does not hold lexically).
        state = SessionState()
        state.add_history_entry(self.attempt("hello", 3.7, score_text="4-"))
        history = state.add_history_entry(self.attempt("hello", 4.0,
                                                       score_text="4"))
        self.assertEqual(history[-1]["trend"], "up")
        history = state.add_history_entry(self.attempt("hello", 4.1,
                                                       score_text="4"))
        self.assertEqual(history[-1]["trend"], "same")

    def test_errors_and_unscored_carry_no_trend(self):
        state = SessionState()
        history = state.add_history_entry({"kind": "error", "message": "boom"})
        self.assertNotIn("trend", history[-1])

    def test_history_is_bounded(self):
        state = SessionState(history_limit=3)
        for i in range(5):
            history = state.add_history_entry(self.attempt("hello", float(i)))
        self.assertEqual(len(history), 3)
        # Oldest entries dropped: scores 2, 3, 4 remain, oldest first.
        self.assertEqual([r["score"] for r in history], [2.0, 3.0, 4.0])

    def test_comparison_base_can_drop_off_the_deque(self):
        # Once every earlier attempt of the phrase has been evicted, the next
        # take is treated as a first attempt again (no trend).
        state = SessionState(history_limit=2)
        state.add_history_entry(self.attempt("hello", 50.0))
        state.add_history_entry({"kind": "error", "message": "x"})
        state.add_history_entry({"kind": "error", "message": "y"})
        history = state.add_history_entry(self.attempt("hello", 90.0))
        self.assertIsNone(history[-1]["trend"])
