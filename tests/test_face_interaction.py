# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Unit tests for the interactive logic in mimora/face_widget.py.

Covers only the pure, Tk-free helpers (cursor gaze, face hit-test and the
balloon-pop cartoon timeline): no Tk root and no Pillow rendering here.

Run: python -m unittest tests.test_face_interaction
"""

import unittest

from mimora.face_widget import (
    APPEAR_S,
    POP_BURST_S,
    POP_EYE_S,
    POP_GRIN_S,
    POP_TOTAL_S,
    appear_scale,
    face_hit,
    gaze_state,
    pop_scene,
)


class TestGazeState(unittest.TestCase):
    """Cursor position -> quantized gaze (direction, widening, cross-eye)."""

    R = 100.0  # head radius; the face is centred at the origin

    def test_cursor_to_the_right_looks_right(self):
        gqx, gqy, _, cross = gaze_state(500.0, 0.0, 0.0, 0.0, self.R)
        self.assertGreater(gqx, 0)
        self.assertEqual(gqy, 0)
        self.assertFalse(cross)

    def test_cursor_above_looks_up(self):
        gqx, gqy, _, _ = gaze_state(0.0, -400.0, 0.0, 0.0, self.R)
        self.assertEqual(gqx, 0)
        self.assertLess(gqy, 0)

    def test_beyond_reach_rests(self):
        self.assertEqual(gaze_state(10_000.0, 0.0, 0.0, 0.0, self.R),
                         (0, 0, 0, False))

    def test_cursor_on_the_nose_crosses_eyes(self):
        gqx, gqy, wide, cross = gaze_state(10.0, 5.0, 0.0, 0.0, self.R)
        self.assertTrue(cross)
        self.assertEqual((gqx, gqy), (0, 0))
        self.assertEqual(wide, 2)

    def test_gaze_offsets_stay_quantized(self):
        for px, py in ((150.0, 0.0), (0.0, -150.0), (300.0, 300.0),
                       (-90.0, 120.0), (599.0, 0.0)):
            gqx, gqy, _, _ = gaze_state(px, py, 0.0, 0.0, self.R)
            self.assertLessEqual(abs(gqx), 2)
            self.assertLessEqual(abs(gqy), 2)

    def test_eyes_widen_as_the_cursor_nears(self):
        far = gaze_state(450.0, 0.0, 0.0, 0.0, self.R)[2]
        mid = gaze_state(200.0, 0.0, 0.0, 0.0, self.R)[2]
        near = gaze_state(110.0, 0.0, 0.0, 0.0, self.R)[2]
        self.assertLessEqual(far, mid)
        self.assertLessEqual(mid, near)
        self.assertEqual(near, 2)

    def test_degenerate_radius_is_safe(self):
        self.assertEqual(gaze_state(5.0, 5.0, 0.0, 0.0, 0.0),
                         (0, 0, 0, False))


class TestFaceHit(unittest.TestCase):
    """Clicks count only when they land on the face disc."""

    def test_centre_hits(self):
        self.assertTrue(face_hit(50.0, 50.0, 50.0, 50.0, 40.0))

    def test_edge_hits(self):
        self.assertTrue(face_hit(90.0, 50.0, 50.0, 50.0, 40.0))

    def test_canvas_corner_misses(self):
        # Inside the square canvas but outside the disc.
        self.assertFalse(face_hit(85.0, 85.0, 50.0, 50.0, 40.0))


class TestPopScene(unittest.TestCase):
    """The balloon-pop timeline plays its stages in order."""

    def test_burst_starts_with_shards_and_startled_mouth(self):
        scene = pop_scene(0.0)
        self.assertIsNotNone(scene["shards"])
        self.assertEqual(scene["mouth"][0], "open")
        left, right = scene["eyes"]
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertFalse(scene["done"])

    def test_left_eye_falls_first(self):
        rest = pop_scene(0.0)["eyes"]
        scene = pop_scene(POP_BURST_S + 0.1 * POP_EYE_S)
        left, right = scene["eyes"]
        self.assertIsNotNone(left)
        self.assertNotEqual(left, rest[0])  # moving already
        self.assertEqual(right, rest[1])    # still in place

    def test_falling_eye_lands_in_the_mouth(self):
        # Just before the gulp the eye must sit next to the mouth centre
        # (x ~ 0, y ~ the mouth line at 0.38 head radii).
        scene = pop_scene(POP_BURST_S + 0.79 * POP_EYE_S)
        x, y = scene["eyes"][0]
        self.assertAlmostEqual(x, 0.0, delta=0.06)
        self.assertAlmostEqual(y, 0.38, delta=0.06)
        self.assertEqual(scene["mouth"][0], "open")

    def test_mouth_opens_to_catch(self):
        scene = pop_scene(POP_BURST_S + 0.6 * POP_EYE_S)
        kind, openness = scene["mouth"]
        self.assertEqual(kind, "open")
        self.assertGreater(openness, 0.5)

    def test_first_eye_swallowed_second_still_waiting(self):
        rest = pop_scene(0.0)["eyes"]
        scene = pop_scene(POP_BURST_S + 0.9 * POP_EYE_S)
        left, right = scene["eyes"]
        self.assertIsNone(left)
        self.assertEqual(right, rest[1])

    def test_second_eye_falls_after_the_first(self):
        scene = pop_scene(POP_BURST_S + 1.2 * POP_EYE_S)
        left, right = scene["eyes"]
        self.assertIsNone(left)
        self.assertIsNotNone(right)

    def test_grin_grows_once_both_eyes_are_eaten(self):
        scene = pop_scene(POP_BURST_S + 2 * POP_EYE_S + 0.5 * POP_GRIN_S)
        self.assertEqual(scene["eyes"], [None, None])
        kind, k = scene["mouth"]
        self.assertEqual(kind, "grin")
        self.assertGreater(k, 0.0)

    def test_grin_pops_at_the_end(self):
        scene = pop_scene(POP_BURST_S + 2 * POP_EYE_S + POP_GRIN_S + 0.01)
        self.assertIsNone(scene["mouth"])
        self.assertIsNotNone(scene["mouth_shards"])
        self.assertFalse(scene["done"])

    def test_everything_gone_after_the_total(self):
        for t in (POP_TOTAL_S, POP_TOTAL_S + 5.0):
            scene = pop_scene(t)
            self.assertTrue(scene["done"])
            self.assertEqual(scene["eyes"], [None, None])
            self.assertIsNone(scene["mouth"])
            self.assertIsNone(scene["shards"])
            self.assertIsNone(scene["mouth_shards"])


class TestAppearScale(unittest.TestCase):
    """Re-inflate curve: 0 -> small overshoot -> exactly 1."""

    def test_starts_collapsed(self):
        self.assertEqual(appear_scale(0.0), 0.0)

    def test_ends_at_exactly_one(self):
        self.assertEqual(appear_scale(APPEAR_S), 1.0)
        self.assertEqual(appear_scale(APPEAR_S + 1.0), 1.0)

    def test_overshoots_like_a_balloon(self):
        peak = max(appear_scale(APPEAR_S * i / 100) for i in range(101))
        self.assertGreater(peak, 1.0)
        self.assertLessEqual(peak, 1.1)

    def test_grows_monotonically_until_the_peak(self):
        samples = [appear_scale(APPEAR_S * 0.7 * i / 20) for i in range(21)]
        self.assertEqual(samples, sorted(samples))


if __name__ == "__main__":
    unittest.main()
