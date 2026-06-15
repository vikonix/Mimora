"""Unit tests for mimora/prosody_utils.py (pure, no tkinter / no ML stack).

Run: python -m unittest tests.test_prosody_utils
"""

import unittest

from mimora import prosody_utils


class TestToSemitones(unittest.TestCase):
    """F0 (Hz) -> semitones relative to the contour's own median."""

    def test_octave_is_twelve_semitones(self):
        # Doubling the frequency must be exactly +12 ST regardless of register.
        out = prosody_utils.to_semitones([200.0, 400.0])  # median = 400 -> [-12, 0]
        self.assertAlmostEqual(out[1] - out[0], 12.0)

    def test_flat_contour_centres_on_zero(self):
        self.assertEqual(prosody_utils.to_semitones([180.0, 180.0, 180.0]),
                         [0.0, 0.0, 0.0])

    def test_unvoiced_frames_map_to_centre(self):
        # Zero (unvoiced) frames are placed at 0 ST, not -inf; median over the
        # voiced frames [200, 400] is 400.
        out = prosody_utils.to_semitones([0.0, 200.0, 400.0])
        self.assertEqual(out[0], 0.0)
        self.assertAlmostEqual(out[1], -12.0)
        self.assertAlmostEqual(out[2], 0.0)

    def test_empty_and_all_silent_are_unchanged(self):
        self.assertEqual(prosody_utils.to_semitones([]), [])
        self.assertEqual(prosody_utils.to_semitones([0.0, 0.0]), [0.0, 0.0])


class TestResampleSeries(unittest.TestCase):
    """Even down-sampling of a contour to a fixed point count for plotting."""

    def test_short_series_returned_unchanged(self):
        self.assertEqual(prosody_utils.resample_series([1.0, 2.0, 3.0], target=160),
                         [1.0, 2.0, 3.0])

    def test_long_series_is_thinned_to_target(self):
        values = list(range(1000))
        out = prosody_utils.resample_series(values, target=160)
        self.assertEqual(len(out), 160)
        # Endpoints are preserved so the sparkline keeps its overall span.
        self.assertEqual(out[0], 0)
        self.assertEqual(out[-1], 999)


if __name__ == "__main__":
    unittest.main()
