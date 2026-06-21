"""Unit tests for mimora/prosody.py — the engine-agnostic prosody layer.

These need numpy/librosa/scikit-learn but never torch or the recognition model,
so they run offline and quickly on short synthetic signals.

Run: python -m unittest tests.test_prosody
"""

import unittest

import numpy as np

from mimora import prosody


class TestInterpolateF0(unittest.TestCase):
    def test_fills_gaps(self):
        f0 = np.array([0.0, 100.0, 0.0, 200.0, 0.0])
        out = prosody.interpolate_f0(f0)
        self.assertEqual(len(out), len(f0))
        self.assertTrue((out > 0).all())  # zeros between voiced frames get filled

    def test_handles_all_silent(self):
        f0 = np.zeros(5)
        out = prosody.interpolate_f0(f0)
        self.assertEqual(len(out), 5)  # no crash on fully unvoiced input


class TestWaveformPrep(unittest.TestCase):
    def test_prepare_waveform_downmixes_and_resamples(self):
        # 2 channels of 24 kHz audio -> mono 16 kHz.
        stereo_24k = np.ones((2, 24_000), dtype=np.float32)
        out = prosody._prepare_waveform(stereo_24k, orig_sr=24_000)
        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.dtype, np.float32)
        self.assertAlmostEqual(len(out), 16_000, delta=50)

    def test_trim_silence_keeps_near_silent_input(self):
        # Below the 0.1 s floor, trimming would empty the clip -> original kept.
        wav = np.zeros(int(0.05 * prosody.TARGET_SAMPLE_RATE), dtype=np.float32)
        out = prosody._trim_silence(wav)
        self.assertEqual(len(out), len(wav))


class TestExtractEnergy(unittest.TestCase):
    def test_energy_is_scaled_into_range(self):
        # A 0.5 s tone gives several RMS frames; MinMax scaling caps them at 250.
        t = np.linspace(0, 0.5, int(0.5 * prosody.TARGET_SAMPLE_RATE), dtype=np.float32)
        tone = np.sin(2 * np.pi * 220 * t).astype(np.float32)
        energy = prosody.extract_energy(tone)
        self.assertGreater(len(energy), 0)
        self.assertGreaterEqual(float(energy.min()), 0.0)
        self.assertLessEqual(float(energy.max()), 250.0 + 1e-6)


class TestComputeProsody(unittest.TestCase):
    def test_returns_four_contours_as_lists(self):
        sr = prosody.TARGET_SAMPLE_RATE
        t = np.linspace(0, 0.4, int(0.4 * sr), dtype=np.float32)
        user = np.sin(2 * np.pi * 150 * t).astype(np.float32)
        reference = np.sin(2 * np.pi * 200 * t).astype(np.float32)

        out = prosody.compute_prosody(user, sr, reference, sr)

        self.assertEqual(set(out), {"f0", "energy", "ref_f0", "ref_energy"})
        for key, series in out.items():
            self.assertIsInstance(series, list, key)
            self.assertGreater(len(series), 0, key)


if __name__ == "__main__":
    unittest.main()
