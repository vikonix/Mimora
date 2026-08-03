# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for the layering helpers in mimora/config.py.

config.py is mostly constants built at import time, so what is testable here is
the small amount of logic that decides them. _model_device is the whole of it
today: it is the rule that keeps a per-model device from outliving the DEVICE it
is derived from. Both module globals it reads are patched per case, so nothing
here depends on this machine's hardware_config.json or on torch. Run from the
project root with:

    python -m unittest tests.test_config
"""

import unittest
from unittest import mock

from mimora import config


class ModelDeviceTests(unittest.TestCase):
    def _device(self, device: str, hw: dict, key: str, default: str) -> str:
        with mock.patch.object(config, "DEVICE", device), \
                mock.patch.object(config, "_HW", hw):
            return config._model_device(key, default)

    def test_pin_is_used_when_device_is_cuda(self):
        self.assertEqual(
            self._device("cuda", {"WAV2VEC2_DEVICE": "cuda"},
                         "WAV2VEC2_DEVICE", "cuda"),
            "cuda")

    def test_stale_cuda_pin_cannot_exceed_a_cpu_device(self):
        # The case this helper exists for: DEVICE stepped down because torch
        # turned out to be a CPU build, while the same file still pins the
        # model to the GPU. Believing the pin would move the crash from Kokoro
        # to Wav2Vec2 rather than prevent it.
        self.assertEqual(
            self._device("cpu", {"WAV2VEC2_DEVICE": "cuda"},
                         "WAV2VEC2_DEVICE", "cpu"),
            "cpu")

    def test_cpu_pin_is_kept_on_a_cuda_machine(self):
        # Pinning one model to the CPU is the VRAM-contention decision the
        # detector is there to make, not staleness.
        self.assertEqual(
            self._device("cuda", {"WAV2VEC2_DEVICE": "cpu"},
                         "WAV2VEC2_DEVICE", "cuda"),
            "cpu")

    def test_default_applies_when_the_key_is_absent(self):
        self.assertEqual(
            self._device("cuda", {}, "TRANSLATOR_DEVICE", "cpu"), "cpu")
        self.assertEqual(
            self._device("cuda", {}, "WAV2VEC2_DEVICE", "cuda"), "cuda")

    def test_cuda_default_is_capped_too(self):
        # WAV2VEC2_DEVICE passes DEVICE itself as the default, so this can only
        # differ if the two ever disagree - which is exactly what a future
        # caller with a literal "cuda" default would do.
        self.assertEqual(
            self._device("cpu", {}, "WAV2VEC2_DEVICE", "cuda"), "cpu")


if __name__ == "__main__":
    unittest.main()
