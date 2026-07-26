# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Unit tests for the hardware probe (tools/detect_hardware.py).

Two things are worth testing without a machine to probe: the decision table of
the llama-server offload probe (which of True/False/None each situation
deserves) and the thresholds build_config turns the answer into. Everything
that would touch a disk or spawn a process is stubbed. Run from the project
root with:

    python -m unittest tests.test_detect_hardware
"""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from mimora import llama_server_fetch
from tools import detect_hardware

# Fake install location: the probe only ever reads its .name for a message.
EXE = Path("/opt/mimora/bin/llama/llama-server")

# Variants the probe branches on. Named explicitly so that renaming one in
# llama_server_fetch.VARIANTS fails here loudly instead of silently skipping.
GPU_VARIANT = "win-cuda-12.4-x64"
CPU_VARIANT = "win-cpu-x64"

CUDA_LISTING = ("Available devices:\n"
                "  CUDA0: NVIDIA GeForce RTX 3090 (24576 MiB, 23000 MiB free)")
EMPTY_LISTING = "Available devices:\n  (none)"


def hardware(vram_gb=None, present=None, offload=None, torch_cuda=False):
    """Minimal hardware dict in the shape build_config expects."""
    return {
        "gpu": {
            "present": bool(vram_gb) if present is None else present,
            "vram_gb": vram_gb,
            "torch_cuda": torch_cuda,
            "llama_gpu_offload": offload,
        }
    }


class VariantTableTests(unittest.TestCase):
    """The probe's branching assumes these two shapes exist in the table."""

    def test_named_variants_still_exist(self):
        self.assertIn(GPU_VARIANT, llama_server_fetch.VARIANTS)
        self.assertIn(CPU_VARIANT, llama_server_fetch.VARIANTS)

    def test_gpu_variant_promises_a_device_and_cpu_variant_does_not(self):
        self.assertIsNotNone(
            llama_server_fetch.VARIANTS[GPU_VARIANT].device_pattern)
        self.assertIsNone(
            llama_server_fetch.VARIANTS[CPU_VARIANT].device_pattern)


class ProbeLlamaOffloadTests(unittest.TestCase):
    """Decision table of _probe_llama_offload.

    The verdict feeds build_config, so False is not a neutral answer: it zeroes
    the LLM's VRAM budget. It is therefore reserved for the two cases where the
    installed binary provably cannot offload, and everything else says None.
    """

    def _probe(self, gpu_present=True, exe=EXE, variant=GPU_VARIANT,
               listing=CUDA_LISTING, error=None):
        """Run the probe with the fetch module's three lookups stubbed out.

        Each stub stands in for a file on disk or a subprocess, neither of
        which a unit test should need.
        """
        warnings = []
        list_devices = (Mock(side_effect=error) if error
                        else Mock(return_value=listing))
        with patch.multiple(llama_server_fetch,
                            installed_exe=Mock(return_value=exe),
                            installed_variant=Mock(return_value=variant),
                            list_devices=list_devices):
            verdict = detect_hardware._probe_llama_offload(warnings, gpu_present)
        return verdict, warnings

    def test_promised_device_is_present(self):
        verdict, warnings = self._probe()
        self.assertIs(verdict, True)
        self.assertEqual(warnings, [])

    def test_promised_device_is_missing(self):
        # The silent CPU fallback: a CUDA build with the wrong cudart DLLs
        # still starts and still answers, about three times slower.
        verdict, warnings = self._probe(listing=EMPTY_LISTING)
        self.assertIs(verdict, False)
        self.assertEqual(len(warnings), 1)
        self.assertIn("cudart", warnings[0])

    def test_cpu_build_cannot_offload(self):
        verdict, warnings = self._probe(variant=CPU_VARIANT)
        self.assertIs(verdict, False)
        self.assertEqual(len(warnings), 1)
        self.assertIn(CPU_VARIANT, warnings[0])

    def test_cpu_build_is_not_worth_a_warning_without_a_gpu(self):
        # The overwhelmingly common case: no GPU, so the CPU build was the
        # right choice and there is nothing to tell the user about.
        verdict, warnings = self._probe(variant=CPU_VARIANT, gpu_present=False)
        self.assertIs(verdict, False)
        self.assertEqual(warnings, [])

    def test_missing_install_is_unknown_not_negative(self):
        # None keeps build_config on its "physical GPU presence" fallback,
        # which is what makes the tool usable before install.py's step 8.
        verdict, warnings = self._probe(exe=None)
        self.assertIsNone(verdict)
        self.assertEqual(len(warnings), 1)
        self.assertIn("llama_server_fetch", warnings[0])

    def test_missing_install_is_silent_without_a_gpu(self):
        verdict, warnings = self._probe(exe=None, gpu_present=False)
        self.assertIsNone(verdict)
        self.assertEqual(warnings, [])

    def test_unknown_variant_is_unknown(self):
        # A stamp naming a variant this build dropped: there is no documented
        # expectation to compare a device listing against.
        verdict, warnings = self._probe(variant=None)
        self.assertIsNone(verdict)
        self.assertEqual(len(warnings), 1)

    def test_probe_failure_is_unknown(self):
        error = llama_server_fetch.LlamaServerFetchError("binary would not run")
        verdict, warnings = self._probe(error=error)
        self.assertIsNone(verdict)
        self.assertIn("binary would not run", warnings[0])


class BuildConfigLlmTests(unittest.TestCase):
    """How the probe's verdict and the card's VRAM become LLM parameters."""

    def test_offload_capable_card_gets_every_layer(self):
        config = detect_hardware.build_config(
            hardware(vram_gb=24.0, offload=True))
        self.assertEqual(config["EXTERNAL_N_GPU_LAYERS"], -1)
        self.assertEqual(config["EXTERNAL_N_CTX"], 4096)

    def test_unknown_verdict_trusts_the_physical_card(self):
        # None must behave exactly like True here - this is what lets
        # detect_hardware run before the binary is installed without changing
        # the numbers it writes.
        self.assertEqual(
            detect_hardware.build_config(hardware(vram_gb=24.0, offload=None)),
            detect_hardware.build_config(hardware(vram_gb=24.0, offload=True)))

    def test_negative_verdict_zeroes_the_llm_budget(self):
        config = detect_hardware.build_config(
            hardware(vram_gb=24.0, offload=False))
        self.assertEqual(config["EXTERNAL_N_GPU_LAYERS"], 0)
        self.assertEqual(config["EXTERNAL_N_CTX"], 2048)

    def test_absent_gpu_zeroes_the_llm_budget(self):
        config = detect_hardware.build_config(
            hardware(vram_gb=None, present=False, offload=None))
        self.assertEqual(config["EXTERNAL_N_GPU_LAYERS"], 0)
        self.assertEqual(config["EXTERNAL_N_CTX"], 2048)

    def test_layer_count_follows_the_vram_ladder(self):
        for vram, expected in ((8.0, -1), (6.0, -1), (5.0, 20), (4.0, 20),
                               (3.0, 12), (2.0, 8), (1.0, 0)):
            with self.subTest(vram=vram):
                config = detect_hardware.build_config(
                    hardware(vram_gb=vram, offload=True))
                self.assertEqual(config["EXTERNAL_N_GPU_LAYERS"], expected)

    def test_context_size_needs_eight_gigabytes(self):
        self.assertEqual(
            detect_hardware.build_config(
                hardware(vram_gb=8.0, offload=True))["EXTERNAL_N_CTX"], 4096)
        self.assertEqual(
            detect_hardware.build_config(
                hardware(vram_gb=6.0, offload=True))["EXTERNAL_N_CTX"], 2048)


class BuildConfigTorchTests(unittest.TestCase):
    """The torch side is independent of the LLM verdict, by design."""

    def test_devices_follow_torch_not_the_llm_probe(self):
        config = detect_hardware.build_config(
            hardware(vram_gb=24.0, offload=False, torch_cuda=True))
        self.assertEqual(config["DEVICE"], "cuda")
        self.assertEqual(config["WAV2VEC2_DEVICE"], "cuda")

    def test_cpu_only_torch_build_falls_back(self):
        config = detect_hardware.build_config(
            hardware(vram_gb=24.0, offload=True, torch_cuda=False))
        self.assertEqual(config["DEVICE"], "cpu")
        self.assertEqual(config["WAV2VEC2_DEVICE"], "cpu")

    def test_small_card_keeps_wav2vec2_off_the_gpu(self):
        # Below 6 GB the LLM and Wav2Vec2 would fight for the same card.
        config = detect_hardware.build_config(
            hardware(vram_gb=4.0, offload=True, torch_cuda=True))
        self.assertEqual(config["DEVICE"], "cuda")
        self.assertEqual(config["WAV2VEC2_DEVICE"], "cpu")


if __name__ == "__main__":
    unittest.main()
