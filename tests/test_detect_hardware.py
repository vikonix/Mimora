# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for the hardware probe (mimora/detect_hardware.py).

Two things are worth testing without a machine to probe: the decision table of
the llama-server offload probe (which of True/False/None each situation
deserves) and the thresholds build_config turns the answer into. Everything
that would touch a disk or spawn a process is stubbed. Run from the project
root with:

    python -m unittest tests.test_detect_hardware
"""

import contextlib
import io
import logging
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from mimora import detect_hardware, llama_server_fetch

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


class UnwritableDataRootTests(unittest.TestCase):
    """`mimora --detect-hardware` on a data root the machine cannot write to.

    This command is named to the user by warn_if_gpu_unused, so it is run by
    people following advice rather than by people debugging. It also does NOT
    import config, which means paths.ensure_dirs() never runs on this path and
    its own reporting cannot cover it: everything here has to be handled where
    it happens. An unreachable MIMORA_HOME used to end the command in a
    three-deep pathlib traceback about a drive letter.
    """

    def setUp(self):
        # The module logger is process-global and this test empties it, which
        # is also what makes _setup_logging's idempotence guard let us in.
        logger = detect_hardware.logger
        saved = logger.handlers[:]
        logger.handlers.clear()
        self.addCleanup(lambda: (logger.handlers.clear(),
                                 logger.handlers.extend(saved)))

    def test_an_unwritable_log_directory_costs_the_file_and_nothing_else(self):
        stderr = io.StringIO()
        with patch.object(detect_hardware.Path, "mkdir",
                          side_effect=OSError(3, "no such drive")), \
                contextlib.redirect_stderr(stderr):
            detect_hardware._setup_logging()  # must not raise

        self.assertIn("no such drive", stderr.getvalue())
        # The probe is the point of the command and does not need the file.
        self.assertEqual(detect_hardware.logger.level, logging.INFO)
        # No file handler, but not an empty handler list either: logging falls
        # back to lastResort when it finds no handler anywhere, which printed
        # main()'s failure to stderr a second time in different words.
        self.assertFalse(any(isinstance(handler, logging.FileHandler)
                             for handler in detect_hardware.logger.handlers))
        self.assertTrue(any(isinstance(handler, logging.NullHandler)
                            for handler in detect_hardware.logger.handlers))

    def test_the_failure_is_not_reported_twice(self):
        # One failure, one message. The duplicate this guards against was not
        # cosmetic: two spellings of one error read as two errors.
        stderr = io.StringIO()
        with patch.object(detect_hardware.Path, "mkdir",
                          side_effect=OSError(3, "no such drive")), \
                patch.object(detect_hardware, "probe_and_write",
                             side_effect=OSError(3, "no such drive")), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(stderr):
            detect_hardware.main()

        # Lower-cased before counting: the print and the log record differ in
        # capitalisation, and a case-sensitive count would match only one of
        # them and pass while the duplicate was still there.
        self.assertEqual(stderr.getvalue().lower().count("could not write"), 1)

    def test_an_unwritable_output_file_is_reported_not_raised(self):
        stderr = io.StringIO()
        # _setup_logging is stubbed rather than allowed to run: the real one
        # opens the project's own logs/hwdetect.log with mode="w", and a test
        # that truncates a log file is a test with a side effect.
        with patch.object(detect_hardware, "_setup_logging"), \
                patch.object(detect_hardware, "probe_and_write",
                             side_effect=OSError(3, "no such drive")), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(stderr):
            exit_code = detect_hardware.main()

        self.assertEqual(exit_code, 1)
        self.assertIn("no such drive", stderr.getvalue())
        self.assertIn(str(detect_hardware.OUTPUT_FILE), stderr.getvalue())


class WarnIfGpuUnusedTests(unittest.TestCase):
    """The startup warning for a CPU-only torch on a machine with a card.

    The failure it exists for is silent by nature, so what matters is that it
    fires in exactly one situation and stays quiet - and cheap - in the others.
    nvidia-smi is stubbed throughout; the real one would answer for whatever
    machine happens to run the suite.
    """

    def _run(self, device, driver_cuda, config_written=False,
             platform="win32"):
        # config_written is stubbed rather than left to the filesystem: the
        # machine running the suite may well have a hardware_config.json, and
        # then the branch under test would be whichever one it happens to have.
        #
        # platform is pinned for the same class of reason: the reinstall advice
        # differs by it, because the uv flag it names is only meaningful on
        # Windows. Left to the machine running the suite, the assertions below
        # would pass or fail according to who ran them. Windows is the default
        # here because it is the platform this warning exists for.
        with patch.object(detect_hardware.sys, "platform", platform), \
                patch.object(llama_server_fetch, "detect_driver_cuda",
                             return_value=driver_cuda) as detect, \
                patch.object(detect_hardware, "_stored_device_may_be_stale",
                             return_value=config_written):
            with self.assertLogs(level="WARNING") as captured:
                detect_hardware.warn_if_gpu_unused(device)
                # assertLogs fails an empty block, so every path needs one
                # record of its own to compare against.
                logging.warning("sentinel")
        return [line for line in captured.output if "sentinel" not in line], detect

    def test_cpu_torch_with_a_driver_present_is_warned_about(self):
        warnings, _ = self._run("cpu", (12, 4))
        self.assertEqual(len(warnings), 1)
        self.assertIn("--torch-backend auto", warnings[0])
        self.assertIn("CPU-only build", warnings[0])

    def test_a_written_hardware_config_makes_the_message_name_both_causes(self):
        # With that file present, "cpu" can also mean detect_device took its
        # word without asking torch - which is what `uv tool upgrade` from a
        # CPU build onto a CUDA one looks like. Telling such a user to
        # reinstall with --torch-backend auto is telling them to redo what they
        # just did, so the refresh command has to be named as well.
        with patch.object(detect_hardware.paths, "repo_mode",
                          return_value=False):
            warnings, _ = self._run("cpu", (12, 4), config_written=True)
        self.assertEqual(len(warnings), 1)
        self.assertIn("mimora --detect-hardware", warnings[0])
        self.assertIn("--torch-backend auto", warnings[0])

    def test_the_uv_flag_is_advised_on_windows_only(self):
        # PyPI already serves a CUDA build of torch on Linux and macOS has no
        # CUDA at all, so --torch-backend auto cannot fix what this warning is
        # about anywhere but Windows - and on Linux it does harm, moving the
        # whole resolution onto the PyTorch index with results that differed
        # between runs. See tasks/release-1.1.0.md, finding 2 of stage 2.
        for platform in ("linux", "darwin"):
            with self.subTest(platform=platform):
                warnings, _ = self._run("cpu", (12, 4), platform=platform)
                self.assertEqual(len(warnings), 1)
                self.assertNotIn("--torch-backend", warnings[0])
                self.assertNotIn("UV_TORCH_BACKEND", warnings[0])
                # Still actionable: the advice that does apply everywhere.
                self.assertIn("pytorch.org", warnings[0])
                self.assertIn("install.py", warnings[0])

    def test_the_refresh_command_matches_how_mimora_is_installed(self):
        # The two forms are not interchangeable. An installed tool has no
        # interpreter on PATH that can import mimora, and a checkout's venv
        # that happens to be active would rewrite a different file; out of a
        # clone the console script may not exist at all.
        with patch.object(detect_hardware.paths, "repo_mode",
                          return_value=True):
            self.assertEqual(detect_hardware._refresh_command(),
                             "`python -m mimora.detect_hardware`")
        with patch.object(detect_hardware.paths, "repo_mode",
                          return_value=False):
            self.assertEqual(detect_hardware._refresh_command(),
                             "`mimora --detect-hardware`")

    def test_staleness_is_decided_by_the_file_existing_at_all(self):
        # Not by what it says: the ambiguity comes from detect_device having
        # been allowed to short-circuit, which any existing file permits.
        missing = Path(__file__).with_name("no-such-hardware-config.json")
        with patch.object(detect_hardware, "OUTPUT_FILE", missing):
            self.assertFalse(detect_hardware._stored_device_may_be_stale())
        with patch.object(detect_hardware, "OUTPUT_FILE", Path(__file__)):
            self.assertTrue(detect_hardware._stored_device_may_be_stale())

    def test_a_machine_without_an_nvidia_driver_is_not_warned(self):
        # The overwhelmingly common case: no card, so the CPU is correct.
        warnings, _ = self._run("cpu", None)
        self.assertEqual(warnings, [])

    def test_a_gpu_run_is_silent(self):
        warnings, _ = self._run("cuda", (12, 4))
        self.assertEqual(warnings, [])

    def test_nvidia_smi_is_not_consulted_when_torch_uses_the_gpu(self):
        # Ordering, not cosmetics: this is what keeps the check free at every
        # startup on a machine that is already fine.
        _, detect = self._run("cuda", (12, 4))
        detect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
