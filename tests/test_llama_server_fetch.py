# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Unit tests for the llama-server downloader (mimora/llama_server_fetch.py).

What can be checked without the machine the build is for: which variant this
platform resolves to, how a binary that dies on startup is reported, and the
shape of the pinned macOS rows. Everything that would reach the network, the
disk or a subprocess is stubbed, so the whole file runs on any OS.

The macOS rows are the reason this file exists. They were added from the
release page and the llama.cpp sources rather than from a Mac, so the facts
that a Mac would have confirmed are written down here instead: that Apple
Silicon resolves to the Metal build even under Rosetta 2, that the Metal device
pattern matches what --list-devices is expected to print, and that no macOS
asset is a .zip - zipfile drops the unix mode bits, which would unpack a
llama-server nobody can execute (see _extract).

Run from the project root with:

    python -m unittest tests.test_llama_server_fetch
"""

import re
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from mimora import llama_server_fetch as fetch

# Fake install location: nothing here executes it, the probes only read .name
# for their messages.
EXE = Path("/opt/mimora/bin/llama/llama-server")

# What --list-devices is expected to print on an Apple Silicon Mac. The device
# name comes from ggml-metal.cpp (GGML_METAL_NAME "MTL") and ggml-metal-device.m
# (formatted as "MTL%d"); the description is the Metal device's own name.
METAL_LISTING = ("Available devices:\n"
                 "  MTL0: Apple M2 Pro (16384 MiB, 12000 MiB free)")
# What a CPU build prints: the header and nothing under it. Copied from a real
# linux-cpu-x64 install rather than invented, because the negative case is what
# a wrong pattern would pass.
EMPTY_LISTING = "Available devices:\n"


class SelectVariantOnMacosTests(unittest.TestCase):
    """select_variant() on darwin, where the architecture is the whole choice.

    The Rosetta case is the one worth pinning: platform.machine() answers for
    the interpreter, so an x86_64 Python on an M-series Mac reports x86_64, and
    taking that at face value would install a CPU-only build on a machine with a
    usable GPU - the silent slow path the module's device check exists to
    prevent.
    """

    def _select(self, platform_name, machine, rosetta=False, macos=None):
        """Run select_variant() with the four machine facts stubbed.

        Each stub stands in for something this process cannot honestly answer
        in a test: which OS it runs on, what the CPU is, whether it is being
        translated (a sysctl subprocess), and which macOS this is. macos
        defaults to None - "version unknown", which never blocks a choice - so
        that the tests below are about the architecture and nothing else.
        """
        rosetta_probe = Mock(return_value=rosetta)
        with patch.object(fetch.sys, "platform", platform_name), \
             patch.object(fetch.platform, "machine", Mock(return_value=machine)), \
             patch.object(fetch, "is_rosetta", rosetta_probe), \
             patch.object(fetch, "detect_macos_version",
                          Mock(return_value=macos)):
            return fetch.select_variant(), rosetta_probe

    def test_apple_silicon_gets_the_metal_build(self):
        variant, rosetta_probe = self._select("darwin", "arm64")
        self.assertEqual(variant, fetch.MACOS_ARM64_DEFAULT)
        # An interpreter that already reports arm64 settles the question, so
        # the sysctl subprocess must not be spawned to re-answer it.
        rosetta_probe.assert_not_called()

    def test_intel_mac_gets_the_cpu_build(self):
        variant, _ = self._select("darwin", "x86_64")
        self.assertEqual(variant, fetch.MACOS_X64_DEFAULT)

    def test_rosetta_gets_the_metal_build_despite_reporting_x86_64(self):
        variant, rosetta_probe = self._select("darwin", "x86_64", rosetta=True)
        rosetta_probe.assert_called_once()
        self.assertEqual(variant, fetch.MACOS_ARM64_DEFAULT)
        # Stated as the intent rather than as the name: the point is that a Mac
        # with a GPU does not end up on a CPU build, whatever the row is called.
        self.assertEqual(fetch.VARIANTS[variant].backend, "Metal")

    def test_an_unpublished_mac_architecture_is_refused(self):
        # No asset exists for anything but arm64 and x64, and saying so is
        # better than downloading a build that cannot run.
        with self.assertRaises(fetch.UnsupportedPlatformError):
            self._select("darwin", "ppc")

    def test_a_mac_older_than_the_build_is_refused_before_downloading(self):
        # The distinction that makes this worth checking early:
        # UnsupportedPlatformError is what install.py records as a manual step
        # and carries on from, while the same fact discovered from dyld after
        # the download arrives as a LlamaServerFetchError and fails the step.
        required = fetch.VARIANTS[fetch.MACOS_ARM64_DEFAULT].min_macos
        self.assertIsNotNone(required)
        older = (required[0] - 1, 0)
        with self.assertRaises(fetch.UnsupportedPlatformError) as caught:
            self._select("darwin", "arm64", macos=older)
        # The message has to name both numbers, or the reader cannot tell
        # whether it is their machine or the release that is the problem.
        self.assertIn(f"{required[0]}.{required[1]}", str(caught.exception))
        self.assertIn(f"{older[0]}.{older[1]}", str(caught.exception))

    def test_the_minimum_version_itself_is_accepted(self):
        # A ">=" written as ">" would refuse exactly the machines the pin was
        # chosen for, and nothing else would notice.
        required = fetch.VARIANTS[fetch.MACOS_X64_DEFAULT].min_macos
        variant, _ = self._select("darwin", "x86_64", macos=required)
        self.assertEqual(variant, fetch.MACOS_X64_DEFAULT)

    def test_an_unreadable_version_does_not_block_the_install(self):
        # detect_macos_version() returning None means "could not read it", and
        # a failed reading must not masquerade as an old Mac.
        variant, _ = self._select("darwin", "arm64", macos=None)
        self.assertEqual(variant, fetch.MACOS_ARM64_DEFAULT)


class IsRosettaTests(unittest.TestCase):
    """is_rosetta() reads sysctl.proc_translated, which may not exist.

    The key is that only a clean "1" means "translated": the sysctl key is
    absent on Intel Macs, where the command exits non-zero, and absent entirely
    off macOS.
    """

    def _is_rosetta(self, platform_name="darwin", returncode=0, stdout="",
                    error=None):
        run = (Mock(side_effect=error) if error
               else Mock(return_value=Mock(returncode=returncode,
                                           stdout=stdout, stderr="")))
        with patch.object(fetch.sys, "platform", platform_name), \
             patch.object(fetch.subprocess, "run", run):
            return fetch.is_rosetta(), run

    def test_translated_process_is_recognised(self):
        translated, _ = self._is_rosetta(stdout="1\n")
        self.assertIs(translated, True)

    def test_intel_mac_reports_no_such_key(self):
        # sysctl exits 1 with nothing on stdout when the key does not exist.
        translated, _ = self._is_rosetta(returncode=1, stdout="")
        self.assertIs(translated, False)

    def test_a_missing_sysctl_is_not_an_error(self):
        # The answer only steers a choice between two builds, so a machine
        # whose sysctl cannot be run must fall back to what machine() said
        # rather than fail the whole install.
        translated, _ = self._is_rosetta(error=FileNotFoundError("sysctl"))
        self.assertIs(translated, False)

    def test_nothing_is_spawned_off_macos(self):
        translated, run = self._is_rosetta(platform_name="win32")
        self.assertIs(translated, False)
        run.assert_not_called()


class DetectMacosVersionTests(unittest.TestCase):
    """platform.mac_ver() is a string, and not always the shape expected."""

    def _version(self, release):
        with patch.object(fetch.platform, "mac_ver",
                          Mock(return_value=(release, ("", "", ""), ""))):
            return fetch.detect_macos_version()

    def test_major_and_minor_are_parsed(self):
        self.assertEqual(self._version("15.5"), (15, 5))

    def test_a_bare_major_counts_as_dot_zero(self):
        self.assertEqual(self._version("26"), (26, 0))

    def test_the_patch_level_is_ignored(self):
        # Only major.minor is compared; a third component must not confuse it.
        self.assertEqual(self._version("13.3.1"), (13, 3))

    def test_an_empty_answer_is_not_a_version(self):
        # This is what mac_ver() returns off macOS, and what a failed reading
        # looks like there.
        self.assertIsNone(self._version(""))


class ProbeFailureTests(unittest.TestCase):
    """_probe reports a binary that starts and then dies on a signal.

    subprocess.run does not raise for that: it returns an ordinary result whose
    output is empty, which used to surface from verify_build as "printed no
    recognisable version" over an empty string. The case worth naming is SIGILL
    from an Intel Mac older than the CPU the release was compiled for.
    """

    def _probe(self, returncode, platform_name="darwin", stdout="", stderr=""):
        result = Mock(returncode=returncode, stdout=stdout, stderr=stderr)
        with patch.object(fetch.sys, "platform", platform_name), \
             patch.object(fetch, "_no_window", Mock(return_value={})), \
             patch.object(fetch.subprocess, "run", Mock(return_value=result)):
            return fetch._probe(EXE, ["--version"])

    def test_a_signal_is_reported_as_such(self):
        with self.assertRaises(fetch.LlamaServerFetchError) as caught:
            self._probe(-4)
        message = str(caught.exception)
        self.assertIn("signal 4", message)
        # The hint is the only place the user is told what to look at, and this
        # is the failure it exists for.
        self.assertIn("illegal instruction", message)

    def test_a_normal_run_returns_both_streams(self):
        output = self._probe(0, stdout="version: 10099 (1a064ab09)\n",
                             stderr="build info\n")
        self.assertIn("version: 10099", output)
        self.assertIn("build info", output)

    def test_a_windows_exit_code_is_not_read_as_a_signal(self):
        # Windows exit codes are not signal numbers, and a crash there reports
        # a large positive value anyway; the branch must stay POSIX-only.
        output = self._probe(-1, platform_name="win32", stdout="whatever")
        self.assertEqual(output, "whatever")


class StartFailureHintTests(unittest.TestCase):
    """Each platform is told about its own reason, not a generic one."""

    def _hint(self, platform_name):
        with patch.object(fetch.sys, "platform", platform_name):
            return fetch._start_failure_hint()

    def test_every_platform_gets_a_distinct_hint(self):
        hints = [self._hint(name) for name in ("win32", "darwin", "linux")]
        self.assertTrue(all(hint.strip() for hint in hints))
        self.assertEqual(len(set(hints)), len(hints))

    def test_the_macos_hint_names_both_failure_modes(self):
        hint = self._hint("darwin")
        # A system older than the build (dyld refuses to load it) and a CPU
        # older than the build (the process dies on an instruction it lacks)
        # look nothing alike and send the reader to different places.
        self.assertIn("dyld", hint)
        self.assertIn("illegal instruction", hint)


class MacosVariantShapeTests(unittest.TestCase):
    """The pinned macOS rows, spelled out because no Mac has confirmed them."""

    def test_both_defaults_exist_in_the_table(self):
        # Named through the constants so that renaming a row moves both.
        self.assertIn(fetch.MACOS_ARM64_DEFAULT, fetch.VARIANTS)
        self.assertIn(fetch.MACOS_X64_DEFAULT, fetch.VARIANTS)

    def test_the_metal_pattern_matches_the_expected_listing(self):
        pattern = fetch.VARIANTS[fetch.MACOS_ARM64_DEFAULT].device_pattern
        self.assertIsNotNone(pattern)
        self.assertRegex(METAL_LISTING, pattern)
        self.assertIsNone(re.search(pattern, EMPTY_LISTING))

    def test_the_intel_build_promises_no_device(self):
        # llama.cpp compiles the macOS x64 asset with GGML_METAL=OFF, so there
        # is nothing for a device check to look for and nothing to fall back to.
        variant = fetch.VARIANTS[fetch.MACOS_X64_DEFAULT]
        self.assertEqual(variant.backend, "CPU")
        self.assertIsNone(variant.device_pattern)
        self.assertIsNone(variant.fallback)

    def test_the_metal_build_has_no_fallback(self):
        # Deliberate: no CPU-only arm64 asset is published, and a Mac quietly
        # running on its CPU is the outcome the device check exists to refuse.
        self.assertIsNone(fetch.VARIANTS[fetch.MACOS_ARM64_DEFAULT].fallback)

    def test_min_macos_is_set_exactly_on_the_macos_rows(self):
        # It is read from the asset's own LC_BUILD_VERSION, so a macOS row
        # without it is an unmeasured one, and a non-macOS row with it is a
        # number nothing will ever compare against.
        for name, variant in fetch.VARIANTS.items():
            with self.subTest(variant=name):
                self.assertEqual(name.startswith("macos-"),
                                 variant.min_macos is not None)

    def test_posix_assets_are_tarballs(self):
        # zipfile.extractall ignores a member's unix mode, so a POSIX build
        # delivered as .zip would unpack a binary nobody can execute, and the
        # failure would surface as a bare PermissionError from _probe rather
        # than as anything about unpacking. The prefix is how the table names
        # its platforms; a new POSIX row under another prefix has to be added
        # here too.
        for name, variant in fetch.VARIANTS.items():
            if not name.startswith(("linux-", "macos-")):
                continue
            for asset in variant.assets:
                with self.subTest(variant=name, asset=asset.name):
                    self.assertTrue(asset.name.endswith(".tar.gz"))


if __name__ == "__main__":
    unittest.main()
