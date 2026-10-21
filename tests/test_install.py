# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Tests for the torch steps of install.py: the CUDA series choice and the
removal of a leftover torchaudio.

No nvidia-smi and no pip: the series choice is pure, and the torchaudio step
runs with dist_version and run_command replaced. The rules under test are in
the comments above TORCH_CU_SERIES, MIN_COMPUTE_CAP_FOR_NEW_SERIES and in the
docstring of step_remove_stale_torchaudio.

Run from the project root with:

    python -m unittest tests.test_install
"""

import unittest
from unittest import mock

import install

TURING = (7, 5)    # GTX 16xx, RTX 20xx: the oldest GPUs in CUDA 13.x builds
AMPERE = (8, 6)    # RTX 30xx
PASCAL = (6, 1)    # GTX 10xx: not in cu128 and newer


class PickCuSeriesTests(unittest.TestCase):
    def pick(self, driver_cuda, compute_cap=None):
        return install.pick_cu_series(install.TORCH_CU_SERIES, driver_cuda,
                                      compute_cap)

    def test_a_new_driver_and_gpu_get_the_newest_series(self):
        # The case that started this: driver 13.3 and an RTX 3090 got cu128
        # while uv --torch-backend auto installed cu132.
        self.assertEqual(self.pick((13, 3), AMPERE), "cu132")

    def test_turing_is_not_limited(self):
        # The GTX 1650 Ti is Turing: the "GTX" name does not make it legacy.
        self.assertEqual(self.pick((13, 3), TURING), "cu132")

    def test_a_driver_between_two_series_takes_the_older_one(self):
        self.assertEqual(self.pick((12, 9), AMPERE), "cu128")
        self.assertEqual(self.pick((13, 1), AMPERE), "cu130")

    def test_a_12_1_to_12_5_driver_gets_cu118(self):
        # cu124 and cu121 stop at torch 2.6 / 2.5, below what transformers
        # 5.x needs, so they are not in the list.
        for driver in [(12, 1), (12, 4), (12, 5)]:
            self.assertEqual(self.pick(driver, AMPERE), "cu118", driver)

    def test_a_pre_turing_gpu_stops_at_cu126(self):
        self.assertEqual(self.pick((13, 3), PASCAL),
                         install.LEGACY_GPU_NEWEST_SERIES)
        self.assertEqual(self.pick((12, 4), PASCAL), "cu118")

    def test_an_unknown_driver_takes_the_newest_allowed_series(self):
        self.assertEqual(self.pick(None, AMPERE), "cu132")
        self.assertEqual(self.pick(None, PASCAL),
                         install.LEGACY_GPU_NEWEST_SERIES)

    def test_an_unknown_compute_capability_does_not_limit(self):
        # Old drivers do not know the compute_cap query field; the choice
        # then rests on the driver alone, as it did before the field.
        self.assertEqual(self.pick((13, 3), None), "cu132")

    def test_a_driver_older_than_every_series_gets_none(self):
        self.assertIsNone(self.pick((11, 7), AMPERE))

    def test_every_series_is_newest_first(self):
        versions = [install._series_to_version(s)
                    for s in install.TORCH_CU_SERIES]
        self.assertEqual(versions, sorted(versions, reverse=True))


class ParseComputeCapTests(unittest.TestCase):
    def test_one_gpu(self):
        self.assertEqual(install.parse_compute_cap("7.5\n"), (7, 5))

    def test_the_first_of_several_gpus(self):
        self.assertEqual(install.parse_compute_cap("8.6\n6.1\n"), (8, 6))

    def test_an_unknown_field_or_empty_output_is_none(self):
        # nvidia-smi without the field answers with an error text, and the
        # caller reads only stdout, which can be empty.
        self.assertIsNone(install.parse_compute_cap(""))
        self.assertIsNone(install.parse_compute_cap(
            'Field "compute_cap" is not a valid field to query.'))


class RemoveStaleTorchaudioTests(unittest.TestCase):
    def run_step(self, torchaudio_version, confirmed=True, uninstall_ok=True):
        versions = {"torchaudio": torchaudio_version, "torch": "2.11.0+cu128"}
        log = mock.Mock()
        confirmer = mock.Mock()
        confirmer.confirm.return_value = confirmed
        report = install.StepReport()
        with mock.patch.object(install, "dist_version", versions.get), \
                mock.patch.object(install, "run_command",
                                  return_value=uninstall_ok) as run:
            install.step_remove_stale_torchaudio(log, confirmer, report)
        return log, confirmer, report, run

    def test_nothing_happens_without_torchaudio(self):
        log, confirmer, report, run = self.run_step(None)
        confirmer.confirm.assert_not_called()
        run.assert_not_called()
        log.banner.assert_not_called()
        self.assertEqual(report.statuses(), [])

    def test_a_leftover_is_uninstalled_after_the_question(self):
        _, confirmer, report, run = self.run_step("2.11.0+cu128")
        confirmer.confirm.assert_called_once()
        (cmd, _), _ = run.call_args
        self.assertEqual(cmd[-3:], ["uninstall", "-y", "torchaudio"])
        self.assertEqual(report.statuses(), [install.DONE])

    def test_a_no_keeps_it_and_says_so(self):
        _, _, report, run = self.run_step("2.11.0+cu128", confirmed=False)
        run.assert_not_called()
        self.assertEqual(report.statuses(), [install.SKIPPED])

    def test_a_failed_uninstall_does_not_stop_the_installer(self):
        # run_command, not run_or_fail: no InstallError here.
        _, _, report, _ = self.run_step("2.11.0+cu128", uninstall_ok=False)
        self.assertEqual(report.statuses(), [install.FAILED])


if __name__ == "__main__":
    unittest.main()
