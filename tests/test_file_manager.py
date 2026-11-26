# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Tests for mimora/file_manager.py.

No file manager is started: the platform is stubbed, and os.startfile and
subprocess.Popen are mocks, so every branch runs on any machine.

Run from the project root with:

    python -m unittest tests.test_file_manager
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mimora import file_manager


def _platform(name):
    """Make file_manager see ``name`` as sys.platform."""
    return mock.patch.object(file_manager.sys, "platform", name)


class OpenFolderTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name) / "logs"

    def test_windows_uses_startfile(self):
        # os.startfile exists only on Windows, hence create=True.
        with _platform("win32"), mock.patch.object(
                file_manager.os, "startfile", create=True) as startfile:
            file_manager.open_folder(self.folder)
        startfile.assert_called_once_with(self.folder)

    def test_macos_uses_open(self):
        with _platform("darwin"), mock.patch.object(
                file_manager.subprocess, "Popen") as popen:
            file_manager.open_folder(self.folder)
        popen.assert_called_once_with(["open", str(self.folder)])

    def test_linux_uses_xdg_open(self):
        with _platform("linux"), mock.patch.object(
                file_manager.subprocess, "Popen") as popen:
            file_manager.open_folder(self.folder)
        popen.assert_called_once_with(["xdg-open", str(self.folder)])

    def test_a_missing_folder_is_created_first(self):
        with _platform("linux"), mock.patch.object(
                file_manager.subprocess, "Popen"):
            file_manager.open_folder(self.folder)
        self.assertTrue(self.folder.is_dir())

    def test_a_missing_file_manager_raises_oserror(self):
        # The settings window shows an OSError in its footer.
        with _platform("linux"), mock.patch.object(
                file_manager.subprocess, "Popen",
                side_effect=FileNotFoundError("xdg-open")):
            with self.assertRaises(OSError):
                file_manager.open_folder(self.folder)


if __name__ == "__main__":
    unittest.main()
