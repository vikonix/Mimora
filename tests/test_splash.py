# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Tests for mimora/splash.py.

Only the parent side is tested: how the child process is started and stopped.
No test starts the child, because it opens a real window. What the window looks
like is checked by eye with ``python mimora/splash.py`` (a click closes it).
"""

import ast
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from mimora import splash


class ShowAndCloseTests(unittest.TestCase):
    def setUp(self):
        # The module keeps the child in a global. Each test starts without one
        # and must not leave one to the next test.
        patcher = mock.patch.object(splash, "_process", None)
        patcher.start()
        self.addCleanup(patcher.stop)
        # atexit must not keep a handler that refers to a mock.
        patcher = mock.patch.object(splash.atexit, "register")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_show_starts_this_file_with_a_pipe_and_no_console(self):
        with mock.patch.object(splash.subprocess, "Popen") as popen:
            splash.show()

        (command,), kwargs = popen.call_args
        self.assertEqual(command, [sys.executable, "-P", splash.__file__])
        # The pipe is what ends the child when this process is killed.
        self.assertEqual(kwargs["stdin"], subprocess.PIPE)
        self.assertEqual(kwargs["creationflags"],
                         getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def test_show_twice_starts_one_process(self):
        with mock.patch.object(splash.subprocess, "Popen") as popen:
            splash.show()
            splash.show()

        popen.assert_called_once()

    def test_a_failed_start_does_not_stop_the_application(self):
        with mock.patch.object(splash.subprocess, "Popen",
                               side_effect=OSError("no interpreter")):
            splash.show()  # Must not raise.

        splash.close()  # Must not raise either: there is nothing to close.

    def test_close_ends_the_child_and_forgets_it(self):
        with mock.patch.object(splash.subprocess, "Popen") as popen:
            splash.show()
            splash.close()
            splash.close()  # The second call finds nothing to close.

        process = popen.return_value
        process.stdin.close.assert_called_once_with()
        process.terminate.assert_called_once_with()

    def test_close_accepts_a_child_that_has_already_gone(self):
        with mock.patch.object(splash.subprocess, "Popen") as popen:
            popen.return_value.terminate.side_effect = OSError("gone")
            splash.show()
            splash.close()  # Must not raise.


class ModuleImportTests(unittest.TestCase):
    def test_module_level_imports_are_stdlib_only_and_without_tkinter(self):
        # cli.py imports this module before it parses its arguments, so
        # --version must not pay for tkinter, and a computer without tkinter
        # must reach the message that cli.py has for that case.
        tree = ast.parse(Path(splash.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module)
        self.assertNotIn("tkinter", imported)
        for module in imported:
            self.assertIn(module.split(".")[0], sys.stdlib_module_names)


if __name__ == "__main__":
    unittest.main()
