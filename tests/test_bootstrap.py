# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for the early process setup (mimora/bootstrap.py).

What is worth pinning here is the one failure that used to end the startup:
a log file that cannot be opened. It is reachable through a data root the
machine cannot write to - most often a MIMORA_HOME naming a drive that is not
there - and that variable exists to be the way OUT of a bad automatic choice,
so a traceback from it is the opposite of what it is for.

setup_logging reconfigures the ROOT logger with force=True, so every test here
saves and restores it; without that the configuration would leak into whatever
runs next in the same process. Run from the project root with:

    python -m unittest tests.test_bootstrap
"""

import contextlib
import io
import logging
import unittest
from pathlib import Path
from unittest import mock

from mimora import bootstrap


class LogFileFailureTests(unittest.TestCase):
    """An unopenable log file costs the file and not the application."""

    def setUp(self):
        root = logging.getLogger()
        saved_handlers, saved_level = root.handlers[:], root.level
        self.addCleanup(self._restore, root, saved_handlers, saved_level)

    @staticmethod
    def _restore(root, handlers, level):
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        for handler in handlers:
            root.addHandler(handler)
        root.setLevel(level)

    def _setup_with_a_failing_file(self, console):
        with mock.patch.object(logging, "FileHandler",
                               side_effect=OSError(3, "no such drive")), \
                contextlib.redirect_stdout(console):
            bootstrap.setup_logging(Path("Z:/nope/logs/main.log"))
            logging.info("the application kept going")

    def test_the_console_handler_survives_and_logging_still_works(self):
        console = io.StringIO()
        self._setup_with_a_failing_file(console)

        root = logging.getLogger()
        self.assertTrue(root.handlers, "logging was left with no handler")
        self.assertFalse(
            any(isinstance(handler, logging.FileHandler)
                for handler in root.handlers),
            "a FileHandler was installed although opening it failed")
        self.assertIn("the application kept going", console.getvalue())

    def test_the_failure_is_reported_and_names_what_can_be_changed(self):
        # The reason and the variable, both: without the second the reader is
        # told that something failed and not what they can do about it, and
        # this runs before any window exists to ask in.
        console = io.StringIO()
        self._setup_with_a_failing_file(console)

        printed = console.getvalue()
        self.assertIn("no such drive", printed)
        self.assertIn("MIMORA_HOME", printed)

    def test_a_working_log_file_is_used(self):
        # The negative case is what a too-eager except would pass: the file
        # handler must still be the normal outcome.
        with contextlib.redirect_stdout(io.StringIO()):
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                bootstrap.setup_logging(Path(tmp) / "main.log")
                root = logging.getLogger()
                self.assertTrue(
                    any(isinstance(handler, logging.FileHandler)
                        for handler in root.handlers))
                # Closed here rather than at cleanup: on Windows the directory
                # cannot be removed while the handler holds the file open.
                for handler in root.handlers[:]:
                    if isinstance(handler, logging.FileHandler):
                        root.removeHandler(handler)
                        handler.close()


if __name__ == "__main__":
    unittest.main()
