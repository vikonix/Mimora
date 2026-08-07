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


class LogHeaderTests(unittest.TestCase):
    """Every log opens with a rule and says what wrote it.

    The header is the only part of a log that is read before anything is known
    about the run, so its shape is worth pinning: a bare rule on line one (no
    timestamp, no level - it is a rule, not a record), then the build, the pid
    and the command line. The restored formatter is the other half: swapping it
    for the header and forgetting to put it back would cost every timestamp for
    the rest of the run, which nothing else in the suite would notice.
    """

    def setUp(self):
        root = logging.getLogger()
        saved_handlers, saved_level = root.handlers[:], root.level
        self.addCleanup(self._restore, root, saved_handlers, saved_level)
        # setup_logging also sets the process-global append flag that
        # log_file_mode() reads, and the append case below would otherwise
        # leave every later log file in this process opening with mode "a".
        self.addCleanup(setattr, bootstrap, "_append_logs",
                        bootstrap._append_logs)

    @staticmethod
    def _restore(root, handlers, level):
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        for handler in handlers:
            root.addHandler(handler)
        root.setLevel(level)

    def _log_lines(self, append=False):
        """Set logging up in a temporary directory and return the file's lines."""
        import tempfile

        with contextlib.redirect_stdout(io.StringIO()):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "main.log"
                bootstrap.setup_logging(path, append=append)
                logging.info("an ordinary record")
                root = logging.getLogger()
                # Closed before the directory goes away: on Windows it cannot
                # be removed while the handler holds the file open.
                for handler in root.handlers[:]:
                    if isinstance(handler, logging.FileHandler):
                        root.removeHandler(handler)
                        handler.close()
                return path.read_text(encoding="utf-8").splitlines()

    def test_the_first_line_is_the_bare_rule(self):
        self.assertEqual(self._log_lines()[0], bootstrap._HEADER_RULE)

    def test_the_header_names_the_build_the_pid_and_the_command(self):
        import os

        from mimora import __version__

        lines = self._log_lines()
        self.assertIn(__version__, lines[1])
        self.assertIn(str(os.getpid()), lines[1])
        self.assertTrue(lines[2].startswith("Launched: "))

    def test_ordinary_records_keep_their_timestamps(self):
        # i.e. the bare formatter was put back. The record is the last line,
        # and every real line carries the level in brackets.
        self.assertIn("[INFO]", self._log_lines()[-1])

    def test_appending_separates_the_runs(self):
        # A blank line only when continuing a file: a fresh log opens with the
        # rule, a continued one gets a gap so the seam is visible.
        lines = self._log_lines(append=True)
        self.assertEqual(lines[0], "")
        self.assertEqual(lines[1], bootstrap._HEADER_RULE)
        self.assertIn("restarted in-session", lines[2])


if __name__ == "__main__":
    unittest.main()
