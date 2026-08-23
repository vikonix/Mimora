# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for the early process setup (mimora/bootstrap.py).

What is worth pinning here is the one failure that can end the startup:
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


class InstallLogTests(unittest.TestCase):
    """The second log file: what an installation did, kept across launches.

    Its whole reason to exist is that main.log is one run long. What the
    first-run window downloaded has to still be readable after the next launch
    has truncated main.log, so the two properties worth pinning are that the
    file is appended rather than opened fresh, and that everything logged
    inside the block reaches it - from any module and any thread, because the
    download runs on a worker.
    """

    def setUp(self):
        import tempfile

        root = logging.getLogger()
        saved_handlers, saved_level = root.handlers[:], root.level
        self.addCleanup(self._restore, root, saved_handlers, saved_level)
        # The block's own records have to pass the root logger's level, which
        # is WARNING in a bare test process and INFO in the running app.
        root.setLevel(logging.INFO)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "install.log"

    @staticmethod
    def _restore(root, handlers, level):
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        for handler in handlers:
            root.addHandler(handler)
        root.setLevel(level)

    def _lines(self):
        return self.path.read_text(encoding="utf-8").splitlines()

    def test_records_from_inside_the_block_reach_the_file(self):
        with bootstrap.install_log(self.path):
            logging.getLogger("mimora.first_run_download").info(
                "Fetching the recognizer (1264 MB) ...")
        body = "\n".join(self._lines())
        self.assertIn("Fetching the recognizer (1264 MB) ...", body)
        # A real record, so it carries the level and the timestamp - only the
        # header is bare.
        self.assertIn("[INFO]", body)

    def test_nothing_reaches_the_file_after_the_block(self):
        # The handler is on the root logger, so leaving it there would send the
        # whole session into a file that is supposed to hold one installation.
        with bootstrap.install_log(self.path):
            pass
        # A named logger rather than logging.info: the module-level call
        # configures the root logger when it has no handler, which is exactly
        # the state this test leaves it in.
        logging.getLogger("mimora.app").info(
            "an ordinary record of the session that follows")
        self.assertNotIn("an ordinary record", "\n".join(self._lines()))

    def test_the_handler_is_removed_even_when_the_block_raises(self):
        root = logging.getLogger()
        before = len(root.handlers)
        with self.assertRaises(RuntimeError):
            with bootstrap.install_log(self.path):
                raise RuntimeError("the download failed")
        self.assertEqual(len(root.handlers), before)

    def test_the_file_opens_with_the_rule_and_the_build(self):
        from mimora import __version__

        with bootstrap.install_log(self.path):
            pass
        lines = self._lines()
        self.assertEqual(lines[0], bootstrap._HEADER_RULE)
        self.assertIn(__version__, lines[1])
        self.assertTrue(lines[2].startswith("Launched: "))

    def test_the_intro_lines_land_under_the_header_and_nowhere_else(self):
        # The caller's summary of what is about to be installed. It goes into
        # this file only: the same lines are already in main.log, logged by
        # build_plan before this file had a handler.
        console = io.StringIO()
        with contextlib.redirect_stdout(console):
            with bootstrap.install_log(self.path,
                                       intro=["Startup plan: 1 of 4"]):
                pass
        self.assertEqual(self._lines()[3], "Startup plan: 1 of 4")
        self.assertNotIn("Startup plan", console.getvalue())

    def test_a_second_run_is_appended_below_the_first(self):
        # The property the whole file exists for. A blank line separates the
        # two sections; a fresh file still opens with the rule.
        log = logging.getLogger("mimora.first_run_download")
        with bootstrap.install_log(self.path):
            log.info("the first installation")
        with bootstrap.install_log(self.path):
            log.info("the second installation")
        lines = self._lines()
        self.assertIn("the first installation", "\n".join(lines))
        self.assertEqual(lines.count(bootstrap._HEADER_RULE), 2)
        self.assertEqual(lines[4], "")

    def test_the_muted_loggers_are_kept_out_of_this_file_only(self):
        # httpx logs one line per request, and an install is hundreds of them
        # with signed CDN URLs in each. They are noise in main.log too, but
        # main.log is replaced every launch while this file only grows.
        console = io.StringIO()
        with contextlib.redirect_stdout(console):
            bootstrap.setup_logging(self.path.parent / "main.log")
            with bootstrap.install_log(self.path):
                logging.getLogger("httpx").info("HTTP Request: GET https://...")
                logging.getLogger("mimora.first_run_download").info(
                    "-> done: Kokoro-82M")
            for handler in logging.getLogger().handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    logging.getLogger().removeHandler(handler)
                    handler.close()
        written = "\n".join(self._lines())
        self.assertNotIn("HTTP Request", written)
        self.assertIn("-> done: Kokoro-82M", written)
        # The filter is on this handler alone: the session log and the console
        # still carry the request line, which is where it is worth having.
        self.assertIn("HTTP Request", console.getvalue())

    def test_an_unopenable_file_costs_the_file_and_not_the_download(self):
        console = io.StringIO()
        with mock.patch.object(logging, "FileHandler",
                               side_effect=OSError(3, "no such drive")), \
                contextlib.redirect_stdout(console):
            bootstrap.setup_logging(self.path.parent / "main.log")
            with bootstrap.install_log(Path("Z:/nope/logs/install.log")):
                logging.getLogger("mimora.first_run_download").info(
                    "the download kept going")
        printed = console.getvalue()
        self.assertIn("the download kept going", printed)
        self.assertIn("no such drive", printed)


if __name__ == "__main__":
    unittest.main()
