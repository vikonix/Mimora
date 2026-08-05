# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for the relaunch command (mimora/lifecycle.py).

The application restarts itself in two situations - a restart-only setting
changed, and the first run has just written hardware_config.json - and both go
through ``spawn_replacement()``. What it must reconstruct is not "a command
that starts Mimora" but "the command that started THIS process", and that
differs by launch form:

* ``python main.py`` needs the interpreter in front;
* ``python -m mimora`` needs ``-m`` again, because executing __main__.py as a
  file path puts the package's own directory on sys.path instead of the
  project root;
* the ``mimora`` console script must be re-run on its own, since prepending
  the interpreter yields ``python.exe mimora.exe``.

Only the first form is observable on a machine running from a checkout, and
the third one cannot exist until the package is published - which is exactly
why it is stubbed here rather than left to be discovered after publication.

The stubbing has to be right to be worth anything, though, and it was not:
these tests originally gave the console script ``__spec__ = None``, and the
first live run of an installed package showed that a Windows one carries a
spec named ``"__main__"`` (the .exe is a launcher with a zip archive appended,
and the __main__.py inside it is imported like any module). The relaunch came
out as ``python.exe -m __main__`` and the application closed instead of
restarting. Both halves of that are pinned below.

Run from the project root with:

    python -m unittest tests.test_lifecycle
"""

import sys
import types
import unittest
from importlib.machinery import ModuleSpec
from unittest import mock

from mimora import lifecycle

FAKE_PYTHON = "/opt/python/bin/python"


def _launched_as(argv, spec=None):
    """Context in which sys.argv, the interpreter and __main__.__spec__ are stubbed.

    ``spec`` is what the import system leaves on the main module: a ModuleSpec
    named ``<package>.__main__`` for ``python -m ...``, a ModuleSpec named
    ``"__main__"`` for a zipapp (which is what a Windows console script is),
    and None for a plain script path.
    """
    fake_main = types.SimpleNamespace(__spec__=spec)
    return (
        mock.patch.object(sys, "argv", list(argv)),
        mock.patch.object(sys, "executable", FAKE_PYTHON),
        mock.patch.dict(sys.modules, {"__main__": fake_main}),
    )


def _command_for(argv, spec=None):
    patches = _launched_as(argv, spec)
    for patch in patches:
        patch.start()
    try:
        return lifecycle.relaunch_command()
    finally:
        for patch in reversed(patches):
            patch.stop()


class ScriptLaunchTests(unittest.TestCase):
    """python main.py - the form that worked before and must keep working."""

    def test_interpreter_is_prepended(self):
        self.assertEqual(
            _command_for(["main.py"]),
            [FAKE_PYTHON, "main.py"],
        )

    def test_absolute_path_and_arguments_survive(self):
        self.assertEqual(
            _command_for(["/src/mimora/main.py", "--flag", "value"]),
            [FAKE_PYTHON, "/src/mimora/main.py", "--flag", "value"],
        )

    def test_pythonw_scripts_count_as_scripts(self):
        # .pyw is the same thing without a console window; it is still a
        # source file that cannot start itself.
        self.assertEqual(
            _command_for(["run.pyw"]),
            [FAKE_PYTHON, "run.pyw"],
        )


class ModuleLaunchTests(unittest.TestCase):
    """python -m mimora - reconstructed as -m, never as the __main__.py path."""

    def test_module_form_is_rebuilt(self):
        command = _command_for(
            ["/src/mimora/__main__.py"],
            ModuleSpec("mimora.__main__", loader=None),
        )
        self.assertEqual(command, [FAKE_PYTHON, "-m", "mimora"])

    def test_the_main_file_path_is_not_reused(self):
        # The point of the branch: running that path directly would put
        # mimora/ on sys.path, and `import mimora` would fail from a checkout.
        command = _command_for(
            ["/src/mimora/__main__.py"],
            ModuleSpec("mimora.__main__", loader=None),
        )
        self.assertNotIn("/src/mimora/__main__.py", command)

    def test_arguments_after_the_module_survive(self):
        command = _command_for(
            ["/src/mimora/__main__.py", "--flag"],
            ModuleSpec("mimora.__main__", loader=None),
        )
        self.assertEqual(command, [FAKE_PYTHON, "-m", "mimora", "--flag"])

    def test_a_plain_module_keeps_its_own_name(self):
        # python -m mimora.cli: the name does not end in __main__, so it is
        # the module to re-run, not its parent.
        command = _command_for(
            ["/src/mimora/cli.py"],
            ModuleSpec("mimora.cli", loader=None),
        )
        self.assertEqual(command, [FAKE_PYTHON, "-m", "mimora.cli"])


class ConsoleScriptLaunchTests(unittest.TestCase):
    """The installed entry point, which starts the interpreter itself."""

    def test_windows_executable_runs_alone(self):
        # A Windows console script carries a __spec__ after all: the .exe is a
        # launcher with a zip archive appended, and the __main__.py inside it is
        # imported like any module, so the spec exists and is named literally
        # "__main__". The first live run of an installed package relaunched as
        # `python.exe -m __main__` because of it, and the process died on the
        # spot. Passing this with spec=None would prove nothing - that is the
        # POSIX case below.
        argv = [r"C:\venv\Scripts\mimora.exe"]
        self.assertEqual(
            _command_for(argv, ModuleSpec("__main__", loader=None)), argv)

    def test_the_windows_launcher_drops_the_extension(self):
        # Observed on a live run: launching Scripts\mimora.exe leaves
        # sys.argv[0] WITHOUT the extension, so the real Windows form looks
        # exactly like the POSIX one. It still takes the same branch (not a
        # .py suffix), and CreateProcess appends .exe on its own - but the
        # test above spells a shape Windows does not actually produce.
        argv = [r"D:\venv\Scripts\mimora"]
        self.assertEqual(
            _command_for(argv, ModuleSpec("__main__", loader=None)), argv)

    def test_posix_console_script_runs_alone(self):
        # No extension at all on POSIX; the shebang does the work, the script
        # is started by path, and __spec__ really is None there.
        argv = ["/home/user/.local/bin/mimora"]
        self.assertEqual(_command_for(argv), argv)

    def test_interpreter_is_never_prepended_to_an_executable(self):
        # python.exe mimora.exe does not run at all - this is the failure the
        # branch exists to prevent.
        command = _command_for([r"C:\venv\Scripts\mimora.exe"])
        self.assertNotIn(FAKE_PYTHON, command)

    def test_the_zipapp_spec_is_not_treated_as_a_module(self):
        # The other half of the same bug: "__main__" says nothing about what to
        # pass to -m, so the -m branch must decline it rather than echo it.
        command = _command_for([r"C:\venv\Scripts\mimora.exe"],
                               ModuleSpec("__main__", loader=None))
        self.assertNotIn("-m", command)

    def test_arguments_survive(self):
        argv = ["/home/user/.local/bin/mimora", "--version"]
        self.assertEqual(_command_for(argv), argv)


class AppendLogFlagTests(unittest.TestCase):
    """The replacement is told to continue the log rather than start a new one.

    The flag is added by spawn_replacement(), not by relaunch_command(), so it
    is only observable through the command handed to Popen - which is what
    these tests read. What they pin is the difference between "one session,
    restarts included" and the previous behaviour, where the child truncated
    main.log and the first-run download vanished from it.
    """

    def _spawned_command(self, argv):
        """The command spawn_replacement() would start, with Popen stubbed."""
        fake_main = types.SimpleNamespace(__spec__=None)
        with mock.patch.object(sys, "argv", list(argv)), \
                mock.patch.object(sys, "executable", FAKE_PYTHON), \
                mock.patch.dict(sys.modules, {"__main__": fake_main}), \
                mock.patch.object(lifecycle.subprocess, "Popen") as popen:
            lifecycle.spawn_replacement()
        # Positional argument 0 of the single call, on either platform branch.
        return popen.call_args[0][0]

    def test_the_flag_is_added(self):
        command = self._spawned_command(["main.py"])
        self.assertEqual(
            command, [FAKE_PYTHON, "main.py", lifecycle.bootstrap.APPEND_LOG_FLAG])

    def test_an_existing_flag_is_not_repeated(self):
        # This process was itself started by a restart, so sys.argv already
        # carries the flag and relaunch_command() copies it over.
        argv = ["main.py", lifecycle.bootstrap.APPEND_LOG_FLAG]
        command = self._spawned_command(argv)
        self.assertEqual(
            command.count(lifecycle.bootstrap.APPEND_LOG_FLAG), 1)

    def test_the_original_arguments_survive(self):
        command = self._spawned_command(["main.py", "--flag", "value"])
        self.assertEqual(command[:3], [FAKE_PYTHON, "main.py", "--flag"])


class CommandOwnershipTests(unittest.TestCase):
    """The result is a new list, whoever built it."""

    def test_console_script_result_is_a_copy(self):
        argv = ["/home/user/.local/bin/mimora"]
        with mock.patch.object(sys, "argv", argv):
            with mock.patch.dict(
                    sys.modules,
                    {"__main__": types.SimpleNamespace(__spec__=None)}):
                command = lifecycle.relaunch_command()
        # Popen would not mutate it, but a caller appending an argument to
        # sys.argv by accident is a bug worth ruling out cheaply.
        self.assertIsNot(command, argv)


if __name__ == "__main__":
    unittest.main()
