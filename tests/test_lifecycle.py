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
    for ``python -m ...`` and None for every other launch form.
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
        argv = [r"C:\venv\Scripts\mimora.exe"]
        self.assertEqual(_command_for(argv), argv)

    def test_posix_console_script_runs_alone(self):
        # No extension at all on POSIX; the shebang does the work.
        argv = ["/home/user/.local/bin/mimora"]
        self.assertEqual(_command_for(argv), argv)

    def test_interpreter_is_never_prepended_to_an_executable(self):
        # python.exe mimora.exe does not run at all - this is the failure the
        # branch exists to prevent.
        command = _command_for([r"C:\venv\Scripts\mimora.exe"])
        self.assertNotIn(FAKE_PYTHON, command)

    def test_arguments_survive(self):
        argv = ["/home/user/.local/bin/mimora", "--version"]
        self.assertEqual(_command_for(argv), argv)


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
