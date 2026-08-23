# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for the command line entry point (mimora/cli.py).

One property, worth a file of its own: **`mimora --version` must answer without
loading the application.** Printing a version string is the first thing a bug
report asks for, and on a slow machine importing mimora.app - torch,
transformers, Kokoro - is tens of seconds. That is the entire reason cli.py
exists as a separate module: `bootstrap.early_init()` only works before the
libraries it configures are imported, and argument parsing cannot live in the
module it is meant to protect.

Which makes `from mimora import app`, sitting inside `main()` rather than at the
top of the file, load-bearing rather than a matter of style - and a one-line
edit away from being wrong. Moving it up breaks nothing visibly: every launch
form still works, `--version` still prints, it just takes ten seconds first. So
the test below reads cli.py's module-level imports and says so out loud, and two
behavioural tests pin the order around them.

A second property lives here because it belongs to the same entry point: the
two native pieces no wheel can supply - tkinter and PortAudio - must fail with
the command that installs them rather than with a traceback, and everything
else must keep its traceback. `install.py` checks both before the first launch,
but the recommended install path is the published wheel, which never runs it.

Run from the project root with:

    python -m unittest tests.test_cli
"""

import ast
import contextlib
import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import mimora
from mimora import cli

# What cli.py is allowed to take from its own package at module level. Both are
# stdlib-only themselves: __init__ defines nothing but __version__, and
# bootstrap is the early process setup that has to run before anything heavy.
ALLOWED_FROM_PACKAGE = {"__version__", "bootstrap"}


def _fake_app(run):
    """A stand-in for mimora.app, which in reality imports the whole stack.

    ``run`` is called the way cli.main() calls the real one, with keyword
    arguments (``append_log``), so a stand-in has to accept them - hence the
    ``**kwargs`` on every stub below. A stub that took none would turn a
    changed signature into a TypeError raised from inside the test's own
    lambda, which says nothing about what broke.
    """
    module = types.ModuleType("mimora.app")
    module.run = run
    return module


@contextlib.contextmanager
def _launched_as(argv, app):
    """Run cli.main() with a given command line and a stubbed application.

    The stand-in is registered both in sys.modules and as an attribute of the
    package, because ``from mimora import app`` reads the attribute first and
    only then falls back to sys.modules.

    Stubbing it at all is the point: without it these tests would import the
    real mimora.app, pull in torch and open a Tk window from inside a test run.
    """
    with mock.patch.object(cli.bootstrap, "early_init"), \
            mock.patch.object(sys, "argv", list(argv)), \
            mock.patch.dict(sys.modules, {"mimora.app": app}), \
            mock.patch.object(mimora, "app", app, create=True), \
            redirect_stdout(io.StringIO()):
        yield


class ModuleLevelImportTests(unittest.TestCase):
    """What cli.py imports before it has parsed anything."""

    def _module_level_imports(self):
        """(module, imported name) pairs from the top level of cli.py.

        Read from the source rather than from the imported module: by the time
        a test runs, sys.modules says nothing about WHERE an import was
        written, and "inside the function" is exactly the fact under test.
        Only ``tree.body`` is walked, so an import nested in a function or a
        conditional is invisible here - which is the intended reading.
        """
        tree = ast.parse(Path(cli.__file__).read_text(encoding="utf-8"))
        pairs = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                pairs.extend((alias.name, None) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                pairs.extend((node.module or "", alias.name)
                             for alias in node.names)
        return pairs

    def test_the_application_is_not_imported_at_module_level(self):
        # The regression this file exists for. `from mimora import app` belongs
        # inside main(), after parse_args, or --version pays for the whole
        # application load before it can print one line.
        for module, name in self._module_level_imports():
            with self.subTest(module=module, name=name):
                self.assertNotEqual(module, "mimora.app")
                self.assertFalse(module == "mimora" and name == "app")

    def test_nothing_heavier_than_the_standard_library_is_imported(self):
        # The same rule stated in full, which is what makes the test survive
        # somebody importing torch here directly rather than through app.py.
        for module, name in self._module_level_imports():
            with self.subTest(module=module, name=name):
                if module == "mimora":
                    self.assertIn(name, ALLOWED_FROM_PACKAGE)
                    continue
                self.assertIn(module.split(".")[0], sys.stdlib_module_names)


class VersionFlagTests(unittest.TestCase):
    """--version answers, and answers without waking the application."""

    def test_version_prints_the_package_version_and_exits_zero(self):
        def must_not_run(**kwargs):
            raise AssertionError("--version reached the application")

        app = _fake_app(must_not_run)
        printed = io.StringIO()
        with mock.patch.object(cli.bootstrap, "early_init"), \
                mock.patch.object(sys, "argv", ["mimora", "--version"]), \
                mock.patch.dict(sys.modules, {"mimora.app": app}), \
                mock.patch.object(mimora, "app", app, create=True), \
                redirect_stdout(printed):
            with self.assertRaises(SystemExit) as caught:
                cli.main()
        # argparse's version action exits with 0.
        self.assertEqual(caught.exception.code, 0)
        self.assertIn(mimora.__version__, printed.getvalue())


class DetectHardwareFlagTests(unittest.TestCase):
    """The maintenance command that only the console script can reach."""

    def test_the_flag_runs_the_probe_and_never_starts_the_application(self):
        def must_not_run(**kwargs):
            raise AssertionError("--detect-hardware started the application")

        probe = types.ModuleType("mimora.detect_hardware")
        calls = []
        probe.main = lambda: calls.append("probe") or 0
        app = _fake_app(must_not_run)
        with mock.patch.object(cli.bootstrap, "early_init"), \
                mock.patch.object(sys, "argv",
                                  ["mimora", "--detect-hardware"]), \
                mock.patch.dict(sys.modules,
                                {"mimora.app": app,
                                 "mimora.detect_hardware": probe}), \
                mock.patch.object(mimora, "app", app, create=True), \
                mock.patch.object(mimora, "detect_hardware", probe,
                                  create=True), \
                redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                cli.main()
        self.assertEqual(calls, ["probe"])
        self.assertEqual(caught.exception.code, 0)


class HandoverTests(unittest.TestCase):
    """The other half: a normal launch does reach the application."""

    def test_a_normal_launch_hands_over_to_the_application(self):
        calls = []
        with _launched_as(["mimora"],
                          _fake_app(lambda **kwargs: calls.append("run"))):
            cli.main()
        self.assertEqual(calls, ["run"])

    def test_early_init_runs_before_the_application(self):
        # early_init switches console encodings and installs warning filters,
        # and both only take effect while the libraries it configures are still
        # unimported. Order, not merely presence, is what is asserted.
        order = []
        app = _fake_app(lambda **kwargs: order.append("run"))
        with mock.patch.object(cli.bootstrap, "early_init",
                               side_effect=lambda: order.append("early_init")), \
                mock.patch.object(sys, "argv", ["mimora"]), \
                mock.patch.dict(sys.modules, {"mimora.app": app}), \
                mock.patch.object(mimora, "app", app, create=True), \
                redirect_stdout(io.StringIO()):
            cli.main()
        self.assertEqual(order, ["early_init", "run"])


class AppendLogFlagTests(unittest.TestCase):
    """--append-log reaches the application, and is off unless asked for.

    The switch exists for the restart the app performs on itself (see
    lifecycle.spawn_replacement), so the half tested here is only the wiring:
    argparse to app.run. What it selects - continuing logs/main.log instead of
    truncating it - belongs to bootstrap.setup_logging.
    """

    def _run_kwargs(self, argv):
        """The keyword arguments cli.main() passes to app.run for a command line."""
        seen = {}
        app = _fake_app(lambda **kwargs: seen.update(kwargs))
        with _launched_as(argv, app):
            cli.main()
        return seen

    def test_the_flag_is_passed_through(self):
        self.assertEqual(
            self._run_kwargs(["mimora", "--append-log"]),
            {"append_log": True})

    def test_a_plain_launch_leaves_it_off(self):
        # The deliberate half: a launch nobody restarted starts a clean log,
        # so one file holds one session.
        self.assertEqual(self._run_kwargs(["mimora"]), {"append_log": False})

    def test_the_flag_matches_the_constant_the_relaunch_uses(self):
        # cli.py declares the argument from bootstrap.APPEND_LOG_FLAG and
        # lifecycle.spawn_replacement() appends the same constant. This pins
        # the spelling argparse derives "append_log" from: renaming the
        # constant alone would keep both sides agreeing with each other and
        # silently stop matching the destination read in cli.main().
        self.assertEqual(cli.bootstrap.APPEND_LOG_FLAG, "--append-log")


class NativeDependencyHintTests(unittest.TestCase):
    """The two Linux failures a wheel install cannot prevent, made readable."""

    def _tkinter_hint(self, manager):
        """The hint for a missing tkinter on a machine that has `manager`."""
        with mock.patch.object(sys, "platform", "linux"), \
                mock.patch.object(cli.shutil, "which",
                                  side_effect=lambda name: name == manager):
            return cli._native_hint_for(
                ModuleNotFoundError("No module named 'tkinter'", name="tkinter"))

    def test_the_package_name_follows_the_distribution(self):
        # The reason this is a lookup and not one string: the same missing
        # module has three different cures.
        self.assertIn("python3-tk", self._tkinter_hint("apt-get"))
        self.assertIn("python3-tkinter", self._tkinter_hint("dnf"))
        self.assertIn("pacman -S tk", self._tkinter_hint("pacman"))

    def test_an_unknown_distribution_still_gets_prose(self):
        with mock.patch.object(sys, "platform", "linux"), \
                mock.patch.object(cli.shutil, "which", return_value=None):
            hint = cli._native_hint_for(
                ModuleNotFoundError("No module named 'tkinter'", name="tkinter"))
        self.assertIn("package manager", hint)

    def test_a_missing_portaudio_names_its_package(self):
        with mock.patch.object(cli.shutil, "which",
                               side_effect=lambda name: name == "apt-get"):
            hint = cli._native_hint_for(OSError("PortAudio library not found"))
        self.assertIn("libportaudio2", hint)

    def test_anything_else_keeps_its_traceback(self):
        # The half that matters more: a real bug must not be dressed up as a
        # missing system package and stripped of its traceback.
        self.assertIsNone(cli._native_hint_for(
            ModuleNotFoundError("No module named 'torch'", name="torch")))
        self.assertIsNone(cli._native_hint_for(OSError("disk full")))

    def test_main_reports_the_missing_package_instead_of_a_traceback(self):
        """The wiring: main() must actually consult the hint.

        Without this, deleting the try/except in main() leaves every test
        above green while the user gets the traceback back.

        The stand-in below is a package, not an application module: `from
        mimora import app` reads the attribute off the package first and only
        falls back to importing the submodule when that raises AttributeError.
        So `__path__` answers AttributeError (the import machinery asks for it
        first, and treating the stand-in as a plain module is what routes the
        lookup through the attribute) while `app` raises the failure under
        test - the same exception mimora/app.py raises on a Linux box without
        python3-tk, and nothing heavy is imported to produce it.
        """
        class PackageWithoutTkinter(types.ModuleType):
            def __getattr__(self, name):
                if name != "app":
                    raise AttributeError(name)
                raise ModuleNotFoundError(
                    "No module named 'tkinter'", name="tkinter")

        with mock.patch.object(cli.bootstrap, "early_init"), \
                mock.patch.object(sys, "argv", ["mimora"]), \
                mock.patch.object(sys, "platform", "linux"), \
                mock.patch.object(cli.shutil, "which",
                                  side_effect=lambda name: name == "apt-get"), \
                mock.patch.dict(sys.modules,
                                {"mimora": PackageWithoutTkinter("mimora")}), \
                redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                cli.main()
        # SystemExit carries the message as its code, which is what the
        # interpreter prints to stderr before exiting with 1.
        self.assertIn("python3-tk", str(caught.exception.code))


if __name__ == "__main__":
    unittest.main()
