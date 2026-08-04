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
    """A stand-in for mimora.app, which in reality imports the whole stack."""
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
        def must_not_run():
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
        def must_not_run():
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
        with _launched_as(["mimora"], _fake_app(lambda: calls.append("run"))):
            cli.main()
        self.assertEqual(calls, ["run"])

    def test_early_init_runs_before_the_application(self):
        # early_init switches console encodings and installs warning filters,
        # and both only take effect while the libraries it configures are still
        # unimported. Order, not merely presence, is what is asserted.
        order = []
        app = _fake_app(lambda: order.append("run"))
        with mock.patch.object(cli.bootstrap, "early_init",
                               side_effect=lambda: order.append("early_init")), \
                mock.patch.object(sys, "argv", ["mimora"]), \
                mock.patch.dict(sys.modules, {"mimora.app": app}), \
                mock.patch.object(mimora, "app", app, create=True), \
                redirect_stdout(io.StringIO()):
            cli.main()
        self.assertEqual(order, ["early_init", "run"])


if __name__ == "__main__":
    unittest.main()
