# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Tests for the entry point checks of tools/check_wheel.py.

The wheel is a small zip built in memory, and pyproject is a dict, so nothing
is built, installed or read from dist/. The rule under test is in the comment
inside check_layout: the console scripts and the GUI scripts of the wheel are
both compared with pyproject.

Run from the project root with:

    python -m unittest tests.test_check_wheel
"""

import contextlib
import importlib.util
import io
import unittest
import zipfile
from pathlib import Path

# Loaded by path: tools/ is not a package, and "from tools import ..." would
# pick up any installed package that happens to be named "tools".
_SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "check_wheel.py"
_SPEC = importlib.util.spec_from_file_location("check_wheel", _SCRIPT)
check_wheel = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_wheel)

CONSOLE = "[console_scripts]\nmimora = mimora.cli:main\n"
GUI = "[gui_scripts]\nmimora-gui = mimora.cli:main\n"


def make_wheel(entry_points: str) -> zipfile.ZipFile:
    """An in-memory wheel with one module and the given entry points."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("mimora/__init__.py", "")
        archive.writestr("mimora-1.0.dist-info/entry_points.txt", entry_points)
    buffer.seek(0)
    return zipfile.ZipFile(buffer)


def make_config(gui_scripts: dict | None) -> dict:
    """The part of pyproject that check_layout reads."""
    project = {"scripts": {"mimora": "mimora.cli:main"}}
    if gui_scripts is not None:
        project["gui-scripts"] = gui_scripts
    return {"project": project}


class EntryPointTests(unittest.TestCase):
    def setUp(self):
        # report() counts failures in a module-level variable.
        check_wheel._failures = 0

    def failures_after_check(self, entry_points: str, config: dict) -> int:
        with contextlib.redirect_stdout(io.StringIO()):
            check_wheel.check_layout(make_wheel(entry_points), config)
        return check_wheel._failures

    def test_both_sections_as_declared_pass(self):
        config = make_config({"mimora-gui": "mimora.cli:main"})
        failures = self.failures_after_check(CONSOLE + "\n" + GUI, config)
        self.assertEqual(failures, 0)

    def test_missing_gui_script_fails(self):
        # The case this check exists for: the installer shortcuts would point
        # at an executable that the wheel does not create.
        config = make_config({"mimora-gui": "mimora.cli:main"})
        self.assertEqual(self.failures_after_check(CONSOLE, config), 1)

    def test_gui_script_with_another_target_fails(self):
        config = make_config({"mimora-gui": "mimora.cli:main"})
        wrong = "[gui_scripts]\nmimora-gui = mimora.app:run\n"
        failures = self.failures_after_check(CONSOLE + "\n" + wrong, config)
        self.assertEqual(failures, 1)

    def test_no_gui_scripts_anywhere_passes(self):
        failures = self.failures_after_check(CONSOLE, make_config(None))
        self.assertEqual(failures, 0)


if __name__ == "__main__":
    unittest.main()
