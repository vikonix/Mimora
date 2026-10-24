# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Tests for mimora/splash.py.

The parent side is tested: how the child process is started and stopped. Of
the child, only the two helpers that decide the window style and the icon are
tested, with a stand-in for the Tk root. No test starts the child, because it
opens a real window. What the window looks like is checked by eye with
``python mimora/splash.py`` (a click closes it).
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


def _fake_root(windowing_system):
    """A stand-in for a Tk root that reports *windowing_system*."""
    root = mock.Mock()
    root.tk.call.return_value = windowing_system
    return root


def _window_calls(root):
    """The window methods called on *root*, in order."""
    names = ("withdraw", "overrideredirect", "deiconify")
    return [name for name, _args, _kwargs in root.mock_calls if name in names]


class RemoveTitleBarTests(unittest.TestCase):
    def test_windows_and_x11_set_overrideredirect_on_the_new_root(self):
        for system in ("win32", "x11"):
            with self.subTest(system=system):
                root = _fake_root(system)
                splash._remove_title_bar(root)
                root.overrideredirect.assert_called_once_with(True)
                self.assertEqual(_window_calls(root), ["overrideredirect"])

    def test_macos_sets_it_while_the_root_is_withdrawn(self):
        # Set on the shown root, Tk 9 on Aqua keeps the title bar and its
        # buttons; the order is what removes them.
        root = _fake_root("aqua")
        splash._remove_title_bar(root)
        root.overrideredirect.assert_called_once_with(True)
        self.assertEqual(_window_calls(root),
                         ["withdraw", "overrideredirect", "deiconify"])


class SetIconTests(unittest.TestCase):
    def test_windows_gets_no_icon(self):
        with mock.patch.object(splash.sys, "platform", "win32"), \
                mock.patch("mimora.window_icon.apply_window_icon") as apply:
            splash._set_icon(mock.Mock())
        apply.assert_not_called()

    def test_macos_and_linux_get_the_mimora_icon(self):
        for platform in ("darwin", "linux"):
            with self.subTest(platform=platform), \
                    mock.patch.object(splash.sys, "platform", platform), \
                    mock.patch.object(splash.sys, "path", list(sys.path)), \
                    mock.patch("mimora.window_icon.apply_window_icon") as apply:
                root = mock.Mock()
                splash._set_icon(root)
                apply.assert_called_once_with(root)

    def test_the_project_root_is_added_at_the_end_of_sys_path(self):
        # At the front, a file in the project root could take the place of a
        # standard library module in the child.
        project_root = str(Path(splash.__file__).resolve().parent.parent)
        others = [entry for entry in sys.path if entry != project_root]
        with mock.patch.object(splash.sys, "platform", "darwin"), \
                mock.patch.object(splash.sys, "path", list(others)), \
                mock.patch("mimora.window_icon.apply_window_icon"):
            splash._set_icon(mock.Mock())
            self.assertEqual(splash.sys.path[:-1], others)
            self.assertEqual(splash.sys.path[-1], project_root)


if __name__ == "__main__":
    unittest.main()
