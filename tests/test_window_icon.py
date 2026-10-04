# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Tests for mimora/window_icon.py.

No test opens a window: the Tk root is a mock, and the platform is stubbed so
that both branches run on any machine. What the icon looks like is checked by
eye in the running application.

Run from the project root with:

    python -m unittest tests.test_window_icon
"""

import ctypes
import tkinter as tk
import unittest
from unittest import mock

from mimora import paths, window_icon


def _platform(name):
    """Make window_icon see ``name`` as sys.platform."""
    return mock.patch.object(window_icon.sys, "platform", name)


class ShippedIconFilesTests(unittest.TestCase):
    def test_icon_files_are_inside_the_package(self):
        # A wheel carries a data file only from inside a package.
        icons_dir = paths.shipped_icons_dir()
        self.assertEqual(icons_dir, paths.shipped_root() / "icons")
        for name in (window_icon.ICON_ICO_NAME, window_icon.ICON_PNG_NAME):
            self.assertTrue((icons_dir / name).is_file(), name)


class ApplyWindowIconTests(unittest.TestCase):
    def test_windows_sets_the_ico_as_the_default_icon(self):
        root = mock.Mock()
        with _platform("win32"):
            window_icon.apply_window_icon(root)

        expected = paths.shipped_icons_dir() / window_icon.ICON_ICO_NAME
        root.iconbitmap.assert_called_once_with(default=str(expected))
        root.iconphoto.assert_not_called()

    def test_windows_sets_the_sharp_icons_once_when_the_root_appears(self):
        root = mock.Mock()
        with _platform("win32"), mock.patch.object(
                window_icon, "_set_sharp_windows_icons") as set_sharp:
            window_icon.apply_window_icon(root)
            (sequence, on_map), kwargs = root.bind.call_args
            # A child widget first, then the root two times (a restore from
            # the taskbar sends <Map> again).
            on_map(mock.Mock(widget=mock.Mock()))
            on_map(mock.Mock(widget=root))
            on_map(mock.Mock(widget=root))

        self.assertEqual(sequence, "<Map>")
        # "+" keeps the <Map> handlers that the application adds.
        self.assertEqual(kwargs, {"add": "+"})
        expected = paths.shipped_icons_dir() / window_icon.ICON_ICO_NAME
        set_sharp.assert_called_once_with(root, expected)

    def test_a_sharp_icon_failure_is_logged_and_not_raised(self):
        root = mock.Mock()
        with _platform("win32"), mock.patch.object(
                window_icon, "_set_sharp_windows_icons",
                side_effect=OSError("no user32")):
            window_icon.apply_window_icon(root)
            (_, on_map), _ = root.bind.call_args
            with self.assertLogs(level="WARNING"):
                on_map(mock.Mock(widget=root))

    def test_other_platforms_set_the_png_and_keep_a_reference(self):
        root = mock.Mock()
        with _platform("linux"), \
                mock.patch.object(window_icon.tk, "PhotoImage") as photo_image:
            window_icon.apply_window_icon(root)

        expected = paths.shipped_icons_dir() / window_icon.ICON_PNG_NAME
        photo_image.assert_called_once_with(master=root, file=str(expected))
        root.iconphoto.assert_called_once_with(True, photo_image.return_value)
        self.assertIs(root._mimora_icon_image, photo_image.return_value)

    def test_a_failure_is_logged_and_does_not_stop_the_window(self):
        root = mock.Mock()
        root.iconbitmap.side_effect = tk.TclError("bitmap not defined")
        with _platform("win32"), self.assertLogs(level="WARNING"):
            window_icon.apply_window_icon(root)


class SharpenWindowIconTests(unittest.TestCase):
    def test_windows_waits_for_the_window_to_appear(self):
        window = mock.Mock()
        with _platform("win32"), mock.patch.object(
                window_icon, "_set_sharp_windows_icons") as set_sharp:
            window_icon.sharpen_window_icon(window)
            set_sharp.assert_not_called()
            (_, on_map), _ = window.bind.call_args
            on_map(mock.Mock(widget=window))

        expected = paths.shipped_icons_dir() / window_icon.ICON_ICO_NAME
        set_sharp.assert_called_once_with(window, expected)

    def test_does_nothing_off_windows(self):
        window = mock.Mock()
        with _platform("linux"):
            window_icon.sharpen_window_icon(window)

        window.bind.assert_not_called()


class SetAppUserModelIdTests(unittest.TestCase):
    def test_does_nothing_off_windows(self):
        with _platform("linux"), \
                mock.patch.object(ctypes, "windll", create=True) as windll:
            window_icon.set_app_user_model_id()

        windll.shell32.SetCurrentProcessExplicitAppUserModelID.assert_not_called()

    def test_windows_sets_the_identity(self):
        with _platform("win32"), \
                mock.patch.object(ctypes, "windll", create=True) as windll:
            window_icon.set_app_user_model_id()

        windll.shell32.SetCurrentProcessExplicitAppUserModelID \
            .assert_called_once_with(window_icon.APP_USER_MODEL_ID)

    def test_a_failure_is_logged_and_not_raised(self):
        with _platform("win32"), \
                mock.patch.object(ctypes, "windll", create=True) as windll, \
                self.assertLogs(level="WARNING"):
            windll.shell32.SetCurrentProcessExplicitAppUserModelID \
                .side_effect = OSError("no shell32")
            window_icon.set_app_user_model_id()


if __name__ == "__main__":
    unittest.main()
