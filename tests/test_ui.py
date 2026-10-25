# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Tests for the scrollbar layout helper in mimora/ui.py.

Only the pure helper is tested: no test creates a Tk root or a style. How
the bar looks is checked by eye on a machine with Tk 9 (macOS with the
Python from uv).
"""

import unittest

from mimora import ui

# The layout ttkbootstrap builds for "Vertical.TScrollbar", as Tk 9.1 on
# macOS returns it from style.layout().
BOOTSTRAP_LAYOUT = [
    ("Vertical.Scrollbar.trough", {
        "sticky": "ns",
        "children": [
            ("Vertical.Scrollbar.uparrow", {"side": "top", "sticky": ""}),
            ("Vertical.Scrollbar.downarrow", {"side": "bottom", "sticky": ""}),
            ("Vertical.TScrollbar.thumb", {"expand": "1", "sticky": "nswe"}),
        ],
    }),
]


class WithoutArrowsTests(unittest.TestCase):
    def test_arrows_are_removed_and_thumb_is_kept(self):
        self.assertEqual(ui.without_arrows(BOOTSTRAP_LAYOUT), [
            ("Vertical.Scrollbar.trough", {
                "sticky": "ns",
                "children": [
                    ("Vertical.TScrollbar.thumb",
                     {"expand": "1", "sticky": "nswe"}),
                ],
            }),
        ])

    def test_input_layout_is_not_changed(self):
        # The style keeps the layout it was given; the helper must not edit it.
        before = repr(BOOTSTRAP_LAYOUT)
        ui.without_arrows(BOOTSTRAP_LAYOUT)
        self.assertEqual(repr(BOOTSTRAP_LAYOUT), before)

    def test_applying_twice_gives_the_same_layout(self):
        # _apply_ttk_palette runs twice, so the second pass sees its own result.
        once = ui.without_arrows(BOOTSTRAP_LAYOUT)
        self.assertEqual(ui.without_arrows(once), once)

    def test_horizontal_arrows_are_removed_too(self):
        layout = [("Horizontal.Scrollbar.trough", {"children": [
            ("Horizontal.Scrollbar.leftarrow", {"side": "left"}),
            ("Horizontal.Scrollbar.rightarrow", {"side": "right"}),
            ("Horizontal.TScrollbar.thumb", {"expand": "1"}),
        ]})]
        self.assertEqual(ui.without_arrows(layout), [
            ("Horizontal.Scrollbar.trough", {"children": [
                ("Horizontal.TScrollbar.thumb", {"expand": "1"}),
            ]}),
        ])


if __name__ == "__main__":
    unittest.main()
