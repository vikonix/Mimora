# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Shared UI building blocks for the Mimora view layer.

Everything here is panel-agnostic: the resolved color palette, the typographic
scale, cross-platform mouse-wheel helpers and the hover tooltip. The view
modules (ui.py facade and the ui_* panel classes) import from this module, so
it must stay free of imports from any of them.

Importing this module also applies the ttkbootstrap "autostyle" patch (see the
comment at the import below), so it must be imported before any Tk widget is
created - which holds automatically because every view module imports it at
module level.
"""
import platform
import tkinter as tk

# ttkbootstrap is a drop-in replacement for tkinter.ttk (same widget classes,
# modern flat themes). Imported here so the autostyle patch below runs exactly
# once, before any widget exists.
from ttkbootstrap.style import Bootstyle

# On import ttkbootstrap patches the constructors of the classic tk widgets
# ("autostyle"): right after creation every widget is repainted with the base
# theme's colors, discarding the explicit THEME bg/fg this view passes (buttons
# turned theme-blue, the phrase label white, panels grey). The view themes
# every classic widget itself, so the hook is unwanted globally - including for
# widgets created inside libraries (scrolledtext internals, the FaceWidget
# canvas), which the per-widget ``autostyle=False`` flag cannot reach. Only the
# classic-widget hook is disabled; ttk widgets (the comboboxes) keep their
# ttkbootstrap styling. Verified against ttkbootstrap 1.x internals - see the
# version pin in requirements.txt.
Bootstyle.update_tk_widget_style = staticmethod(lambda widget=None: None)

from mimora import config

# Resolved UI color palette (semantic name -> hex), selected by the
# "color_theme" setting in settings.json; see config.py.
THEME = config.THEME

# ttkbootstrap base theme per Mimora color theme. The base theme supplies only
# the ttk widget geometry/elements (combobox arrow, focus behaviour); every
# visible color is still overridden from THEME in the facade's
# _apply_ttk_palette, so the palette keeps coming from config/themes/ exactly
# as before.
_BOOTSTRAP_THEMES = {"dark": "darkly", "light": "flatly"}
BOOTSTRAP_THEME = _BOOTSTRAP_THEMES.get(config.COLOR_THEME, "darkly")

# UI font family, chosen per platform. "Segoe UI" exists only on Windows;
# without an explicit choice Tk would silently substitute an arbitrary font
# on other systems, so each platform gets its standard UI face instead.
_FONT_FAMILIES = {
    "Windows": "Segoe UI",
    "Darwin": "Helvetica Neue",   # macOS
}
FONT_FAMILY = _FONT_FAMILIES.get(platform.system(), "DejaVu Sans")  # Linux/other

# Typographic scale (Tk points). Kept in one place so the redesign pulls from a
# single ladder instead of scattering magic sizes across widgets. Tk points
# render larger in pixels than the original design's CSS px (Windows ~1.3x),
# so the 21pt phrase lands near the original 27px hero text.
FONT_SIZE_PHRASE = 21        # hero practice phrase (original design ~27px)
FONT_SIZE_SCORE = 26         # big verdict score in the score row (added in the
                             # hero-card stage; defined here to keep the scale whole)
FONT_SIZE_TRANSLATION = 11   # translation line under the phrase
FONT_SIZE_BODY = 10          # normal body text
FONT_SIZE_CAPTION = 8        # sublabels, captions, legends

# The wheel-event sequences a scrollable widget must listen to: <MouseWheel>
# covers Windows and macOS; X11 (Linux) delivers Button-4/Button-5 instead.
WHEEL_EVENTS = ("<MouseWheel>", "<Button-4>", "<Button-5>")


def wheel_scroll_units(event) -> int:
    """Mouse-wheel event -> ``yview_scroll`` units, cross-platform.

    Windows reports ``event.delta`` in multiples of 120 per notch; macOS
    reports small per-event deltas (typically 1..3); X11 sends Button-4/5
    events whose ``delta`` stays 0, so the button number decides. Negative
    units scroll up, matching Tk's ``yview_scroll`` convention.
    """
    num = getattr(event, "num", 0)
    if num == 4:
        return -1
    if num == 5:
        return 1
    delta = getattr(event, "delta", 0)
    if delta == 0:
        return 0
    if abs(delta) >= 120:          # Windows: n * 120 per notch
        return -int(delta / 120)
    return -1 if delta > 0 else 1  # macOS: small per-event deltas


def bind_hover(widget, enter: dict, leave: dict) -> None:
    """Give a widget a hover state: ``config(**enter)`` on ``<Enter>``,
    ``config(**leave)`` on ``<Leave>``.

    Tk buttons have no native hover (``activebackground`` shows only while
    pressed), so without this flat buttons read as static text. ``add="+"``
    keeps other Enter/Leave bindings (e.g. a Tooltip) working on the same
    widget.
    """
    widget.bind("<Enter>", lambda _e: widget.config(**enter), add="+")
    widget.bind("<Leave>", lambda _e: widget.config(**leave), add="+")


class Tooltip:
    """A minimal hover tooltip for a Tk widget (Tk has no native ``title=``).

    Shows a small borderless ``Toplevel`` with ``text`` just below the widget on
    ``<Enter>`` and hides it on ``<Leave>`` or click. One instance per widget;
    the popup is created lazily and destroyed on hide, so no stray windows leak.
    """

    def __init__(self, widget, text: str):
        self._widget = widget
        self._text = text
        self._tip = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _show(self, _event=None):
        if self._tip or not self._text:
            return
        # Position just under the widget's bottom-left corner.
        x = self._widget.winfo_rootx()
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._tip = tk.Toplevel(self._widget)
        self._tip.wm_overrideredirect(True)   # no title bar / border
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self._tip, text=self._text, justify=tk.LEFT,
                 font=(FONT_FAMILY, FONT_SIZE_CAPTION),
                 bg=THEME["bg_panel"], fg=THEME["text"],
                 highlightthickness=1, highlightbackground=THEME["border"],
                 padx=6, pady=3).pack()

    def _hide(self, _event=None):
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None
