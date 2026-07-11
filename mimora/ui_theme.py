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

# Monospace family for aligned/technical text (phoneme rows in the history
# panel). Kept next to FONT_FAMILY so the one hard-coded face lives here too.
FONT_FAMILY_MONO = "Consolas"

# Typographic scale (Tk points), ordered large -> small. This is the single
# source of truth for every font size in the UI: widgets must pull from these
# names instead of hard-coding numbers, so the whole interface can be rescaled
# from one place. Tk points render larger in pixels than the original design's
# CSS px (Windows ~1.3x), so e.g. the 16pt phrase lands near the original hero
# text. Sizes that share a value are still kept as separate, role-named
# constants so each can be tuned independently later.
FONT_SIZE_SCORE = 26         # hero verdict score - the single largest element
FONT_SIZE_EMOJI = 20         # emoji glyph drawn on the round record button
FONT_SIZE_PHRASE = 16        # hero practice phrase (the sentence being trained)
FONT_SIZE_ICON = 15          # header gear / settings glyph
FONT_SIZE_TITLE = 14         # header brand title ("MIMORA + language")
FONT_SIZE_SUBTITLE = 11      # emphasized secondary text: translation line, history rows
FONT_SIZE_BODY = 10          # default body / label text
FONT_SIZE_SMALL = 9          # compact controls: status bar, settings fields and buttons
FONT_SIZE_CAPTION = 8        # captions, help text, legends, sublabels

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


class FlatButton(tk.Label):
    """A flat, fully colour-controllable button built on :class:`tk.Label`.

    On macOS the classic :class:`tk.Button` is drawn by the native Aqua engine,
    which ignores the ``bg``/``activebackground`` options - every button renders
    as a white system button regardless of the THEME colour passed, so the app's
    purple buttons came out almost white on Mac. :class:`tk.Label` honours
    ``bg``/``fg`` on every platform, so this emulates the slice of the tk.Button
    API the view relies on - ``command``, ``state`` (normal/disabled), the
    ``active*`` press colours and ``disabledforeground`` - on top of a Label.
    The result is identical on Windows/Linux and finally-purple on macOS.

    Faithful to tk.Button semantics: the label shows its ``active*`` colours
    while pressed and fires ``command`` on release *inside* the widget; a
    disabled button neither highlights nor fires. ``config(state=...)`` and
    ``widget["state"]`` work as before, so callers need no changes beyond the
    class name. Only ``<ButtonPress-1>``/``<ButtonRelease-1>``/``<Leave>`` are
    bound here (not ``<Enter>``), leaving :func:`bind_hover` free to add a hover
    treatment on the same widget without clashing.
    """

    def __init__(self, parent, command=None, activebackground=None,
                 activeforeground=None, disabledforeground=None,
                 state=tk.NORMAL, **kwargs):
        self._command = command
        self._normal_bg = kwargs.get("bg", kwargs.get("background"))
        self._normal_fg = kwargs.get("fg", kwargs.get("foreground"))
        self._active_bg = activebackground if activebackground is not None else self._normal_bg
        self._active_fg = activeforeground if activeforeground is not None else self._normal_fg
        self._disabled_fg = disabledforeground if disabledforeground is not None else self._normal_fg
        self._state = state
        self._pressed = False
        super().__init__(parent, **kwargs)
        self.bind("<ButtonPress-1>", self._on_press, add="+")
        self.bind("<ButtonRelease-1>", self._on_release, add="+")
        self.bind("<Leave>", self._on_leave, add="+")
        self._apply_state()

    def _paint(self, bg, fg):
        """Set bg/fg on the underlying Label, skipping any unset (None) colour."""
        opts = {}
        if bg is not None:
            opts["bg"] = bg
        if fg is not None:
            opts["fg"] = fg
        if opts:
            tk.Label.configure(self, **opts)

    def _apply_state(self):
        """Repaint in the resting look for the current enabled/disabled state."""
        self._pressed = False
        fg = self._disabled_fg if self._state == tk.DISABLED else self._normal_fg
        self._paint(self._normal_bg, fg)

    def _on_press(self, _event=None):
        if self._state == tk.DISABLED:
            return
        self._pressed = True
        self._paint(self._active_bg, self._active_fg)

    def _on_release(self, event=None):
        was_pressed = self._pressed
        self._apply_state()  # restore the resting look (also clears _pressed)
        if self._state == tk.DISABLED or not was_pressed:
            return
        # tk.Button fires only when the release lands inside the widget.
        if event is not None and not (
                0 <= event.x < self.winfo_width()
                and 0 <= event.y < self.winfo_height()):
            return
        if self._command is not None:
            self._command()

    def _on_leave(self, _event=None):
        # Dragging off a pressed button drops the pressed look and cancels the
        # click, matching tk.Button.
        if self._pressed:
            self._apply_state()

    def configure(self, cnf=None, **kw):
        """Accept the tk.Button-only options this view sets at runtime.

        ``command``/``activebackground``/``activeforeground``/
        ``disabledforeground``/``state`` are intercepted (a Label would reject
        them); ``bg``/``fg`` updates are mirrored into the resting-colour store
        so hover/press stay consistent. Everything else falls through to Label.
        """
        if cnf:
            kw = {**cnf, **kw}
        if "command" in kw:
            self._command = kw.pop("command")
        if "activebackground" in kw:
            self._active_bg = kw.pop("activebackground")
        if "activeforeground" in kw:
            self._active_fg = kw.pop("activeforeground")
        if "disabledforeground" in kw:
            self._disabled_fg = kw.pop("disabledforeground")
        if "state" in kw:
            self._state = kw.pop("state")
        if "bg" in kw:
            self._normal_bg = kw["bg"]
        if "background" in kw:
            self._normal_bg = kw["background"]
        if "fg" in kw:
            self._normal_fg = kw["fg"]
        if "foreground" in kw:
            self._normal_fg = kw["foreground"]
        if kw:
            tk.Label.configure(self, **kw)
        self._apply_state()

    config = configure

    def cget(self, key):
        if key == "state":
            return self._state
        if key == "command":
            return self._command
        if key == "activebackground":
            return self._active_bg
        if key == "activeforeground":
            return self._active_fg
        if key == "disabledforeground":
            return self._disabled_fg
        return tk.Label.cget(self, key)

    __getitem__ = cget


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
