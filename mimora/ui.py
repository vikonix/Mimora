# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""View layer for the Mimora pronunciation trainer.

This module holds the UI as a standalone, passive :class:`TrainerView`, composed
into the controller (``self.view``) in ``main.py`` rather than inherited. The
view owns every widget and renders state; the controller owns the application
logic. Both directions across the boundary are explicit, typed contracts - the
view shares no implicit namespace with the controller:

* view → controller: widget callbacks call only ``self._cb.<handler>`` on a
  :class:`ViewCallbacks` of plain callables passed in at construction. The view
  never references the controller object, so the two sides form no cycle and the
  view can be exercised with stand-in callbacks.
* controller → view: the controller drives the UI through named *intent*
  methods (``enter_recording``, ``enter_phrase_ready``, ``show_feedback`` …) and
  small read accessors (``get_practice_text`` …). It never touches a widget or a
  status/colour string directly - every UI state and its copy live here.
"""
import logging
import platform
import tkinter as tk
# ttkbootstrap is a drop-in replacement for tkinter.ttk (same widget classes,
# modern flat themes). Aliased as ``ttk`` so every ttk.Combobox / ttk.Style
# reference below keeps working unchanged.
import ttkbootstrap as ttk
from ttkbootstrap.style import Bootstyle

# On import ttkbootstrap patches the constructors of the classic tk widgets
# ("autostyle"): right after creation every widget is repainted with the base
# theme's colors, discarding the explicit THEME bg/fg this view passes (buttons
# turned theme-blue, the phrase label white, panels grey). This view themes
# every classic widget itself, so the hook is unwanted globally - including for
# widgets created inside libraries (scrolledtext internals, the FaceWidget
# canvas), which the per-widget ``autostyle=False`` flag cannot reach. Only the
# classic-widget hook is disabled; ttk widgets (the comboboxes) keep their
# ttkbootstrap styling. Verified against ttkbootstrap 1.x internals - see the
# version pin in requirements.txt.
Bootstyle.update_tk_widget_style = staticmethod(lambda widget=None: None)

from tkinter import scrolledtext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from mimora import config, prosody_utils
from mimora.face_widget import FaceWidget
from mimora.phoneme_examples import example_for

# Resolved UI color palette (semantic name -> hex), selected by the
# "color_theme" setting in settings.json; see config.py.
THEME = config.THEME

# ttkbootstrap base theme per Mimora color theme. The base theme supplies only
# the ttk widget geometry/elements (combobox arrow, focus behaviour); every
# visible color is still overridden from THEME in _apply_ttk_palette, so the
# palette keeps coming from config/themes/ exactly as before.
_BOOTSTRAP_THEMES = {"dark": "darkly", "light": "flatly"}
BOOTSTRAP_THEME = _BOOTSTRAP_THEMES.get(config.COLOR_THEME, "darkly")

if TYPE_CHECKING:  # only for the show_feedback annotation; no runtime import
    from pronunciation.common import PronunciationResult


@dataclass(frozen=True)
class ViewCallbacks:
    """Typed view→controller contract: the handlers the widgets invoke.

    The view stores only this bundle of callables (never the controller object),
    so the two sides share no implicit namespace. Event-bound handlers receive a
    Tk event argument and are also called with none elsewhere, hence the
    ``Callable[..., None]`` signatures for those.
    """
    on_settings_clicked: Callable[[], None]
    on_practice_collapsed_toggled: Callable[[], None]
    on_gui_btn_press: Callable[[], None]
    on_gui_btn_release: Callable[[], None]
    on_show_face_toggled: Callable[[], None]
    on_prosody_toggled: Callable[[], None]
    on_test_reference: Callable[[], None]
    play_user_recording: Callable[[], None]
    play_reference: Callable[[], None]
    play_reference_slow: Callable[[], None]
    on_generate_phrase: Callable[[], None]
    # Click on an underlined ("miss") word inside the hero phrase: the
    # controller synthesizes that single word and plays it slowly.
    on_word_clicked: Callable[[str], None]
    # Click on a "WORK ON" phoneme badge: the controller synthesizes the
    # phoneme's example word (e.g. "put" for /ʊ/) and plays it at normal speed.
    on_sound_example: Callable[[str], None]
    # A take was scored: the view reports the phrase and its user-facing score so
    # the controller can keep the session tally (unique phrases, running average)
    # that the status bar shows. Only called for actually-scored takes.
    on_take_scored: Callable[[str, float], None]
    # A new entry for the attempt history: a scored/unscored take or an error
    # message. The controller owns the bounded history (last 10), computes the
    # per-phrase trend and hands the full list back via view.render_history.
    on_history_entry: Callable[[dict], None]

# Phrase-length selector labels. The label maps to generate_phrase's ``length``
# mode: LENGTH_FULL → "full" sentence, LENGTH_FEW_WORDS → "fragment".
LENGTH_FULL = "Full phrase"
LENGTH_FEW_WORDS = "Few words"

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
# render larger in pixels than the CSS px in the mockup (Windows ~1.3x), so the
# 21pt phrase lands near the mockup's 27px hero text.
FONT_SIZE_PHRASE = 21        # hero practice phrase (mockup ~27px)
FONT_SIZE_SCORE = 26         # big verdict score in the score row (added in the
                             # hero-card stage; defined here to keep the scale whole)
FONT_SIZE_TRANSLATION = 11   # translation line under the phrase
FONT_SIZE_BODY = 10          # normal body text
FONT_SIZE_CAPTION = 8        # sublabels, captions, legends


class _Tooltip:
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


class TrainerView:
    """Passive view: builds the Tk widgets and renders UI state.

    Owns the widgets (built in ``__init__``). Widget callbacks forward to the
    :class:`ViewCallbacks` passed in (``self._cb``); the controller drives the UI
    through the public intent methods and read accessors below. The view holds no
    reference to the controller.
    """

    def __init__(self, root, callbacks: ViewCallbacks):
        """Build the UI under ``root``, wiring widgets to ``callbacks``.

        Args:
            root: the Tk root window the widgets are placed in.
            callbacks: the view→controller handlers the widgets invoke.
        """
        self.root = root
        self._cb = callbacks
        # Last analysis prosody, cached so the canvases can redraw on resize.
        self._last_prosody = None
        self.setup_styles()
        self.build_ui()
        # Second palette pass: the comboboxes created in build_ui made
        # ttkbootstrap build its default TCombobox style, which can override
        # the colors applied in setup_styles (see _apply_ttk_palette).
        self._apply_ttk_palette()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def setup_styles(self):
        # The first Style() instantiation applies the ttkbootstrap base theme
        # (replacing the old ttk theme_use("clam")); later calls return the
        # same singleton. All visible colors are then overridden from THEME.
        self.style = ttk.Style(theme=BOOTSTRAP_THEME)
        self._apply_ttk_palette()
        # The popdown list is a classic Tk Listbox, themed via the option DB.
        self.root.option_add("*TCombobox*Listbox.background", THEME["bg_panel"])
        self.root.option_add("*TCombobox*Listbox.foreground", THEME["text_accent"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", THEME["bg_accent_active"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", THEME["text_bright"])

    def _apply_ttk_palette(self):
        """Apply the THEME colors to the ttk widget styles.

        Called once before the widgets are built and once after: ttkbootstrap
        builds a widget class's default style lazily when the first widget of
        that class is created, which would overwrite configure() calls made
        beforehand. The second pass makes the THEME colors win.
        """
        self.style.configure("Vertical.TScrollbar",
                             gripcount=0,
                             background=THEME["bg_panel"],
                             troughcolor=THEME["bg_main"],
                             bordercolor=THEME["bg_main"],
                             arrowcolor=THEME["accent"])

        # Theme styling for the reference-speed combobox.
        self.style.configure("TCombobox",
                             fieldbackground=THEME["bg_accent"],
                             background=THEME["bg_accent"],
                             foreground=THEME["text_accent"],
                             arrowcolor=THEME["accent"],
                             bordercolor=THEME["border"],
                             relief="flat")
        self.style.map("TCombobox",
                       fieldbackground=[("readonly", THEME["bg_accent"]),
                                        ("disabled", THEME["bg_main"])],
                       foreground=[("readonly", THEME["text_accent"]),
                                   ("disabled", THEME["text_disabled"])],
                       arrowcolor=[("disabled", THEME["text_disabled"])])

    def _make_button(self, parent, text, command, width=None, padx=12):
        """Create a consistently styled themed button.

        ``width`` (in text characters) is optional; pass it to give a group of
        buttons a uniform width so they line up regardless of label length.
        ``padx`` is the internal horizontal padding (default 12); the bottom
        control row passes a smaller value so its four columns fit the 600px
        window without clipping the last button.
        """
        button = tk.Button(parent, text=text, command=command,
                           font=(FONT_FAMILY, 10, "bold"),
                           bg=THEME["bg_button"], fg=THEME["text_button"],
                           activebackground=THEME["bg_button_active"], activeforeground=THEME["text"],
                           bd=0, padx=padx, pady=6, cursor="hand2",
                           disabledforeground=THEME["text_disabled"])
        if width is not None:
            button.config(width=width)
        return button

    def _control_column(self, parent, caption):
        """A vertical control cell: the control(s) on top, an 8pt caption below.

        Used by the bottom control panel (v2c step 4) so each action - Reference,
        the mic, My recording, Next phrase - carries a one-line hint underneath.
        The caption is packed at the bottom up-front, so callers can simply pack
        their control widgets (default ``side=TOP``) into the returned frame
        without worrying about ordering. Columns are top-aligned by the caller
        (``anchor=N``) so short buttons and the taller mic each keep their caption
        directly beneath them, as in the mockup.
        """
        col = tk.Frame(parent, bg=THEME["bg_main"])
        # padx=6 on each side yields a ~12px gap between columns - tighter than
        # the mockup's 22px so all four columns fit the 600px window.
        col.pack(side=tk.LEFT, anchor=tk.N, padx=6)
        tk.Label(col, text=caption, font=(FONT_FAMILY, FONT_SIZE_CAPTION),
                 fg=THEME["text_muted"], bg=THEME["bg_main"]).pack(side=tk.BOTTOM, pady=(5, 0))
        return col

    def build_ui(self):
        # Window background (shows through wherever no widget covers it). The
        # view owns the window chrome so the controller need not know the palette.
        self.root.configure(bg=THEME["bg_main"])

        # 1. Header
        header_frame = tk.Frame(self.root, bg=THEME["bg_main"], height=60)
        header_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)

        tk.Label(header_frame, text="MIMORA • Pronunciation Trainer",
                 font=(FONT_FAMILY, 14, "bold"), fg=THEME["accent"], bg=THEME["bg_main"]).pack(side=tk.LEFT)

        # Settings gear at the right edge of the header; the language chip sits
        # just left of it. Opens the settings window (see main.py
        # on_settings_clicked).
        tk.Button(header_frame, text="⚙", command=self._cb.on_settings_clicked,
                  font=(FONT_FAMILY, 12), bg=THEME["bg_main"], fg=THEME["text_dim"],
                  activebackground=THEME["bg_accent_active"],
                  activeforeground=THEME["text_bright"],
                  bd=0, padx=6, pady=0, cursor="hand2").pack(side=tk.RIGHT)

        tk.Label(header_frame, text=config.TARGET_LANGUAGE,
                 font=(FONT_FAMILY, 9, "bold"), fg=THEME["text_dim"], bg=THEME["bg_panel"],
                 padx=10, pady=4, bd=0).pack(side=tk.RIGHT, padx=(0, 8))

        # 2. Status bar (absolute bottom)
        self.status_bar = tk.Frame(self.root, bg=THEME["bg_panel"], height=30)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = tk.Label(self.status_bar, text="Starting...",
                                     font=(FONT_FAMILY, 9), fg=THEME["ready"], bg=THEME["bg_panel"])
        self.status_label.pack(side=tk.LEFT, padx=15, pady=4)

        # Session tally (right side): the number of distinct phrases practiced
        # this run and the mean of their latest scores. The per-take score itself
        # moved to the hero card (see _set_score_row), so this line no longer
        # duplicates it. Driven by update_session_stats from the controller.
        self.stats_label = tk.Label(self.status_bar,
                                    text="Phrases: 0 · Avg: -",
                                    font=(FONT_FAMILY, 9), fg=THEME["text_dim"], bg=THEME["bg_panel"])
        self.stats_label.pack(side=tk.RIGHT, padx=15, pady=4)

        # 2a. Tip line - a single static hint sitting directly above the status
        # bar. Packed side=BOTTOM after the status bar so it lands just above it,
        # regardless of the TOP-packed content built later. It restates the two
        # least discoverable actions (space-to-record, Reference-replays), the
        # role the removed instruction line used to fill.
        self.tip_label = tk.Label(
            self.root,
            text='Tip: hold SPACE or click the mic to record. '
                 '"Reference ▶" replays the example.',
            font=(FONT_FAMILY, FONT_SIZE_CAPTION), fg=THEME["text_muted"],
            bg=THEME["bg_main"], anchor=tk.W)
        self.tip_label.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=(0, 4))

        # 3. Control panel (v2c step 4): one row holding every phrase-level
        # action, each in its own column with an 8pt caption beneath. Left to
        # right: Reference (+ slow-replay turtle), the mic, My recording, Next
        # phrase. The old single instruction line is gone - its guidance now
        # lives in these per-control captions, the Tip line and the status bar.
        #
        # The widgets are built here, but the frame is packed later - right below
        # the hero card (see the control_frame.pack() call after the hero
        # section) - so the on-screen order matches the mockup:
        # header -> practice text -> hero -> controls -> prosody -> history.
        control_frame = tk.Frame(self.root, bg=THEME["bg_main"])

        # Self-test enabled state. The self-test has no visible button; it stays
        # reachable via the 't' hotkey, gated by this flag (kept in sync by
        # _set_actions; see is_test_enabled).
        self._test_enabled = False

        # Equal-width buttons keep the row balanced regardless of label length;
        # 14 chars is the width of the longest label ("My recording ▶"). A tight
        # internal padding keeps all four columns within the 600px window.
        action_btn_width = 13
        action_btn_padx = 5

        # Centered row of control columns (each built by _control_column, which
        # top-aligns them and hangs the caption underneath).
        controls_row = tk.Frame(control_frame, bg=THEME["bg_main"])
        controls_row.pack()

        # -- Reference column: the replay button paired with the slow (0.7x)
        #    turtle. Both are gated together by _set_actions(reference=...); the
        #    exact slow speed lives in Settings. --
        ref_col = self._control_column(controls_row, "Listen to the example")
        ref_pair = tk.Frame(ref_col, bg=THEME["bg_main"])
        ref_pair.pack()
        self.ref_btn = self._make_button(
            ref_pair, "Reference ▶", self._cb.play_reference,
            width=action_btn_width, padx=action_btn_padx)
        self.ref_btn.pack(side=tk.LEFT, padx=(0, 4))
        self.ref_btn.config(state=tk.DISABLED)
        self.slow_btn = self._make_button(ref_pair, "🐢", self._cb.play_reference_slow,
                                          padx=action_btn_padx)
        self.slow_btn.pack(side=tk.LEFT)
        self.slow_btn.config(state=tk.DISABLED)

        # -- Mic column: the press-and-hold record button (canvas-drawn). --
        mic_col = self._control_column(controls_row, "Hold SPACE or click")
        self.btn_canvas = tk.Canvas(mic_col, width=100, height=100, bg=THEME["bg_main"],
                                    highlightthickness=0, cursor="hand2")
        self.btn_canvas.pack()
        self.btn_canvas.bind("<ButtonPress-1>", lambda e: self._cb.on_gui_btn_press())
        self.btn_canvas.bind("<ButtonRelease-1>", lambda e: self._cb.on_gui_btn_release())
        self.draw_mic_button("loading")

        # -- My recording column: always visible; enabled only once a take
        #    exists (_set_actions(user=...)). It used to sit in its own row above
        #    the feedback log; step 4 folds it into this single panel. --
        user_col = self._control_column(controls_row, "Record first to listen")
        self.user_btn = self._make_button(
            user_col, "My recording ▶", self._cb.play_user_recording,
            width=action_btn_width, padx=action_btn_padx)
        self.user_btn.pack()
        self.user_btn.config(state=tk.DISABLED)

        # -- Next phrase column: generate the next phrase. --
        gen_col = self._control_column(controls_row, "Skip to the next phrase")
        self.generate_btn = self._make_button(
            gen_col, "Next phrase ▶", self._cb.on_generate_phrase,
            width=action_btn_width, padx=action_btn_padx)
        self.generate_btn.pack()
        self.generate_btn.config(state=tk.DISABLED)

        # 4. Source text panel (editable)
        source_frame = tk.Frame(self.root, bg=THEME["bg_main"])
        source_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(0, 8))

        # Header row above the practice text: the collapse caption on the left
        # with the Paste/Clear affordances. Translation, phrase length, voice,
        # speed and the user name all moved to the Settings window (the header
        # here stays deliberately minimal - see the settings_window Field model).
        text_header = tk.Frame(source_frame, bg=THEME["bg_main"])
        text_header.pack(fill=tk.X)
        # The caption doubles as the collapse toggle for the text box: clicking
        # it hides/shows the editor (Paste/Clear go with it) to free vertical
        # space; the arrow prefix mirrors the state (see toggle_practice_text).
        self.practice_collapsed = tk.BooleanVar(value=config.PRACTICE_TEXT_COLLAPSED)
        self._practice_caption = tk.Button(
            text_header, text="▾ Practice text:",
            command=self._on_practice_caption_clicked,
            font=(FONT_FAMILY, 9, "bold"), fg=THEME["text_dim"], bg=THEME["bg_main"],
            activebackground=THEME["bg_main"], activeforeground=THEME["text"],
            bd=0, padx=0, pady=0, cursor="hand2")
        self._practice_caption.pack(side=tk.LEFT)

        # Quick-edit affordances next to the caption. Their real job is
        # discoverability: a control visible even while the field still shows the
        # pre-filled welcome text is the clearest signal that the box is editable
        # (reviewers kept reading the box as static help text). Both act on
        # self.source_text directly - clipboard paste and clearing are view-local
        # widget operations with no application logic, so they belong in the view
        # (which owns the widget); the controller still reads via get_practice_text.
        self._paste_btn = tk.Button(
            text_header, text="Paste", command=self._paste_practice_text,
            font=(FONT_FAMILY, 9), bg=THEME["bg_accent"], fg=THEME["text_accent"],
            activebackground=THEME["bg_accent_active"], activeforeground=THEME["text_bright"],
            bd=0, width=10, padx=8, pady=1, cursor="hand2")
        self._paste_btn.pack(side=tk.LEFT, padx=(10, 0))
        self._clear_btn = tk.Button(
            text_header, text="Clear text", command=self._clear_practice_text,
            font=(FONT_FAMILY, 9), bg=THEME["bg_accent"], fg=THEME["text_accent"],
            activebackground=THEME["bg_accent_active"], activeforeground=THEME["text_bright"],
            bd=0, width=10, padx=8, pady=1, cursor="hand2")
        self._clear_btn.pack(side=tk.LEFT, padx=(6, 0))

        # Collapsed-state preview: when the editor is hidden, this shows the first
        # part of the current text (grey, italic, quoted) so the panel still hints
        # at its content without taking the editor's vertical space - the same
        # affordance the mockup's collapsed <summary> line provides. Only visible
        # while collapsed; toggle_practice_text swaps it against the editor and the
        # Paste/Clear buttons, and refreshes its text on each collapse.
        self._practice_preview = tk.Label(
            text_header, text="", font=(FONT_FAMILY, 9, "italic"),
            fg=THEME["text_muted"], bg=THEME["bg_main"], anchor=tk.W)

        self.source_text = scrolledtext.ScrolledText(
            source_frame, bg=THEME["bg_panel"], fg=THEME["text"], insertbackground=THEME["text_bright"],
            font=(FONT_FAMILY, 10), wrap=tk.WORD, bd=0, height=7,
            highlightthickness=1, highlightbackground=THEME["border"], highlightcolor=THEME["accent"],
            padx=10, pady=8)
        self.source_text.pack(fill=tk.X, pady=4)

        # Translation language, phrase length, voice and reference speed used to
        # sit in a selector row here; they now live in the Settings window and
        # are read straight from config (the live source of truth), so the main
        # window keeps only the practice text and the pronunciation loop.

        # The editor is packed above by default; hide it now if the persisted
        # state says collapsed (same late-apply idiom as the prosody toggles).
        self.toggle_practice_text()

        # 5. Hero card - THE screen object ("one screen - one task"): the current
        # phrase with its translation, and the result row beneath them. Drawn on
        # bg_card, the lightest of the three surfaces (bg_main < bg_panel <
        # bg_card), so the card visually leads the whole window.
        self.hero_frame = tk.Frame(self.root, bg=THEME["bg_card"],
                                   highlightthickness=1, highlightbackground=THEME["border"])
        self.hero_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(8, 10))

        # Phrase: a read-only tk.Text (not a Label) so individual words can carry
        # tags - the "miss" underline for problem words and its click-to-hear
        # binding. Disabled + takefocus=0 keeps it from swallowing the spacebar
        # record toggle (main.py gates hotkeys on a focused Text widget).
        self.phrase_text = tk.Text(
            self.hero_frame, font=(FONT_FAMILY, FONT_SIZE_PHRASE, "bold"),
            fg=THEME["phrase"], bg=THEME["bg_card"], wrap=tk.WORD, bd=0,
            highlightthickness=0, height=1, cursor="arrow", takefocus=0,
            padx=16, pady=0)
        self.phrase_text.pack(fill=tk.X, padx=10, pady=(18, 0))
        # All content is inserted with the "center" tag: tk.Text has no
        # widget-level justify, only per-tag.
        self.phrase_text.tag_configure("center", justify=tk.CENTER)
        # "miss" marks a mispronounced word: a colored underline (kept subtle -
        # the word itself stays in the phrase color). underlinefg needs
        # Tk >= 8.6.6; older Tks fall back to painting the word itself red,
        # which keeps the signal without the separate underline color.
        try:
            self.phrase_text.tag_configure("miss", underline=True,
                                           underlinefg=THEME["bad"])
        except tk.TclError:
            self.phrase_text.tag_configure("miss", underline=True,
                                           foreground=THEME["bad"])
        self.phrase_text.tag_bind("miss", "<Button-1>", self._on_miss_word_clicked)
        # Hover affordance: hand cursor + highlight while over any miss word.
        # Tag options are per-tag (not per-range), so hovering one miss word
        # highlights them all - acceptable for the few misses a phrase has.
        self.phrase_text.tag_bind(
            "miss", "<Enter>", lambda e: (self.phrase_text.configure(cursor="hand2"),
                                          self.phrase_text.tag_configure(
                                              "miss", background=THEME["bg_accent_active"])))
        self.phrase_text.tag_bind(
            "miss", "<Leave>", lambda e: (self.phrase_text.configure(cursor="arrow"),
                                          self.phrase_text.tag_configure(
                                              "miss", background="")))
        # The Text is disabled between set_phrase calls; its height follows the
        # wrapped content (recomputed on every set_phrase and on resize).
        self.phrase_text.configure(state=tk.DISABLED)
        self.phrase_text.bind("<Configure>", lambda e: self._fit_phrase_height())
        # Right-click the phrase to copy it (whole phrase - a tk.Text has no
        # single "text" option like a Label).
        self._bind_copy_menu(self.phrase_text,
                             getter=lambda: self.phrase_text.get("1.0", "end-1c"))
        # Same "-" placeholder the old Label started with.
        self.set_phrase("-")

        # Translation inside the same card, right under the phrase: just dimmer,
        # smaller text - no separator needed. Packed/unpacked by
        # refresh_translation_ui() when a translation language is (de)selected;
        # "-" stands in until a translation arrives with the next phrase.
        self.translation_label = tk.Label(
            self.hero_frame, text="-", font=(FONT_FAMILY, FONT_SIZE_TRANSLATION),
            fg=THEME["text_dim"], bg=THEME["bg_card"], wraplength=560,
            justify=tk.CENTER)
        # Wrap the translation to the label's real width so a long translation
        # wraps onto more lines instead of being clipped on both sides. A fixed
        # wraplength did not track the card width, so once the card was narrower
        # than that value the centred line overflowed and got cut off.
        self.translation_label.bind("<Configure>", self._wrap_translation)
        # Right-click the translation to copy it (independently of the phrase).
        self._bind_copy_menu(self.translation_label)
        self.refresh_translation_ui()

        # Bottom padding of the phrase block, then a 1px divider above the
        # score row (the card's only internal separator, as in the mockup).
        tk.Frame(self.hero_frame, height=14, bg=THEME["bg_card"]).pack(fill=tk.X)
        tk.Frame(self.hero_frame, height=1, bg=THEME["border"]).pack(fill=tk.X)

        # Score row: [SCORE column] [WORK ON badges] [face]. Filled by
        # show_feedback; _reset_score_row shows the empty state ("--" and
        # "record to get a score") until the first take.
        score_row = tk.Frame(self.hero_frame, bg=THEME["bg_card"])
        score_row.pack(fill=tk.X, padx=18, pady=(10, 12))

        score_col = tk.Frame(score_row, bg=THEME["bg_card"])
        score_col.pack(side=tk.LEFT)
        tk.Label(score_col, text="SCORE", font=(FONT_FAMILY, FONT_SIZE_CAPTION, "bold"),
                 fg=THEME["text_dim"], bg=THEME["bg_card"]).pack()
        self.score_num_label = tk.Label(
            score_col, text="--", font=(FONT_FAMILY, FONT_SIZE_SCORE, "bold"),
            fg=THEME["text_dim"], bg=THEME["bg_card"])
        self.score_num_label.pack()
        self.score_verdict_label = tk.Label(
            score_col, text="record to get a score",
            font=(FONT_FAMILY, FONT_SIZE_CAPTION),
            fg=THEME["text_dim"], bg=THEME["bg_card"])
        self.score_verdict_label.pack()

        # WORK ON: caption + a row of flat phoneme badges (top problem sounds).
        # The badges are rebuilt per result (_set_badges); clicking one will
        # speak an example word - wired in the sounds-feedback stage.
        workon_col = tk.Frame(score_row, bg=THEME["bg_card"])
        workon_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(20, 8))
        self.workon_caption = tk.Label(
            workon_col, text="", font=(FONT_FAMILY, FONT_SIZE_CAPTION, "bold"),
            fg=THEME["warn"], bg=THEME["bg_card"])
        self.workon_caption.pack(anchor=tk.W)
        self.badges_frame = tk.Frame(workon_col, bg=THEME["bg_card"])
        self.badges_frame.pack(anchor=tk.W, pady=(4, 0))
        # Hint under the badges (mockup): how to use the two interactive feedback
        # affordances. Packed only while a scored take is on the card (see
        # _set_hint), so the empty/unscored states stay quiet.
        self.hint_label = tk.Label(
            workon_col,
            text="Click an underlined word to hear it slowly; "
                 "click a sound for an example.",
            font=(FONT_FAMILY, FONT_SIZE_CAPTION), fg=THEME["text_muted"],
            bg=THEME["bg_card"], justify=tk.LEFT, wraplength=320)

        # The face lives in the score row now (verdict indicator + talking
        # mouth), not in the prosody block. Same single FaceWidget instance the
        # controller drives via face_play_levels/face_rest; ~78px as mocked up.
        self.face = FaceWidget(score_row, size=78, bg=THEME["bg_card"],
                               face_color=THEME["face"], face_outline=THEME["border"],
                               eye_color=THEME["eyes"], mouth_color=THEME["mouth"])
        self.face.set_expression(":)")  # waiting state
        # The face has no on-main control anymore: its visibility is a settings
        # value ("Show articulation face"), mirrored into this var and applied by
        # toggle_face(). Not packed yet - toggle_face() below packs it if enabled.
        self.show_face = tk.BooleanVar(value=config.SHOW_FACE)

        # 5a. Control panel, packed here so it sits directly under the hero card
        # (mockup order). The frame and its buttons were built in section 3; only
        # the placement was deferred to this point in the packing order.
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(0, 8))

        # 5b. Prosody panel - pitch (F0) and energy sparklines, you vs reference,
        # folded under one collapse header (v2c step 6). Both charts live together
        # now (the per-chart checkboxes and the on-main "Face" toggle are gone);
        # one flag (show_prosody) shows/hides the whole body and gates the
        # (expensive) prosody computation. Default collapsed for a calmer view.
        prosody_frame = tk.Frame(self.root, bg=THEME["bg_main"])
        prosody_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(0, 8))

        # Header caption doubles as the collapse toggle (same idiom as the
        # practice-text caption): clicking it expands/collapses the body; the
        # arrow prefix mirrors the state.
        self.show_prosody = tk.BooleanVar(value=config.SHOW_PROSODY)
        self._prosody_caption = tk.Button(
            prosody_frame, text="▾ Intonation & stress",
            command=self._on_prosody_caption_clicked,
            font=(FONT_FAMILY, 9, "bold"), fg=THEME["accent"], bg=THEME["bg_main"],
            activebackground=THEME["bg_main"], activeforeground=THEME["accent"],
            bd=0, padx=0, pady=0, cursor="hand2")
        self._prosody_caption.pack(anchor=tk.W)

        # Body: everything hidden while collapsed. toggle_prosody() packs/forgets
        # it as a whole, so the individual children are packed once here.
        self.prosody_body = tk.Frame(prosody_frame, bg=THEME["bg_main"])

        # Legend lives inside the body (only meaningful when charts are visible),
        # not in the always-shown header.
        legend_row = tk.Frame(self.prosody_body, bg=THEME["bg_main"])
        legend_row.pack(fill=tk.X, pady=(4, 2))
        tk.Label(legend_row, text="● you", font=(FONT_FAMILY, 8),
                 fg=THEME["info"], bg=THEME["bg_main"]).pack(side=tk.LEFT)
        tk.Label(legend_row, text="● reference", font=(FONT_FAMILY, 8),
                 fg=THEME["reference"], bg=THEME["bg_main"]).pack(side=tk.LEFT, padx=(8, 0))

        # Pitch chart: plain title label above its canvas (no longer a checkbox).
        tk.Label(self.prosody_body, text="Pitch - intonation (semitones vs your median)",
                 font=(FONT_FAMILY, 8), fg=THEME["text_dim"], bg=THEME["bg_main"]).pack(anchor=tk.W)
        self.f0_canvas = tk.Canvas(self.prosody_body, height=46, bg=THEME["bg_panel"],
                                   highlightthickness=1, highlightbackground=THEME["border"])
        self.f0_canvas.pack(fill=tk.X, pady=(0, 4))

        # Energy chart: same, below the pitch chart.
        tk.Label(self.prosody_body, text="Energy - stress pattern",
                 font=(FONT_FAMILY, 8), fg=THEME["text_dim"], bg=THEME["bg_main"]).pack(anchor=tk.W)
        self.en_canvas = tk.Canvas(self.prosody_body, height=46, bg=THEME["bg_panel"],
                                   highlightthickness=1, highlightbackground=THEME["border"])
        self.en_canvas.pack(fill=tk.X)

        # Reading hint: horizontal axis is time (stretched to equal width for both),
        # so the goal is matching the *shape* of the reference, not exact overlap.
        tk.Label(self.prosody_body,
                  text="Time runs (stretched to equal width). Aim to match the reference shape.",
                 font=(FONT_FAMILY, 8), fg=THEME["text_dim"], bg=THEME["bg_main"],
                 wraplength=540, justify=tk.LEFT).pack(anchor=tk.W, pady=(3, 0))

        # fill=X canvases change width on resize, so redraw from the cached prosody.
        self.f0_canvas.bind("<Configure>", lambda e: self._redraw_prosody())
        self.en_canvas.bind("<Configure>", lambda e: self._redraw_prosody())

        # Late-apply the persisted state: show the body only if expanded, and
        # pack the face (built unpacked in the hero card's score row) if enabled.
        self.toggle_prosody()
        self.toggle_face()

        # 6. My recording, Reference and Next phrase now all live together in the
        # single bottom control panel (control_frame, section 3, v2c step 4);
        # there is no separate action row here anymore. The diagnostic self-test
        # still has no visible button - it stays reachable via the 't' hotkey,
        # gated by self._test_enabled (see is_test_enabled / _set_actions).

        # 7. Attempt history (fills remaining space). Packed AFTER the control
        # panel so its expand=True only consumes space left over above the button
        # row. A scrollable Canvas hosts one inner frame of per-attempt rows
        # (mirrors the SettingsWindow scroll pattern): the inner frame is kept as
        # wide as the canvas and the scrollregion as tall as the content. Rows are
        # rebuilt by render_history from the controller-owned history list.
        history_outer = tk.Frame(self.root, bg=THEME["bg_main"])
        history_outer.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=(8, 16))

        self.history_canvas = tk.Canvas(
            history_outer, bg=THEME["bg_panel"], highlightthickness=1,
            highlightbackground=THEME["border"], bd=0)
        history_scroll = tk.Scrollbar(history_outer, orient=tk.VERTICAL,
                                      command=self.history_canvas.yview)
        self.history_canvas.configure(yscrollcommand=history_scroll.set)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.history_frame = tk.Frame(self.history_canvas, bg=THEME["bg_panel"])
        self._history_window = self.history_canvas.create_window(
            (0, 0), window=self.history_frame, anchor="nw")
        self.history_frame.bind("<Configure>", lambda e: self.history_canvas.configure(
            scrollregion=self.history_canvas.bbox("all")))
        self.history_canvas.bind("<Configure>", lambda e: self.history_canvas.itemconfigure(
            self._history_window, width=e.width))
        # Wheel scrolling is bound directly on the canvas and (in render_history)
        # on every row widget, rather than via a global bind_all toggled on
        # Enter/Leave: the latter flickers off whenever the pointer crosses into a
        # child row, so the wheel would stop working over the rows themselves.
        self.history_canvas.bind("<MouseWheel>", self._on_history_mousewheel)
        self.history_frame.bind("<MouseWheel>", self._on_history_mousewheel)

        # Empty-state hint shown until the first attempt lands.
        self._history_placeholder = tk.Label(
            self.history_frame, text="Your attempts will appear here.",
            font=(FONT_FAMILY, 10, "italic"), fg=THEME["text_muted"],
            bg=THEME["bg_panel"], anchor="w")
        self._history_placeholder.pack(fill=tk.X, padx=12, pady=12)

    # Mic button geometry, shared by draw_mic_button and set_record_level.
    _MIC_CENTER = 50
    _MIC_R_OUTER = 42
    _MIC_R_INNER = 34
    # Live-level mapping for the recording indicator: the fixed outer ring stays
    # at full radius, and a solid red disc inside it grows with the input level,
    # from _MIC_LEVEL_MIN_R up to the inner radius (just short of the ring). Input
    # RMS at or above _MIC_LEVEL_FULL_RMS fills it to the inner radius; it never
    # shrinks below the min radius, so the mic stays visibly "open" in silence.
    _MIC_LEVEL_FULL_RMS = 0.08
    _MIC_LEVEL_MIN_R = 10

    def draw_mic_button(self, state):
        self.btn_canvas.delete("all")
        cx = cy = self._MIC_CENTER
        r_outer, r_inner = self._MIC_R_OUTER, self._MIC_R_INNER
        palette = {
            "loading":   (THEME["mic_loading_bg"], THEME["mic_loading_outline"], "⌛"),
            "idle":      (THEME["bg_accent"], THEME["accent"], "🎤"),
            "recording": (THEME["mic_recording_bg"], THEME["bad"], "🔴"),
            "processing":(THEME["mic_processing_bg"], THEME["warn"], "⚡"),
            "speaking":  (THEME["mic_speaking_bg"], THEME["good"], "🔊"),
        }
        bg_color, outline_color, emoji = palette.get(state, (THEME["mic_loading_bg"], THEME["mic_loading_outline"], "🎤"))
        self.btn_canvas.create_oval(cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer,
                                    fill="", outline=outline_color, width=3)
        self.btn_canvas.create_oval(cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner,
                                    fill=bg_color, outline="")
        self.btn_canvas.create_text(cx, cy, text=emoji, font=(FONT_FAMILY, 20), fill=THEME["text_bright"])

    def set_record_level(self, level: float):
        """Redraw the recording button with a level-driven red fill.

        Recording now stops on its own after silence, so the static red glyph
        no longer tells the user the mic is actually hearing them. The outer ring
        is drawn exactly as the recording state (full radius, red); inside it a
        solid red disc grows with the live input level (``level`` is RMS in 0..1
        from the recorder): quiet -> a small disc (auto-stop is near), louder ->
        it fills up to the inner radius, just short of the ring. Leaving the
        recording state is handled by the next draw_mic_button call
        (enter_analyzing / idle), which repaints the button from scratch.
        """
        cx = cy = self._MIC_CENTER
        r_outer, r_inner = self._MIC_R_OUTER, self._MIC_R_INNER
        # Map RMS to a 0..1 fraction, then to the disc radius; clamp so a loud
        # spike cannot grow the fill past the inner radius (into the ring).
        frac = max(0.0, min(1.0, level / self._MIC_LEVEL_FULL_RMS))
        r_level = self._MIC_LEVEL_MIN_R + frac * (r_inner - self._MIC_LEVEL_MIN_R)

        self.btn_canvas.delete("all")
        # Outer ring: identical to draw_mic_button("recording") - fixed, red.
        self.btn_canvas.create_oval(cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer,
                                    fill="", outline=THEME["bad"], width=3)
        # Dark inner track, so the red fill is visible shrinking against it.
        self.btn_canvas.create_oval(cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner,
                                    fill=THEME["mic_recording_bg"], outline="")
        # The pulsing level indicator: a solid red disc, min radius -> inner
        # radius. No glyph on top - the disc itself is the indicator, and an
        # overlaid emoji's text bounding box does not share the disc's center.
        self.btn_canvas.create_oval(cx - r_level, cy - r_level, cx + r_level, cy + r_level,
                                    fill=THEME["bad"], outline="")

    # ------------------------------------------------------------------
    # Read accessors (controller queries widget values through these)
    # ------------------------------------------------------------------
    def get_practice_text(self) -> str:
        """Return the editable practice text (without trailing whitespace)."""
        return self.source_text.get("1.0", tk.END).strip()

    def set_practice_text(self, text: str):
        """Replace the practice-text panel contents."""
        self.source_text.delete("1.0", tk.END)
        self.source_text.insert("1.0", text)
        # Keep the collapsed preview in sync when text is loaded while hidden.
        if self.practice_collapsed.get():
            self._update_practice_preview()

    def _paste_practice_text(self):
        """Insert clipboard text into the practice field (the 'Paste' button).

        Standard paste semantics: any current selection is replaced, then the
        clipboard text is inserted at the caret. The field is focused afterwards
        so the user can keep editing. A missing or non-text clipboard is a no-op.
        """
        try:
            text = self.root.clipboard_get()
        except tk.TclError:
            return  # clipboard empty or holds non-text data
        if not text:
            return
        if self.source_text.tag_ranges(tk.SEL):
            self.source_text.delete(tk.SEL_FIRST, tk.SEL_LAST)
        self.source_text.insert(tk.INSERT, text)
        self.source_text.focus_set()

    def _clear_practice_text(self):
        """Empty the practice field and focus it, ready for the user's own text."""
        self.source_text.delete("1.0", tk.END)
        self.source_text.focus_set()

    def _on_practice_caption_clicked(self):
        """Flip the collapse flag and route it through the controller.

        Unlike the Face Checkbutton, the caption Button has no variable of its
        own, so the view flips the flag here; the controller then applies and
        persists it exactly like the other visibility toggles.
        """
        self.practice_collapsed.set(not self.practice_collapsed.get())
        self._cb.on_practice_collapsed_toggled()

    def toggle_practice_text(self):
        """Show/hide the practice-text editor to match the collapse flag."""
        # Return focus to the window so the spacebar record toggle keeps working.
        self.root.focus_set()
        collapsed = self.practice_collapsed.get()
        self._practice_caption.config(
            text="▸ Practice text:" if collapsed else "▾ Practice text:")
        # ScrolledText delegates pack/pack_forget to its outer .frame but NOT
        # winfo_manager: asking the Text itself always answers "pack" (it is
        # permanently packed inside that frame), so probe the frame instead.
        shown = self.source_text.frame.winfo_manager() == "pack"
        if collapsed and shown:
            self.source_text.pack_forget()
            self._paste_btn.pack_forget()
            self._clear_btn.pack_forget()
            # Swap the editor row for the one-line text preview next to the caption.
            self._update_practice_preview()
            self._practice_preview.pack(side=tk.LEFT, padx=(10, 0))
        elif not collapsed and not shown:
            # Restore the build order: the editor is the last child of
            # source_frame, and the buttons pack after the caption (LEFT packing
            # preserves their order).
            self._practice_preview.pack_forget()
            self.source_text.pack(fill=tk.X, pady=4)
            self._paste_btn.pack(side=tk.LEFT, padx=(10, 0))
            self._clear_btn.pack(side=tk.LEFT, padx=(6, 0))

    def _update_practice_preview(self):
        """Refresh the collapsed-state preview from the current editor content.

        Shows the first ~60 characters of the practice text on a single line
        (newlines and runs of whitespace collapsed to single spaces), quoted, with
        an ellipsis when truncated. Empty text falls back to a neutral placeholder.
        """
        raw = self.source_text.get("1.0", "end-1c")
        collapsed_ws = " ".join(raw.split())
        if not collapsed_ws:
            self._practice_preview.config(text="(empty)")
            return
        limit = 60
        shown = collapsed_ws[:limit].rstrip()
        if len(collapsed_ws) > limit:
            shown += "..."
        self._practice_preview.config(text=f'"{shown}"')

    def set_practice_collapsed(self, flag: bool):
        self.practice_collapsed.set(bool(flag))

    def get_practice_collapsed(self) -> bool:
        return bool(self.practice_collapsed.get())

    # ------------------------------------------------------------------
    # Read accessors for settings moved to the Settings window. config is the
    # live source of truth (the controller updates it when a setting changes),
    # so these read config directly instead of a main-window widget var.
    # ------------------------------------------------------------------
    def get_user_name(self) -> str:
        return config.USER_NAME.strip()

    def get_voice(self) -> str:
        return config.KOKORO_VOICE

    def get_length_label(self) -> str:
        return LENGTH_FEW_WORDS if config.PHRASE_LENGTH == "fragment" else LENGTH_FULL

    def get_translation_language(self) -> str:
        """Return the current translation language label ('' when off)."""
        return config.TRANSLATION_LANGUAGE

    def set_translation(self, text: str):
        """Set the translation panel text, falling back to '-' when empty.

        '-' marks "language is on, but no translation yet" (e.g. right after the
        language was switched - the translation arrives with the next phrase).
        """
        self.translation_label.config(text=text.strip() if text and text.strip() else "-")

    def _wrap_translation(self, event):
        """Keep the translation wraplength equal to the label's current width.

        Guarded against the feedback loop where changing wraplength alters the
        label's height, which fires <Configure> again: only the width matters
        here and it does not change on a height-only event, so re-setting the
        same value is skipped.
        """
        if int(self.translation_label.cget("wraplength")) != event.width:
            self.translation_label.config(wraplength=max(1, event.width))

    # ------------------------------------------------------------------
    # Hero card: phrase text and score row
    # ------------------------------------------------------------------
    def set_phrase(self, text: str):
        """Replace the hero phrase (centered, no tags) and refit its height.

        Every phrase starts untagged; the "miss" underlines are applied per
        analysis result (the words-feedback stage maps them from the engine's
        per-word breakdown).
        """
        self.phrase_text.configure(state=tk.NORMAL)
        self.phrase_text.delete("1.0", tk.END)
        self.phrase_text.insert("1.0", text or "-", ("center",))
        self.phrase_text.configure(state=tk.DISABLED)
        self._fit_phrase_height()

    def _fit_phrase_height(self):
        """Size the phrase Text (in lines) to its wrapped content.

        A tk.Text does not auto-grow like a Label: its height is a fixed line
        count, so it is recomputed after every set_phrase and on width changes
        (the <Configure> binding). No-op before the widget is laid out.
        """
        if self.phrase_text.winfo_width() <= 1:
            return
        # "update" makes Tk compute the line metrics first, so the display-line
        # count is accurate right after an insert; without it the answer can
        # lag one layout pass behind.
        lines = self.phrase_text.count("1.0", "end", "update", "displaylines")
        # Tkinter's Text.count may return the value tuple-wrapped.
        if isinstance(lines, tuple):
            lines = lines[0]
        self.phrase_text.configure(height=max(1, int(lines or 1)))

    def _mark_miss_words(self, reference_words: list) -> int:
        """Re-render the hero phrase, underlining the mispronounced words.

        Rebuilds the phrase from the engine's per-word breakdown so the "miss"
        tag lands on exactly the target tokens the engine judged: a word is a
        miss when its three-level ``level`` is "bad" (the phoneme engine) or,
        without a level, when ``correct`` is False (the acoustic engine) - "ok"
        words stay unmarked. With no breakdown the phrase is left as set (no
        underlines). The Text is disabled again afterwards so it never steals the
        spacebar record toggle. Returns the number of underlined words, so the
        caller can decide whether the "click an underlined word" hint applies.
        """
        if not reference_words:
            return 0
        misses = 0
        self.phrase_text.configure(state=tk.NORMAL)
        self.phrase_text.delete("1.0", tk.END)
        last = len(reference_words) - 1
        for i, word in enumerate(reference_words):
            token = word.get("word", "")
            level = word.get("level")
            if level in ("good", "ok", "bad"):
                is_miss = level == "bad"
            else:
                is_miss = not word.get("correct", True)
            if is_miss:
                misses += 1
            self.phrase_text.insert(tk.END, token,
                                    ("center", "miss") if is_miss else ("center",))
            if i < last:
                self.phrase_text.insert(tk.END, " ", ("center",))
        self.phrase_text.configure(state=tk.DISABLED)
        self._fit_phrase_height()
        return misses

    def _on_miss_word_clicked(self, event):
        """Speak the clicked underlined word slowly (via the controller).

        The word is taken from the exact "miss" tag range under the click, so
        in-word punctuation (apostrophes etc.) cannot split it the way a plain
        wordstart/wordend probe would.
        """
        index = self.phrase_text.index(f"@{event.x},{event.y}")
        tag_range = self.phrase_text.tag_prevrange("miss", f"{index}+1c")
        if not tag_range:
            return
        word = self.phrase_text.get(*tag_range).strip(".,!?;:\"'()").strip()
        # Return focus to the window so the spacebar record toggle keeps
        # working (main.py's global click handler skips Text widgets).
        self.root.focus_set()
        if word:
            self._cb.on_word_clicked(word)

    def _set_score_row(self, number: str, verdict: str, color: str):
        """Fill the SCORE column: the big number and its verdict, both in the
        already-resolved verdict color (never green for a "needs work")."""
        self.score_num_label.configure(text=number, fg=color)
        self.score_verdict_label.configure(text=verdict, fg=color)

    def _set_badges(self, phonemes: list[str]):
        """Rebuild the WORK ON badges from the top problem phonemes.

        An empty list clears the row and hides the caption (the empty state and
        clean takes show no badges). Each badge whose phoneme has a known example
        word gets a hover tooltip ("as in 'put'") and speaks that example at
        normal speed on click (via ``on_sound_example``). An unknown phoneme
        stays a plain, non-playing badge.
        """
        for child in self.badges_frame.winfo_children():
            child.destroy()
        self.workon_caption.configure(text="WORK ON" if phonemes else "")
        for phoneme in phonemes:
            example = example_for(phoneme)
            badge = tk.Button(self.badges_frame, text=f"/{phoneme}/",
                              font=(FONT_FAMILY, FONT_SIZE_BODY),
                              bg=THEME["bg_accent"], fg=THEME["text_accent"],
                              activebackground=THEME["bg_button"],
                              activeforeground=THEME["text_button"],
                              bd=0, padx=10, pady=2, cursor="hand2",
                              highlightthickness=1,
                              highlightbackground=THEME["accent"])
            badge.pack(side=tk.LEFT, padx=(0, 6))
            if example:
                # Bind the current example per button via a default argument so
                # every badge keeps its own word (not the loop's last one).
                badge.configure(
                    command=lambda w=example: self._cb.on_sound_example(w))
                _Tooltip(badge, f"as in '{example}'")
            else:
                badge.configure(cursor="arrow")   # nothing to play or name

    def _set_hint(self, show: bool):
        """Show or hide the interactive-feedback hint under the badges.

        Shown only while a scored take is on the card, where both affordances it
        describes (underlined words, sound badges) actually exist.
        """
        if show:
            self.hint_label.pack(anchor=tk.W, pady=(6, 0))
        else:
            self.hint_label.pack_forget()

    def _reset_score_row(self):
        """Empty state: no take scored yet (or a new recording just started)."""
        self.score_num_label.configure(text="--", fg=THEME["text_dim"])
        self.score_verdict_label.configure(text="record to get a score",
                                           fg=THEME["text_dim"])
        self._set_badges([])
        self._set_hint(False)
        # Drop any underlines from the previous take (the phrase itself stays;
        # a new phrase is replaced wholesale by set_phrase).
        self.phrase_text.tag_remove("miss", "1.0", tk.END)

    # ------------------------------------------------------------------
    # Copy-to-clipboard (right-click menu on the phrase / translation)
    # ------------------------------------------------------------------
    def _bind_copy_menu(self, widget, getter: Optional[Callable[[], str]] = None):
        """Attach a right-click 'Copy' menu that copies the widget's text.

        ``getter`` returns the text to copy; it defaults to the widget's own
        "text" option (a Label). The phrase tk.Text passes an explicit getter
        (its whole content), since a Text has no such option. Phrase and
        translation get their own binding, so each is copied independently.
        """
        if getter is None:
            getter = lambda: widget.cget("text")
        widget.bind("<Button-3>", lambda event: self._show_copy_menu(event, getter))

    def _show_copy_menu(self, event, getter: Callable[[], str]):
        """Pop up a one-item 'Copy' menu for the clicked widget, if it has text."""
        text = getter().strip()
        # Nothing useful to copy from an empty card or the "-" placeholder.
        if not text or text == "-":
            return
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Copy", command=lambda: self._copy_text(text))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _copy_text(self, text: str):
        """Replace the clipboard contents with ``text``."""
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def refresh_translation_ui(self):
        """Show or hide the translation panel to match the current setting.

        Shows the panel whenever a language is selected - in both "Full phrase"
        and "Few words" modes (fragments are translated too). Safe to call
        repeatedly - it reconciles to config.TRANSLATION_LANGUAGE without side
        effects.
        """
        show = bool(config.TRANSLATION_LANGUAGE)
        packed = self.translation_label.winfo_manager() == "pack"
        if show and not packed:
            # Inside the hero card, right under the phrase (no separator - the
            # dimmer, smaller type is the whole distinction).
            self.translation_label.pack(fill=tk.X, padx=26, pady=(8, 0),
                                        after=self.phrase_text)
        elif not show and packed:
            self.translation_label.pack_forget()

    # ------------------------------------------------------------------
    # Write accessors for the checkbox settings the main window still owns
    # (the prosody/face toggles); the controller mirrors settings-window
    # changes into them. Pure value updates, no callbacks fire.
    # ------------------------------------------------------------------
    def set_show_face(self, flag: bool):
        self.show_face.set(bool(flag))

    def set_show_prosody(self, flag: bool):
        self.show_prosody.set(bool(flag))

    def is_reference_enabled(self) -> bool:
        """True when the Reference replay button is currently clickable."""
        return str(self.ref_btn["state"]) == str(tk.NORMAL)

    def is_generate_enabled(self) -> bool:
        """True when the New phrase button is currently clickable."""
        return str(self.generate_btn["state"]) == str(tk.NORMAL)

    def is_test_enabled(self) -> bool:
        """True when the self-test is currently allowed. There is no visible
        button; the 't' hotkey reads this flag (kept in sync by _set_actions)."""
        return self._test_enabled

    def is_user_enabled(self) -> bool:
        """True when the My phrase replay button is currently clickable."""
        return str(self.user_btn["state"]) == str(tk.NORMAL)

    def get_show_prosody(self) -> bool:
        return bool(self.show_prosody.get())

    def get_show_face(self) -> bool:
        return bool(self.show_face.get())

    # ------------------------------------------------------------------
    # Intent methods: named UI states the controller transitions into.
    # Each owns the buttons, mic glyph, status and instruction copy for one
    # state, so the controller only names the state. All run on the main
    # thread (the controller schedules them via root.after from workers).
    # ------------------------------------------------------------------
    def _set_actions(self, *, generate=None, reference=None, user=None, test=None):
        """Enable/disable action buttons; ``None`` leaves a button unchanged."""
        def state(flag):
            return tk.NORMAL if flag else tk.DISABLED
        if generate is not None:
            self.generate_btn.config(state=state(generate))
        if reference is not None:
            self.ref_btn.config(state=state(reference))
            self.slow_btn.config(state=state(reference))
        if user is not None:
            self.user_btn.config(state=state(user))
        if test is not None:
            self._test_enabled = bool(test)

    def enter_app_ready(self):
        """Models loaded, no phrase yet: only New phrase is available."""
        self.draw_mic_button("idle")
        self.update_status("Ready", THEME["ready"])
        self.update_instruction("Edit the text, then click 'New phrase' to begin.")
        self._set_actions(generate=True)

    def enter_generating(self):
        """LLM is producing a phrase: every action is locked out."""
        self._set_actions(generate=False, reference=False, user=False, test=False)
        self.draw_mic_button("processing")
        self.update_status("Generating phrase (LLM)...", THEME["info"])
        self.update_instruction("Generating a new phrase...")

    def enter_reference_playing(self, phrase: str, translation: str = ""):
        """A fresh phrase is shown and its reference is being played.

        ``translation`` is the phrase rendered in the selected translation
        language (empty when translation is off or unavailable); it is shown in
        the panel under the phrase card when a language is active.
        """
        # Clear the previous phrase's result first: its score, verdict, WORK ON
        # badges, hint, word underlines, prosody charts and face must not linger
        # under the new phrase. (Recording start clears these too; a new phrase is
        # the other entry point where the old result becomes stale.)
        self.clear_previous_result()
        self.set_phrase(phrase)
        self.set_translation(translation)
        self.update_status("Listen to the reference...", THEME["reference"])
        self.draw_mic_button("speaking")
        self.update_instruction("Listening to the example...")

    def enter_phrase_ready(self):
        """Reference done: the user can record, replay or self-test."""
        self.draw_mic_button("idle")
        self.update_status("Your turn", THEME["ready"])
        self.update_instruction("Press SPACE or click the mic, then repeat the phrase.")
        self._set_actions(generate=True, reference=True, test=True)

    def generation_failed(self, message: str):
        """Phrase generation failed: surface the error and offer a retry."""
        self.append_error_msg(message)
        self.draw_mic_button("idle")
        self.update_status("Ready", THEME["ready"])
        self.update_instruction("Click 'New phrase' to try again.")
        self._set_actions(generate=True)

    def enter_recording(self):
        """Microphone is open: lock out playback so it cannot bleed into the take."""
        # Reset only the single-value indicators (charts, score read-out, face)
        # the moment the mic opens. The feedback panel is left intact: it is a
        # running history of all attempts and must not be wiped on each take.
        self.clear_previous_result()
        self._set_actions(generate=False, reference=False, user=False, test=False)
        # Paint the level indicator straight away (at zero) instead of the
        # "recording" glyph: the first live level frame overwrites the whole
        # button anyway, so drawing the emoji first only flashes a redundant
        # symbol. set_record_level(0.0) draws the same red ring + dark track with
        # a minimal level disc, so the look is identical from the first frame.
        self.set_record_level(0.0)
        self.update_status("Recording...", THEME["bad"])
        self.update_instruction("Speak now - recording stops automatically (press again to stop).")

    def enter_analyzing(self, status: str = "Analyzing pronunciation..."):
        """Analysis is running (also used for the reference self-test)."""
        self.draw_mic_button("processing")
        self.update_status(status, THEME["warn"])

    def enter_playing(self, status: str):
        """Reference or recorded take is being played back during a flow."""
        self.draw_mic_button("speaking")
        self.update_status(status, THEME["reference"])

    def enter_retry(self, *, has_phrase: bool, has_recording: bool):
        """Return to a recordable idle state, enabling only what is available."""
        self.draw_mic_button("idle")
        self.update_status("Ready", THEME["ready"])
        self._set_actions(generate=True)
        if has_phrase:
            self._set_actions(reference=True, test=True)
            self.update_instruction("Press SPACE or click the mic to repeat the phrase.")
        else:
            self.update_instruction("Click 'New phrase' to begin.")
        if has_recording:
            self._set_actions(user=True)

    # ------------------------------------------------------------------
    # Startup / failure status intents (status line only; the action
    # buttons start disabled during init, so these deliberately leave them
    # untouched). They keep the controller from knowing the palette.
    # ------------------------------------------------------------------
    def enter_loading(self):
        """Models are loading."""
        self.update_status("Loading models...", THEME["warn"])

    def enter_server_starting(self):
        """The local LLM server is being launched."""
        self.update_status("Starting LLM server...", THEME["warn"])

    def enter_warming_up(self):
        """Models are loaded and being warmed up."""
        self.update_status("Warming up models...", THEME["warn"])

    def server_failed(self):
        """The local LLM server failed to start."""
        self.update_status("LLM Server Error", THEME["bad"])

    def init_failed(self):
        """Initialization aborted on an unexpected error."""
        self.update_status("Initialization Failed", THEME["bad"])

    def recording_failed(self):
        """The microphone input stream failed."""
        self.update_status("Recording Error", THEME["bad"])

    def playing_status(self, status: str):
        """Set the status line for an ad-hoc playback (no button changes).

        Unlike enter_playing, this only touches the status line; it is used by
        the standalone replay path, which manages the mic button elsewhere.
        """
        self.update_status(status, THEME["reference"])

    def restore_ready_status(self):
        """Restore the default "Ready" status line without a full transition."""
        self.update_status("Ready", THEME["ready"])

    # ------------------------------------------------------------------
    # Talking face (driven by the controller from a loudness envelope)
    # ------------------------------------------------------------------
    def face_fps(self):
        """Frame rate of the talking-mouth animation, or None if no face is shown.

        Returns None while the face panel is hidden (the "Face" checkbox), not
        just when no face exists: the caller skips building the loudness track
        then, so a hidden face costs no per-playback work (envelope computation
        plus a 30 fps after-loop animating an invisible mouth).
        """
        face = getattr(self, "face", None)
        if face is None or not self.get_show_face():
            return None
        return face.fps

    def face_play_levels(self, levels, fps):
        """Drive the talking mouth from a pre-computed loudness track."""
        face = getattr(self, "face", None)
        if face is not None:
            face.play_levels(levels, fps=fps)

    def face_rest(self):
        """Close the talking mouth (no-op if there is no face)."""
        face = getattr(self, "face", None)
        if face is not None:
            face.rest()

    # ------------------------------------------------------------------
    # Feedback / status helpers (always called on the main thread)
    # ------------------------------------------------------------------
    def append_system_msg(self, text: str):
        """Log a progress/status message.

        Intentionally *not* shown in the feedback panel: routine progress
        ("Loading models...", "New phrase: ...") would drown out the
        pronunciation feedback. Only errors appear on screen - see
        append_error_msg.
        """
        logging.info(f"[System] {text}")

    def append_error_msg(self, text: str):
        """Show an error as a red row in the attempt history and log it.

        Errors like "Audio is too short" or "LM Studio is offline" must reach
        the user, not only the log file. They share the bounded history list with
        the takes: the controller owns the storage, so this just forwards the
        entry and the render follows from render_history.
        """
        logging.warning(f"[Error] {text}")
        self._cb.on_history_entry({"kind": "error", "text": text})

    # ------------------------------------------------------------------
    # Attempt history (scrollable list; model owned by the controller)
    # ------------------------------------------------------------------
    def _on_history_mousewheel(self, event):
        """Scroll the history list with the wheel."""
        self.history_canvas.yview_scroll(-int(event.delta / 120), "units")

    def _bind_history_wheel(self, widget):
        """Bind wheel scrolling on ``widget`` and all its descendants.

        Rows and their labels are separate widgets, so the wheel is bound on each
        one; otherwise it would only scroll when the pointer sat on the bare
        canvas between rows.
        """
        widget.bind("<MouseWheel>", self._on_history_mousewheel)
        for child in widget.winfo_children():
            self._bind_history_wheel(child)

    def render_history(self, entries: list):
        """Rebuild the attempt-history rows from the controller-owned list.

        ``entries`` is the full bounded history (oldest first); each is a dict with
        a ``kind`` of "attempt", "unscored" or "error". Only the most recent take
        (attempt/unscored) is expanded by default; errors are flat red rows. The
        list is small (<= 10), so a full rebuild per update is cheap and keeps the
        render a pure function of the model.
        """
        for child in self.history_frame.winfo_children():
            child.destroy()
        if not entries:
            self._history_placeholder = tk.Label(
                self.history_frame, text="Your attempts will appear here.",
                font=(FONT_FAMILY, 10, "italic"), fg=THEME["text_muted"],
                bg=THEME["bg_panel"], anchor="w")
            self._history_placeholder.pack(fill=tk.X, padx=12, pady=12)
            return
        # Newest first: the most recent take sits at the top, so a short list
        # fills from the top of the panel instead of leaving it blank above the
        # single bottom row.
        ordered = list(reversed(entries))
        # Expand the latest take (skip leading errors) by default: it is now the
        # first attempt/unscored entry from the top.
        expand_index = None
        for i, entry in enumerate(ordered):
            if entry.get("kind") in ("attempt", "unscored"):
                expand_index = i
                break
        for i, entry in enumerate(ordered):
            self._build_history_row(entry, expanded=(i == expand_index))
        # Wheel scrolling over any part of the freshly built rows.
        self._bind_history_wheel(self.history_frame)
        # Keep the newest row (at the top) in view.
        self.history_canvas.update_idletasks()
        self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all"))
        self.history_canvas.yview_moveto(0.0)

    def _build_history_row(self, entry: dict, expanded: bool):
        """Build one history row: [score chip][trend][phrase] plus a detail area."""
        kind = entry.get("kind")
        row = tk.Frame(self.history_frame, bg=THEME["bg_panel"])
        row.pack(fill=tk.X, padx=8, pady=(4, 0))

        if kind == "error":
            tk.Label(row, text=entry.get("text", ""), fg=THEME["bad"],
                     bg=THEME["bg_panel"], font=(FONT_FAMILY, 10), anchor="w",
                     justify=tk.LEFT, wraplength=520).pack(fill=tk.X, padx=4, pady=2)
            return

        header = tk.Frame(row, bg=THEME["bg_panel"], cursor="hand2")
        header.pack(fill=tk.X)

        # Score chip: a 1px outline in the verdict colour with the digit inside.
        if kind == "attempt":
            chip_text = f"{entry.get('score', 0):.0f}"
            chip_color = entry.get("color") or THEME["text_dim"]
        else:  # unscored ("none" engine): no number, no verdict colour.
            chip_text, chip_color = "--", THEME["text_dim"]
        chip = tk.Label(header, text=chip_text, fg=chip_color, bg=THEME["bg_panel"],
                        font=(FONT_FAMILY, 11, "bold"), padx=8, pady=1,
                        highlightthickness=1, highlightbackground=chip_color)
        chip.pack(side=tk.LEFT)

        # Trend arrow vs the previous attempt of the same phrase (controller-set).
        trend = entry.get("trend")
        if trend == "up":
            arrow_text, arrow_color = "▲", THEME["good"]
        elif trend == "down":
            arrow_text, arrow_color = "▼", THEME["bad"]
        else:  # "same", or no earlier attempt to compare against.
            arrow_text, arrow_color = "–", THEME["text_dim"]
        tk.Label(header, text=arrow_text, fg=arrow_color, bg=THEME["bg_panel"],
                 font=(FONT_FAMILY, 11)).pack(side=tk.LEFT, padx=(8, 8))

        tk.Label(header, text=entry.get("phrase") or "-", fg=THEME["text_emph"],
                 bg=THEME["bg_panel"], font=(FONT_FAMILY, 11), anchor="w",
                 justify=tk.LEFT).pack(side=tk.LEFT, fill=tk.X, expand=True)

        detail = tk.Frame(row, bg=THEME["bg_panel"])
        detail_text = self._fill_history_detail(detail, entry)
        if expanded:
            detail.pack(fill=tk.X, padx=(4, 4), pady=(2, 4))
            self.root.after_idle(lambda t=detail_text: self._fit_history_text(t))

        # Clicking anywhere on the header toggles the detail below it.
        def toggle(event=None, d=detail, t=detail_text):
            if d.winfo_manager() == "pack":
                d.pack_forget()
            else:
                d.pack(fill=tk.X, padx=(4, 4), pady=(2, 4))
                self.root.after_idle(lambda: self._fit_history_text(t))
            self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all"))

        for widget in (header, *header.winfo_children()):
            widget.bind("<Button-1>", toggle)

    def _fill_history_detail(self, parent, entry: dict):
        """Fill a row's detail with the coloured word breakdown and problem sounds.

        Returns the inner read-only Text so its height can be fitted once visible.
        """
        txt = tk.Text(parent, bg=THEME["bg_panel"], fg=THEME["text"], bd=0,
                      highlightthickness=0, wrap=tk.WORD, height=1, cursor="arrow",
                      font=(FONT_FAMILY, 10), padx=4, pady=0, spacing3=2)
        txt.tag_configure("label", foreground=THEME["text_dim"])
        txt.tag_configure("good", foreground=THEME["good"])
        txt.tag_configure("ok", foreground=THEME["text_dim"])
        txt.tag_configure("bad", foreground=THEME["bad"], font=(FONT_FAMILY, 10, "bold"))
        txt.tag_configure("text", foreground=THEME["text_emph"])
        txt.tag_configure("mono", foreground=THEME["text_dim"], font=("Consolas", 10))

        if entry.get("kind") == "unscored":
            txt.insert(tk.END, "Heard: ", "label")
            txt.insert(tk.END, "scoring is off - compare the takes by ear", "text")
        else:
            # "Words": the target phrase coloured by per-word correctness
            # (good/ok/bad, three-level "level" or the acoustic engine's boolean
            # "correct"); falls back to the plain phrase if tags are empty.
            txt.insert(tk.END, "Words: ", "label")
            ref_words = entry.get("reference_words") or []
            if ref_words:
                for word in ref_words:
                    level = word.get("level")
                    if level in ("good", "ok", "bad"):
                        tag = level
                    else:
                        tag = "good" if word.get("correct") else "bad"
                    txt.insert(tk.END, word["word"] + " ", tag)
            else:
                txt.insert(tk.END, entry.get("phrase") or "-", "text")
            phonemes = entry.get("phonemes") or []
            if phonemes:
                txt.insert(tk.END, "\nWork on: ", "label")
                for phoneme in phonemes:
                    txt.insert(tk.END, f"/{phoneme}/ ", "mono")
        txt.configure(state=tk.DISABLED)
        txt.pack(fill=tk.X)
        return txt

    def _fit_history_text(self, txt):
        """Size a detail Text to its number of logical lines.

        Height is the logical line count (the coloured "Words" line plus an
        optional "Work on" line), not the wrapped display-line count: a
        display-line count measured before the canvas-embedded Text gets its real
        width balloons (every word wraps onto its own line), which would leave a
        tall gap under the expanded row and push the older rows to the bottom.
        Logical lines are width-independent, so the row height stays stable.
        """
        if not txt.winfo_exists():
            return
        lines = int(txt.index("end-1c").split(".")[0])
        txt.configure(height=max(1, lines))
        self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all"))

    def update_status(self, text: str, color: str = THEME["text_dim"]):
        self.status_label.configure(text=text, fg=color)

    def update_instruction(self, text: str):
        # No-op since v2c step 4: the standalone instruction line was removed.
        # Its guidance now lives in the per-control captions of the bottom panel,
        # the Tip line and the status bar. Kept as a harmless stub so the many
        # enter_* state methods (and the LLM-startup path) can keep calling it
        # without change; the dead calls are cleaned up in the step 8 pass.
        pass

    def update_session_stats(self, count: int, average: float):
        """Show the session tally in the status bar: ``Phrases: 4 · Avg: 78``.

        ``count`` is the number of distinct phrases practiced this run and
        ``average`` the mean over every scored attempt this run (both supplied by
        the controller, which owns the session data). The average is rounded to a
        whole number to match the mockup.
        """
        self.stats_label.configure(text=f"Phrases: {count} · Avg: {average:.0f}",
                                   fg=THEME["text_dim"])

    def clear_previous_result(self):
        """Reset the per-take indicators before a new recording starts.

        Resets the hero-card score row to its empty state, erases both prosody
        charts and resets the face to its neutral waiting smile, so a stale
        single-value indicator can never be mistaken for the take in progress.
        The session tally in the status bar is deliberately NOT touched here: it
        accumulates across the whole run. The feedback panel is likewise NOT
        cleared: it is a running history of every attempt, appended to by
        show_feedback / append_error_msg and kept across takes. The status *line*
        is set separately by the caller (enter_recording -> "Recording...").
        """
        # Hero-card score row back to its empty state ("--", no badges).
        self._reset_score_row()
        self._last_prosody = None
        self.f0_canvas.delete("all")
        self.en_canvas.delete("all")
        # Reset the face to the neutral waiting smile so it no longer reflects
        # the previous take's score. rest() drops any leftover talking mouth.
        self.face.rest()
        self.face.set_expression(":)")

    # Qualitative rating bands on the raw 0-100 score, used by the acoustic engine
    # (no calibrated bucket) and as the fallback colour for the score read-out. The
    # phoneme engine uses _bucket_quality instead; show_feedback picks one basis so
    # the status line, the face and the Score bar always agree (see show_feedback).
    def _quality_label(self, score: float) -> tuple[str, str]:
        if score < 40:
            return "Weak", THEME["bad"]
        if score < 55:
            return "Poor", THEME["bad"]
        if score < 70:
            return "Needs work", THEME["warn"]
        if score < 85:
            return "Good", THEME["good"]
        return "Excellent", THEME["good"]

    def _bucket_quality(self, bucket: int) -> tuple[str, str]:
        """Qualitative label + colour for a calibrated 0-5 bucket (phoneme engine).

        Keyed to the engine's pass bucket (4): bucket >= 4 is a pass and reads
        green, so the label, the score colour and the face can never contradict
        result.passed. Bucket 3 is the amber near-miss; buckets 0-2 are red.
        """
        if bucket >= 5:
            return "Excellent", THEME["good"]
        if bucket >= 4:
            return "Good", THEME["good"]
        if bucket >= 3:
            return "Needs work", THEME["warn"]
        if bucket >= 2:
            return "Poor", THEME["bad"]
        return "Weak", THEME["bad"]

    def _quality_expression(self, color: str) -> str:
        """Map a resolved quality colour to a face expression so the two agree."""
        if color == THEME["good"]:
            return "happy"
        if color == THEME["warn"]:
            return "neutral"
        return "sad"

    # ------------------------------------------------------------------
    # Prosody drawing
    # ------------------------------------------------------------------
    def _draw_prosody(self, canvas, series):
        """Draw contours onto ``canvas``. ``series`` is a list of (values, color).

        All series share one vertical scale so they are directly comparable.
        No-op until the canvas has been laid out (winfo_width > 1).
        """
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width <= 1 or height <= 1:
            return
        pad_x, pad_y = 4, 4
        plot_w = max(1, width - 2 * pad_x)
        plot_h = max(1, height - 2 * pad_y)

        all_values = [v for values, _ in series for v in values]
        if not all_values:
            return
        lo, hi = min(all_values), max(all_values)
        span = (hi - lo) or 1.0

        for values, color in series:
            points = prosody_utils.resample_series(values)
            if len(points) < 2:
                continue
            coords = []
            for i, value in enumerate(points):
                x = pad_x + (i / (len(points) - 1)) * plot_w
                y = pad_y + (1 - (value - lo) / span) * plot_h
                coords.extend((x, y))
            canvas.create_line(*coords, fill=color, width=3, smooth=True)

    def _on_prosody_caption_clicked(self):
        """Flip the prosody collapse flag and route it through the controller.

        Mirrors the practice-text caption: the Button owns no variable, so the
        view flips show_prosody here; the controller then applies (toggle_prosody)
        and persists it, and updates the worker-visible compute flag.
        """
        self.show_prosody.set(not self.show_prosody.get())
        self._cb.on_prosody_toggled()

    def toggle_prosody(self):
        """Show/hide the whole prosody body to match the show_prosody flag."""
        # Return focus to the window so the spacebar record toggle keeps working.
        self.root.focus_set()
        expanded = self.show_prosody.get()
        self._prosody_caption.config(
            text="▾ Intonation & stress" if expanded else "▸ Intonation & stress")
        shown = self.prosody_body.winfo_manager() == "pack"
        if expanded and not shown:
            self.prosody_body.pack(fill=tk.X)
        elif not expanded and shown:
            self.prosody_body.pack_forget()

    def toggle_face(self):
        """Show/hide the face in the hero card's score row (the "Face" setting)."""
        self.root.focus_set()
        shown = self.face.winfo_manager() == "pack"
        if self.show_face.get() and not shown:
            # The only side=RIGHT child of the score row, so packing order
            # relative to the other columns does not matter.
            self.face.pack(side=tk.RIGHT, padx=(8, 0))
        elif not self.show_face.get() and shown:
            self.face.pack_forget()

    def _redraw_prosody(self):
        """Redraw both prosody canvases from the cached result (e.g. after resize)."""
        prosody = self._last_prosody
        if not prosody:
            return
        # Normalize each pitch contour to semitones vs its own median so the
        # reference and the user (different vocal registers) become directly
        # comparable in shape; _draw_prosody's shared scale then centres both
        # on 0 ST. Energy is already per-utterance scaled, so it is left as-is.
        self._draw_prosody(self.f0_canvas, [
            (prosody_utils.to_semitones(prosody.get("ref_f0", [])), THEME["reference"]),  # reference
            (prosody_utils.to_semitones(prosody.get("f0", [])), THEME["info"]),           # you
        ])
        self._draw_prosody(self.en_canvas, [
            (prosody.get("ref_energy", []), THEME["reference"]),
            (prosody.get("energy", []), THEME["info"]),
        ])

    # ------------------------------------------------------------------
    # Feedback rendering
    # ------------------------------------------------------------------
    def show_feedback(self, result: "PronunciationResult", current_phrase,
                      has_recording: bool = True, is_self_test: bool = False):
        # An unscored result (the "none" engine) has no verdict to present:
        # score/passed carry no meaning, so render the neutral read-out instead
        # of a quality label that would pretend the take was judged.
        if not getattr(result, "scored", True):
            self._show_unscored_feedback(result, current_phrase, has_recording,
                                         is_self_test)
            return
        # One consistent presentation for the whole result, so the score read-out,
        # quality label, face and the passed/try-again line never contradict each
        # other. The phoneme engine grades into a calibrated 0-5 bucket that also
        # decides result.passed, so the bucket drives the user-facing percent, the
        # label and the colour; the acoustic engine has no bucket and uses its raw
        # 0-100 score (consistent with its passed = score >= threshold).
        bucket = getattr(result, "bucket", -1)
        if bucket >= 0:
            display_score = result.user_percent
            quality, quality_color = self._bucket_quality(bucket)
        else:
            display_score = result.score
            quality, quality_color = self._quality_label(result.score)
        # The face follows the same quality band, so a passed take always smiles.
        self.face.set_expression(self._quality_expression(quality_color))
        # Hero card: the big score digit and its verdict share the resolved
        # quality color (never green for a "needs work"), and the WORK ON row
        # shows the top-3 problem sounds.
        self._set_score_row(f"{display_score:.0f}", quality.lower(), quality_color)
        badges = [entry["phoneme"] for entry in result.weak_phonemes[:3]]
        self._set_badges(badges)
        # Underline the mispronounced words directly in the phrase; show the
        # how-to hint only when at least one interactive affordance exists (an
        # underlined word or a sound badge), so a clean take stays uncluttered.
        misses = self._mark_miss_words(result.reference_words)
        self._set_hint(bool(misses) or bool(badges))
        # The Phrase/Work-on breakdown is no longer written to a running text
        # panel: it now lives in the attempt-history list, emitted below once per
        # take (see on_history_entry). The hero card already carries the score,
        # verdict and problem-sound badges for the current take.

        # Cache prosody and draw the sparklines (you vs reference).
        self._last_prosody = result.prosody or {}
        self.root.update_idletasks()  # ensure the canvases have a real width/height
        self._redraw_prosody()

        # The number itself now lives in the hero card (set above via
        # _set_score_row); report the scored take to the controller so it can
        # update the session tally shown in the status bar. The reference
        # self-test reaches here without a real recording - it is a pipeline
        # sanity check, not a practice attempt, so it must not touch the tally.
        if not is_self_test:
            self._cb.on_take_scored(current_phrase, display_score)
            # Record the take into the attempt history (controller-owned). The
            # colour is the resolved verdict colour; the detail keeps the coloured
            # word breakdown and the top problem sounds shown on the hero card.
            self._cb.on_history_entry({
                "kind": "attempt",
                "phrase": current_phrase or "",
                "score": display_score,
                "color": quality_color,
                "reference_words": result.reference_words,
                "phonemes": [entry["phoneme"] for entry in result.weak_phonemes[:3]],
            })

        # Re-enable everything disabled while recording: the reference replay,
        # the self-test and new-phrase generation. "My recording" is enabled only
        # when a user take actually exists - the reference self-test reaches here
        # without recording, so enabling it there would offer a dead button.
        self._set_actions(generate=True, reference=True, user=has_recording, test=True)
        self.draw_mic_button("idle")

        self.update_status(quality, quality_color)
        if result.passed:
            self.update_instruction("Nice! Click 'New phrase' to continue, or repeat to refine.")
        else:
            self.update_instruction("Try again: press SPACE or click the mic to repeat. ▶ Reference replays the example.")

    def _show_unscored_feedback(self, result: "PronunciationResult", current_phrase,
                                has_recording: bool, is_self_test: bool = False):
        """Feedback for an unscored take (``result.scored`` is False; "none" engine).

        Keeps everything that does not depend on scoring - the prosody charts
        (still computed by the host from the raw waveforms), the neutral hero-card
        read-out and re-enabling the controls - and adds a neutral "--" row to the
        attempt history, so this mode never pretends the take was judged.
        """
        self.face.set_expression("neutral")
        # Hero card: neutral "scoring off" read-out - no number, no badges, no
        # word underlines and no how-to hint (nothing was judged to act on).
        self._set_score_row("--", "scoring off", THEME["text_dim"])
        self._set_badges([])
        self._set_hint(False)
        # A neutral history row (no score, no trend). The reference self-test is a
        # pipeline check, not a practice take, so it stays out of the history for
        # the same reason it stays out of the session tally.
        if not is_self_test:
            self._cb.on_history_entry({
                "kind": "unscored",
                "phrase": current_phrase or "",
            })

        # Prosody still works without scoring; cache and draw it as usual.
        self._last_prosody = result.prosody or {}
        self.root.update_idletasks()  # ensure the canvases have a real width/height
        self._redraw_prosody()

        # Unscored takes ("none" engine) do not contribute to the session tally,
        # so the status bar's Phrases/Avg line is left untouched here.

        self._set_actions(generate=True, reference=True, user=has_recording, test=True)
        self.draw_mic_button("idle")

        self.update_status("Recorded (scoring off)", THEME["text_dim"])
        self.update_instruction("Compare by ear: ▶ Reference and ▶ My recording, "
                                "then repeat or click 'New phrase' to continue.")
