# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""View layer for the Mimora pronunciation trainer (facade).

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

TrainerView is a facade over per-panel classes, each owning its widgets:

* :class:`mimora.ui_practice.PracticePanel` - the collapsible source-text editor.
* :class:`mimora.ui_hero.HeroCard` - phrase, translation, score row, face,
  session-average progress ring.
* :class:`mimora.ui_prosody.ProsodyPanel` - the pitch/energy sparklines.
* :class:`mimora.ui_history.HistoryPanel` - the scrollable attempt list.
* :mod:`mimora.ui_theme` - shared palette, fonts, wheel helpers, tooltip.

The facade keeps the window chrome (header, status bar, tip line), the control
row with the mic button, the intent methods and the feedback orchestration, and
delegates everything panel-local. The controller-facing API is unchanged by the
split: ``main.py`` and ``settings_window.py`` import exactly what they did
before (TrainerView, ViewCallbacks, LENGTH_*; THEME, FONT_FAMILY, WHEEL_EVENTS,
wheel_scroll_units are re-exported from ui_theme for settings_window).
"""
import logging
import tkinter as tk

# Importing ui_theme also disables ttkbootstrap's classic-widget "autostyle"
# hook (see the comment there), so it must stay the first view import.
from mimora.ui_theme import (  # noqa: F401  (re-exported for settings_window)
    BOOTSTRAP_THEME,
    FONT_FAMILY,
    FONT_SIZE_BODY,
    FONT_SIZE_CAPTION,
    FONT_SIZE_EMOJI,
    FONT_SIZE_ICON,
    FONT_SIZE_SMALL,
    FONT_SIZE_TITLE,
    THEME,
    WHEEL_EVENTS,
    bind_hover,
    wheel_scroll_units,
)

# ttkbootstrap is a drop-in replacement for tkinter.ttk (same widget classes,
# modern flat themes). Aliased as ``ttk`` so ttk.Style keeps working unchanged.
import ttkbootstrap as ttk

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from mimora import config
from mimora.ui_hero import HeroCard
from mimora.ui_history import HistoryPanel
from mimora.ui_practice import PracticePanel
from mimora.ui_prosody import ProsodyPanel

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
    # Click on any word inside the hero phrase (mispronounced ones are
    # underlined): the controller synthesizes that single word and plays it
    # slowly.
    on_word_clicked: Callable[[str], None]
    # Click on a "WORK ON" phoneme badge: the controller synthesizes the
    # phoneme's example word (e.g. "put" for /ʊ/) and plays it at normal speed.
    on_sound_example: Callable[[str], None]
    # A take was scored: the view reports the phrase, the numeric value behind the
    # displayed mark (grade_value on the 0-5 axis for the phoneme engine, the raw
    # 0-100 score otherwise) and whether it is a 0-5 grade, so the controller can
    # keep the session tally (unique phrases, running average) that the status bar
    # shows on the right scale. Only called for actually-scored takes.
    on_take_scored: Callable[[str, float, bool], None]
    # A new entry for the attempt history: a scored/unscored take or an error
    # message. The controller owns the bounded history (last 10), computes the
    # per-phrase trend and hands the full list back via view.render_history.
    on_history_entry: Callable[[dict], None]

# Phrase-length selector labels. The label maps to generate_phrase's ``length``
# mode: LENGTH_FULL → "full" sentence, LENGTH_FEW_WORDS → "fragment".
LENGTH_FULL = "Full phrase"
LENGTH_FEW_WORDS = "Few words"


class TrainerView:
    """Passive view facade: builds the Tk widgets and renders UI state.

    Owns the window chrome and the control row directly, and composes the
    panel classes (PracticePanel, HeroCard, ProsodyPanel, HistoryPanel) for
    the rest. Widget callbacks forward to the :class:`ViewCallbacks` passed in
    (``self._cb``); the controller drives the UI through the public intent
    methods and read accessors below. The view holds no reference to the
    controller.
    """

    def __init__(self, root, callbacks: ViewCallbacks):
        """Build the UI under ``root``, wiring widgets to ``callbacks``.

        Args:
            root: the Tk root window the widgets are placed in.
            callbacks: the view→controller handlers the widgets invoke.
        """
        self.root = root
        self._cb = callbacks
        self.setup_styles()
        self.build_ui()
        # Second palette pass: any ttk widget created in build_ui makes
        # ttkbootstrap build its default style, which can override the colors
        # applied in setup_styles (see _apply_ttk_palette).
        self._apply_ttk_palette()

    # ------------------------------------------------------------------
    # Styles
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

        # Theme styling for the reference-speed combobox (settings window).
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

    # ------------------------------------------------------------------
    # UI construction (window chrome + control row; panels build themselves)
    # ------------------------------------------------------------------
    def _make_button(self, parent, text, command, width=None, padx=12):
        """Create a consistently styled themed button.

        ``width`` (in text characters) is optional; pass it to give a group of
        buttons a uniform width so they line up regardless of label length.
        ``padx`` is the internal horizontal padding (default 12); the bottom
        control row passes a smaller value so its four columns fit the 600px
        window without clipping the last button.
        """
        button = tk.Button(parent, text=text, command=command,
                           font=(FONT_FAMILY, FONT_SIZE_BODY, "bold"),
                           bg=THEME["bg_button"], fg=THEME["text_button"],
                           activebackground=THEME["bg_button_active"], activeforeground=THEME["text"],
                           bd=0, padx=padx, pady=6, cursor="hand2",
                           disabledforeground=THEME["text_disabled"])
        if width is not None:
            button.config(width=width)
        return button

    def _control_column(self, parent, caption):
        """A vertical control cell: the control(s) on top, an 8pt caption below.

        Used by the bottom control panel so each action - Reference,
        the mic, My recording, Next phrase - carries a one-line hint underneath.
        The caption is packed at the bottom up-front, so callers can simply pack
        their control widgets (default ``side=TOP``) into the returned frame
        without worrying about ordering. Columns are top-aligned by the caller
        (``anchor=N``) so short buttons and the taller mic each keep their caption
        directly beneath them.
        """
        col = tk.Frame(parent, bg=THEME["bg_main"])
        # padx=4 on each side yields a ~8px gap between columns, tight enough
        # that all four columns fit the 600px window.
        col.pack(side=tk.LEFT, anchor=tk.N, padx=4)
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

        # Title carries the fixed training language (no dynamic language picker):
        # brand + language shown prominently, with the descriptor as a smaller
        # subtitle. The language reads from config.TARGET_LANGUAGE so the title
        # stays the single source of truth (replacing the old language chip).
        tk.Label(header_frame, text=f"MIMORA · {config.TARGET_LANGUAGE}",
                 font=(FONT_FAMILY, FONT_SIZE_TITLE, "bold"), fg=THEME["accent"], bg=THEME["bg_main"]).pack(side=tk.LEFT)

        tk.Label(header_frame, text="- Pronunciation Trainer",
                 font=(FONT_FAMILY, FONT_SIZE_BODY, "bold"), fg=THEME["accent"], bg=THEME["bg_main"]).pack(
                     side=tk.LEFT, padx=(0, 0), pady=(6, 0))

        # Settings gear at the right edge of the header. Opens the settings
        # window (see main.py on_settings_clicked). Tk buttons have no native
        # hover state (activebackground only shows while pressed), so a small
        # <Enter>/<Leave> pair brightens the glyph and fills the button like a
        # quiet outline-less icon button - otherwise the gear reads as a stray
        # character rather than a control.
        gear = tk.Button(header_frame, text="⚙",
                         command=self._cb.on_settings_clicked,
                         font=(FONT_FAMILY, FONT_SIZE_ICON), bg=THEME["bg_main"],
                         fg=THEME["text_dim"],
                         activebackground=THEME["bg_accent_active"],
                         activeforeground=THEME["text_bright"],
                         bd=0, padx=8, pady=2, cursor="hand2")
        gear.pack(side=tk.RIGHT)
        bind_hover(gear,
                   enter={"bg": THEME["bg_panel"], "fg": THEME["text_bright"]},
                   leave={"bg": THEME["bg_main"], "fg": THEME["text_dim"]})

        # 2. Status bar (absolute bottom)
        self.status_bar = tk.Frame(self.root, bg=THEME["bg_panel"], height=30)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = tk.Label(self.status_bar, text="Starting...",
                                     font=(FONT_FAMILY, FONT_SIZE_SMALL), fg=THEME["ready"], bg=THEME["bg_panel"])
        self.status_label.pack(side=tk.LEFT, padx=15, pady=4)

        # The session tally (distinct-phrase count + running average) used to
        # live here on the right of the status bar; it now lives in the hero
        # card's progress ring (see HeroCard / update_session_stats), so the
        # status bar carries only the transient status line on the left.

        # 2a. Tip line - a single static hint sitting directly above the status
        # bar. Packed side=BOTTOM after the status bar so it lands just above it,
        # regardless of the TOP-packed content built later. It restates the two
        # least discoverable actions (space-to-record, Reference-replays), the
        # role the removed instruction line used to fill.
        self.tip_label = tk.Label(
            self.root,
            text='Tip: press SPACE or click the mic to record. '
                 '"Reference ▶" replays the example.',
            font=(FONT_FAMILY, FONT_SIZE_CAPTION), fg=THEME["text_muted"],
            bg=THEME["bg_main"], anchor=tk.W)
        self.tip_label.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=(0, 4))

        # 3. Control panel: one row holding every phrase-level
        # action, each in its own column with an 8pt caption beneath. Left to
        # right: Reference (+ slow-replay button), the mic, My recording, Next
        # phrase. The old single instruction line is gone - its guidance now
        # lives in these per-control captions, the Tip line and the status bar.
        #
        # The widgets are built here, but the frame is packed later - right below
        # the hero card (see the control_frame.pack() call after the hero
        # section) - so the on-screen order is:
        # header -> practice text -> hero -> controls -> prosody -> history.
        control_frame = tk.Frame(self.root, bg=THEME["bg_main"])

        # Self-test enabled state. The self-test has no visible button; it stays
        # reachable via the 't' hotkey, gated by this flag (kept in sync by
        # _set_actions; see is_test_enabled).
        self._test_enabled = False

        # Equal-width buttons keep the row balanced regardless of label length;
        # the longest label ("My recording ▶") is 14 chars, so it still sizes
        # its button to fit rather than the requested 12. A tight internal
        # padding keeps all four columns within the 600px window.
        action_btn_width = 12
        action_btn_padx = 4

        # Centered row of control columns (each built by _control_column, which
        # top-aligns them and hangs the caption underneath).
        controls_row = tk.Frame(control_frame, bg=THEME["bg_main"])
        controls_row.pack()

        # -- Reference column: the replay button paired with the "Slow ▶"
        #    replay. Both are gated together by _set_actions(reference=...); the
        #    exact slow speed lives in Settings. --
        ref_col = self._control_column(controls_row, "Listen to the example")
        ref_pair = tk.Frame(ref_col, bg=THEME["bg_main"])
        ref_pair.pack()
        self.ref_btn = self._make_button(
            ref_pair, "Reference ▶", self._cb.play_reference,
            width=action_btn_width, padx=action_btn_padx)
        self.ref_btn.pack(side=tk.LEFT, padx=(0, 4))
        self.ref_btn.config(state=tk.DISABLED)
        self.slow_btn = self._make_button(ref_pair, "Slow ▶", self._cb.play_reference_slow,
                                          padx=action_btn_padx)
        self.slow_btn.pack(side=tk.LEFT)
        self.slow_btn.config(state=tk.DISABLED)

        # -- Mic column: the press-to-toggle record button (canvas-drawn). --
        mic_col = self._control_column(controls_row, "Press SPACE or click")
        self.btn_canvas = tk.Canvas(mic_col, width=100, height=100, bg=THEME["bg_main"],
                                    highlightthickness=0, cursor="hand2")
        self.btn_canvas.pack()
        self.btn_canvas.bind("<ButtonPress-1>", lambda e: self._cb.on_gui_btn_press())
        self.btn_canvas.bind("<ButtonRelease-1>", lambda e: self._cb.on_gui_btn_release())
        self.draw_mic_button("loading")

        # -- My recording column: always visible; enabled only once a take
        #    exists (_set_actions(user=...)). --
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

        # 4. Source text panel (editable, collapsible) - PracticePanel packs
        # itself here, so the on-screen order stays header -> practice text.
        self._practice = PracticePanel(
            self.root, on_collapsed_toggled=self._cb.on_practice_collapsed_toggled)

        # 5. Hero card - phrase, translation, score row and the face.
        self._hero = HeroCard(
            self.root,
            on_word_clicked=self._cb.on_word_clicked,
            on_sound_example=self._cb.on_sound_example)

        # 5a. Control panel, packed here so it sits directly under the hero card.
        # The frame and its buttons were built in section 3; only the placement
        # was deferred to this point in the packing order.
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(0, 8))

        # 5b. Prosody panel - pitch (F0) and energy sparklines under one
        # collapse header. Default collapsed for a calmer view.
        self._prosody = ProsodyPanel(
            self.root, on_prosody_toggled=self._cb.on_prosody_toggled)

        # Late-apply the persisted state: show the prosody body only if
        # expanded, and pack the face (built unpacked in the hero card's score
        # row) if enabled.
        self._prosody.toggle()
        self._hero.toggle_face()

        # 6. Attempt history (fills remaining space). Created AFTER the control
        # panel so its expand=True only consumes space left over above the
        # button row.
        self._history = HistoryPanel(self.root)

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
        self.btn_canvas.create_text(cx, cy, text=emoji, font=(FONT_FAMILY, FONT_SIZE_EMOJI), fill=THEME["text_bright"])

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
    # Practice text (delegated to PracticePanel)
    # ------------------------------------------------------------------
    def get_practice_text(self) -> str:
        """Return the editable practice text (without trailing whitespace)."""
        return self._practice.get_text()

    def set_practice_text(self, text: str):
        """Replace the practice-text panel contents."""
        self._practice.set_text(text)

    def toggle_practice_text(self):
        """Show/hide the practice-text editor to match the collapse flag."""
        self._practice.toggle()

    def set_practice_collapsed(self, flag: bool):
        self._practice.set_collapsed(flag)

    def get_practice_collapsed(self) -> bool:
        return self._practice.get_collapsed()

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

    # ------------------------------------------------------------------
    # Hero card (delegated to HeroCard)
    # ------------------------------------------------------------------
    def set_translation(self, text: str):
        """Set the translation panel text, falling back to '-' when empty."""
        self._hero.set_translation(text)

    def refresh_translation_ui(self):
        """Show or hide the translation panel to match the current setting."""
        self._hero.refresh_translation_ui()

    def set_phrase(self, text: str):
        """Replace the hero phrase (every word clickable)."""
        self._hero.set_phrase(text)

    # ------------------------------------------------------------------
    # Write accessors for the visibility settings (prosody/face); the
    # controller mirrors settings-window changes into them. Pure value
    # updates, no callbacks fire.
    # ------------------------------------------------------------------
    def set_show_face(self, flag: bool):
        self._hero.set_show_face(flag)

    def get_show_face(self) -> bool:
        return self._hero.get_show_face()

    def toggle_face(self):
        """Show/hide the face in the hero card's score row (the "Face" setting)."""
        self._hero.toggle_face()

    def set_show_prosody(self, flag: bool):
        self._prosody.set_show(flag)

    def get_show_prosody(self) -> bool:
        return self._prosody.get_show()

    def toggle_prosody(self):
        """Show/hide the whole prosody body to match the show_prosody flag."""
        self._prosody.toggle()

    # ------------------------------------------------------------------
    # Enabled-state queries (hotkey gating mirrors button gating)
    # ------------------------------------------------------------------
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
        """True when the My recording replay button is currently clickable."""
        return str(self.user_btn["state"]) == str(tk.NORMAL)

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
        self._set_actions(generate=True)

    def enter_generating(self):
        """LLM is producing a phrase: every action is locked out."""
        self._set_actions(generate=False, reference=False, user=False, test=False)
        self.draw_mic_button("processing")
        self.update_status("Generating phrase (LLM)...", THEME["info"])

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
        self._hero.set_phrase(phrase)
        self._hero.set_translation(translation)
        self.update_status("Listen to the reference...", THEME["reference"])
        self.draw_mic_button("speaking")

    def enter_phrase_ready(self):
        """Reference done: the user can record, replay or self-test."""
        self.draw_mic_button("idle")
        self.update_status("Your turn", THEME["ready"])
        self._set_actions(generate=True, reference=True, test=True)

    def generation_failed(self, message: str):
        """Phrase generation failed: surface the error and offer a retry."""
        self.append_error_msg(message)
        self.draw_mic_button("idle")
        self.update_status("Ready", THEME["ready"])
        self._set_actions(generate=True)

    def enter_recording(self):
        """Microphone is open: lock out playback so it cannot bleed into the take."""
        # Reset only the single-value indicators (charts, score read-out, face)
        # the moment the mic opens. The attempt history is left intact: it is a
        # running record of all attempts and must not be wiped on each take.
        self.clear_previous_result()
        self._set_actions(generate=False, reference=False, user=False, test=False)
        # Paint the level indicator straight away (at zero) instead of the
        # "recording" glyph: the first live level frame overwrites the whole
        # button anyway, so drawing the emoji first only flashes a redundant
        # symbol. set_record_level(0.0) draws the same red ring + dark track with
        # a minimal level disc, so the look is identical from the first frame.
        self.set_record_level(0.0)
        self.update_status("Recording...", THEME["bad"])

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
    # Talking face (delegated to HeroCard; driven by the controller from a
    # loudness envelope)
    # ------------------------------------------------------------------
    def face_fps(self):
        """Frame rate of the talking-mouth animation, or None if no face is shown."""
        return self._hero.face_fps()

    def face_play_levels(self, levels, fps):
        """Drive the talking mouth from a pre-computed loudness track."""
        self._hero.face_play_levels(levels, fps)

    def face_rest(self):
        """Close the talking mouth."""
        self._hero.face_rest()

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

    def render_history(self, entries: list):
        """Rebuild the attempt-history rows from the controller-owned list."""
        self._history.render(entries)

    def update_status(self, text: str, color: str = THEME["text_dim"]):
        self.status_label.configure(text=text, fg=color)

    def update_session_stats(self, count: int, average: float, maximum: float):
        """Show the session tally in the hero card's progress ring.

        ``count`` is the number of distinct phrases practiced this run,
        ``average`` the running mean over every scored attempt and ``maximum``
        its scale (5 for graded takes, 100 for raw-percent ones) - plain
        numbers straight from SessionState.record_take. The ring formats the
        value itself from the scale (see ProgressRing.set_progress), so no
        display string has to be parsed back here.
        """
        self._hero.set_progress(average, maximum, count)

    def clear_previous_result(self):
        """Reset the per-take indicators before a new recording starts.

        Resets the hero-card score row to its empty state, erases both prosody
        charts and resets the face to its neutral waiting smile, so a stale
        single-value indicator can never be mistaken for the take in progress.
        The session tally in the status bar is deliberately NOT touched here: it
        accumulates across the whole run. The attempt-history list is likewise
        NOT cleared: it is a running, controller-owned record of every attempt,
        fed by show_feedback / append_error_msg and kept across takes. The
        status *line* is set separately by the caller (enter_recording ->
        "Recording...").
        """
        # Hero-card score row back to its empty state ("--", no badges).
        self._hero.reset_score_row()
        self._prosody.clear()
        # Reset the face to the neutral waiting smile so it no longer reflects
        # the previous take's score. rest() drops any leftover talking mouth.
        self._hero.face.rest()
        self._hero.face.set_expression(":)")

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
    # Feedback rendering (orchestrates hero card, prosody and history)
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
        # decides result.passed, so the bucket drives the displayed mark (the
        # "4+"-style grade; six buckets shown as a percent read as a precise
        # measurement, which the engine cannot deliver), the label and the colour;
        # the acoustic engine has no bucket and uses its raw 0-100 score
        # (consistent with its passed = score >= threshold).
        bucket = getattr(result, "bucket", -1)
        graded = bucket >= 0 and bool(getattr(result, "grade", ""))
        if graded:
            display_text = result.grade
            tally_value = result.grade_value
            quality, quality_color = self._bucket_quality(bucket)
        elif bucket >= 0:  # bucketized but ungraded (defensive): band midpoint
            display_text = f"{result.user_percent:.0f}"
            tally_value = result.user_percent
            quality, quality_color = self._bucket_quality(bucket)
        else:
            display_text = f"{result.score:.0f}"
            tally_value = result.score
            quality, quality_color = self._quality_label(result.score)
        # The face follows the same quality band, so a passed take always smiles.
        self._hero.face.set_expression(self._quality_expression(quality_color))
        # Hero card: the big score digit and its verdict share the resolved
        # quality color (never green for a "needs work"), and the WORK ON row
        # shows the top-3 problem sounds.
        self._hero.set_score_row(display_text, quality.lower(), quality_color)
        badges = [entry["phoneme"] for entry in result.weak_phonemes[:3]]
        self._hero.set_badges(badges)
        # Underline the mispronounced words directly in the phrase; show the
        # how-to hint only when at least one interactive affordance exists (an
        # underlined word or a sound badge), so a clean take stays uncluttered.
        misses = self._hero.mark_miss_words(result.reference_words)
        self._hero.set_hint(bool(misses) or bool(badges))

        # Cache prosody and draw the sparklines (you vs reference).
        self._prosody.set_prosody(result.prosody or {})
        self.root.update_idletasks()  # ensure the canvases have a real width/height
        self._prosody.redraw()

        # The number itself lives in the hero card (set above via
        # set_score_row); report the scored take to the controller so it can
        # update the session tally shown in the status bar. The reference
        # self-test reaches here without a real recording - it is a pipeline
        # sanity check, not a practice attempt, so it must not touch the tally.
        if not is_self_test:
            self._cb.on_take_scored(current_phrase, tally_value, graded)
            # Record the take into the attempt history (controller-owned). The
            # colour is the resolved verdict colour; the detail keeps the coloured
            # word breakdown and the top problem sounds shown on the hero card.
            # "score" is the numeric value behind the mark (for the trend arrow),
            # "score_text" the mark exactly as the hero card shows it.
            self._cb.on_history_entry({
                "kind": "attempt",
                "phrase": current_phrase or "",
                "score": tally_value,
                "score_text": display_text,
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

    def _show_unscored_feedback(self, result: "PronunciationResult", current_phrase,
                                has_recording: bool, is_self_test: bool = False):
        """Feedback for an unscored take (``result.scored`` is False; "none" engine).

        Keeps everything that does not depend on scoring - the prosody charts
        (still computed by the host from the raw waveforms), the neutral hero-card
        read-out and re-enabling the controls - and adds a neutral "--" row to the
        attempt history, so this mode never pretends the take was judged.
        """
        self._hero.face.set_expression("neutral")
        # Hero card: neutral "scoring off" read-out - no number, no badges, no
        # word underlines and no how-to hint (nothing was judged to act on).
        self._hero.set_score_row("--", "scoring off", THEME["text_dim"])
        self._hero.set_badges([])
        self._hero.set_hint(False)
        # A neutral history row (no score, no trend). The reference self-test is a
        # pipeline check, not a practice take, so it stays out of the history for
        # the same reason it stays out of the session tally.
        if not is_self_test:
            self._cb.on_history_entry({
                "kind": "unscored",
                "phrase": current_phrase or "",
            })

        # Prosody still works without scoring; cache and draw it as usual.
        self._prosody.set_prosody(result.prosody or {})
        self.root.update_idletasks()  # ensure the canvases have a real width/height
        self._prosody.redraw()

        # Unscored takes ("none" engine) do not contribute to the session tally,
        # so the hero card's progress ring is left untouched here.

        self._set_actions(generate=True, reference=True, user=has_recording, test=True)
        self.draw_mic_button("idle")

        self.update_status("Recorded (scoring off)", THEME["text_dim"])
