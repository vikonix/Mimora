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
    on_prosody_charts_toggled: Callable[[], None]
    on_test_reference: Callable[[], None]
    play_user_recording: Callable[[], None]
    play_reference: Callable[[], None]
    play_reference_slow: Callable[[], None]
    on_generate_phrase: Callable[[], None]
    # Click on an underlined ("miss") word inside the hero phrase: the
    # controller synthesizes that single word and plays it slowly.
    on_word_clicked: Callable[[str], None]

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

    def _make_button(self, parent, text, command, width=None):
        """Create a consistently styled themed button.

        ``width`` (in text characters) is optional; pass it to give a group of
        buttons a uniform width so they line up regardless of label length.
        """
        button = tk.Button(parent, text=text, command=command,
                           font=(FONT_FAMILY, 10, "bold"),
                           bg=THEME["bg_button"], fg=THEME["text_button"],
                           activebackground=THEME["bg_button_active"], activeforeground=THEME["text"],
                           bd=0, padx=12, pady=6, cursor="hand2",
                           disabledforeground=THEME["text_disabled"])
        if width is not None:
            button.config(width=width)
        return button

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

        self.stats_label = tk.Label(self.status_bar,
                                    text="Score: - (-)",
                                    font=(FONT_FAMILY, 9), fg=THEME["text_dim"], bg=THEME["bg_panel"])
        self.stats_label.pack(side=tk.RIGHT, padx=15, pady=4)

        # 3. Bottom control panel: the mic flanked by its two phrase-level
        # actions - Reference (replay the example) on the left, New phrase
        # (generate the next one) on the right - with the instruction line below.
        control_frame = tk.Frame(self.root, bg=THEME["bg_main"])
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=(0, 10))

        # Equal-width buttons keep the mic visually centred between them.
        action_btn_width = 14

        # Horizontal row: [Reference] [mic] [New phrase].
        mic_row = tk.Frame(control_frame, bg=THEME["bg_main"])
        mic_row.pack()

        self.ref_btn = self._make_button(
            mic_row, "▶ Reference", self._cb.play_reference, width=action_btn_width)
        self.ref_btn.pack(side=tk.LEFT, padx=(0, 4))
        self.ref_btn.config(state=tk.DISABLED)

        # One-tap slow replay of the same reference, one step (0.1x) slower than
        # the normal speed. Gated together with the Reference button (see
        # _set_actions); the normal speed lives in Settings.
        self.slow_btn = self._make_button(mic_row, "🐢", self._cb.play_reference_slow)
        self.slow_btn.pack(side=tk.LEFT, padx=(0, 15))
        self.slow_btn.config(state=tk.DISABLED)

        self.btn_canvas = tk.Canvas(mic_row, width=100, height=100, bg=THEME["bg_main"],
                                    highlightthickness=0, cursor="hand2")
        self.btn_canvas.pack(side=tk.LEFT, pady=5)
        self.btn_canvas.bind("<ButtonPress-1>", lambda e: self._cb.on_gui_btn_press())
        self.btn_canvas.bind("<ButtonRelease-1>", lambda e: self._cb.on_gui_btn_release())
        self.draw_mic_button("loading")

        self.generate_btn = self._make_button(
            mic_row, "🎲 New phrase", self._cb.on_generate_phrase, width=action_btn_width)
        self.generate_btn.pack(side=tk.LEFT, padx=(15, 0))
        self.generate_btn.config(state=tk.DISABLED)

        self.instruction_label = tk.Label(control_frame, text="Loading components...",
                                          font=(FONT_FAMILY, 10), fg=THEME["text_dim"], bg=THEME["bg_main"])
        self.instruction_label.pack(pady=5)

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

        # The face lives in the score row now (verdict indicator + talking
        # mouth), not in the prosody block. Same single FaceWidget instance the
        # controller drives via face_play_levels/face_rest; ~78px as mocked up.
        self.face = FaceWidget(score_row, size=78, bg=THEME["bg_card"],
                               face_color=THEME["face"], face_outline=THEME["border"],
                               eye_color=THEME["eyes"], mouth_color=THEME["mouth"])
        self.face.set_expression(":)")  # waiting state
        # Not packed yet: toggle_face() applies the persisted "Face" setting
        # once its checkbox variable exists (created with the prosody header
        # below, which still owns the toggle until the prosody-stage rework).

        # 5b. Prosody panel - pitch (F0) and energy sparklines, you vs reference.
        prosody_frame = tk.Frame(self.root, bg=THEME["bg_main"])
        prosody_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(0, 8))

        prosody_header = tk.Frame(prosody_frame, bg=THEME["bg_main"])
        prosody_header.pack(fill=tk.X)
        tk.Label(prosody_header, text="Prosody", font=(FONT_FAMILY, 9, "bold"),
                 fg=THEME["accent"], bg=THEME["bg_main"]).pack(side=tk.LEFT)
        # "Face" toggles the articulation face (now living in the hero card's
        # score row; the checkbox stays here until the prosody-stage rework).
        # Packed first among the right-aligned items so it sits at the far
        # right, after the legend.
        self.show_face = tk.BooleanVar(value=config.SHOW_FACE)
        tk.Checkbutton(prosody_header, text="Face", variable=self.show_face,
                       command=self._cb.on_show_face_toggled,
                       font=(FONT_FAMILY, 8), fg=THEME["text_dim"], bg=THEME["bg_main"],
                       activebackground=THEME["bg_main"], activeforeground=THEME["text_dim"],
                       selectcolor=THEME["bg_panel"], bd=0, highlightthickness=0,
                       cursor="hand2").pack(side=tk.RIGHT, padx=(12, 0))
        tk.Label(prosody_header, text="● reference", font=(FONT_FAMILY, 8),
                 fg=THEME["reference"], bg=THEME["bg_main"]).pack(side=tk.RIGHT, padx=(8, 0))
        tk.Label(prosody_header, text="● you", font=(FONT_FAMILY, 8),
                 fg=THEME["info"], bg=THEME["bg_main"]).pack(side=tk.RIGHT)

        # Each chart title doubles as a checkbox: unchecking hides that chart's
        # canvas to free vertical space; checking restores it in place
        # (see toggle_prosody_charts). Initial state is the persisted setting.
        # The Pitch title sits above the row so the face panel beside the charts
        # lines up with the chart areas, not the labels.
        self.show_f0 = tk.BooleanVar(value=config.SHOW_PITCH_CHART)
        self.f0_check = self._make_chart_checkbox(
            prosody_frame, "Pitch - intonation (semitones vs your median)", self.show_f0)
        self.f0_check.pack(anchor=tk.W)

        # Body row: the charts (the face moved to the hero card's score row).
        prosody_body = tk.Frame(prosody_frame, bg=THEME["bg_main"])
        prosody_body.pack(fill=tk.X)

        charts_frame = tk.Frame(prosody_body, bg=THEME["bg_main"])
        charts_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.f0_canvas = tk.Canvas(charts_frame, height=46, bg=THEME["bg_panel"],
                                   highlightthickness=1, highlightbackground=THEME["border"])
        self.f0_canvas.pack(fill=tk.X, pady=(0, 4))

        self.show_energy = tk.BooleanVar(value=config.SHOW_ENERGY_CHART)
        self.en_check = self._make_chart_checkbox(
            charts_frame, "Energy - stress pattern", self.show_energy)
        self.en_check.pack(anchor=tk.W)
        self.en_canvas = tk.Canvas(charts_frame, height=46, bg=THEME["bg_panel"],
                                   highlightthickness=1, highlightbackground=THEME["border"])
        self.en_canvas.pack(fill=tk.X)

        # Reading hint: horizontal axis is time (stretched to equal width for both),
        # so the goal is matching the *shape* of the reference, not exact overlap.
        tk.Label(prosody_frame,
                  text="Time runs (stretched to equal width). Aim to match the reference shape.",
                 font=(FONT_FAMILY, 8), fg=THEME["text_dim"], bg=THEME["bg_main"],
                 wraplength=540, justify=tk.LEFT).pack(anchor=tk.W, pady=(3, 0))

        # fill=X canvases change width on resize, so redraw from the cached prosody.
        self.f0_canvas.bind("<Configure>", lambda e: self._redraw_prosody())
        self.en_canvas.bind("<Configure>", lambda e: self._redraw_prosody())

        # Late-apply the persisted visibility settings: hide whichever chart is
        # off, and pack the face (built unpacked in the hero card's score row)
        # if the "Face" setting is on.
        self.toggle_prosody_charts()
        self.toggle_face()

        # 6. Action row directly under the result window: replays the user's own
        # attempt. Reference and New phrase live down by the mic; the diagnostic
        # self-test has no visible button - it stays reachable via the 't' hotkey,
        # gated by self._test_enabled (see is_test_enabled / _set_actions).
        #
        # Packed BEFORE the feedback panel below and at side=BOTTOM on purpose:
        # the feedback panel uses expand=True with a large default height and
        # would otherwise claim all the space, clipping a later-packed row to
        # zero height. Reserving this row first keeps it visible just above the
        # mic control_frame, so it sits right under the result window.
        action_frame = tk.Frame(self.root, bg=THEME["bg_main"])
        action_frame.pack(side=tk.BOTTOM, padx=20, pady=(0, 8))

        # Self-test enabled state. The visible button was removed, but the
        # feature lives on via the 't' hotkey; the enter_* states still toggle this
        # through _set_actions(test=...), and is_test_enabled gates the hotkey.
        self._test_enabled = False

        self.user_btn = self._make_button(
            action_frame, "▶ My recording", self._cb.play_user_recording, width=action_btn_width)
        self.user_btn.pack()
        self.user_btn.config(state=tk.DISABLED)

        # 7. Feedback log (fills remaining space). Packed AFTER action_frame so
        # its expand=True only consumes space left over above the button row.
        feedback_frame = tk.Frame(self.root, bg=THEME["bg_main"])
        feedback_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=(8, 16))

        self.feedback_display = scrolledtext.ScrolledText(
            feedback_frame, bg=THEME["bg_panel"], fg=THEME["text"], insertbackground=THEME["text_bright"],
            font=(FONT_FAMILY, 11), wrap=tk.WORD, bd=0,
            highlightthickness=1, highlightbackground=THEME["border"], highlightcolor=THEME["accent"],
            padx=15, pady=15, spacing2=4, spacing3=8)
        self.feedback_display.pack(fill=tk.BOTH, expand=True)
        self.feedback_display.configure(state=tk.DISABLED)

        self.feedback_display.tag_configure("system", foreground=THEME["text_muted"], font=(FONT_FAMILY, 10, "italic"))
        self.feedback_display.tag_configure("good", foreground=THEME["good"], font=(FONT_FAMILY, 11, "bold"))
        self.feedback_display.tag_configure("bad", foreground=THEME["bad"], font=(FONT_FAMILY, 11, "bold"))
        # "ok" == acceptable word in the three-level phoneme highlight: light grey,
        # sitting between green (good) and red (bad).
        self.feedback_display.tag_configure("ok", foreground=THEME["text_dim"], font=(FONT_FAMILY, 11, "bold"))
        self.feedback_display.tag_configure("label", foreground=THEME["text_dim"], font=(FONT_FAMILY, 10))
        self.feedback_display.tag_configure("text", foreground=THEME["text_emph"], font=(FONT_FAMILY, 11))
        # Monospace tag for phoneme strings so they align and read clearly.
        self.feedback_display.tag_configure("mono", foreground=THEME["text_dim"], font=("Consolas", 10))
        # Amber tag for "no word errors but score still low" guidance.
        self.feedback_display.tag_configure("warn", foreground=THEME["warn"], font=(FONT_FAMILY, 11))
        # Red tag for error messages (append_error_msg); non-bold so the score
        # line ("bad" tag) still stands out above errors.
        self.feedback_display.tag_configure("error", foreground=THEME["bad"], font=(FONT_FAMILY, 10))

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
        elif not collapsed and not shown:
            # Restore the build order: the editor is the last child of
            # source_frame, and the buttons pack after the caption (LEFT packing
            # preserves their order).
            self.source_text.pack(fill=tk.X, pady=4)
            self._paste_btn.pack(side=tk.LEFT, padx=(10, 0))
            self._clear_btn.pack(side=tk.LEFT, padx=(6, 0))

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
        clean takes show no badges). Clicking a badge will speak an example
        word - wired in the sounds-feedback stage; until then the buttons are
        display-only.
        """
        for child in self.badges_frame.winfo_children():
            child.destroy()
        self.workon_caption.configure(text="WORK ON" if phonemes else "")
        for phoneme in phonemes:
            tk.Button(self.badges_frame, text=f"/{phoneme}/",
                      font=(FONT_FAMILY, FONT_SIZE_BODY),
                      bg=THEME["bg_accent"], fg=THEME["text_accent"],
                      activebackground=THEME["bg_button"],
                      activeforeground=THEME["text_button"],
                      bd=0, padx=10, pady=2, cursor="hand2",
                      highlightthickness=1,
                      highlightbackground=THEME["accent"]).pack(side=tk.LEFT,
                                                                padx=(0, 6))

    def _reset_score_row(self):
        """Empty state: no take scored yet (or a new recording just started)."""
        self.score_num_label.configure(text="--", fg=THEME["text_dim"])
        self.score_verdict_label.configure(text="record to get a score",
                                           fg=THEME["text_dim"])
        self._set_badges([])

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

    def set_show_pitch(self, flag: bool):
        self.show_f0.set(bool(flag))

    def set_show_energy(self, flag: bool):
        self.show_energy.set(bool(flag))

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

    def get_show_pitch(self) -> bool:
        return bool(self.show_f0.get())

    def get_show_energy(self) -> bool:
        return bool(self.show_energy.get())

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
        """Show an error in the feedback panel (in red) and log it.

        Errors like "Audio is too short" or "LM Studio is offline" must reach
        the user, not only the log file.
        """
        logging.warning(f"[Error] {text}")
        self.feedback_display.configure(state=tk.NORMAL)
        self.feedback_display.insert(tk.END, f"{text}\n", "error")
        self.feedback_display.configure(state=tk.DISABLED)
        self.feedback_display.see(tk.END)

    def update_status(self, text: str, color: str = THEME["text_dim"]):
        self.status_label.configure(text=text, fg=color)

    def update_instruction(self, text: str):
        self.instruction_label.configure(text=text)

    def update_score_stats(self, score: float, bucket: int = -1,
                           color: Optional[str] = None):
        """Show the user-facing score and its 0-5 bucket: ``Score: 92 (4)``.

        ``score`` is the engine's user-facing number (the phoneme engine's
        calibrated percent, the acoustic engine's raw 0-100). ``bucket`` is the
        0-5 grade, shown as a dash when absent (``bucket == -1``). ``color`` is
        the already-resolved quality colour so this read-out matches the status
        line and face exactly; when omitted it falls back to the raw-score band.
        """
        bucket_text = str(bucket) if bucket >= 0 else "-"
        if color is None:
            _, color = self._quality_label(score)
        self.stats_label.configure(text=f"Score: {score:.0f} ({bucket_text})", fg=color)

    def clear_previous_result(self):
        """Reset the per-take indicators before a new recording starts.

        Resets the Score read-out in the status bar to its placeholder, erases
        both prosody charts and resets the face to its neutral waiting smile, so
        a stale single-value indicator can never be mistaken for the take in
        progress. The feedback panel is deliberately NOT cleared here: it is a
        running history of every attempt, appended to by show_feedback /
        append_error_msg and kept across takes. The status *line* is set
        separately by the caller (enter_recording -> "Recording...").
        """
        self.stats_label.configure(text="Score: - (-)", fg=THEME["text_dim"])
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

    def _make_chart_checkbox(self, parent, text, variable):
        """Create a themed chart-title checkbox that toggles its chart's visibility.

        The command goes through the controller (on_prosody_charts_toggled in
        main.py) so the new state is also persisted to settings.json.
        """
        return tk.Checkbutton(parent, text=text, variable=variable,
                              command=self._cb.on_prosody_charts_toggled,
                              font=(FONT_FAMILY, 8), fg=THEME["text_dim"], bg=THEME["bg_main"],
                              activebackground=THEME["bg_main"], activeforeground=THEME["text_dim"],
                              selectcolor=THEME["bg_panel"], bd=0, highlightthickness=0,
                              cursor="hand2")

    def toggle_prosody_charts(self):
        """Show/hide each prosody canvas to match its title checkbox."""
        # Return focus to the window so the spacebar record toggle keeps working
        # (a focused checkbox would otherwise capture the spacebar and toggle).
        self.root.focus_set()
        # Pitch chart leads the charts column (its title is above the row), so
        # re-pack it before the Energy title to preserve order.
        f0_shown = self.f0_canvas.winfo_manager() == "pack"
        if self.show_f0.get() and not f0_shown:
            self.f0_canvas.pack(fill=tk.X, pady=(0, 4), before=self.en_check)
        elif not self.show_f0.get() and f0_shown:
            self.f0_canvas.pack_forget()
        # Energy chart sits right after its own title.
        en_shown = self.en_canvas.winfo_manager() == "pack"
        if self.show_energy.get() and not en_shown:
            self.en_canvas.pack(fill=tk.X, after=self.en_check)
        elif not self.show_energy.get() and en_shown:
            self.en_canvas.pack_forget()

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
                      has_recording: bool = True):
        # An unscored result (the "none" engine) has no verdict to present:
        # score/passed carry no meaning, so render the neutral read-out instead
        # of a quality label that would pretend the take was judged.
        if not getattr(result, "scored", True):
            self._show_unscored_feedback(result, current_phrase, has_recording)
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
        self._set_badges([entry["phoneme"] for entry in result.weak_phonemes[:3]])
        self.feedback_display.configure(state=tk.NORMAL)
        # The numeric score moved out of this panel: the raw score/bucket now lives
        # in the status bar and the qualitative label. This panel keeps only the
        # actionable Phrase/Heard breakdown.
        # First line: the target phrase, highlighting what was said well (green)
        # vs mispronounced (red). Driven by the engine-neutral reference_words
        # tags; falls back to the raw phrase if an engine left them empty.
        # Separate consecutive results with a blank line, but not before the
        # first one. Each block leaves its last line unterminated, so the
        # separator needs two newlines: one to close that line, one for the blank
        # row. (A single leading "\n" only closed the previous line, which is why
        # it appeared to fire just once - on the initially empty panel.)
        if self.feedback_display.get("1.0", "end-1c"):
            self.feedback_display.insert(tk.END, "\n\n")
        self.feedback_display.insert(tk.END, "Phrase: ", "label")
        if result.reference_words:
            for entry in result.reference_words:
                # Three-level colour when the engine supplies a "level"
                # (good/ok/bad); fall back to the boolean "correct" (acoustic engine).
                level = entry.get("level")
                if level in ("good", "ok", "bad"):
                    tag = {"good": "good", "ok": "ok", "bad": "bad"}[level]
                else:
                    tag = "good" if entry.get("correct") else "bad"
                self.feedback_display.insert(tk.END, entry["word"] + " ", tag)
        else:
            for token in (current_phrase or "-").split():
                self.feedback_display.insert(tk.END, token + " ", "text")
        self.feedback_display.insert(tk.END, "\n")

        # Second line: the few phonemes worth working on, instead of the
        # full recognised transcription -- the panel now says "what to fix", not
        # "what was heard". The phoneme engine supplies weak_phonemes (worst
        # first); the acoustic engine has no per-phone breakdown, so it keeps the
        # old unit-by-unit "Heard" readout as a meaningful fallback.
        if result.weak_phonemes:
            self.feedback_display.insert(tk.END, "Work on: ", "label")
            for entry in result.weak_phonemes:
                self.feedback_display.insert(tk.END, f"/{entry['phoneme']}/ ", "bad")
        elif getattr(result, "bucket", -1) >= 0:
            # Phoneme engine with no weak phones -> a clean read.
            self.feedback_display.insert(tk.END, "Work on: ", "label")
            self.feedback_display.insert(tk.END, "nothing major - nice work ✓", "good")
        else:
            # Acoustic engine fallback: the original recognised-units "Heard" line.
            self.feedback_display.insert(tk.END, "Heard: ", "label")
            # "matches the target" only when there are no mispronounced words AND the
            # take passed -- so a low score can no longer sit next to a "matches ✓".
            if not result.word_diff and result.passed:
                self.feedback_display.insert(tk.END, "matches the target ✓", "good")
            elif result.recognized_units:
                for entry in result.recognized_units:
                    tag = "good" if entry.get("correct") else "bad"
                    self.feedback_display.insert(tk.END, entry["unit"] + " ", tag)
            else:
                self.feedback_display.insert(tk.END, f"{result.transcription or '-'}", "text")

        self.feedback_display.configure(state=tk.DISABLED)
        self.feedback_display.see(tk.END)

        # Cache prosody and draw the sparklines (you vs reference).
        self._last_prosody = result.prosody or {}
        self.root.update_idletasks()  # ensure the canvases have a real width/height
        self._redraw_prosody()

        self.update_score_stats(display_score, bucket, quality_color)

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
                                has_recording: bool):
        """Feedback for an unscored take (``result.scored`` is False; "none" engine).

        Keeps everything that does not depend on scoring - the phrase line, the
        prosody charts (still computed by the host from the raw waveforms) and
        re-enabling the controls - and shows neutral placeholders where a verdict
        would go, so this mode never pretends the take was judged.
        """
        self.face.set_expression("neutral")
        # Hero card: neutral "scoring off" read-out - no number, no badges.
        self._set_score_row("--", "scoring off", THEME["text_dim"])
        self._set_badges([])
        self.feedback_display.configure(state=tk.NORMAL)
        # Same result-separator convention as show_feedback (see the comment there).
        if self.feedback_display.get("1.0", "end-1c"):
            self.feedback_display.insert(tk.END, "\n\n")
        self.feedback_display.insert(tk.END, "Phrase: ", "label")
        for token in (current_phrase or "-").split():
            self.feedback_display.insert(tk.END, token + " ", "text")
        self.feedback_display.insert(tk.END, "\n")
        self.feedback_display.insert(tk.END, "Heard: ", "label")
        self.feedback_display.insert(tk.END, "scoring is off - compare the takes by ear", "text")
        self.feedback_display.configure(state=tk.DISABLED)
        self.feedback_display.see(tk.END)

        # Prosody still works without scoring; cache and draw it as usual.
        self._last_prosody = result.prosody or {}
        self.root.update_idletasks()  # ensure the canvases have a real width/height
        self._redraw_prosody()

        # Neutral score placeholder (same style clear_previous_result uses).
        self.stats_label.configure(text="Score: - (-)", fg=THEME["text_dim"])

        self._set_actions(generate=True, reference=True, user=has_recording, test=True)
        self.draw_mic_button("idle")

        self.update_status("Recorded (scoring off)", THEME["text_dim"])
        self.update_instruction("Compare by ear: ▶ Reference and ▶ My recording, "
                                "then repeat or click 'New phrase' to continue.")
