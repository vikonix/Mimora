# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""View layer for the Mimora pronunciation trainer.

This module holds the UI as a standalone, passive :class:`TrainerView`, composed
into the controller (``self.view``) in ``main.py`` rather than inherited. The
view owns every widget and renders state; the controller owns the application
logic. Both directions across the boundary are explicit, typed contracts — the
view shares no implicit namespace with the controller:

* view → controller: widget callbacks call only ``self._cb.<handler>`` on a
  :class:`ViewCallbacks` of plain callables passed in at construction. The view
  never references the controller object, so the two sides form no cycle and the
  view can be exercised with stand-in callbacks.
* controller → view: the controller drives the UI through named *intent*
  methods (``enter_recording``, ``enter_phrase_ready``, ``show_feedback`` …) and
  small read accessors (``get_practice_text`` …). It never touches a widget or a
  status/colour string directly — every UI state and its copy live here.
"""
import logging
import platform
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from mimora import config, prosody_utils
from mimora.face_widget import FaceWidget

# Resolved UI color palette (semantic name -> hex), selected by the
# "color_theme" setting in settings.json; see config.py.
THEME = config.THEME

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
    on_open_practice_text: Callable[[], None]
    quit_app: Callable[[], None]
    on_gui_btn_press: Callable[[], None]
    on_gui_btn_release: Callable[[], None]
    on_user_name_changed: Callable[..., None]
    on_length_changed: Callable[..., None]
    on_translation_language_changed: Callable[..., None]
    on_voice_changed: Callable[..., None]
    on_speed_changed: Callable[..., None]
    on_show_face_toggled: Callable[[], None]
    on_prosody_charts_toggled: Callable[[], None]
    on_test_reference: Callable[[], None]
    play_user_recording: Callable[[], None]
    play_reference: Callable[[], None]
    on_generate_phrase: Callable[[], None]

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

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
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
        # The popdown list is a classic Tk Listbox, themed via the option DB.
        self.root.option_add("*TCombobox*Listbox.background", THEME["bg_panel"])
        self.root.option_add("*TCombobox*Listbox.foreground", THEME["text_accent"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", THEME["bg_accent_active"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", THEME["text_bright"])

    def _make_button(self, parent, text, command):
        """Create a consistently styled themed button."""
        return tk.Button(parent, text=text, command=command,
                         font=(FONT_FAMILY, 10, "bold"),
                         bg=THEME["bg_accent"], fg=THEME["text_accent"],
                         activebackground=THEME["bg_accent_active"], activeforeground=THEME["text_bright"],
                         bd=0, padx=12, pady=6, cursor="hand2",
                         disabledforeground=THEME["text_disabled"])

    def build_ui(self):
        # Window background (shows through wherever no widget covers it). The
        # view owns the window chrome so the controller need not know the palette.
        self.root.configure(bg=THEME["bg_main"])

        # 0. Menu bar (cross-platform tk.Menu). On Windows it is drawn by the
        # OS, so it does not follow the app's dark theme — that is expected.
        # The handler (on_open_practice_text) lives in the controller (main.py).
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open practice text…",
                              command=self._cb.on_open_practice_text)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._cb.quit_app)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

        # 1. Header
        header_frame = tk.Frame(self.root, bg=THEME["bg_main"], height=60)
        header_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)

        tk.Label(header_frame, text="MIMORA • Pronunciation",
                 font=(FONT_FAMILY, 16, "bold"), fg=THEME["accent"], bg=THEME["bg_main"]).pack(side=tk.LEFT)

        tk.Label(header_frame, text=config.TARGET_LANGUAGE,
                 font=(FONT_FAMILY, 9, "bold"), fg=THEME["text_dim"], bg=THEME["bg_panel"],
                 padx=10, pady=4, bd=0).pack(side=tk.RIGHT)

        # 2. Status bar (absolute bottom)
        self.status_bar = tk.Frame(self.root, bg=THEME["bg_panel"], height=30)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = tk.Label(self.status_bar, text="Status: Starting...",
                                     font=(FONT_FAMILY, 9), fg=THEME["ready"], bg=THEME["bg_panel"])
        self.status_label.pack(side=tk.LEFT, padx=15, pady=4)

        self.stats_label = tk.Label(self.status_bar,
                                    text="Score: - (-)",
                                    font=(FONT_FAMILY, 9), fg=THEME["text_dim"], bg=THEME["bg_panel"])
        self.stats_label.pack(side=tk.RIGHT, padx=15, pady=4)

        # 3. Bottom control panel (mic + instruction + replay buttons)
        control_frame = tk.Frame(self.root, bg=THEME["bg_main"])
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=10)

        self.btn_canvas = tk.Canvas(control_frame, width=100, height=100, bg=THEME["bg_main"],
                                    highlightthickness=0, cursor="hand2")
        self.btn_canvas.pack(pady=5)
        self.btn_canvas.bind("<ButtonPress-1>", lambda e: self._cb.on_gui_btn_press())
        self.btn_canvas.bind("<ButtonRelease-1>", lambda e: self._cb.on_gui_btn_release())
        self.draw_mic_button("loading")

        self.instruction_label = tk.Label(control_frame, text="Loading components...",
                                          font=(FONT_FAMILY, 10), fg=THEME["text_dim"], bg=THEME["bg_main"])
        self.instruction_label.pack(pady=5)

        # 4. Source text panel (editable)
        source_frame = tk.Frame(self.root, bg=THEME["bg_main"])
        source_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(0, 5))

        # Header row above the practice text: caption on the left, the user-name
        # field on the right edge. The name is persisted to settings.json when
        # editing finishes (see on_user_name_changed).
        text_header = tk.Frame(source_frame, bg=THEME["bg_main"])
        text_header.pack(fill=tk.X)
        tk.Label(text_header, text="Practice text (edit freely):",
                 font=(FONT_FAMILY, 9, "bold"), fg=THEME["text_dim"], bg=THEME["bg_main"]).pack(side=tk.LEFT)

        self.user_name_var = tk.StringVar(value=config.USER_NAME)
        self.user_name_entry = tk.Entry(
            text_header, textvariable=self.user_name_var, width=14,
            font=(FONT_FAMILY, 9), bg=THEME["bg_accent"], fg=THEME["text_accent"],
            insertbackground=THEME["text_bright"], bd=0, highlightthickness=1,
            highlightbackground=THEME["border"], highlightcolor=THEME["accent"])
        self.user_name_entry.pack(side=tk.RIGHT)
        tk.Label(text_header, text="Name:", font=(FONT_FAMILY, 9),
                 fg=THEME["text_dim"], bg=THEME["bg_main"]).pack(side=tk.RIGHT, padx=(0, 6))
        # Save on focus loss; Enter just drops focus (which triggers the save
        # and returns the spacebar to record-toggle duty).
        self.user_name_entry.bind("<FocusOut>", self._cb.on_user_name_changed)
        self.user_name_entry.bind("<Return>", lambda e: self.root.focus_set())

        self.source_text = scrolledtext.ScrolledText(
            source_frame, bg=THEME["bg_panel"], fg=THEME["text"], insertbackground=THEME["text_bright"],
            font=(FONT_FAMILY, 10), wrap=tk.WORD, bd=0, height=6,
            highlightthickness=1, highlightbackground=THEME["border"], highlightcolor=THEME["accent"],
            padx=10, pady=8)
        self.source_text.pack(fill=tk.X, pady=4)

        # Selector row: translation language, phrase length, voice and
        # reference-playback speed share a single line. Labels are kept terse and
        # padding tight so all four fit the fixed 600px window width.
        selectors_frame = tk.Frame(source_frame, bg=THEME["bg_main"])
        selectors_frame.pack(anchor=tk.E, pady=(2, 0))

        # Translation-language selector (leftmost). No caption — the value itself
        # ("Russian", "Spanish", …) names the language; the empty first choice
        # means "translation off". Selecting a language shows the translation
        # panel under the phrase card (see refresh_translation_ui); the panel and
        # the extra LLM work stay off until a language is picked. Disabled in
        # "Few words" mode, where fragments are not translated.
        self.translation_var = tk.StringVar(value=config.TRANSLATION_LANGUAGE)
        self.translation_selector = ttk.Combobox(
            selectors_frame, textvariable=self.translation_var, state="readonly",
            width=10, values=config.TRANSLATION_LANGUAGES)
        self.translation_selector.pack(side=tk.LEFT, padx=(0, 10))
        self.translation_selector.bind(
            "<<ComboboxSelected>>", self._cb.on_translation_language_changed)

        # Phrase-length selector. "Few words" requests a short fragment instead
        # of a full sentence; changing it regenerates the phrase (see
        # on_length_changed).
        tk.Label(selectors_frame, text="Phrase length:", font=(FONT_FAMILY, 9),
                 fg=THEME["text_dim"], bg=THEME["bg_main"]).pack(side=tk.LEFT, padx=(0, 6))
        self.length_var = tk.StringVar(
            value=LENGTH_FEW_WORDS if config.PHRASE_LENGTH == "fragment" else LENGTH_FULL)
        self.length_selector = ttk.Combobox(
            selectors_frame, textvariable=self.length_var, state="readonly",
            width=12, values=(LENGTH_FULL, LENGTH_FEW_WORDS))
        self.length_selector.pack(side=tk.LEFT, padx=(0, 10))
        self.length_selector.bind("<<ComboboxSelected>>", self._cb.on_length_changed)

        # Voice selector for the reference speech. Changing it regenerates the
        # phrase (see on_voice_changed) so the new voice is heard right away.
        tk.Label(selectors_frame, text="Voice:", font=(FONT_FAMILY, 9),
                 fg=THEME["text_dim"], bg=THEME["bg_main"]).pack(side=tk.LEFT, padx=(0, 6))
        self.voice_var = tk.StringVar(value=config.KOKORO_VOICE)
        self.voice_selector = ttk.Combobox(
            selectors_frame, textvariable=self.voice_var, state="readonly",
            width=12, values=tuple(config.KOKORO_VOICES))
        self.voice_selector.pack(side=tk.LEFT, padx=(0, 10))
        self.voice_selector.bind("<<ComboboxSelected>>", self._cb.on_voice_changed)

        # Lower values slow the reference playback (see play_reference). Stored as
        # the displayed label and parsed back to a float by _selected_speed().
        tk.Label(selectors_frame, text="Speed:", font=(FONT_FAMILY, 9),
                 fg=THEME["text_dim"], bg=THEME["bg_main"]).pack(side=tk.LEFT, padx=(0, 6))
        # Options come from config so the persisted value is always one of them.
        self.playback_speed = tk.StringVar(value=f"{config.REFERENCE_SPEED:.1f}×")
        self.speed_selector = ttk.Combobox(
            selectors_frame, textvariable=self.playback_speed, state="readonly",
            width=5, values=tuple(f"{s:.1f}×" for s in config.REFERENCE_SPEED_CHOICES))
        self.speed_selector.pack(side=tk.LEFT)
        # Changing the speed replays the reference so the difference is heard
        # immediately (see on_speed_changed).
        self.speed_selector.bind("<<ComboboxSelected>>", self._cb.on_speed_changed)

        # 5. Current phrase card
        self.phrase_frame = tk.Frame(self.root, bg=THEME["bg_panel"])
        self.phrase_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=5)

        self.phrase_label = tk.Label(self.phrase_frame, text="—", font=(FONT_FAMILY, 15, "bold"),
                                     fg=THEME["info"], bg=THEME["bg_panel"], wraplength=520, justify=tk.LEFT)
        # Match the translation card's vertical padding so the two panels are the
        # same height (the phrase still reads larger via its bigger, bold font).
        self.phrase_label.pack(anchor=tk.W, padx=12, pady=(8, 8))
        # Right-click the phrase to copy it (independently of the translation).
        self._bind_copy_menu(self.phrase_label)

        # 5a. Translation card — a SEPARATE panel below the phrase card (its own
        # frame, not a sub-section of it), shown only when a translation language
        # is selected. Same panel background as the phrase card; the
        # small gap between the two (their pack pady) reads as two independent
        # cards. Dimmer and smaller text so the phrase stays the focus. The frame
        # is packed/unpacked right after the phrase card by refresh_translation_ui();
        # "—" stands in until a translation arrives with the next phrase.
        self.translation_frame = tk.Frame(self.root, bg=THEME["bg_panel"])
        self.translation_label = tk.Label(
            self.translation_frame, text="—", font=(FONT_FAMILY, 11),
            fg=THEME["text_dim"], bg=THEME["bg_panel"], wraplength=520, justify=tk.LEFT)
        self.translation_label.pack(anchor=tk.W, padx=12, pady=(8, 8))
        # Right-click the translation to copy it (independently of the phrase).
        self._bind_copy_menu(self.translation_label)
        # Force the readonly combobox to render its persisted value: a readonly
        # ttk.Combobox does not always paint the initial textvariable value until
        # the user opens the list, which made a loaded language look unselected
        # while its panel was already shown. Then reconcile panel visibility.
        self.translation_selector.set(config.TRANSLATION_LANGUAGE)
        # The translation frame is not packed yet — refresh_translation_ui()
        # decides based on the selected language and phrase-length mode.
        self.refresh_translation_ui()

        # 5b. Prosody panel — pitch (F0) and energy sparklines, you vs reference.
        prosody_frame = tk.Frame(self.root, bg=THEME["bg_main"])
        prosody_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(0, 5))

        prosody_header = tk.Frame(prosody_frame, bg=THEME["bg_main"])
        prosody_header.pack(fill=tk.X)
        tk.Label(prosody_header, text="Prosody", font=(FONT_FAMILY, 9, "bold"),
                 fg=THEME["accent"], bg=THEME["bg_main"]).pack(side=tk.LEFT)
        # "Face" toggles the articulation panel on the right. Packed first among
        # the right-aligned items so it sits at the far right, after the legend.
        self.show_face = tk.BooleanVar(value=config.SHOW_FACE)
        tk.Checkbutton(prosody_header, text="Face", variable=self.show_face,
                       command=self._cb.on_show_face_toggled,
                       font=(FONT_FAMILY, 8), fg=THEME["text_muted"], bg=THEME["bg_main"],
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
            prosody_frame, "Pitch — intonation (semitones vs your median)", self.show_f0)
        self.f0_check.pack(anchor=tk.W)

        # Body row: charts on the left (flexible width), face panel on the right.
        prosody_body = tk.Frame(prosody_frame, bg=THEME["bg_main"])
        prosody_body.pack(fill=tk.X)

        charts_frame = tk.Frame(prosody_body, bg=THEME["bg_main"])
        charts_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.f0_canvas = tk.Canvas(charts_frame, height=46, bg=THEME["bg_panel"],
                                   highlightthickness=1, highlightbackground=THEME["border"])
        self.f0_canvas.pack(fill=tk.X, pady=(0, 4))

        self.show_energy = tk.BooleanVar(value=config.SHOW_ENERGY_CHART)
        self.en_check = self._make_chart_checkbox(
            charts_frame, "Energy — stress pattern", self.show_energy)
        self.en_check.pack(anchor=tk.W)
        self.en_canvas = tk.Canvas(charts_frame, height=46, bg=THEME["bg_panel"],
                                   highlightthickness=1, highlightbackground=THEME["border"])
        self.en_canvas.pack(fill=tk.X)

        # Face panel: a bordered box (matching the charts) spanning the row
        # height, with the articulation face centred inside. The face fill uses
        # the brightest theme colour and its features the panel colour, so it
        # reads as a white face on dark themes and a dark face on light ones.
        self.face_frame = tk.Frame(prosody_body, width=105, height=100,
                                   bg=THEME["bg_panel"],
                                   highlightthickness=1, highlightbackground=THEME["border"])
        # Fix the panel width at 105px: turn off geometry propagation so the
        # FaceWidget inside cannot stretch the frame to its own requested width.
        # The 100px height is a *minimum*: with fill=Y the panel still grows to
        # match a taller charts row, but when both charts are hidden the row
        # would otherwise collapse to a checkbox's height and shrink the face to
        # nothing, so this floor keeps the face a usable size on its own.
        # The face stays circular/centred for whatever box it gets.
        self.face_frame.pack_propagate(False)
        self.face = FaceWidget(self.face_frame, size=110, bg=THEME["bg_panel"],
                               face_color=THEME["face"], face_outline=THEME["border"],
                               eye_color=THEME["eyes"], mouth_color=THEME["mouth"])
        # Let the charts column decide the row height: a tiny requested height
        # stops the face from inflating the row, while fill=BOTH makes it fill
        # whatever height the panel gets. Its responsive rebuild keeps the face
        # circular and centred, so the panel bottom lines up with the charts.
        self.face.configure(height=1)
        self.face.pack(expand=True, fill=tk.BOTH, padx=10, pady=6)
        self.face.set_expression(":)")  # waiting state

        # Reading hint: horizontal axis is time (stretched to equal width for both),
        # so the goal is matching the *shape* of the reference, not exact overlap.
        tk.Label(prosody_frame,
                 text="Time runs left→right (stretched to equal width). Aim to match the reference shape.",
                 font=(FONT_FAMILY, 8), fg=THEME["text_muted"], bg=THEME["bg_main"],
                 wraplength=540, justify=tk.LEFT).pack(anchor=tk.W, pady=(3, 0))

        # fill=X canvases change width on resize, so redraw from the cached prosody.
        self.f0_canvas.bind("<Configure>", lambda e: self._redraw_prosody())
        self.en_canvas.bind("<Configure>", lambda e: self._redraw_prosody())

        # Both canvases and the face panel are packed above by default; hide
        # whichever the persisted checkbox state says is off.
        self.toggle_prosody_charts()
        self.toggle_face()

        # 6. Action row directly under the result window: groups every action
        # button on one line — Test diagnostic, replay of the user's recording,
        # reference replay and new-phrase generation.
        #
        # Packed BEFORE the feedback panel below and at side=BOTTOM on purpose:
        # the feedback panel uses expand=True with a large default height and
        # would otherwise claim all the space, clipping a later-packed row to
        # zero height. Reserving this row first keeps it visible just above the
        # mic control_frame, so it sits right under the result window.
        action_frame = tk.Frame(self.root, bg=THEME["bg_main"])
        action_frame.pack(side=tk.BOTTOM, padx=20, pady=(0, 5))

        # Small diagnostic button: run the reference through analysis instead
        # of a recording (it should score near 100 against itself).
        self.test_btn = tk.Button(action_frame, text="Test", command=self._cb.on_test_reference,
                                  font=(FONT_FAMILY, 8), bg=THEME["bg_panel"], fg=THEME["text_muted"],
                                  activebackground=THEME["border"], activeforeground=THEME["info"],
                                  bd=0, padx=8, pady=3, cursor="hand2",
                                  disabledforeground=THEME["text_disabled_dim"])
        self.test_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.test_btn.config(state=tk.DISABLED)

        self.user_btn = self._make_button(action_frame, "▶ My recording", self._cb.play_user_recording)
        self.user_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.user_btn.config(state=tk.DISABLED)

        self.ref_btn = self._make_button(action_frame, "▶ Reference", self._cb.play_reference)
        self.ref_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.ref_btn.config(state=tk.DISABLED)

        self.generate_btn = self._make_button(action_frame, "🎲 New phrase", self._cb.on_generate_phrase)
        self.generate_btn.pack(side=tk.LEFT)
        self.generate_btn.config(state=tk.DISABLED)

        # 7. Feedback log (fills remaining space). Packed AFTER action_frame so
        # its expand=True only consumes space left over above the button row.
        feedback_frame = tk.Frame(self.root, bg=THEME["bg_main"])
        feedback_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=5)

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
        # Outer ring: identical to draw_mic_button("recording") — fixed, red.
        self.btn_canvas.create_oval(cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer,
                                    fill="", outline=THEME["bad"], width=3)
        # Dark inner track, so the red fill is visible shrinking against it.
        self.btn_canvas.create_oval(cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner,
                                    fill=THEME["mic_recording_bg"], outline="")
        # The pulsing level indicator: a solid red disc, min radius -> inner
        # radius. No glyph on top — the disc itself is the indicator, and an
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

    def get_user_name(self) -> str:
        return self.user_name_var.get().strip()

    def get_voice(self) -> str:
        return self.voice_var.get()

    def get_length_label(self) -> str:
        return self.length_var.get()

    def get_translation_language(self) -> str:
        """Return the selected translation language label ('' when off)."""
        return self.translation_var.get()

    def set_translation(self, text: str):
        """Set the translation panel text, falling back to '—' when empty.

        '—' marks "language is on, but no translation yet" (e.g. right after the
        language was switched — the translation arrives with the next phrase).
        """
        self.translation_label.config(text=text.strip() if text and text.strip() else "—")

    # ------------------------------------------------------------------
    # Copy-to-clipboard (right-click menu on the phrase / translation)
    # ------------------------------------------------------------------
    def _bind_copy_menu(self, label: tk.Label):
        """Attach a right-click 'Copy' menu that copies ``label``'s own text.

        Phrase and translation get their own binding, so each is copied
        independently of the other.
        """
        label.bind("<Button-3>", lambda event: self._show_copy_menu(event, label))

    def _show_copy_menu(self, event, label: tk.Label):
        """Pop up a one-item 'Copy' menu for the clicked label, if it has text."""
        text = label.cget("text").strip()
        # Nothing useful to copy from an empty card or the "—" placeholder.
        if not text or text == "—":
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
        """Sync the translation panel and selector to the current UI state.

        Shows the panel whenever a language is selected — in both "Full phrase"
        and "Few words" modes (fragments are translated too). Safe to call
        repeatedly — it reconciles to the current vars without side effects.
        """
        # The language selector is always usable; translation applies to whatever
        # the next phrase is, fragment or full sentence.
        self.translation_selector.config(state="readonly")
        show = bool(self.translation_var.get())
        packed = self.translation_frame.winfo_manager() == "pack"
        if show and not packed:
            # Insert directly under the phrase card. Top gap of 3 plus the phrase
            # card's own 5px bottom gap make an ~8px interval between the panels.
            self.translation_frame.pack(side=tk.TOP, fill=tk.X, padx=20,
                                        pady=(3, 5), after=self.phrase_frame)
        elif not show and packed:
            self.translation_frame.pack_forget()

    def get_speed_label(self) -> str:
        return self.playback_speed.get()

    def is_reference_enabled(self) -> bool:
        """True when the Reference replay button is currently clickable."""
        return str(self.ref_btn["state"]) == str(tk.NORMAL)

    def is_generate_enabled(self) -> bool:
        """True when the New phrase button is currently clickable."""
        return str(self.generate_btn["state"]) == str(tk.NORMAL)

    def is_test_enabled(self) -> bool:
        """True when the Test (self-test) button is currently clickable."""
        return str(self.test_btn["state"]) == str(tk.NORMAL)

    def is_user_enabled(self) -> bool:
        """True when the My recording replay button is currently clickable."""
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
        if user is not None:
            self.user_btn.config(state=state(user))
        if test is not None:
            self.test_btn.config(state=state(test))

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
        self.phrase_label.config(text=phrase)
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
        # Drop the previous take's charts and result the moment the mic opens,
        # so the user starts each recording from a clean slate.
        self.clear_previous_result()
        self._set_actions(generate=False, reference=False, user=False, test=False)
        self.draw_mic_button("recording")
        self.update_status("Recording...", THEME["bad"])
        self.update_instruction("Speak now — recording stops automatically (press again to stop).")

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
        """Frame rate of the talking-mouth animation, or None if no face."""
        face = getattr(self, "face", None)
        return face.fps if face is not None else None

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
        pronunciation feedback. Only errors appear on screen — see
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
        self.status_label.configure(text=f"Status: {text}", fg=color)

    def update_instruction(self, text: str):
        self.instruction_label.configure(text=text)

    def update_score_stats(self, score: float, bucket: int = -1):
        """Show the raw score and its 0-5 bucket: ``Score: 85 (4)``.

        The phoneme engine supplies a bucket (0-5); the acoustic engine leaves
        ``bucket == -1``, shown as ``Score: 55 (-)`` (a dash instead of ``-1``).
        """
        bucket_text = str(bucket) if bucket >= 0 else "-"
        # Colour the score by the same quality band as the status line, so
        # the two score read-outs always agree visually.
        _, color = self._quality_label(score)
        self.stats_label.configure(text=f"Score: {score:.0f} ({bucket_text})", fg=color)

    def clear_previous_result(self):
        """Wipe the previous take's result before a new recording starts.

        Clears the Phrase/Work-on feedback panel, resets the Score read-out in
        the status bar to its placeholder, erases both prosody charts and resets
        the face to its neutral waiting smile, so a stale result can never be
        mistaken for the take in progress. The status *line* is set separately
        by the caller (enter_recording -> "Recording...").
        """
        self.feedback_display.configure(state=tk.NORMAL)
        self.feedback_display.delete("1.0", tk.END)
        self.feedback_display.configure(state=tk.DISABLED)
        self.stats_label.configure(text="Score: - (-)", fg=THEME["text_dim"])
        self._last_prosody = None
        self.f0_canvas.delete("all")
        self.en_canvas.delete("all")
        # Reset the face to the neutral waiting smile so it no longer reflects
        # the previous take's score. rest() drops any leftover talking mouth.
        self.face.rest()
        self.face.set_expression(":)")

    # Qualitative rating bands for the status line, replacing the old
    # Passed/Keep-practicing copy. Bands run on the raw 0-100 score and the
    # colour escalates with quality, so the status line, the face (neutral at
    # 50) and the Score bar all tell a consistent story.
    def _quality_label(self, score: float) -> tuple[str, str]:
        if score < 40:
            return "Weak", THEME["bad"]
        if score < 55:
            return "Poor", THEME["bad"]
        if score < 70:
            return "OK", THEME["warn"]
        if score < 85:
            return "Good", THEME["good"]
        return "Excellent", THEME["good"]

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
            canvas.create_line(*coords, fill=color, width=2, smooth=True)

    def _make_chart_checkbox(self, parent, text, variable):
        """Create a themed chart-title checkbox that toggles its chart's visibility.

        The command goes through the controller (on_prosody_charts_toggled in
        main.py) so the new state is also persisted to settings.json.
        """
        return tk.Checkbutton(parent, text=text, variable=variable,
                              command=self._cb.on_prosody_charts_toggled,
                              font=(FONT_FAMILY, 8), fg=THEME["text_muted"], bg=THEME["bg_main"],
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
        """Show/hide the face panel; the charts column reflows to the free width."""
        self.root.focus_set()
        shown = self.face_frame.winfo_manager() == "pack"
        if self.show_face.get() and not shown:
            self.face_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        elif not self.show_face.get() and shown:
            self.face_frame.pack_forget()

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
    def show_feedback(self, result: "PronunciationResult", current_phrase):
        # Reflect the score on the face: above 50 smiles, below frowns, 50 flat.
        self.face.set_score(result.score)
        self.feedback_display.configure(state=tk.NORMAL)
        # The numeric score moved out of this panel: the raw score/bucket now lives
        # in the status bar and the qualitative label. This panel keeps only the
        # actionable Phrase/Heard breakdown.
        # First line: the target phrase, highlighting what was said well (green)
        # vs mispronounced (red). Driven by the engine-neutral reference_words
        # tags; falls back to the raw phrase if an engine left them empty.
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
            for token in (current_phrase or "—").split():
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
            self.feedback_display.insert(tk.END, "\n")
        elif getattr(result, "bucket", -1) >= 0:
            # Phoneme engine with no weak phones -> a clean read.
            self.feedback_display.insert(tk.END, "Work on: ", "label")
            self.feedback_display.insert(tk.END, "nothing major — nice work ✓\n", "good")
        else:
            # Acoustic engine fallback: the original recognised-units "Heard" line.
            self.feedback_display.insert(tk.END, "Heard: ", "label")
            # "matches the target" only when there are no mispronounced words AND the
            # take passed -- so a low score can no longer sit next to a "matches ✓".
            if not result.word_diff and result.passed:
                self.feedback_display.insert(tk.END, "matches the target ✓\n", "good")
            elif result.recognized_units:
                for entry in result.recognized_units:
                    tag = "good" if entry.get("correct") else "bad"
                    self.feedback_display.insert(tk.END, entry["unit"] + " ", tag)
                self.feedback_display.insert(tk.END, "\n")
            else:
                self.feedback_display.insert(tk.END, f"{result.transcription or '—'}\n", "text")

        self.feedback_display.insert(tk.END, "\n")
        self.feedback_display.configure(state=tk.DISABLED)
        self.feedback_display.see(tk.END)

        # Cache prosody and draw the sparklines (you vs reference).
        self._last_prosody = result.prosody or {}
        self.root.update_idletasks()  # ensure the canvases have a real width/height
        self._redraw_prosody()

        self.update_score_stats(result.score, getattr(result, "bucket", -1))

        # Re-enable everything disabled while recording: replay buttons (both
        # signals exist now), the self-test and new-phrase generation.
        self._set_actions(generate=True, reference=True, user=True, test=True)
        self.draw_mic_button("idle")

        quality, quality_color = self._quality_label(result.score)
        self.update_status(quality, quality_color)
        if result.passed:
            self.update_instruction("Nice! Click 'New phrase' to continue, or repeat to refine.")
        else:
            self.update_instruction("Try again: hold SPACE or click the mic to repeat. ▶ Reference replays the example.")
