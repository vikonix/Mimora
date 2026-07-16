# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Settings window for the Mimora pronunciation trainer.

A declarative, passive view over config/settings.json: every editable setting
is described by a :class:`Field` (key, label, widget kind, constraints), the
fields are grouped into :class:`Section` blocks, and :class:`SettingsWindow`
renders them into one scrollable column - most-used sections on top, technical
ones at the bottom. Adding a setting is one Field line in build_sections().

The window follows the same architecture as TrainerView (ui.py): it holds no
controller reference and no application logic. Every change is forwarded to
the controller through :class:`SettingsCallbacks.on_setting_changed`, which
persists the value and applies any live effect; the window only tracks which
restart-only fields were touched so it can offer a restart on close.

Values are committed immediately on change (comboboxes/checkboxes) or when
editing finishes (entries, on FocusOut/Return) - there is no OK/Cancel buffer,
matching how the main-window toggles already persist their settings. The
Cancel button instead replays the window-opening values back through the same
commit path (see _cancel).
"""

import logging
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from dataclasses import dataclass
from typing import Callable, Optional

from mimora import config
# Reuse the main view's resolved palette, platform font and wheel-event
# helpers so the settings window matches the app theme exactly and scrolls
# on every platform (ui.py builds/hosts all of these).
from mimora.ui import (
    FONT_FAMILY,
    FONT_SIZE_BODY,
    FONT_SIZE_CAPTION,
    FONT_SIZE_SMALL,
    THEME,
    WHEEL_EVENTS,
    FlatButton,
    wheel_scroll_units,
)


@dataclass(frozen=True)
class SettingsCallbacks:
    """Typed view->controller contract for the settings window.

    on_setting_changed(key, value) - persist the settings.json *key* and apply
        any live effect (the controller owns both; the window never writes).
    on_preview_voice(voice) - speak a short test phrase with *voice* so the
        user can compare voices without leaving the window.
    on_restart_requested() - restart the application to apply restart-only
        settings (the user has already confirmed).
    on_reset_settings() - remove every settings.json override and apply the
        built-in defaults live; returns True on success (the user has already
        confirmed; the window then repaints itself from the defaults).
    """
    on_setting_changed: Callable[[str, object], None]
    on_preview_voice: Callable[[str], None]
    on_restart_requested: Callable[[], None]
    on_reset_settings: Callable[[], bool]


@dataclass(frozen=True)
class Field:
    """One editable setting: its settings.json key and how to render it.

    kind:
        "bool"   - Checkbutton
        "choice" - read-only Combobox over choices()
        "number" - Entry validated against minimum/maximum (int when integer)
        "scale"  - Scale slider over [minimum, maximum], snapped to step and
                   committed when a mouse drag or key step ends; shows the
                   live value beside it
        "text"   - free-text Entry
        "path"   - read-only Entry + Browse... file picker
    get_value returns the current effective value from config (called when the
    window opens, so a reopened window always shows the live state).
    restart marks settings that only take effect after an app restart; every
    restart field also carries runtime_value - the validated constant the
    running process was started with - so the pending-restart hint reflects
    "saved differs from running", even across window close/reopen.
    step is the snap granularity of a "scale" slider (ignored by other kinds).
    enabled, when given, is checked at render time; a field that answers False
    is shown disabled (greyed, non-interactive) - used for settings that do not
    apply to the running language/variant, e.g. Random voice with one voice.
    """
    key: str
    label: str
    kind: str
    get_value: Callable[[], object]
    choices: Optional[Callable[[], tuple]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    integer: bool = False
    restart: bool = False
    runtime_value: Optional[Callable[[], object]] = None
    file_types: tuple = ()
    help: str = ""
    step: Optional[float] = None
    enabled: Optional[Callable[[], bool]] = None


@dataclass(frozen=True)
class Section:
    """A titled group of fields, rendered as one block in the column."""
    title: str
    fields: tuple


def _pending_language() -> str:
    """The persisted practice language: what the next start will run with.

    The window edits the SAVED configuration, so every per-language choice
    list (accent, voice, engine, translation targets) follows the saved
    language rather than the running one - the two differ when the window is
    reopened after saving a language change but before the restart. Building
    the lists from the running language would then show a persisted value
    (e.g. "castilian") among another language's choices.
    """
    return config.user_setting("practice_language", config.PRACTICE_LANGUAGE)


def _pending_accent() -> str:
    """The persisted variant, validated against the pending language.

    A persisted accent that does not belong to the pending language (possible
    after hand-editing settings.json) falls back to that language's default
    variant - mirroring the startup validation in config.
    """
    language = _pending_language()
    accent = config.user_setting("accent", config.ACCENT)
    if accent not in config.accent_choices(language):
        accent = config.default_accent(language)
    return accent


def build_sections() -> tuple:
    """The full settings model: every editable key, grouped and ordered.

    Order matters: the everyday sections come first, the most technical last.
    Choices are callables into config so valid values are never duplicated
    here (e.g. the voice list always matches the accent profiles).

    get_value reads through config.user_setting (the persisted value, kept
    current in memory by save_user_setting) with the validated constant as the
    fallback, so a reopened window shows what is saved - including restart-only
    changes the running process has not picked up yet.
    """
    return (
        Section("General", (
            Field("user_name", "Your name", "text",
                  lambda: config.user_setting("user_name", config.USER_NAME),
                  help="Used in the greeting and the per-user calibration."),
            Field("practice_language", "Language", "choice",
                  lambda: config.user_setting("practice_language",
                                              config.PRACTICE_LANGUAGE),
                  choices=config.language_choices, restart=True,
                  runtime_value=lambda: config.PRACTICE_LANGUAGE,
                  help="The language you practice. Changing it needs a restart."),
            # Regional variant of the language. Hidden by _build_body when the
            # language has a single variant (nothing to choose).
            Field("accent", "Accent", "choice",
                  lambda: config.user_setting("accent", config.ACCENT),
                  choices=lambda: config.accent_choices(_pending_language()),
                  restart=True,
                  runtime_value=lambda: config.ACCENT,
                  help="Regional variant of the language."),
            Field("voice", "Voice", "choice",
                  lambda: config.user_setting("voice", config.TTS_VOICE),
                  choices=lambda: config.accent_voices(_pending_accent(),
                                                       _pending_language())),
            Field("translation_language", "Translation", "choice",
                  lambda: config.user_setting("translation_language",
                                              config.TRANSLATION_LANGUAGE),
                  choices=lambda: config.translation_targets(
                      _pending_language()),
                  help="Empty = translation off."),
            Field("phrase_length", "Phrase length", "choice",
                  lambda: config.user_setting("phrase_length",
                                              config.PHRASE_LENGTH),
                  choices=lambda: config.PHRASE_LENGTH_CHOICES,
                  help="full = whole sentence, fragment = 2-4 words."),
            Field("phrase_gen_level", "Language level", "scale",
                  lambda: config.user_setting("phrase_gen_level",
                                              config.PHRASE_GEN_LEVEL),
                  minimum=0, maximum=5, step=1,
                  help="Difficulty of generated phrases: 0 = complete "
                       "beginner (simplest words, present tense only), "
                       "5 = advanced (rich vocabulary, complex sentences). "
                       "Applies from the next phrase."),
            Field("reference_speed", "Reference speed", "scale",
                  lambda: config.user_setting("reference_speed",
                                              config.REFERENCE_SPEED),
                  minimum=config.REFERENCE_SPEED_MIN,
                  maximum=config.REFERENCE_SPEED_MAX,
                  step=config.REFERENCE_SPEED_STEP,
                  help="Normal reference playback speed. The Slow ▶ "
                       "button next to Reference plays 0.1 below this "
                       "(e.g. 0.9 -> 0.8)."),
            Field("random_voice", "Random voice per phrase", "bool",
                  lambda: config.user_setting("random_voice",
                                              config.RANDOM_VOICE),
                  enabled=lambda: len(config.TTS_VOICES) >= 2,
                  help="Each new phrase speaks with a different voice of "
                       "the current language. The Voice choice above is kept "
                       "and used again when this is off. Needs at least two "
                       "voices."),
            Field("playback_own_recording", "Play back your recording", "bool",
                  lambda: config.user_setting("playback_own_recording",
                                              config.PLAYBACK_OWN_RECORDING),
                  help="Hear your take played back before the score. "
                       "Off skips straight to analysis."),
            Field("practice_text_file", "Practice text file", "path",
                  lambda: config.user_setting("practice_text_file",
                                              config.PRACTICE_TEXT_FILE),
                  file_types=(("Text files", "*.txt"), ("All files", "*.*")),
                  help="Loaded into the practice panel right away."),
        )),
        Section("Pronunciation", (
            Field("engine", "Scoring engine", "choice",
                  lambda: config.user_setting("engine", config.ENGINE),
                  choices=lambda: config.available_engines(
                      _pending_language()),
                  restart=True,
                  runtime_value=lambda: config.ENGINE,
                  help="phoneme = default; none = scoring off (fast start). "
                       "Only engines available for the language are shown."),
            Field("phoneme_good_mode", "Phoneme anchor mode", "choice",
                  lambda: config.user_setting("phoneme_good_mode",
                                              config.PHONEME_GOOD_MODE),
                  choices=lambda: config.PHONEME_GOOD_MODE_CHOICES,
                  restart=True,
                  runtime_value=lambda: config.PHONEME_GOOD_MODE,
                  help="global = one calibrated anchor (default, faster); "
                       "ceiling = a flawless read maps to 100 per phrase."),
        )),
        Section("Appearance", (
            Field("color_theme", "Color theme", "choice",
                  lambda: config.user_setting("color_theme",
                                              config.COLOR_THEME),
                  choices=config.available_themes, restart=True,
                  runtime_value=lambda: config.COLOR_THEME),
            Field("show_face", "Show articulation face", "bool",
                  lambda: config.user_setting("show_face", config.SHOW_FACE)),
            Field("show_prosody", "Show intonation & stress", "bool",
                  lambda: config.user_setting("show_prosody",
                                              config.SHOW_PROSODY),
                  help="Pitch and energy charts under the recording controls."),
            Field("practice_text_collapsed", "Collapse practice text", "bool",
                  lambda: config.user_setting("practice_text_collapsed",
                                              config.PRACTICE_TEXT_COLLAPSED),
                  help="Hide the editable text box under its caption."),
        )),
        Section("LLM & phrase generation", (
            Field("llm_backend", "LLM backend", "choice",
                  lambda: config.user_setting("llm_backend",
                                              config.LLM_BACKEND),
                  choices=lambda: config.LLM_BACKEND_CHOICES, restart=True,
                  runtime_value=lambda: config.LLM_BACKEND,
                  help="lm-studio requires LM Studio running separately."),
            Field("external_model_path", "GGUF model file", "path",
                  lambda: config.user_setting("external_model_path",
                                              config.EXTERNAL_MODEL_PATH),
                  restart=True,
                  runtime_value=lambda: config.EXTERNAL_MODEL_PATH,
                  file_types=(("GGUF models", "*.gguf"), ("All files", "*.*"))),
            Field("external_n_ctx", "Context size (n_ctx)", "number",
                  lambda: config.user_setting("external_n_ctx",
                                              config.EXTERNAL_N_CTX),
                  minimum=256, integer=True, restart=True,
                  runtime_value=lambda: config.EXTERNAL_N_CTX,
                  help="LLM context window in tokens."),
            Field("phrase_gen_window_sentences", "Text window, sentences", "number",
                  lambda: config.user_setting(
                      "phrase_gen_window_sentences",
                      config.PHRASE_GEN_WINDOW_SENTENCES),
                  minimum=1, integer=True,
                  help="Source sentences sent to the LLM per request."),
            Field("phrase_gen_window_repeats", "Advance window every N phrases", "number",
                  lambda: config.user_setting(
                      "phrase_gen_window_repeats",
                      config.PHRASE_GEN_WINDOW_REPEATS),
                  minimum=1, integer=True),
        )),
        Section("Technical", (
            Field("save_recordings", "Save recordings (debug)", "bool",
                  lambda: config.user_setting("save_recordings",
                                              config.SAVE_RECORDINGS),
                  help="Writes each take's WAVs and phrase to records/."),
            Field("warm_up", "Warm up models at startup", "bool",
                  lambda: config.user_setting("warm_up", config.WARM_UP),
                  restart=True,
                  runtime_value=lambda: config.WARM_UP,
                  help="Faster first take, slower startup."),
            Field("max_record_seconds", "Max recording, seconds", "number",
                  lambda: config.user_setting("max_record_seconds",
                                              config.MAX_RECORD_SECONDS),
                  minimum=1),
            Field("silence_timeout", "Silence timeout, seconds", "number",
                  lambda: config.user_setting("silence_timeout",
                                              config.SILENCE_TIMEOUT),
                  minimum=0.5,
                  help="Silence after speech before the take auto-stops."),
            Field("silence_threshold", "Silence threshold (RMS)", "number",
                  lambda: config.user_setting("silence_threshold",
                                              config.SILENCE_THRESHOLD),
                  minimum=0.001,
                  help="Chunk RMS above this counts as speech."),
        )),
    )


def all_fields() -> tuple:
    """Flat tuple of every Field across all sections (used by tests too)."""
    return tuple(f for section in build_sections() for f in section.fields)


# Spoken by the voice-preview button; comes from the active language profile
# (config), so the preview is in the language being practiced. Re-exported here
# because main.py imports it from this module (the settings-window preview path).
PREVIEW_PHRASE = config.PREVIEW_PHRASE


class SettingsWindow:
    """The Toplevel settings dialog: one scrollable column of sections.

    Non-modal and transient to the main window. The controller keeps at most
    one instance alive (see main.py on_settings_clicked) and pushes changes
    made in the main window back in through set_value(), so both windows stay
    in sync without ever looping (set_value never re-emits).
    """

    _WIDTH = 540
    _LABEL_WRAP = 200      # label column width, px
    _HELP_WRAP = 300       # help text wrap, px

    def __init__(self, parent, callbacks: SettingsCallbacks):
        # ttkbootstrap is imported here, not at module level, so importing the
        # field model (tests, tooling) does not build any widget machinery;
        # only creating the real window needs it.
        import ttkbootstrap as ttk
        self._ttk = ttk
        self._cb = callbacks
        self._updating = False        # guards set_value against re-emitting
        self._restart_pending = {}    # key -> label of touched restart fields
        self._vars = {}               # key -> tk.Variable
        self._widgets = {}            # key -> main control widget
        self._scale_labels = {}       # key -> value label beside a "scale" slider
        self._committed = {}          # key -> last committed value
        self._fields = {f.key: f for f in all_fields()}

        self.top = tk.Toplevel(parent)
        self.top.title("Mimora - Settings")
        self.top.configure(bg=THEME["bg_main"])
        self.top.transient(parent)    # stays above the main window
        self._place_near(parent)
        # Footer first: it packs at the bottom edge, and the body's widgets
        # (e.g. the voice preview state) may write to the footer labels
        # already during construction.
        self._build_footer()
        self._build_body()
        # Phoneme-only fields start greyed out unless the phoneme engine is the
        # one currently selected (see _sync_engine_dependent_state).
        self._sync_engine_dependent_state()
        # Show the experimental notice if the running engine+language warrants it.
        self._sync_experimental_notice()
        # Snapshot for Cancel: the values every field had when the window
        # opened (committed values are filled during _build_body).
        self._opened_values = dict(self._committed)
        # Saved restart-only values may already differ from what this process
        # is running with (changed in an earlier window session, restart
        # declined) - surface that immediately.
        self._refresh_restart_pending()
        self.top.protocol("WM_DELETE_WINDOW", self._on_close)
        # One binding on the Toplevel serves every child (widget bindtags
        # include their toplevel), so the wheel scrolls anywhere in the window.
        # All WHEEL_EVENTS are bound: <MouseWheel> for Windows/macOS,
        # Button-4/5 for X11 (see ui.wheel_scroll_units).
        for sequence in WHEEL_EVENTS:
            self.top.bind(sequence, self._on_mousewheel)

    # ------------------------------------------------------------------
    # Public API (used by the controller)
    # ------------------------------------------------------------------
    def exists(self) -> bool:
        """True while the Toplevel is alive."""
        try:
            return bool(self.top.winfo_exists())
        except tk.TclError:
            return False

    def lift(self):
        """Raise and focus the window (gear clicked while already open)."""
        self.top.deiconify()
        self.top.lift()
        self.top.focus_set()

    def set_value(self, key: str, value):
        """Reflect a change made elsewhere (main window) without re-emitting."""
        var = self._vars.get(key)
        if var is None:
            return
        field = self._fields[key]
        self._updating = True
        try:
            if field.kind == "bool":
                var.set(bool(value))
            elif field.kind == "scale":
                snapped = self._scale_snap(field, float(value))
                var.set(snapped)
                label = self._scale_labels.get(key)
                if label is not None:
                    label.config(text=self._display_value(field, snapped))
            else:
                var.set(self._display_value(field, value))
            self._committed[key] = value
        finally:
            self._updating = False
        # An engine change pushed in from elsewhere (main window, or the
        # Cancel/Default replay that goes through set_value) must re-evaluate
        # which engine-specific fields are inert and grey them accordingly.
        if key == "engine":
            self._sync_engine_dependent_state()
        # The same replay paths change engine/language/variant, so keep the
        # experimental notice in step with them too.
        if key in ("engine", "practice_language", "accent"):
            self._sync_experimental_notice()

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------
    def _place_near(self, parent):
        """Open beside the main window when it fits, else overlap it."""
        parent.update_idletasks()
        height = min(max(parent.winfo_height(), 500), 760)
        x = parent.winfo_x() + parent.winfo_width() + 8
        if x + self._WIDTH > self.top.winfo_screenwidth():
            x = max(parent.winfo_x() - self._WIDTH - 8, 0)
        y = max(parent.winfo_y(), 0)
        self.top.geometry(f"{self._WIDTH}x{height}+{x}+{y}")

    def _build_body(self):
        """Scrollable column: a Canvas hosting one inner frame of sections."""
        body = tk.Frame(self.top, bg=THEME["bg_main"])
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(body, bg=THEME["bg_main"],
                                 highlightthickness=0, bd=0)
        scrollbar = self._ttk.Scrollbar(body, orient=tk.VERTICAL,
                                        command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(self._canvas, bg=THEME["bg_main"])
        window_id = self._canvas.create_window((0, 0), window=inner, anchor="nw")
        # Keep the inner frame as wide as the canvas, and the scrollregion as
        # tall as the content, through resizes in either direction.
        inner.bind("<Configure>", lambda e: self._canvas.configure(
            scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>", lambda e: self._canvas.itemconfigure(
            window_id, width=e.width))

        inner.columnconfigure(1, weight=1)
        row = 0
        for section in build_sections():
            row = self._add_section_header(inner, section.title, row)
            for field in section.fields:
                # The variant selector is hidden when the PENDING (saved)
                # language has a single variant: there is nothing to choose
                # (see the "accent" field). Still record its saved value so
                # restart-pending and Cancel compare against a real variant,
                # not a missing one. The saved value (get_value) is used, not
                # the runtime one: for a saved-but-not-restarted language
                # change the hidden variant genuinely awaits the restart, so
                # it must count as pending; when nothing was changed the two
                # agree and no pending is shown.
                if field.key == "accent" and \
                        len(config.accent_choices(_pending_language())) <= 1:
                    self._committed[field.key] = field.get_value()
                    continue
                row = self._add_field_row(inner, field, row)

    def _add_section_header(self, parent, title: str, row: int) -> int:
        pad_top = 14 if row else 10
        tk.Label(parent, text=title, font=(FONT_FAMILY, FONT_SIZE_BODY, "bold"),
                 fg=THEME["accent"], bg=THEME["bg_main"]).grid(
            row=row, column=0, columnspan=2, sticky=tk.W,
            padx=14, pady=(pad_top, 4))
        return row + 1

    def _add_field_row(self, parent, field: Field, row: int) -> int:
        """Render one field as label + control (+ optional help line)."""
        value = field.get_value()
        self._committed[field.key] = value

        if field.kind == "bool":
            var = tk.BooleanVar(value=bool(value))
            # highlightthickness=1 keeps Tk's native focus ring: drawn in
            # highlightcolor (accent) while focused, in highlightbackground
            # (the window bg - invisible) otherwise, so tabbing onto the
            # checkbox is visible, as with the sliders.
            control = tk.Checkbutton(
                parent, text=field.label, variable=var,
                command=lambda f=field, v=var: self._emit(f, bool(v.get())),
                font=(FONT_FAMILY, FONT_SIZE_SMALL), fg=THEME["text"], bg=THEME["bg_main"],
                activebackground=THEME["bg_main"],
                activeforeground=THEME["text"],
                selectcolor=THEME["bg_panel"], bd=0, highlightthickness=1,
                highlightbackground=THEME["bg_main"],
                highlightcolor=THEME["accent"],
                cursor="hand2", anchor=tk.W)
            control.grid(row=row, column=0, columnspan=2, sticky=tk.W,
                         padx=14, pady=2)
            # A field that does not apply to the running language/variant (e.g.
            # Random voice with a single voice) renders disabled.
            if field.enabled is not None and not field.enabled():
                control.configure(state=tk.DISABLED, cursor="")
        else:
            tk.Label(parent, text=field.label, font=(FONT_FAMILY, FONT_SIZE_SMALL),
                     fg=THEME["text_dim"], bg=THEME["bg_main"],
                     wraplength=self._LABEL_WRAP, justify=tk.LEFT,
                     anchor=tk.W).grid(
                row=row, column=0, sticky=tk.W, padx=(14, 8), pady=2)
            control = self._make_control(parent, field, value)
            control.grid(row=row, column=1, sticky=tk.EW, padx=(0, 14), pady=2)
        row += 1

        if field.help:
            tk.Label(parent, text=field.help, font=(FONT_FAMILY, FONT_SIZE_CAPTION),
                     fg=THEME["text_muted"], bg=THEME["bg_main"],
                     wraplength=self._HELP_WRAP, justify=tk.LEFT).grid(
                row=row, column=0 if field.kind == "bool" else 1,
                columnspan=2 if field.kind == "bool" else 1,
                sticky=tk.W, padx=(28, 14) if field.kind == "bool" else (0, 14),
                pady=(0, 2))
            row += 1
        return row

    def _make_control(self, parent, field: Field, value):
        """Build the non-bool control for a field row (stored in _widgets)."""
        if field.kind == "choice":
            var = tk.StringVar(value=str(value))
            # The voice row holds two widgets (combobox + Listen), so its
            # control is a frame and the combobox is created INSIDE it. The
            # frame must exist first: a sibling frame created after the
            # combobox would sit above it in the window stacking order and
            # hide it completely.
            container = parent
            if field.key == "voice":
                container = tk.Frame(parent, bg=THEME["bg_main"])
            combo = self._ttk.Combobox(container, textvariable=var,
                                       state="readonly", width=16,
                                       values=tuple(field.choices()))
            combo.bind("<<ComboboxSelected>>",
                       lambda e, f=field, v=var: self._on_choice_selected(f, v))
            # Tk's TCombobox class binding cycles a readonly combobox's value
            # on mouse wheel and fires <<ComboboxSelected>> - so scrolling the
            # settings column with the cursor over a combobox would silently
            # change (and persist) that setting. "break" stops the class (and
            # toplevel) bindings; the column is scrolled here instead, so the
            # wheel behaves the same everywhere in the window.
            def _wheel_scrolls_column(event):
                self._on_mousewheel(event)
                return "break"
            for sequence in WHEEL_EVENTS:
                combo.bind(sequence, _wheel_scrolls_column)
            self._vars[field.key] = var
            self._widgets[field.key] = combo
            if field.key == "voice":
                combo.pack(side=tk.LEFT)
                self._add_preview_button(container)
                return container
            return combo

        if field.kind == "scale":
            # The 1px highlight ring doubles as the slider's focus indicator
            # (see the FocusIn/FocusOut binds below); invisible while unfocused.
            frame = tk.Frame(parent, bg=THEME["bg_main"],
                             highlightthickness=1,
                             highlightbackground=THEME["bg_main"])
            snapped = self._scale_snap(field, float(value))
            var = tk.DoubleVar(value=snapped)
            scale = self._ttk.Scale(
                frame, from_=field.minimum, to=field.maximum,
                orient=tk.HORIZONTAL, variable=var,
                command=lambda _v, f=field: self._on_scale_move(f))
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
            # The ttkbootstrap themes draw no visible focus ring on a scale,
            # so a keyboard user tabbing onto it gets no cue: paint the row's
            # ring in the accent colour while the slider holds focus.
            scale.bind("<FocusIn>", lambda e: frame.config(
                highlightbackground=THEME["accent"]), add="+")
            scale.bind("<FocusOut>", lambda e: frame.config(
                highlightbackground=THEME["bg_main"]), add="+")
            # The TScale class bindings step arrows by a fixed 1.0 unit - more
            # than this slider's whole range, so an arrow press slammed the
            # value to min/max. Rebind them to one snap step ("break" stops
            # the class binding); Home/End (min/max) stay as the class has them.
            for arrows, direction in ((("Left", "Down"), -1),
                                      (("Right", "Up"), +1)):
                for arrow in arrows:
                    scale.bind(f"<KeyPress-{arrow}>",
                               lambda e, f=field, d=direction:
                                   self._step_scale(f, d))
            value_lbl = tk.Label(
                frame, text=self._display_value(field, snapped),
                font=(FONT_FAMILY, FONT_SIZE_SMALL), fg=THEME["text_accent"],
                bg=THEME["bg_main"], width=4, anchor=tk.E)
            value_lbl.pack(side=tk.LEFT, padx=(8, 0))
            # Commit the snapped value only when the drag ends, not on every
            # pixel of movement (each commit persists and re-previews the speed).
            scale.bind("<ButtonRelease-1>",
                       lambda e, f=field: self._commit_scale(f))
            # The scale is keyboard-operable too (arrows/Home/End via the Tk
            # class bindings, which fire on KeyPress); commit on KeyRelease,
            # otherwise a value stepped by keyboard shows on screen but is
            # never persisted. Non-stepping keys commit the unchanged value,
            # which _emit drops as a no-op.
            scale.bind("<KeyRelease>",
                       lambda e, f=field: self._commit_scale(f))
            # Tk's Scale binds the wheel to nudge the value; as with the
            # comboboxes, redirect it to scrolling the settings column so a
            # scroll-past never silently changes (and persists) the setting.
            def _wheel_scrolls_column(event):
                self._on_mousewheel(event)
                return "break"
            for sequence in WHEEL_EVENTS:
                scale.bind(sequence, _wheel_scrolls_column)
            self._vars[field.key] = var
            self._widgets[field.key] = scale
            self._scale_labels[field.key] = value_lbl
            return frame

        if field.kind in ("number", "text"):
            var = tk.StringVar(value=self._display_value(field, value))
            entry = tk.Entry(
                parent, textvariable=var, width=16,
                font=(FONT_FAMILY, FONT_SIZE_SMALL), bg=THEME["bg_accent"],
                fg=THEME["text_accent"], insertbackground=THEME["text_bright"],
                bd=0, highlightthickness=1,
                highlightbackground=THEME["border"],
                highlightcolor=THEME["accent"])
            entry.bind("<FocusOut>", lambda e, f=field: self._commit_entry(f))
            entry.bind("<Return>", lambda e, f=field: self._commit_entry(f))
            self._vars[field.key] = var
            self._widgets[field.key] = entry
            return entry

        if field.kind == "path":
            frame = tk.Frame(parent, bg=THEME["bg_main"])
            var = tk.StringVar(value=str(value))
            entry = tk.Entry(
                frame, textvariable=var, state="readonly",
                font=(FONT_FAMILY, FONT_SIZE_CAPTION), readonlybackground=THEME["bg_panel"],
                fg=THEME["text_dim"], bd=0, highlightthickness=1,
                highlightbackground=THEME["border"])
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            # Show the tail of a long path (the file name matters most).
            entry.xview_moveto(1.0)
            FlatButton(frame, text="Browse…",
                      command=lambda f=field: self._browse_path(f),
                      font=(FONT_FAMILY, FONT_SIZE_CAPTION), bg=THEME["bg_accent"],
                      fg=THEME["text_accent"],
                      activebackground=THEME["bg_accent_active"],
                      activeforeground=THEME["text_bright"],
                      bd=0, padx=8, pady=1, cursor="hand2").pack(
                side=tk.LEFT, padx=(6, 0))
            self._vars[field.key] = var
            self._widgets[field.key] = entry
            return frame

        raise RuntimeError(f"Unknown field kind {field.kind!r} for {field.key!r}")

    def _add_preview_button(self, frame):
        """Add the Listen preview button next to the voice combobox."""
        self._preview_btn = FlatButton(
            frame, text="▶ Listen", command=self._preview_voice,
            font=(FONT_FAMILY, FONT_SIZE_CAPTION), bg=THEME["bg_accent"],
            fg=THEME["text_accent"],
            activebackground=THEME["bg_accent_active"],
            activeforeground=THEME["text_bright"],
            bd=0, padx=8, pady=1, cursor="hand2",
            disabledforeground=THEME["text_disabled"])
        self._preview_btn.pack(side=tk.LEFT, padx=(6, 0))
        self._sync_preview_state()

    def _build_footer(self):
        """Status strip: validation message, restart hint + button, Close."""
        footer = tk.Frame(self.top, bg=THEME["bg_panel"])
        footer.pack(side=tk.BOTTOM, fill=tk.X)

        # Hints stack on their own rows above the button row, so a long hint
        # (e.g. the restart notice) never competes with the buttons for
        # horizontal space and cannot shrink the "Restart now" button.
        self._status_label = tk.Label(footer, text="", font=(FONT_FAMILY, FONT_SIZE_CAPTION),
                                      fg=THEME["bad"], bg=THEME["bg_panel"],
                                      wraplength=self._WIDTH - 24,
                                      justify=tk.LEFT)
        self._status_label.pack(side=tk.TOP, anchor=tk.W, padx=12, pady=(6, 0))

        self._restart_label = tk.Label(footer, text="", font=(FONT_FAMILY, FONT_SIZE_CAPTION),
                                       fg=THEME["warn"], bg=THEME["bg_panel"],
                                       wraplength=self._WIDTH - 24,
                                       justify=tk.LEFT)
        self._restart_label.pack(side=tk.TOP, anchor=tk.W, padx=12, pady=(2, 0))

        # Marks the phoneme engine as experimental for the selected language
        # (no committed model calibration - see config.engine_experimental).
        # A dedicated label so it never competes with the validation/preview
        # status or the restart hint. Text is set by _sync_experimental_notice.
        self._experimental_label = tk.Label(
            footer, text="", font=(FONT_FAMILY, FONT_SIZE_CAPTION),
            fg=THEME["warn"], bg=THEME["bg_panel"],
            wraplength=self._WIDTH - 24, justify=tk.LEFT)
        self._experimental_label.pack(side=tk.TOP, anchor=tk.W, padx=12, pady=(2, 0))

        # Button row: full-width strip below the hint rows.
        button_row = tk.Frame(footer, bg=THEME["bg_panel"])
        button_row.pack(side=tk.TOP, fill=tk.X)

        # "Default" resets everything to the built-in defaults (confirmed).
        FlatButton(button_row, text="Default", command=self._reset_defaults,
                  font=(FONT_FAMILY, FONT_SIZE_SMALL), bg=THEME["bg_accent"],
                  fg=THEME["text_accent"],
                  activebackground=THEME["bg_accent_active"],
                  activeforeground=THEME["text_bright"],
                  bd=0, padx=14, pady=4, cursor="hand2").pack(
            side=tk.LEFT, padx=(12, 0), pady=6)

        FlatButton(button_row, text="Close", command=self._on_close,
                  font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"), bg=THEME["bg_button"],
                  fg=THEME["text_button"],
                  activebackground=THEME["bg_button_active"],
                  activeforeground=THEME["text"],
                  bd=0, padx=14, pady=4, cursor="hand2").pack(
            side=tk.RIGHT, padx=12, pady=6)

        # "Cancel" undoes everything changed in this window session and closes.
        FlatButton(button_row, text="Cancel", command=self._cancel,
                  font=(FONT_FAMILY, FONT_SIZE_SMALL), bg=THEME["bg_accent"],
                  fg=THEME["text_accent"],
                  activebackground=THEME["bg_accent_active"],
                  activeforeground=THEME["text_bright"],
                  bd=0, padx=14, pady=4, cursor="hand2").pack(
            side=tk.RIGHT, pady=6)

        # Restart now uses the strong (filled) button style - the same as Close -
        # so it stands apart from the light Cancel button beside it instead of
        # blending into it. A warn/orange fill was avoided: it fails contrast in
        # the dark theme (light orange fill under white text).
        self._restart_btn = FlatButton(
            button_row, text="Restart now", command=self._restart_now,
            font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"), bg=THEME["bg_button"],
            fg=THEME["text_button"],
            activebackground=THEME["bg_button_active"],
            activeforeground=THEME["text"],
            bd=0, padx=14, pady=4, cursor="hand2")
        # Packed on demand by _update_restart_hint().

    # ------------------------------------------------------------------
    # Change handling
    # ------------------------------------------------------------------
    def _display_value(self, field: Field, value) -> str:
        """Format *value* for an Entry/Combobox var."""
        if field.kind == "number":
            if field.integer:
                return str(int(value))
            return f"{float(value):g}"
        if field.kind == "scale":
            return f"{float(value):g}"
        return str(value)

    def _emit(self, field: Field, value):
        """Commit a changed value: track restart fields, notify the controller."""
        if self._updating:
            return
        if value == self._committed.get(field.key):
            return
        self._committed[field.key] = value
        self._status_label.config(text="")
        if field.restart:
            self._refresh_restart_pending()
        logging.info(f"[Settings] {field.key} -> {value!r}")
        self._cb.on_setting_changed(field.key, value)

    def _on_choice_selected(self, field: Field, var):
        # Drop focus so the main window's space-to-record hotkey keeps working
        # after the dialog closes (mirrors the main-window toggles).
        self.top.focus_set()
        # A Combobox var is always a string; map it back to the original
        # choice object so non-string choices (reference_speed's floats) are
        # emitted and committed with their real type. Without this, re-picking
        # the current value would look like a change ("0.9" != 0.9) and the
        # controller would receive strings it has to coerce.
        raw = var.get()
        value = next((choice for choice in field.choices()
                      if str(choice) == raw), raw)
        self._emit(field, value)
        if field.key == "accent":
            self._apply_accent_change(value)
        elif field.key == "practice_language":
            self._apply_language_change(value)
        elif field.key == "engine":
            self._sync_engine_dependent_state()
        # Engine, language and variant all feed the experimental judgement.
        if field.key in ("engine", "practice_language", "accent"):
            self._sync_experimental_notice()

    def _selected_language(self) -> str:
        """The language chosen in the window (may differ from the running one)."""
        return self._vars["practice_language"].get() \
            if "practice_language" in self._vars else config.PRACTICE_LANGUAGE

    def _apply_accent_change(self, accent: str):
        """Variant switched: repoint the voice list and reset to its default.

        The persisted voice must belong to the persisted variant, otherwise the
        next startup rejects it and falls back with a warning. The new voice is
        emitted as a normal change; the controller decides whether it can apply
        live (it cannot while the running pipeline speaks the old variant). The
        voice list follows the language selected in the window, which may differ
        from the running one until a restart.
        """
        language = self._selected_language()
        voices = config.accent_voices(accent, language)
        combo = self._widgets.get("voice")
        if combo is None or not voices:
            return
        combo.configure(values=voices)
        default_voice = config.accent_default_voice(accent, language)
        self._vars["voice"].set(default_voice)
        self._emit(self._fields["voice"], default_voice)
        self._sync_preview_state()

    def _apply_language_change(self, language: str):
        """Language switched (restart-only): repoint every per-language list
        (variant, voice, engine, translation targets).

        The persisted variant and voice must belong to the persisted language,
        or the next startup rejects them and falls back with a warning. The new
        language's default variant and voice are emitted as normal changes; they
        apply live only after the restart into the new language. The variant
        combobox is present only when the running language has more than one
        variant (see _build_body), so its repoint is guarded.
        """
        default_variant = config.default_accent(language)
        accent_combo = self._widgets.get("accent")
        if accent_combo is not None:
            accent_combo.configure(values=config.accent_choices(language))
        if "accent" in self._vars:
            self._vars["accent"].set(default_variant)
        # Persist the variant of the new language even when the variant combobox
        # is not shown (a single-variant running language hides it): otherwise
        # settings.json keeps the previous language's variant and the next start
        # rejects it with a fallback warning.
        self._emit(self._fields["accent"], default_variant)
        voices = config.accent_voices(default_variant, language)
        combo = self._widgets.get("voice")
        if combo is not None and voices:
            combo.configure(values=voices)
            default_voice = config.accent_default_voice(default_variant, language)
            self._vars["voice"].set(default_voice)
            self._emit(self._fields["voice"], default_voice)
        # The engine and translation-target lists are per-language too. An
        # engine the new language does not offer resets to its first available
        # one (mirroring the startup fallback in config); a translation target
        # equal to the new language resets to "off" - translating a phrase
        # into its own language is pointless and would be rejected at startup.
        engines = config.available_engines(language)
        engine_combo = self._widgets.get("engine")
        if engine_combo is not None and engines:
            engine_combo.configure(values=engines)
            if self._vars["engine"].get() not in engines:
                self._vars["engine"].set(engines[0])
                self._emit(self._fields["engine"], engines[0])
                self._sync_engine_dependent_state()
        targets = config.translation_targets(language)
        translation_combo = self._widgets.get("translation_language")
        if translation_combo is not None:
            translation_combo.configure(values=targets)
            if self._vars["translation_language"].get() not in targets:
                self._vars["translation_language"].set("")
                self._emit(self._fields["translation_language"], "")
        self._sync_preview_state()

    def _sync_engine_dependent_state(self):
        """Grey out phoneme-only fields unless the phoneme engine is selected.

        "Phoneme anchor mode" (phoneme_good_mode) is handed only to the phoneme
        engine in engine.configure(); the acoustic engine ignores it and the
        "none" engine does no scoring at all. Disabling its combobox for any
        non-phoneme engine signals that the setting is inert for the current
        choice. The selected (combobox) value drives this, so the field greys
        the moment the engine is switched, before the pending restart applies.
        """
        combo = self._widgets.get("phoneme_good_mode")
        if combo is None:
            return
        engine = self._vars["engine"].get() if "engine" in self._vars \
            else config.ENGINE
        # A readonly combobox toggles between "readonly" (usable, closed list)
        # and "disabled" (greyed); "normal" would wrongly allow free typing.
        combo.configure(state="readonly" if engine == "phoneme" else "disabled")

    # Text of the experimental notice (see _sync_experimental_notice); kept as a
    # constant so the notice can be recognized and cleared.
    _EXPERIMENTAL_NOTICE = ("The phoneme engine is experimental for this "
                            "language: no tuned calibration yet, scoring falls "
                            "back to English.")

    def _sync_experimental_notice(self):
        """Show or clear the experimental notice for the selected engine+language.

        The phoneme engine scores against a per-language calibration; a language
        without one committed runs on the English fallback - usable, not tuned.
        config.engine_experimental() judges this from the selected (not only the
        running) values, so the notice appears as soon as the user picks an
        uncalibrated language or the phoneme engine, before any restart.
        """
        engine = self._vars["engine"].get() if "engine" in self._vars \
            else config.ENGINE
        language = self._selected_language()
        accent = self._vars["accent"].get() if "accent" in self._vars else None
        experimental = config.engine_experimental(engine, language, accent)
        self._experimental_label.config(
            text=self._EXPERIMENTAL_NOTICE if experimental else "")

    # Footer hint shown while the preview button is disabled; kept as a
    # constant so _sync_preview_state can recognize (and clear) its own text.
    _PREVIEW_HINT = ("Voice preview needs a restart into the selected "
                     "language/variant first.")

    def _sync_preview_state(self):
        """Preview works only for the language and variant the app is running.

        The Kokoro pipeline speaks one language+variant per run, so a voice of a
        selected-but-not-yet-running language or variant cannot be previewed
        until a restart. While disabled, a neutral footer hint explains why
        (kind="info", not an error - nothing went wrong); it is cleared as soon
        as the preview is possible again, so switching back does not leave it
        stale.
        """
        language = self._selected_language()
        accent = self._vars["accent"].get() \
            if "accent" in self._vars else config.ACCENT
        running = (language == config.PRACTICE_LANGUAGE
                   and accent == config.ACCENT)
        state = tk.NORMAL if running else tk.DISABLED
        self._preview_btn.config(state=state)
        if state == tk.DISABLED:
            self._show_status(self._PREVIEW_HINT, kind="info")
        elif self._status_label.cget("text") == self._PREVIEW_HINT:
            self._status_label.config(text="")

    def _preview_voice(self):
        voice = self._vars["voice"].get()
        if voice:
            self._cb.on_preview_voice(voice)

    def _scale_snap(self, field: Field, value: float) -> float:
        """Clamp *value* to [minimum, maximum] and snap it to the field's step.

        The slider itself is continuous; snapping keeps the persisted value on
        the discrete grid the user expects (e.g. 0.05 steps for reference
        speed) instead of an arbitrary float like 0.9137.
        """
        lo, hi = field.minimum, field.maximum
        value = max(lo, min(hi, value))
        if field.step:
            value = round(round((value - lo) / field.step) * field.step + lo, 6)
            value = max(lo, min(hi, value))
        return value

    def _on_scale_move(self, field: Field):
        """Refresh the value label live as the slider is dragged (no commit)."""
        snapped = self._scale_snap(field, float(self._vars[field.key].get()))
        label = self._scale_labels.get(field.key)
        if label is not None:
            label.config(text=self._display_value(field, snapped))

    def _step_scale(self, field: Field, direction: int) -> str:
        """Nudge a scale one snap step by keyboard (replaces the class binding,
        whose fixed 1.0-unit step is wider than the whole slider range).

        Only moves the thumb and the value label; the commit happens on the
        shared <KeyRelease> binding, same as the mouse path. Returns "break"
        so the TScale class binding does not also fire.
        """
        step = field.step or (field.maximum - field.minimum) / 20
        value = self._scale_snap(
            field, float(self._vars[field.key].get()) + direction * step)
        self._updating = True
        try:
            self._vars[field.key].set(value)
        finally:
            self._updating = False
        label = self._scale_labels.get(field.key)
        if label is not None:
            label.config(text=self._display_value(field, value))
        return "break"

    def _commit_scale(self, field: Field):
        """Snap the thumb to the discrete value and commit it.

        Runs when an interaction ends: mouse drag released, or a keyboard
        step (arrows/Home/End) released. Idempotent - committing the already
        committed value is dropped by _emit.
        """
        snapped = self._scale_snap(field, float(self._vars[field.key].get()))
        self._updating = True
        try:
            self._vars[field.key].set(snapped)  # pull the thumb onto the grid
        finally:
            self._updating = False
        label = self._scale_labels.get(field.key)
        if label is not None:
            label.config(text=self._display_value(field, snapped))
        self._emit(field, snapped)

    def _commit_entry(self, field: Field):
        """Validate and commit a number/text Entry when editing finishes."""
        raw = self._vars[field.key].get().strip()
        if field.kind == "text":
            self._emit(field, raw)
            return
        # number: parse, range-check, and normalize the display.
        try:
            number = float(raw.replace(",", "."))
        except ValueError:
            self._reject_entry(field, f"{field.label}: not a number.")
            return
        if field.integer:
            # Reject a fractional value instead of silently truncating it:
            # the user would otherwise see 5.7 turn into 5 with no explanation.
            if number != int(number):
                self._reject_entry(field,
                                   f"{field.label}: must be a whole number.")
                return
            number = int(number)
        if field.minimum is not None and number < field.minimum:
            self._reject_entry(field,
                               f"{field.label}: must be at least {field.minimum:g}.")
            return
        if field.maximum is not None and number > field.maximum:
            self._reject_entry(field,
                               f"{field.label}: must be at most {field.maximum:g}.")
            return
        self._vars[field.key].set(self._display_value(field, number))
        self._emit(field, number)

    def _reject_entry(self, field: Field, message: str):
        """Revert an invalid Entry to its last committed value and explain."""
        self._vars[field.key].set(
            self._display_value(field, self._committed[field.key]))
        self._show_status(message, kind="error")

    def _browse_path(self, field: Field):
        current = str(self._committed.get(field.key) or "")
        # Stored paths may be project-relative (the settings.json convention);
        # resolve against the project root so initialdir opens the right place.
        if current and not os.path.isabs(current):
            current = os.path.join(str(config.BASE_DIR), current)
        path = filedialog.askopenfilename(
            parent=self.top,
            title=field.label,
            initialdir=os.path.dirname(current) or str(config.BASE_DIR),
            filetypes=list(field.file_types) or [("All files", "*.*")],
        )
        if not path:
            return  # dialog cancelled
        self._vars[field.key].set(path)
        widget = self._widgets.get(field.key)
        if widget is not None:
            widget.xview_moveto(1.0)  # keep the file name visible
        self._emit(field, path)

    # ------------------------------------------------------------------
    # Cancel / Default
    # ------------------------------------------------------------------
    def _cancel(self):
        """Undo every change made in this window session, then close.

        Values are reverted through the same on_setting_changed path that
        applied them, so live effects (voice, charts, timeouts, loaded text)
        are rolled back and the old values are persisted again. Field order
        matters and all_fields() provides it: the accent field precedes the
        voice field, so the reverted voice is validated against the reverted
        accent by the controller.
        """
        for field in all_fields():
            old = self._opened_values.get(field.key)
            if old == self._committed.get(field.key):
                continue
            self.set_value(field.key, old)
            logging.info(f"[Settings] cancel: {field.key} -> {old!r}")
            self._cb.on_setting_changed(field.key, old)
        # Close without the restart prompt: Cancel restored the opening state,
        # and a restart already pending when the window opened is not this
        # session's doing - the hint will simply reappear next time.
        self.top.destroy()

    def _reset_defaults(self):
        """Reset every setting to its built-in default (confirmed)."""
        if not messagebox.askyesno(
                "Reset settings?",
                "Reset all settings to their defaults?\n\n"
                "Restart-only settings apply after a restart; "
                "Cancel still restores the previous values.",
                parent=self.top):
            return
        if not self._cb.on_reset_settings():
            self._show_status("Reset failed - see the error in the main window.",
                              kind="error")
            return
        # Repaint the rows from the known defaults (set_value never re-emits;
        # the controller has already applied them). Keys without a fixed
        # default (external_n_ctx - hardware-derived) keep their display and
        # are covered by the restart hint below.
        defaults = config.default_user_settings()
        for field in all_fields():
            if field.key in defaults:
                self.set_value(field.key, defaults[field.key])
        # Every per-language list must match the default language now shown -
        # resolved within the default language, not the running one (a reset
        # from Spanish still shows the English default variant's voices and
        # every English engine).
        default_language = defaults.get("practice_language")
        combo = self._widgets.get("voice")
        if combo is not None:
            combo.configure(values=config.accent_voices(
                defaults["accent"], default_language))
        accent_combo = self._widgets.get("accent")
        if accent_combo is not None:
            accent_combo.configure(
                values=config.accent_choices(default_language))
        engine_combo = self._widgets.get("engine")
        if engine_combo is not None:
            engine_combo.configure(
                values=config.available_engines(default_language))
        translation_combo = self._widgets.get("translation_language")
        if translation_combo is not None:
            translation_combo.configure(
                values=config.translation_targets(default_language))
        self._sync_preview_state()
        self._sync_experimental_notice()
        # Only the defaults that differ from the running values actually
        # need a restart (a machine already on defaults needs none).
        self._refresh_restart_pending()
        self._show_status("Settings were reset to their defaults.")

    def _show_status(self, message: str, *, kind: str = "ok"):
        """One-line footer status: red errors, green confirmations, dim hints."""
        colors = {"error": THEME["bad"], "ok": THEME["good"],
                  "info": THEME["text_dim"]}
        self._status_label.config(text=message, fg=colors[kind])

    # ------------------------------------------------------------------
    # Restart handling / closing
    # ------------------------------------------------------------------
    def _refresh_restart_pending(self):
        """Recompute which restart-only fields await a restart, update the hint.

        A field is pending while its committed (= saved) value differs from
        the value the running process was started with (Field.runtime_value).
        This survives change-and-change-back round trips and window reopens:
        the comparison is always against the actual runtime, never against
        transient window state.
        """
        self._restart_pending.clear()
        for field in all_fields():
            if field.restart and self._differs_from_runtime(
                    field, self._committed.get(field.key)):
                self._restart_pending[field.key] = field.label
        self._update_restart_hint()

    def _differs_from_runtime(self, field: Field, value) -> bool:
        """True when *value* is not what the running process uses for *field*.

        Kind-aware comparison: paths may be stored project-relative while the
        runtime constant is absolute, and numbers may be int in one place and
        float in the other - neither difference is a real change.
        """
        if field.runtime_value is None:
            return False
        runtime = field.runtime_value()
        if field.kind == "path":
            return self._normalize_path(value) != self._normalize_path(runtime)
        if field.kind == "number":
            try:
                return float(value) != float(runtime)
            except (TypeError, ValueError):
                return True
        return value != runtime

    @staticmethod
    def _normalize_path(value) -> str:
        """Absolute, case/separator-normalized form of a stored path value."""
        path = str(value or "")
        if path and not os.path.isabs(path):
            path = os.path.join(str(config.BASE_DIR), path)
        return os.path.normcase(os.path.normpath(path))

    def _update_restart_hint(self):
        """Show, refresh, or clear the pending-restart hint and its button."""
        if not self._restart_pending:
            self._restart_label.config(text="")
            if self._restart_btn.winfo_manager():
                self._restart_btn.pack_forget()
            return
        labels = ", ".join(self._restart_pending.values())
        self._restart_label.config(text=f"Applies after restart: {labels}")
        if not self._restart_btn.winfo_manager():
            # padx keeps a gap from the Cancel button on its right.
            self._restart_btn.pack(side=tk.RIGHT, padx=(0, 8), pady=6)

    def _restart_now(self):
        """Footer button: an explicit click needs no extra confirmation."""
        self._cb.on_restart_requested()

    def _on_close(self):
        if self._restart_pending:
            labels = ", ".join(self._restart_pending.values())
            if messagebox.askyesno(
                    "Restart Mimora?",
                    f"These changes apply after a restart:\n{labels}\n\n"
                    f"Restart now?",
                    parent=self.top):
                self._cb.on_restart_requested()
                return  # the app is exiting; leave the window as-is
        self.top.destroy()

    def _on_mousewheel(self, event):
        """Scroll the settings column with the wheel (see ui.wheel_scroll_units)."""
        units = wheel_scroll_units(event)
        if units:
            self._canvas.yview_scroll(units, "units")
