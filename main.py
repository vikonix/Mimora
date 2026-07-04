# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Mimora application entry point.

Wires together the pronunciation-trainer components and runs the Tkinter GUI:
the local LLM (LLMManager / LLMServerController), text-to-speech (TTSManager,
Kokoro), audio capture (AudioRecorder) and pronunciation analysis via the engine
dispatcher (mimora/engine.py, which binds the backend chosen by config.ENGINE -
"phoneme" by default), all driven from PronunciationTrainerGUI, which composes
the TrainerView (mimora/ui.py) for the widgets.

It also installs the root logging configuration (console + logs/main.log) for
the whole app. Run this module to start the trainer:

    python main.py
"""

import subprocess
import time
import threading
from collections import deque
from typing import Optional
import os
import sys

# Prefer UTF-8 everywhere so non-ASCII (IPA phones, espeak-ng / panphon data) never
# trips a cp1252 default on Windows. We deliberately do NOT re-exec the interpreter
# into UTF-8 mode: os.execv detaches stdout under some launchers (the orphaned
# process then fails any print with "[Errno 22] Invalid argument"). Instead we set
# the hint for child processes and switch our own console streams to UTF-8 where the
# stream supports it. The in-process file reads that mattered (panphon's tables) keep
# their own narrow UTF-8 fallback in pronunciation/phoneme/speech.py.
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass  # stream may be None (pythonw), wrapped by an IDE, or already detached

# Parse CLI arguments before anything heavy: the mimora.* imports below pull in
# torch/transformers/Kokoro and can take many seconds, so `--version` (which
# exits inside parse_args) must run before them to return fast. Importing bare
# `mimora` is free - its __init__.py only defines __version__. Guarded by
# __name__ so importing this module never consumes sys.argv.
from mimora import __version__

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="mimora", description="Mimora pronunciation trainer.")
    parser.add_argument(
        "--version", action="version", version=f"Mimora {__version__}")
    parser.parse_args()

# Print as early as possible: the heavy mimora.* imports below (torch, transformers,
# Kokoro) can take many seconds on slow machines, so this is the first sign of life
# the user gets. flush=True defeats stdout buffering when output is redirected.
print("starting ...", flush=True)

import warnings
import logging
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import numpy as np

# Disable Hugging Face hub symlinks warning for a cleaner console output
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Ignore specific deprecation and model warnings from underlying libraries
warnings.filterwarnings("ignore", message="dropout option adds dropout.*")
warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")

from mimora import config, prosody
from mimora.llm import LLMManager
from mimora.llm_server_ctl import LLMServerController
from mimora.audio_io import KOKORO_SAMPLE_RATE
from mimora.tts import TTSManager, loudness_envelope
from mimora.translator import TranslatorManager
from mimora.recorder import (
    AudioRecorder,
    RECORD_MODEL_FILE,
    RECORD_NORMALIZED_FILE,
    RECORD_PHRASE_FILE,
    RECORD_RAW_FILE,
    dump_record_text,
    dump_record_wav,
    normalize_audio,
)
# Pronunciation runs through the engine dispatcher (mimora/engine.py), which binds
# the backend chosen by config.ENGINE ("acoustic" -> pronunciation/acoustic/, "phoneme" ->
# pronunciation/phoneme/) and exposes one interface. main.py never imports an engine
# directly, so switching is a single settings.json flip.
from mimora import engine
from mimora.ui import TrainerView, ViewCallbacks, LENGTH_FEW_WORDS
from mimora.settings_window import (
    PREVIEW_PHRASE,
    SettingsCallbacks,
    SettingsWindow,
)

# Configure comprehensive events logging (console + file). force=True replaces
# any handlers auto-installed by logging calls during the imports above (e.g.
# the acoustic engine loads calibration.json at import and logs it), which would otherwise
# turn this basicConfig into a silent no-op and leave main.log empty.
log_format = "%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(config.LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ],
    force=True,
)

class PronunciationTrainerGUI:
    """Tkinter front-end for the Mimora pronunciation trainer.

    Holds the controller logic (threads, analysis, state transitions) and owns
    the view by composition: ``self.view`` is a TrainerView (ui.py) that builds
    and renders the widgets. The controller drives the UI through
    ``self.view.*`` and the view forwards widget callbacks back to the
    controller. Microphone capture lives in mimora/recorder.py and the local
    LLM server lifecycle in mimora/llm_server_ctl.py.

    Flow per phrase (state machine):
        Prompt   -> Kokoro speaks an LLM-generated reference phrase.
        Record   -> user repeats it (shared recording path).
        Analyze  -> engine.analyze() runs in a daemon thread.
        Feedback -> score + problem words shown via root.after().
        Loop     -> repeat the same phrase until the score passes the threshold,
                    then the user generates the next phrase.
    """

    def __init__(self):
        logging.info("Starting Mimora Pronunciation Trainer v%s...", __version__)

        # Core Tkinter setup
        self.root = tk.Tk()
        self.root.title(f"Mimora · {config.TARGET_LANGUAGE} - Pronunciation Trainer v{__version__}")

        # Fixed width; the window spans the full usable screen height. We query the
        # Windows desktop work area (screen minus the taskbar) so the window fits
        # without being clipped, and fall back to the full screen height on other
        # platforms or if the query fails. Horizontally centered, pinned to the top
        # of the work area. winfo_screen* are valid before the first mainloop
        # iteration, so the size/position are correct from the start.
        window_width = 600
        work_top, avail_height = 0, self.root.winfo_screenheight()
        try:
            import ctypes
            from ctypes import wintypes
            SPI_GETWORKAREA = 0x0030
            rect = wintypes.RECT()
            if ctypes.windll.user32.SystemParametersInfoW(
                    SPI_GETWORKAREA, 0, ctypes.byref(rect), 0):
                work_top = rect.top
                avail_height = rect.bottom - rect.top
        except Exception:
            logging.debug("Work-area query failed; using full screen height.", exc_info=True)

        x = (self.root.winfo_screenwidth() - window_width) // 2

        # Tk's geometry height is the client area, while the position is the outer
        # frame. With a full work-area height the title bar pushes the client area
        # past the work area, hiding the bottom status bar under the taskbar. So we
        # apply a first guess, measure the actual frame (title bar + borders) once
        # the window is realized, and subtract it so the whole window fits and the
        # status bar stays visible. If the frame can't be measured yet, the guess
        # is kept unchanged.
        self.root.geometry(f"{window_width}x{avail_height}+{x}+{work_top}")
        self.root.update_idletasks()
        caption = max(self.root.winfo_rooty() - self.root.winfo_y(), 0)
        border = max(self.root.winfo_rootx() - self.root.winfo_x(), 0)
        window_height = avail_height - caption - border
        self.root.geometry(f"{window_width}x{window_height}+{x}+{work_top}")

        # Thread management events. playback_stop_event always refers to the
        # *current* playback's stop event; each new playback installs a fresh
        # one via _new_playback_event() (always on the Tk main thread) and
        # stop_playback() sets the current.
        self.shutdown_event = threading.Event()
        self.playback_stop_event = threading.Event()

        # Recording is press-to-start / auto-stop (see the recording controls
        # section). _record_key_held only tracks whether a record key (spacebar
        # or the Down arrow) is physically held, so key-autorepeat does not fire
        # repeated toggles - it is no longer a "hold to record" flag. The actual
        # capture lives in AudioRecorder; all four callbacks fire on the capture
        # thread, so they marshal every UI touch onto the Tk main thread via
        # root.after.
        self._record_key_held = False
        self.recorder = AudioRecorder(
            on_max_duration=self._on_record_max_duration,
            on_stream_error=self._on_record_stream_error,
            on_silence_stop=self._on_record_silence_stop,
            on_level=self._on_record_level,
        )

        # Audio processing guard - prevents concurrent analysis runs
        self.is_processing_audio = False
        self.processing_lock = threading.Lock()

        # Application readiness and per-phrase practice state
        self.app_ready = False
        self.is_generating = False
        self._closing = False  # guards quit_app against double invocation
        self.current_phrase: Optional[str] = None
        # Translation of the current phrase shown under the phrase card. Filled by
        # the phrase generator when a translation language is selected; kept
        # beside current_phrase so the two are always shown together.
        self.current_translation: str = ""
        # Session tally shown in the status bar. "Phrases: N" counts the distinct
        # phrases practiced this run (a set of phrase texts); "Avg" is the running
        # mean over *every* scored attempt this run (repeats of one phrase each
        # add to the average). The two therefore count different things: unique
        # phrases vs total attempts. Empty/zero at construction == reset on app
        # start (no explicit reset action).
        self._session_phrases: set[str] = set()
        self._session_score_sum: float = 0.0
        self._session_attempts: int = 0
        # Bounded attempt history shown in the scrollable list (view.render_history).
        # Holds the last N entries - scored takes, unscored ("none" engine) takes
        # and error messages - oldest first. The controller owns it so the trend
        # arrow (this take vs the previous attempt of the same phrase) can be
        # computed from the retained entries.
        self._history: deque = deque(maxlen=10)
        # Kokoro voice the current reference was synthesized with (logged with
        # every analysis sample - the acoustic calibration is voice-specific).
        self.current_voice: str = config.KOKORO_VOICE
        self.reference_audio: Optional[np.ndarray] = None   # 24 kHz Kokoro output
        self.last_user_audio: Optional[np.ndarray] = None   # 16 kHz recorded attempt
        # Last user name written to settings.json; lets on_user_name_changed
        # skip the file write when the field loses focus without an edit.
        self._saved_user_name: str = config.USER_NAME
        # Whether the prosody block is expanded, mirrored from the show_prosody
        # flag so analysis workers can skip the (expensive) prosody computation
        # while it is collapsed without touching Tk widgets. Written only on the
        # Tk main thread (here and in on_prosody_toggled); workers only read it.
        self._prosody_wanted: bool = config.SHOW_PROSODY

        # Initialize core modular sub-managers
        self.tts_mgr = TTSManager()
        # Offline phrase translator (NLLB-200). Loaded lazily: only when a
        # translation language is selected (see load_components / translate).
        self.translator_mgr = TranslatorManager()

        # LLM backend (used only to generate practice phrases)
        self.llm_backend = config.LLM_BACKEND
        self.llm_server = LLMServerController()  # no-op unless local_server backend

        if self.llm_backend == "local_server":
            logging.info("Using local_server LLM backend (llm_server/server.py subprocess).")
            self.llm_mgr = LLMManager(model=config.LOCAL_SERVER_MODEL)
        else:
            if self.llm_backend != "lm-studio":
                logging.warning(f"Unknown LLM_BACKEND '{self.llm_backend}', falling back to lm-studio.")
                self.llm_backend = "lm-studio"
            logging.info("Using LM Studio LLM backend (LLMManager).")
            self.llm_mgr = LLMManager()

        # Compose the view: it builds and owns the widgets, and forwards widget
        # callbacks back to this controller through an explicit ViewCallbacks
        # bundle (the view never holds the controller itself).
        # Settings window (mimora/settings_window.py); at most one instance,
        # created on demand by on_settings_clicked. _suppress_persist is set
        # only while on_reset_settings re-applies the defaults (see
        # _persist_setting); written and read on the Tk main thread only.
        self._settings_window: Optional[SettingsWindow] = None
        self._suppress_persist = False

        self.view = TrainerView(self.root, ViewCallbacks(
            on_settings_clicked=self.on_settings_clicked,
            on_practice_collapsed_toggled=self.on_practice_collapsed_toggled,
            on_gui_btn_press=self.on_gui_btn_press,
            on_gui_btn_release=self.on_gui_btn_release,
            on_show_face_toggled=self.on_show_face_toggled,
            on_prosody_toggled=self.on_prosody_toggled,
            on_test_reference=self.on_test_reference,
            play_user_recording=self.play_user_recording,
            play_reference=self.play_reference,
            play_reference_slow=self.play_reference_slow,
            on_generate_phrase=self.on_generate_phrase,
            on_word_clicked=self.on_word_clicked,
            on_sound_example=self.on_sound_example,
            on_take_scored=self.on_take_scored,
            on_history_entry=self.on_history_entry,
        ))
        self.bind_events()

        # Bring the freshly launched window to the foreground and put keyboard
        # focus on it, so the space-to-record hotkey works without a first click.
        # This matters most after a settings restart: the replacement process is
        # launched detached (see restart_app), and Windows does not auto-focus a
        # detached process's window, so the record keys stayed dead until the
        # user clicked the window. Scheduled on the loop (focus is only reliable
        # once the window is realized).
        self.root.after(0, self._grab_initial_focus)

        # Load all models in the background to keep the UI responsive.
        threading.Thread(target=self.load_components, daemon=True).start()

    def _grab_initial_focus(self):
        """Foreground the window and give it keyboard focus (startup/restart).

        The record hotkeys are bound to the toplevel and fire only while it holds
        keyboard focus. A brief topmost flip defeats the Windows foreground lock
        that otherwise keeps a detached, self-launched window in the background,
        then releases it so the window is not pinned above everything. Focus goes
        to the window itself (not the practice-text box), so space records rather
        than typing a space.
        """
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.attributes("-topmost", True)
            self.root.after(200, lambda: self.root.attributes("-topmost", False))
            self.root.focus_force()
        except tk.TclError:
            pass  # window torn down before the callback ran

    def bind_events(self):
        # Record keys: the spacebar and the Down arrow both toggle capture,
        # mirroring the mic button. Press/release share an auto-repeat guard so
        # holding the key does not fire repeated toggles.
        self.root.bind("<KeyPress-space>", self.on_keyboard_press)
        self.root.bind("<KeyRelease-space>", self.on_keyboard_release)
        self.root.bind("<KeyPress-Down>", self.on_keyboard_press)
        self.root.bind("<KeyRelease-Down>", self.on_keyboard_release)
        # Shortcut keys mirror the actions. Each is gated the same way its control
        # is: ignored while a text field has focus (so the key does its normal job
        # there) and only fired when the matching action is actually enabled, so a
        # hotkey can never trigger something the UI is currently disallowing
        # (e.g. replaying into an open mic). Mapping:
        #   Left -> Reference replay      Right -> New phrase
        #   Up   -> My recording replay   Down  -> mic / record toggle (above)
        #   t    -> reference self-test
        self.root.bind("<Left>", lambda _: self._hotkey(
            self.view.is_reference_enabled, self.play_reference))
        self.root.bind("<Right>", lambda _: self._hotkey(
            self.view.is_generate_enabled, self.on_generate_phrase))
        self.root.bind("<Up>", lambda _: self._hotkey(
            self.view.is_user_enabled, self.play_user_recording))
        self.root.bind("<KeyPress-t>", lambda _: self._hotkey(
            self.view.is_test_enabled, self.on_test_reference))
        self.root.bind("<KeyPress-T>", lambda _: self._hotkey(
            self.view.is_test_enabled, self.on_test_reference))
        self.root.bind("<Escape>", lambda _: self.quit_app())
        # Clicking anywhere that is not a text input returns keyboard focus to
        # the window, so the hotkeys above resume working after the user edits
        # the source text. Without this, focus would stay in the text field
        # forever: plain frames/labels never take focus on click in Tk, and
        # only button handlers happen to call focus_set().
        self.root.bind_all("<Button-1>", self._on_global_click, add="+")
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

    def _hotkey(self, is_enabled, action):
        """Run an arrow-key action if it is allowed right now.

        Mirrors button gating: skip when typing in a text field (let the arrow
        do its normal caret navigation) and only act when the corresponding
        button reports itself enabled. Returns "break" when the hotkey fired so
        Tk does not also apply the default arrow behavior to the focused widget.
        """
        if self._typing_in_text_field():
            return None
        if is_enabled():
            action()
            return "break"
        return None

    # ------------------------------------------------------------------
    # Startup: LLM server + model loading
    # ------------------------------------------------------------------
    def load_components(self):
        logging.info("Starting model loading thread...")
        self.root.after(0, self.view.enter_loading)
        self.root.after(0, self.view.append_system_msg, "Loading TTS and pronunciation models...")

        try:
            self.tts_mgr.load_model()
            logging.info("TTS model loaded.")

            # The "none" engine loads no model, so the ~1.2 GB message would
            # only confuse; every other engine loads a Wav2Vec2 recognizer.
            if engine.name() != "none":
                self.root.after(0, self.view.append_system_msg,
                                "Loading Wav2Vec2 (pronunciation, ~1.2 GB on first run)...")
            # Inject app settings into the active engine before it loads any model.
            # The dispatcher builds the engine-specific config from app settings;
            # this is the analyzer's composition root.
            engine.configure()
            engine.load_models()
            logging.info("Pronunciation engine ready (engine=%s).", engine.name())

            # Translator (NLLB) is loaded only when a language is selected at
            # startup, so a session with translation off pays no RAM/time cost.
            # If the user enables a language later, translate() loads on demand.
            if config.TRANSLATION_LANGUAGE:
                self.root.after(0, self.view.append_system_msg,
                                "Loading translator (NLLB, ~2.4 GB)...")
                self.translator_mgr.load_model()
                logging.info("Translator model loaded.")

            if self.llm_backend == "local_server":
                model_name = os.path.basename(config.EXTERNAL_MODEL_PATH)
                self.root.after(0, self.view.append_system_msg, f"Starting LLM server with {model_name}...")
                self.root.after(0, self.view.enter_server_starting)
                if not self.llm_server.start(self.llm_mgr):
                    self.root.after(0, self.view.append_error_msg, "Error: LLM server failed to start. Check model path and GPU memory.")
                    self.root.after(0, self.view.server_failed)
                    return
                self.root.after(0, self.view.append_system_msg, "LLM server is ready.")
            else:
                self.llm_mgr.init_client()
                if not self.llm_mgr.check_connection():
                    self.root.after(0, self.view.append_error_msg, "Warning: LM Studio is offline. Start it to generate phrases!")

            if config.WARM_UP:
                self.root.after(0, self.view.enter_warming_up)
                self.tts_mgr.warm_up()
                engine.warm_up()
                if config.TRANSLATION_LANGUAGE:
                    self.translator_mgr.warm_up()
                logging.info("Models warmed up.")
            else:
                # settings.json "warm_up": false - skip the dummy passes so the
                # app is ready sooner on slow machines; the first take pays the
                # first-call latency instead (see config.WARM_UP).
                logging.info("Model warm-up skipped (warm_up=false).")

            self.root.after(0, self.load_practice_text)
            self.root.after(0, self.make_app_ready)
            logging.info("Mimora initialization complete.")

        except Exception as e:
            logging.exception("Error during initialization thread:")
            self.root.after(0, self.view.append_error_msg, f"Initialization Error: {e}")
            self.root.after(0, self.view.init_failed)

    def load_practice_text(self):
        """Pre-fill the source panel from the practice text file (main thread)."""
        text = ""
        try:
            with open(config.PRACTICE_TEXT_FILE, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            logging.warning(f"Could not read practice text file: {e}")
            text = "Hello and welcome to Mimora. Edit this text and click New phrase to begin."

        self.view.set_practice_text(text.strip())

    def _load_practice_file(self, path: str):
        """Load *path* into the source panel and persist it (main thread).

        Called by the settings window's practice-file picker: the file is
        applied immediately and stored in settings.json. A relative *path*
        (the stored settings.json form) is resolved against the project root,
        matching the loader convention.
        """
        if not os.path.isabs(path):
            path = str(config.BASE_DIR / path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except (OSError, UnicodeDecodeError) as e:
            logging.warning(f"Could not read practice text file {path!r}: {e}")
            self.view.append_error_msg(f"Could not read {os.path.basename(path)}: {e}")
            return
        if not text:
            self.view.append_error_msg(f"{os.path.basename(path)} is empty - nothing to load.")
            return

        self.view.set_practice_text(text)
        self.view.append_system_msg(f"Loaded practice text: {os.path.basename(path)}")
        logging.info(f"Practice text loaded from {path!r}.")

        # Persist for the next launch. Files inside the project are stored
        # relative to the root (the settings.json convention - see _user_path);
        # as_posix() keeps the JSON free of escaped backslashes on Windows.
        try:
            saved = Path(path).relative_to(config.BASE_DIR).as_posix()
        except ValueError:
            saved = path  # outside the project - keep the absolute path
        self._persist_setting("practice_text_file", saved)
        # Keep the runtime view current for load_practice_text and the file
        # dialog's initialdir; the settings window reads the persisted value.
        config.PRACTICE_TEXT_FILE = str(path)
        self._sync_settings_window("practice_text_file", saved)

    def make_app_ready(self):
        self.app_ready = True
        self.view.enter_app_ready()
        self.view.append_system_msg("Ready. Generate a phrase, listen, then press SPACE to repeat it.")
        # Speak a personal greeting first, then auto-generate the first phrase.
        # The name and voice are read here, on the main thread (Tk is not
        # thread-safe), and the playback stop event is installed here too (see
        # _new_playback_event); the worker only receives plain values.
        name = self.view.get_user_name()
        voice = self._selected_voice()
        stop_event = self._new_playback_event()
        threading.Thread(target=self._greet_and_start,
                         args=(name, voice, stop_event), daemon=True).start()

    def _greet_and_start(self, name: str, voice: str, stop_event: threading.Event):
        """Speak a greeting, then start the first phrase. (Background thread.)

        The greeting uses the same Kokoro voice as the reference phrases. Any
        failure here is non-fatal - the first phrase is generated regardless,
        so a TTS hiccup cannot leave the app stuck without a phrase.
        """
        try:
            greeting = f"Hello {name}, listen and repeat." if name else "Hello, listen and repeat."
            self.root.after(0, self.view.append_system_msg, greeting)
            audio = self.tts_mgr.synthesize(greeting, voice=voice)
            if audio.size > 0:
                self._play_with_face(audio, KOKORO_SAMPLE_RATE, stop_event)
        except Exception:
            logging.exception("Greeting error:")
        finally:
            # Auto-generate the first phrase so the user isn't met with an
            # empty phrase card; afterwards generation is driven by the New
            # phrase button. Scheduled via after() - on_generate_phrase
            # touches widgets, so it must run on the Tk main thread.
            self.root.after(0, self.on_generate_phrase)

    # ------------------------------------------------------------------
    # Phrase generation + Prompt phase
    # ------------------------------------------------------------------
    def _selected_voice(self) -> str:
        """Return the currently selected Kokoro voice, falling back to the default."""
        try:
            return self.view.get_voice() or config.KOKORO_VOICE
        except AttributeError:
            return config.KOKORO_VOICE

    def _selected_translation_language(self) -> str:
        """Return the selected translation-language label, or "" when off."""
        try:
            return self.view.get_translation_language() or ""
        except AttributeError:
            return ""

    def _persist_setting(self, key: str, value) -> bool:
        """Save one UI setting to settings.json, reporting failure in the UI.

        No-op while _suppress_persist is set (the "Default" reset just removed
        every override from settings.json; the handlers re-run to apply the
        default values live and must not write them back as overrides).
        Returns True on success (or when suppressed).
        """
        if self._suppress_persist:
            return True
        if config.save_user_setting(key, value):
            return True
        self.view.append_error_msg(f"Could not save {key} to settings.json.")
        return False

    def on_voice_changed(self, event=None):
        """Regenerate the phrase with the newly chosen voice.

        Re-using the standard generation path is simpler than re-synthesizing the
        current reference, and it also refreshes the analysis reference. If the app
        is busy the change is ignored here and simply applies to the next phrase.
        """
        logging.info(f"Reference voice changed to {self._selected_voice()}.")
        self._persist_setting("voice", self._selected_voice())
        self._sync_settings_window("voice", self._selected_voice())
        # Return focus to the window so the spacebar record toggle keeps working.
        self.root.focus_set()
        if self.app_ready and not self.is_generating:
            self.on_generate_phrase()

    def _selected_length(self) -> str:
        """Map the phrase-length setting to generate_phrase's mode ('full'/'fragment')."""
        try:
            return "fragment" if self.view.get_length_label() == LENGTH_FEW_WORDS else "full"
        except AttributeError:
            return "full"

    def on_user_name_changed(self, event=None):
        """Persist the user name after it changes in the Settings window."""
        name = self.view.get_user_name()
        if name == self._saved_user_name:
            return  # unchanged - don't rewrite the file
        if self._persist_setting("user_name", name):
            self._saved_user_name = name
            logging.info(f"User name saved: {name!r}.")
            self._sync_settings_window("user_name", name)

    def on_length_changed(self, event=None):
        """Regenerate the phrase when the desired length changes."""
        logging.info(f"Phrase length changed to {self.view.get_length_label()!r}.")
        self._persist_setting("phrase_length", self._selected_length())
        self._sync_settings_window("phrase_length", self._selected_length())
        # Reconcile the translation panel and selector. Fragments are translated
        # too, so the length mode does not affect them; this is a consistency
        # refresh, not a mode switch.
        self.view.refresh_translation_ui()
        # Return focus to the window so the spacebar record toggle keeps working.
        self.root.focus_set()
        if self.app_ready and not self.is_generating:
            self.on_generate_phrase()

    def on_translation_language_changed(self, event=None):
        """Persist the chosen translation language and reflect it in the panel.

        The new language is applied to the *next* generated phrase (matching how
        voice/length changes behave); the current phrase is not re-translated, so
        the panel shows '-' until then. Only the panel visibility and the saved
        setting change here.
        """
        language = self.view.get_translation_language()
        logging.info(f"Translation language changed to {language!r}.")
        self._persist_setting("translation_language", language)
        self._sync_settings_window("translation_language", language)
        # The cached translation belonged to the previous language, so drop it and
        # blank the panel to "-"; the next generated phrase fills it for the new
        # language (translations are applied to the next phrase, like voice/length).
        self.current_translation = ""
        self.view.set_translation("")
        self.view.refresh_translation_ui()
        # Return focus to the window so the spacebar record toggle keeps working
        # (a focused combobox would otherwise capture the spacebar).
        self.root.focus_set()

    def on_prosody_toggled(self):
        """Apply the prosody collapse toggle and persist the show_prosody flag."""
        self.view.toggle_prosody()
        # Keep the worker-visible flag in sync (read by _compute_prosody_safe
        # on analysis threads; written only here, on the Tk main thread).
        self._prosody_wanted = self.view.get_show_prosody()
        # Keep config current too: it is the live source of truth the reset
        # diffing (_default_differs_from_live) compares against.
        config.SHOW_PROSODY = self.view.get_show_prosody()
        self._persist_setting("show_prosody", self.view.get_show_prosody())
        self._sync_settings_window("show_prosody", self.view.get_show_prosody())

    def on_show_face_toggled(self):
        """Apply the face checkbox (show/hide the panel) and persist it."""
        self.view.toggle_face()
        config.SHOW_FACE = self.view.get_show_face()
        self._persist_setting("show_face", self.view.get_show_face())
        self._sync_settings_window("show_face", self.view.get_show_face())

    def on_practice_collapsed_toggled(self):
        """Apply the practice-text collapse toggle and persist it."""
        self.view.toggle_practice_text()
        collapsed = self.view.get_practice_collapsed()
        config.PRACTICE_TEXT_COLLAPSED = collapsed
        self._persist_setting("practice_text_collapsed", collapsed)
        self._sync_settings_window("practice_text_collapsed", collapsed)

    # ------------------------------------------------------------------
    # Settings window (mimora/settings_window.py)
    # ------------------------------------------------------------------
    # settings.json key -> (config attribute, cast) for settings that the
    # runtime re-reads from the config module on every use (recorder loop,
    # phrase generator, recording dumps). Updating the attribute applies the
    # change immediately; everything else in _RESTART-marked fields waits for
    # a restart because it was bound at startup.
    _LIVE_CONFIG_ATTRS = {
        "save_recordings": ("SAVE_RECORDINGS", bool),
        "max_record_seconds": ("MAX_RECORD_SECONDS", float),
        "silence_timeout": ("SILENCE_TIMEOUT", float),
        "silence_threshold": ("SILENCE_THRESHOLD", float),
        "phrase_gen_window_sentences": ("PHRASE_GEN_WINDOW_SENTENCES", int),
        "phrase_gen_window_repeats": ("PHRASE_GEN_WINDOW_REPEATS", int),
    }

    # settings.json key -> config attribute holding the value the running app
    # currently uses, for the handler-driven settings on_setting_changed
    # applies live (the handlers keep these attributes current). Together with
    # _LIVE_CONFIG_ATTRS this lets the "Default" reset skip no-op dispatches -
    # see _default_differs_from_live. Restart-only keys are absent on purpose:
    # their dispatch would be persist-only, and the reset has already removed
    # the persisted overrides.
    _SETTING_LIVE_ATTRS = {
        "user_name": "USER_NAME",
        "voice": "KOKORO_VOICE",
        "phrase_length": "PHRASE_LENGTH",
        "translation_language": "TRANSLATION_LANGUAGE",
        "reference_speed": "REFERENCE_SPEED",
        "show_face": "SHOW_FACE",
        "show_prosody": "SHOW_PROSODY",
        "practice_text_collapsed": "PRACTICE_TEXT_COLLAPSED",
        "practice_text_file": "PRACTICE_TEXT_FILE",
    }

    def on_settings_clicked(self):
        """Open the settings window, or raise the one already open."""
        if self._settings_window is not None and self._settings_window.exists():
            self._settings_window.lift()
            return
        self._settings_window = SettingsWindow(self.root, SettingsCallbacks(
            on_setting_changed=self.on_setting_changed,
            on_preview_voice=self.on_preview_voice,
            on_restart_requested=self.restart_app,
            on_reset_settings=self.on_reset_settings,
        ))

    def _sync_settings_window(self, key: str, value):
        """Mirror a main-window setting change into the settings window.

        No-op when the window is closed. set_value never re-emits, so a change
        that originated in the settings window cannot loop back through here.
        """
        window = self._settings_window
        if window is not None and window.exists():
            window.set_value(key, value)

    def on_setting_changed(self, key: str, value):
        """Apply one settings-window change (Tk main thread).

        The settings window is the sole editor for the phrase-loop settings
        (voice, phrase length, translation language, reference speed, user
        name): each is written to config (the live source of truth) and then
        applied through the matching handler, which persists it and refreshes
        or regenerates as needed. The prosody collapse toggle also has an
        on-main control, and the face is settings-only now; both route through
        their handler so the view and settings stay in sync. Keys the runtime
        re-reads are persisted and applied via _LIVE_CONFIG_ATTRS; restart-only
        keys are just persisted (the window shows the pending-restart hint).
        """
        if key == "user_name":
            config.USER_NAME = value
            self.on_user_name_changed()
        elif key == "voice":
            if value in config.KOKORO_VOICES:
                config.KOKORO_VOICE = value
                self.on_voice_changed()
            else:
                # A voice of the other (not yet active) accent: valid only
                # after restarting into that accent, so only persist it.
                self._persist_setting("voice", value)
        elif key == "phrase_length":
            config.PHRASE_LENGTH = value
            self.on_length_changed()
        elif key == "translation_language":
            config.TRANSLATION_LANGUAGE = value
            self.on_translation_language_changed()
        elif key == "show_face":
            self.view.set_show_face(value)
            self.on_show_face_toggled()
        elif key == "practice_text_collapsed":
            self.view.set_practice_collapsed(value)
            self.on_practice_collapsed_toggled()
        elif key == "show_prosody":
            self.view.set_show_prosody(value)
            self.on_prosody_toggled()
        elif key == "reference_speed":
            # Apply live so the next Reference playback uses it; no replay here
            # (the change originates in the separate Settings window, so there
            # is nothing on-screen to demo against).
            config.REFERENCE_SPEED = float(value)
            self._persist_setting(key, float(value))
        elif key == "practice_text_file":
            self._load_practice_file(value)
        else:
            self._persist_setting(key, value)
            live = self._LIVE_CONFIG_ATTRS.get(key)
            if live is not None:
                attr, cast = live
                setattr(config, attr, cast(value))
                logging.info(f"Applied live setting config.{attr} = {value!r}.")

    def on_reset_settings(self) -> bool:
        """Reset every user setting to its default ("Default" button).

        Two steps, matching the chosen semantics: first every override is
        removed from settings.json (defaults live in the code, so an empty
        file IS the default state), then the known default values are pushed
        through the normal on_setting_changed dispatch to take effect live -
        with persistence suppressed, so the applied defaults are not written
        straight back as overrides. Returns True on success.
        """
        if not config.reset_user_settings():
            self.view.append_error_msg("Could not reset settings.json.")
            return False
        logging.info("Settings reset to defaults; applying live values.")
        self._suppress_persist = True
        try:
            for key, value in config.default_user_settings().items():
                # Dispatch only actual changes: re-applying an already-default
                # voice or phrase length would needlessly regenerate the
                # current phrase (their handlers regenerate on every call).
                if self._default_differs_from_live(key, value):
                    self.on_setting_changed(key, value)
        finally:
            self._suppress_persist = False
        return True

    def _default_differs_from_live(self, key: str, value) -> bool:
        """True when applying default *value* for *key* would change anything.

        The comparison target is the config attribute the running app reads
        (the handlers and _LIVE_CONFIG_ATTRS keep those current). Restart-only
        keys always answer False: their reset dispatch would be persist-only,
        and on_reset_settings already removed the overrides from the file.
        Numbers compare as floats (an int/float mismatch is not a change) and
        the practice-text path compares normalized, mirroring
        SettingsWindow._differs_from_runtime.
        """
        live = self._LIVE_CONFIG_ATTRS.get(key)
        attr = live[0] if live is not None else self._SETTING_LIVE_ATTRS.get(key)
        if attr is None:
            return False
        current = getattr(config, attr)
        if key == "practice_text_file":
            return (os.path.normcase(os.path.normpath(str(current)))
                    != os.path.normcase(os.path.normpath(str(value))))
        if isinstance(value, (int, float)) and not isinstance(value, bool) \
                and isinstance(current, (int, float)):
            return float(current) != float(value)
        return current != value

    def on_preview_voice(self, voice: str):
        """Speak the preview phrase with *voice* (settings-window Listen button).

        Gated like the other playback actions: never into an open microphone,
        never during generation or analysis. Only voices of the active accent
        can be previewed (the Kokoro pipeline speaks one accent per run); the
        settings window disables the button otherwise, this check is the
        thread-safe backstop.
        """
        if not self.app_ready or self.is_generating:
            return
        if voice not in config.KOKORO_VOICES:
            return
        if self.recorder.is_active():
            return
        with self.processing_lock:
            if self.is_processing_audio:
                return
        self.stop_playback()
        stop_event = self._new_playback_event()
        self.view.playing_status(f"Previewing {voice}...")
        threading.Thread(target=self._preview_voice_worker,
                         args=(voice, stop_event), daemon=True).start()

    def _preview_voice_worker(self, voice: str, stop_event: threading.Event):
        """Synthesize and play the preview phrase. (Background thread.)"""
        try:
            audio = self.tts_mgr.synthesize(PREVIEW_PHRASE, voice=voice)
            if audio.size > 0:
                self._play_with_face(audio, KOKORO_SAMPLE_RATE, stop_event)
        except Exception:
            logging.exception("Voice preview error:")
            self.root.after(0, self.view.append_error_msg,
                            f"Could not preview voice {voice}.")
        finally:
            self.root.after(0, self._playback_finished, stop_event)

    def on_generate_phrase(self):
        if not self.app_ready or self.is_generating:
            return
        with self.processing_lock:
            if self.is_processing_audio:
                return  # don't generate mid-analysis

        # Read the editable source text on the main thread (Tk is not thread-safe).
        source_text = self.view.get_practice_text()
        if not source_text:
            self.view.append_error_msg("Please enter some practice text first.")
            return

        self.is_generating = True
        self.current_phrase = None
        self.current_translation = ""
        self.last_user_audio = None
        self.view.enter_generating()

        # Capture every selector value here, on the main thread (Tk is not
        # thread-safe), and once per generation, so the phrase, its audio and
        # its translation all stay consistent even if the user changes a
        # selector mid-generation. The worker never reads a widget. The playback
        # stop event is installed here too (see _new_playback_event).
        length = self._selected_length()
        language = self._selected_translation_language()
        voice = self._selected_voice()
        speed = self._selected_speed()
        stop_event = self._new_playback_event()

        threading.Thread(target=self._generate_and_prompt,
                         args=(source_text, length, language, voice, speed, stop_event),
                         daemon=True).start()

    def _generate_and_prompt(self, source_text: str, length: str,
                             language: str, voice: str, speed: float,
                             stop_event: threading.Event):
        """Generate one phrase, synthesize the reference, and play it. (Background thread.)

        All selector values (``length``, ``language``, ``voice``, ``speed``) are
        captured by the caller on the Tk main thread and passed in as plain
        values, and ``stop_event`` was installed there as well (see
        _new_playback_event): this worker must never read widgets or mutate the
        shared playback state (Tk is not thread-safe).
        """
        try:
            phrase = self.llm_mgr.generate_phrase(source_text, length=length)
            if not phrase:
                self.root.after(0, self._phrase_generation_failed, "The model returned no phrase. Try again.")
                return

            # Synthesize the reference once; reused for playback and analysis.
            reference_audio = self.tts_mgr.synthesize(phrase, voice=voice)
            if reference_audio.size == 0:
                self.root.after(0, self._phrase_generation_failed, "Could not synthesize the reference audio.")
                return

            self.current_phrase = phrase
            # Translation is filled later by _translate_into_panel (after the
            # reference plays); until then the panel shows its "-" placeholder
            # whenever a language is selected.
            self.current_translation = ""
            self.current_voice = voice
            self.reference_audio = reference_audio
            if config.SAVE_RECORDINGS:
                dump_record_wav(reference_audio, RECORD_MODEL_FILE, KOKORO_SAMPLE_RATE)
                dump_record_text(phrase, RECORD_PHRASE_FILE)
            # Show the phrase and play the reference for the user to hear
            # (stop_event was installed by the caller; see _new_playback_event).
            self.root.after(0, self._show_new_phrase, phrase)
            # Honor the selected reference speed (see play_reference for the
            # lowered-sample-rate slowing approach) instead of always 1.0×.
            effective_sr = int(KOKORO_SAMPLE_RATE * speed)
            self._play_with_face(self.reference_audio, effective_sr, stop_event)

            # Re-enable the controls now, before translating: NLLB translation is
            # latency-tolerant and (on its first call) pays a one-time model load,
            # so it must not gate the app's readiness. The panel stays at its "-"
            # placeholder until the translation arrives.
            self.root.after(0, self._phrase_ready)
            self._translate_into_panel(phrase, language)

        except Exception as e:
            logging.exception("Phrase generation / prompt error:")
            self.root.after(0, self._phrase_generation_failed, f"Error: {e}")

    def _translate_into_panel(self, phrase: str, language: str):
        """Translate *phrase* and push it to the panel. (Background thread.)

        Translates both full sentences and "Few words" fragments; only the
        "translation off" choice is skipped. Guards against a stale result: if a
        newer phrase has replaced this one while the (CPU) translation ran, the
        result is dropped, so the panel never shows a translation that does not
        match the phrase on screen. A translator failure returns "" and simply
        leaves the placeholder in place.
        """
        if not language:
            return
        translated = self.translator_mgr.translate(phrase, language)
        # current_phrase is replaced by a fresh generation; if it no longer
        # matches, this translation is for a superseded phrase - drop it.
        if not translated or self.current_phrase != phrase:
            return
        self.current_translation = translated
        self.root.after(0, self.view.set_translation, translated)

    def _show_new_phrase(self, phrase: str):
        self.view.enter_reference_playing(phrase, self.current_translation)
        self.view.append_system_msg(f"New phrase: {phrase}")

    def _phrase_ready(self):
        self.is_generating = False
        # Enables replay + self-test now that a reference exists.
        self.view.enter_phrase_ready()

    def _phrase_generation_failed(self, message: str):
        self.is_generating = False
        self.view.generation_failed(message)

    # ------------------------------------------------------------------
    # Recording controls (shared recording path)
    # ------------------------------------------------------------------
    def _toggle_recording(self):
        """One press toggles capture: start a take, or stop the running one.

        Replaces the old hold-to-talk model. A take now starts on a single
        press and stops on its own after silence (recorder VAD); pressing again
        while it is running is the manual stop. trigger_recording_start /
        trigger_recording_stop keep their own guards, so this only routes.
        """
        if self.recorder.is_active():
            self.trigger_recording_stop()
        else:
            self.trigger_recording_start()

    def on_gui_btn_press(self):
        self._toggle_recording()

    def on_gui_btn_release(self):
        # No-op: recording is press-to-toggle now, not hold-to-talk. Kept so the
        # view's button-release binding (ViewCallbacks.on_gui_btn_release) still
        # has a target.
        pass

    def _on_global_click(self, event):
        """Give keyboard focus back to the window on clicks outside text inputs.

        Tk moves focus into Entry/Text widgets on click but never moves it back
        out when empty space is clicked, so the hotkeys would stay disabled
        (_typing_in_text_field) until some button handler called focus_set().
        Text inputs, comboboxes, and scrollbars are left alone so clicking them
        keeps normal editing/selection behavior. Widgets created inside Tk
        itself (e.g. the combobox dropdown list) reach a bind_all handler as
        path strings rather than instances - those are skipped too.
        """
        if isinstance(event.widget, str):
            return
        if isinstance(event.widget, (tk.Entry, tk.Text, ttk.Combobox, tk.Scrollbar)):
            return
        self.root.focus_set()

    def _typing_in_text_field(self) -> bool:
        """True when a text-input widget owns focus - spacebar should type, not record.

        Disabled Text widgets do not count: nothing can be typed into them, yet
        on Windows Tk focuses a Text on click even when it is disabled (so a
        selection can be shown). Without this exception, clicking the read-only
        hero phrase or the feedback panel would silently kill the space/arrow
        hotkeys until something else took focus.
        """
        widget = self.root.focus_get()
        if isinstance(widget, tk.Text):
            return str(widget.cget("state")) != tk.DISABLED
        return isinstance(widget, tk.Entry)

    def on_keyboard_press(self, event):
        # Holding a record key (space or Down) makes Tk fire KeyPress repeatedly
        # (auto-repeat). _record_key_held gates those out so one physical press is
        # one toggle; it is cleared on the matching KeyRelease. Returning "break"
        # when handled stops Tk from also applying the key's default behavior
        # (e.g. the Down arrow moving focus).
        if event.keysym in ("space", "Down") and not self._record_key_held \
                and not self._typing_in_text_field():
            self._record_key_held = True
            self._toggle_recording()
            return "break"
        return None

    def on_keyboard_release(self, event):
        # Only clears the auto-repeat guard; the take keeps recording until it
        # auto-stops on silence or the user presses a record key / clicks the mic
        # again.
        if event.keysym in ("space", "Down"):
            self._record_key_held = False

    def _can_record(self) -> bool:
        """Recording is only allowed once a phrase is ready and nothing else is busy."""
        if not self.app_ready or self.is_generating or self.current_phrase is None:
            return False
        with self.processing_lock:
            return not self.is_processing_audio

    def trigger_recording_start(self):
        if not self._can_record():
            return
        if self.recorder.is_active():
            return
        self.stop_playback()  # silence any reference playback before recording
        if not self.recorder.start():
            return
        # Lock out every playback/diagnostic action for the duration of the
        # take: playing the reference (or the previous attempt) into an open
        # microphone would end up inside the recording and corrupt analysis.
        # Re-enabled in show_feedback / _reset_to_retry.
        self.view.enter_recording()

    def trigger_recording_stop(self):
        if not self.recorder.stop():
            return

        # Claim the analysis slot *synchronously, before* spawning the worker.
        # Claiming it inside the worker left a gap in which a 'New phrase' click
        # could pass the is_processing_audio check and replace current_phrase /
        # reference_audio mid-analysis (feedback against the wrong phrase).
        with self.processing_lock:
            if self.is_processing_audio:
                logging.warning("Analysis already running, skipping duplicate take.")
                return
            self.is_processing_audio = True

        self.root.after(0, self.view.enter_analyzing)
        # The stop event for the take's playback is installed here, on the main
        # thread (see _new_playback_event); the worker only receives it.
        stop_event = self._new_playback_event()
        threading.Thread(target=self._finalize_recording, args=(stop_event,),
                         daemon=True).start()

    def _finalize_recording(self, stop_event: threading.Event):
        """Join the record thread, then run analysis - off the main thread.

        The is_processing_audio slot was claimed by trigger_recording_stop and
        is always released here, whatever happens. ``stop_event`` (installed by
        the caller) governs the playback of the just-recorded take.
        """
        try:
            if not self.recorder.join():
                # The capture thread is stuck (e.g. the device hangs on close)
                # and its callback may still be appending chunks. Reading them
                # now would race the writer - drop the take.
                self.root.after(0, self.view.append_error_msg,
                                "Audio device did not stop in time - take discarded, please try again.")
                self.root.after(0, self._reset_to_retry)
                return
            self.analyze_recording(stop_event)
        finally:
            with self.processing_lock:
                self.is_processing_audio = False

    def _on_record_max_duration(self):
        """The take hit MAX_RECORD_SECONDS (called on the capture thread).

        Route through the normal stop path on the main thread so the take is
        finalized and analyzed exactly like a manual stop.
        """
        self.root.after(0, self.view.append_error_msg, "Reached maximum record limit - take cut off.")
        self.root.after(0, self.trigger_recording_stop)

    def _on_record_silence_stop(self):
        """The take auto-stopped after silence (called on the capture thread).

        Route through the normal stop path on the main thread so the take is
        finalized and analyzed exactly like a manual stop. Unlike the max-
        duration cutoff this is the expected, designed ending, so no error is
        shown.
        """
        self.root.after(0, self.trigger_recording_stop)

    def _on_record_level(self, level: float):
        """Live mic level during a take (called on the capture thread).

        Forwarded to the recording indicator on the Tk main thread so the user
        can see the mic is hearing them while the silence auto-stop runs.
        """
        self.root.after(0, self._apply_record_level, level)

    def _apply_record_level(self, level: float):
        """Repaint the mic level indicator, but only while still recording. (Tk thread.)

        A level frame can be queued just before the take stops; applying it after
        the stop (enter_analyzing / idle already repainted the mic) would draw the
        red "recording" level disc back on top of the processing/idle glyph. The
        capture thread's stop path clears is_recording before enter_analyzing runs,
        so gating on recorder.is_active() drops these stale late frames.
        """
        if self.recorder.is_active():
            self.view.set_record_level(level)

    def _on_record_stream_error(self):
        """The input stream failed (called on the capture thread).

        Re-enable the buttons disabled at recording start (without this a
        failed stream open would leave the whole UI locked), then show the
        error status on top of the reset's default "Ready".
        """
        self.root.after(0, self._reset_to_retry)
        self.root.after(0, self.view.recording_failed)

    # ------------------------------------------------------------------
    # Analyze phase
    # ------------------------------------------------------------------
    def _compute_prosody_safe(self, user_audio: np.ndarray, user_sr: int) -> dict:
        """Prosody contours for the feedback charts, or ``{}`` when skipped/failed.

        Prosody is the engine-agnostic audio layer: it is computed here, from the
        same waveforms the engine scored, so the charts work identically across
        engines. Two cases degrade to empty contours instead:
          * Both prosody charts are hidden: the pitch tracking (librosa.pyin)
            costs seconds of CPU per take on slow machines, so hidden charts
            must not pay for it. Re-enabling a chart shows data again from the
            next take (the skipped take has nothing cached to draw).
          * The computation failed: a prosody failure must not discard the
            completed analysis - the score is already valid, so the feedback is
            shown without charts.
        Runs on analysis worker threads; reads only plain values (never widgets).
        """
        if not self._prosody_wanted:
            return {}
        try:
            return prosody.compute_prosody(
                user_audio=user_audio,
                user_sr=user_sr,
                reference_audio=self.reference_audio,
                reference_sr=KOKORO_SAMPLE_RATE,
            )
        except Exception:
            logging.exception("Prosody computation failed; showing the result without charts:")
            return {}

    def on_test_reference(self):
        """Diagnostic: feed the reference audio through analysis instead of a
        recording. Since the reference is compared against itself it should score
        near 100 - a quick way to sanity-check the pipeline without speaking.
        """
        if not self.app_ready or self.is_generating:
            return
        if self.current_phrase is None or self.reference_audio is None:
            return
        if self.recorder.is_active():
            return  # never play the reference into an open microphone
        with self.processing_lock:
            if self.is_processing_audio:
                return
            self.is_processing_audio = True

        self.stop_playback()  # silence any current playback first
        # The playback stop event is installed here, on the main thread (see
        # _new_playback_event); the worker only receives it.
        stop_event = self._new_playback_event()
        threading.Thread(target=self._run_reference_test, args=(stop_event,),
                         daemon=True).start()

    def _run_reference_test(self, stop_event: threading.Event):
        """Play the reference, then analyze it against itself (off the main thread)."""
        try:
            # Play the reference back first (stop_event was installed by the
            # caller; see _new_playback_event).
            self.root.after(0, self.view.enter_playing, "Playing reference...")
            self._play_with_face(self.reference_audio, KOKORO_SAMPLE_RATE,
                                 stop_event)

            # Then run analysis with the reference as both inputs.
            self.root.after(0, self.view.enter_analyzing, "Testing with reference...")
            result = engine.analyze(
                user_audio=self.reference_audio,
                expected_text=self.current_phrase,
                reference_audio=self.reference_audio,
                user_sr=KOKORO_SAMPLE_RATE,       # reference is Kokoro's 24 kHz output
                reference_sr=KOKORO_SAMPLE_RATE,
                voice=self.current_voice,
                is_reference=True,                # self-test: excluded from GOOD calibration
            )
            result.prosody = self._compute_prosody_safe(self.reference_audio,
                                                         KOKORO_SAMPLE_RATE)
            self.root.after(0, self.view.show_feedback, result, self.current_phrase,
                            self._has_user_recording(), True)  # is_self_test
        except Exception:
            logging.exception("Reference self-test error:")
            self.root.after(0, self.view.append_error_msg, "Reference test failed.")
            self.root.after(0, self._reset_to_retry)
        finally:
            with self.processing_lock:
                self.is_processing_audio = False

    def analyze_recording(self, stop_event: threading.Event):
        try:
            audio = self.recorder.get_audio()
            if audio is None or len(audio) < config.AUDIO_SAMPLE_RATE * 0.2:
                logging.warning("Captured audio too short or empty.")
                self.root.after(0, self.view.append_error_msg, "Audio is too short. Speak a little longer and try again.")
                self.root.after(0, self._reset_to_retry)
                return

            if config.SAVE_RECORDINGS:
                dump_record_wav(audio, RECORD_RAW_FILE, config.AUDIO_SAMPLE_RATE)

            audio = normalize_audio(audio)
            self.last_user_audio = audio

            if config.SAVE_RECORDINGS:
                dump_record_wav(audio, RECORD_NORMALIZED_FILE, config.AUDIO_SAMPLE_RATE)

            if self.current_phrase is None or self.reference_audio is None:
                self.root.after(0, self.view.append_error_msg, "No active phrase to compare against.")
                self.root.after(0, self._reset_to_retry)
                return

            # Play the just-recorded audio back to the user right away, before the
            # (slower) pronunciation analysis runs (stop_event was installed by
            # trigger_recording_stop; see _new_playback_event).
            self.root.after(0, self.view.enter_playing, "Playing your recording...")
            self._play_with_face(self.last_user_audio, config.AUDIO_SAMPLE_RATE,
                                 stop_event)
            self.root.after(0, self.view.enter_analyzing)

            analyze_start = time.perf_counter()
            result = engine.analyze(
                user_audio=audio,
                expected_text=self.current_phrase,
                reference_audio=self.reference_audio,
                user_sr=config.AUDIO_SAMPLE_RATE,
                reference_sr=KOKORO_SAMPLE_RATE,
                voice=self.current_voice,
            )
            elapsed_ms = (time.perf_counter() - analyze_start) * 1000
            logging.info(f"Pronunciation analysis done in {elapsed_ms:.0f}ms. Score={result.score}")

            result.prosody = self._compute_prosody_safe(audio,
                                                         config.AUDIO_SAMPLE_RATE)

            self.root.after(0, self.view.show_feedback, result, self.current_phrase,
                            self._has_user_recording())

        except Exception:
            logging.exception("Error in analyze_recording:")
            self.root.after(0, self.view.append_error_msg, "Analysis error. Please try again.")
            self.root.after(0, self._reset_to_retry)

    def _has_user_recording(self) -> bool:
        """True when a user take exists to replay or feed back against.

        The reference self-test reaches show_feedback without recording, so the
        "My recording" button must follow this rather than always being enabled.
        """
        return self.last_user_audio is not None and self.last_user_audio.size > 0

    def _reset_to_retry(self):
        """Return to a state where the user can record the current phrase again.

        Also re-enables the buttons disabled while recording, according to what
        is actually available (phrase / last recording).
        """
        self.view.enter_retry(has_phrase=bool(self.current_phrase),
                              has_recording=self._has_user_recording())

    # ------------------------------------------------------------------
    # Playback (reference / own recording)
    # ------------------------------------------------------------------
    def _selected_speed(self) -> float:
        """The reference-playback speed to use, from the Settings value."""
        return config.REFERENCE_SPEED

    def play_reference(self):
        """Replay the reference at the configured speed (Reference button)."""
        self._play_reference_at(self._selected_speed())

    def _slow_speed(self) -> float:
        """The slow-replay speed: one step below the Settings value."""
        return max(config.REFERENCE_SPEED - config.REFERENCE_SLOW_DELTA,
                   config.REFERENCE_SLOW_MIN)

    def play_reference_slow(self):
        """Replay the reference one step slower than normal (the Slow ▶ button)."""
        self._play_reference_at(self._slow_speed())

    def _speak_word_at(self, word: str, speed: float, status: str):
        """Synthesize a single word and play it at ``speed``.

        Shared by the two single-word playbacks - a clicked hero-card word
        and the phoneme-example badge - which differ only in speed and status
        line. ``status`` is the already-formatted status-bar text. Gated like the
        other playbacks: never into an open microphone or while an analysis
        playback is in flight.
        """
        word = word.strip()
        if not word or not self.app_ready or self.is_generating:
            return
        if self.recorder.is_active():
            return  # never play into an open microphone
        with self.processing_lock:
            if self.is_processing_audio:
                return
        self.stop_playback()  # silence any current playback first
        # The stop event is installed here on the main thread (see
        # _new_playback_event); the worker only receives it.
        stop_event = self._new_playback_event()
        self.view.playing_status(status)

        def _worker():
            try:
                audio = self.tts_mgr.synthesize(word, voice=self.current_voice)
                if audio is None or audio.size == 0:
                    return
                self._play_with_face(audio, int(KOKORO_SAMPLE_RATE * speed),
                                     stop_event)
            except Exception:
                logging.exception("Word playback error:")
            finally:
                self.root.after(0, self._playback_finished, stop_event)

        threading.Thread(target=_worker, daemon=True).start()

    def on_word_clicked(self, word: str):
        """Speak one phrase word slowly (click on any hero-card word).

        Synthesizes just that word and plays it at the slow-replay speed, via
        the same lowered-sample-rate slowing as the Slow ▶ reference button.
        """
        self._speak_word_at(word, self._slow_speed(),
                            f"Playing '{word.strip()}' slowly...")

    def on_sound_example(self, word: str):
        """Speak a phoneme's example word at normal speed (WORK ON badge click).

        The example word (e.g. "put" for /ʊ/) is a natural rendering of the
        target sound, so it plays at 1.0x rather than the slowed single-word
        replay used for the clicked phrase words.
        """
        self._speak_word_at(word, 1.0, f"Playing example '{word.strip()}'...")

    def on_take_scored(self, phrase: str, score: float):
        """Record a scored take into the session tally and refresh the status bar.

        Called by the view once a take has a user-facing score. Every attempt
        feeds the running average; the phrase text is added to a set so the count
        reflects distinct phrases. The status bar then shows the unique-phrase
        count and the mean over all attempts this run.
        """
        phrase = (phrase or "").strip()
        if not phrase:
            return
        self._session_phrases.add(phrase)
        self._session_score_sum += score
        self._session_attempts += 1
        average = self._session_score_sum / self._session_attempts
        self.view.update_session_stats(len(self._session_phrases), average)

    def on_history_entry(self, record: dict):
        """Append an entry to the bounded attempt history and re-render the list.

        ``record`` comes from the view with a ``kind`` of "attempt", "unscored" or
        "error". For a scored take, the trend arrow is derived here by comparing
        the new score with the most recent earlier attempt of the *same* phrase:
        "up" if higher, "down" if lower, "same" if equal, and left unset (a dim
        dash) when there is no earlier attempt to compare against. Errors and
        unscored takes carry no trend. The deque is capped at 10, so old entries
        drop off the top; the view then rebuilds every row from the full list.
        """
        if record.get("kind") == "attempt":
            record["trend"] = self._history_trend(record.get("phrase", ""),
                                                  record.get("score", 0.0))
        self._history.append(record)
        self.view.render_history(list(self._history))

    def _history_trend(self, phrase: str, score: float) -> Optional[str]:
        """Trend of ``score`` vs the previous attempt of ``phrase`` in the history.

        Compared on the *displayed* (rounded) score, not the raw float, so the
        arrow always agrees with the two chip numbers the user sees: 82 vs 82
        reads as "same" (a dash), never a stray up/down from a sub-point
        difference like 82.4 vs 81.6.
        """
        current = round(score)
        for past in reversed(self._history):
            if past.get("kind") == "attempt" and past.get("phrase") == phrase:
                previous = round(past.get("score", 0.0))
                if current > previous:
                    return "up"
                if current < previous:
                    return "down"
                return "same"
        return None

    def _play_reference_at(self, speed: float):
        """Play the current reference audio at *speed* (1.0 = normal)."""
        if self.reference_audio is None or self.reference_audio.size == 0:
            return
        # Slowing is done by lowering the effective sample rate: playing 24 kHz
        # audio at, say, 12 kHz (0.5×) makes it twice as long. This is the simple
        # resampling approach - it also shifts the pitch down, no extra deps.
        effective_sr = int(KOKORO_SAMPLE_RATE * speed)
        status = "Playing reference..." if speed == 1.0 else f"Playing reference ({speed:g}×)..."
        self._play_async(self.reference_audio, effective_sr, status)

    def play_user_recording(self):
        if self.last_user_audio is None or self.last_user_audio.size == 0:
            return
        self._play_async(self.last_user_audio, config.AUDIO_SAMPLE_RATE, "Playing your recording...")

    def _new_playback_event(self) -> threading.Event:
        """Install a fresh stop event for a new playback and return it.

        Every playback gets its own event. The previous shared event needed a
        set()-then-clear() dance: an old playback blocked inside a chunk write
        could miss the brief set() entirely and keep playing alongside the new
        one. With per-playback events, stop_playback() sets the current
        playback's event and it stays set - nothing is ever cleared from under
        a still-running playback.

        Must be called on the Tk main thread: it replaces the shared "current
        playback" reference that stop_playback() (also main thread) reads, so
        installing it from a worker would race the stop. Workers receive the
        event as an argument instead of creating it themselves. A stop issued
        between installing the event and the actual start of playback is not
        lost: play_array checks the event before and during playback.
        """
        event = threading.Event()
        self.playback_stop_event = event
        return event

    def _play_async(self, waveform: np.ndarray, sample_rate: int, status: str):
        """Play a waveform in a background thread, stopping any current playback first."""
        self.stop_playback()
        stop_event = self._new_playback_event()
        self.view.playing_status(status)

        def _worker():
            self._play_with_face(waveform, sample_rate, stop_event)
            self.root.after(0, self._playback_finished, stop_event)

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Articulation face (talking mouth driven from the loudness envelope)
    # ------------------------------------------------------------------
    def _play_with_face(self, waveform: np.ndarray, sample_rate: int,
                        stop_event: threading.Event):
        """play_array, with the talking mouth driven from its loudness envelope.

        winsound plays the whole buffer with no per-frame callback, so the mouth
        cannot follow live amplitude on Windows. Instead the envelope is
        pre-computed (the signal is fully known up front) and the face advances
        it on its own wall-clock timer, kept in sync by matching the playback
        lead-in. Safe to call from a background thread: the widget is touched
        only via root.after. Blocks for the playback duration, like play_array.
        """
        self.root.after(0, self._start_face_track, waveform, sample_rate)
        try:
            self.tts_mgr.play_array(waveform, sample_rate, stop_event, self.shutdown_event)
        finally:
            self.root.after(0, self._rest_face_if_current, stop_event)

    def _start_face_track(self, waveform: np.ndarray, sample_rate: int):
        """Build the loudness track and hand it to the face. (Tk thread.)"""
        fps = self.view.face_fps()
        if fps is None or waveform is None or getattr(waveform, "size", 0) == 0:
            return
        levels = loudness_envelope(waveform, sample_rate, fps=fps)
        # Keep the mouth shut during any playback lead-in silence (Windows
        # audio-session warm-up) so the animation lines up with the sound.
        lead_frames = int(round(self.tts_mgr.playback_lead_in_seconds() * fps))
        if lead_frames:
            levels = [0.0] * lead_frames + levels
        self.view.face_play_levels(levels, fps=fps)

    def _rest_face_if_current(self, stop_event: threading.Event):
        """Close the mouth, unless a newer playback has already taken over.

        Guards against an interrupted playback's cleanup clobbering the mouth
        track of the playback that superseded it (same reasoning as
        _playback_finished). (Tk thread.)
        """
        if stop_event is self.playback_stop_event:
            self.view.face_rest()

    def _playback_finished(self, stop_event: threading.Event):
        """Restore the Ready status unless this playback was stopped/superseded.

        Without the check, the worker of an interrupted playback would
        overwrite the status set by whatever replaced it (e.g. a newer
        playback's "Playing..." line).
        """
        if stop_event is self.playback_stop_event and not stop_event.is_set():
            self.view.restore_ready_status()

    def stop_playback(self):
        self.playback_stop_event.set()
        self.tts_mgr.stop_playback()
        # Close the mouth at once on an interrupt; a track left running would
        # keep flapping with no sound. A superseding playback calls this before
        # starting its own track, so the order (rest -> new track) is correct.
        self.view.face_rest()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def _shutdown_runtime(self):
        """Release the external resources before the process goes away.

        Shared by quit_app and restart_app. Stops playback, ends a recording
        in progress (so the capture thread exits its loop and closes the input
        stream), and kills the local LLM server subprocess - the hard exit
        below does NOT terminate children, so without this the llama_cpp
        server would leak, keep holding VRAM, and (on restart) still occupy
        the server port the new process needs.
        """
        self.shutdown_event.set()
        self.stop_playback()
        self.recorder.stop()
        self.recorder.join()
        self.llm_server.shutdown()

    def _hard_exit(self):
        """End the process immediately, bypassing interpreter finalization.

        Hard-exit on the main thread instead of via root.destroy() + the
        interpreter's normal finalization. With CUDA + PyTorch loaded, the
        native CUDA context is torn down while still live and crashes inside
        the C extensions, surfacing as Windows exit code 0xC0000409
        (STATUS_STACK_BUFFER_OVERRUN) with no Python traceback.

        os._exit is NOT enough on Windows: it maps to ExitProcess, which still
        runs DLL_PROCESS_DETACH for every loaded DLL - and the CUDA runtime's
        detach is exactly what crashes. TerminateProcess ends the process at
        the OS level without running any DLL detach handlers, so that crash
        never runs. The external resources that actually need releasing were
        handled by _shutdown_runtime; logs are flushed here first. os._exit is
        the fallback for non-Windows.
        """
        logging.shutdown()
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            kernel32 = ctypes.windll.kernel32
            # Declare the signatures: GetCurrentProcess returns a HANDLE (a
            # 64-bit pointer). Without this, ctypes defaults the result to a
            # 32-bit c_int and TRUNCATES the pseudo-handle, so TerminateProcess
            # gets a bad handle, silently fails (returns FALSE without killing
            # anything), and we fall through to os._exit - which crashes in the
            # CUDA DLL detach. With the correct types the pseudo-handle (-1) is
            # passed intact and the process ends at once with exit code 0.
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
            kernel32.TerminateProcess(kernel32.GetCurrentProcess(), 0)
        os._exit(0)

    def quit_app(self):
        if self._closing:
            return  # Escape and the window-close button can both land here
        self._closing = True
        logging.info("Shutting down Mimora...")
        self._shutdown_runtime()
        logging.info("quit_app: cleanup done, hard-exiting now.")
        self._hard_exit()

    def restart_app(self):
        """Relaunch the app in a new process (settings-window restart).

        Applies restart-only settings without the user quitting and starting
        the app by hand: the same cleanup as quit_app runs first (crucially
        freeing the LLM server port for the new process), then a detached
        replacement process is spawned and this one hard-exits.
        subprocess.Popen is used instead of os.execv: on Windows execv detaches
        the console under some launchers and mangles arguments with spaces.

        The replacement must not share the dying parent's console/stdio: when
        launched from an IDE, the IDE closes those pipes as soon as the parent
        exits and the child's first print would crash with [Errno 22] (the
        same failure mode the module-top comment describes for os.execv). So
        stdio is pointed at DEVNULL - the app logs to logs/main.log anyway -
        and on Windows the child is detached from the console and, when the
        launcher allows it, broken out of the IDE's job object so "stop" in
        the IDE cannot kill the restarted app.
        """
        if self._closing:
            return
        self._closing = True
        logging.info("Restarting Mimora to apply changed settings...")
        self._shutdown_runtime()
        try:
            command = [sys.executable] + sys.argv
            logging.info(f"Relaunching: {command}")
            popen_kwargs = {
                "cwd": os.getcwd(),
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            }
            if sys.platform == "win32":
                flags = (subprocess.DETACHED_PROCESS
                         | subprocess.CREATE_NEW_PROCESS_GROUP)
                try:
                    # Escape the launcher's job object (IDEs kill the whole
                    # tree on stop). Denied by some jobs - retry without.
                    subprocess.Popen(
                        command,
                        creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB,
                        **popen_kwargs)
                except OSError:
                    logging.info("Job breakaway denied; relaunching attached "
                                 "to the current job.")
                    subprocess.Popen(command, creationflags=flags,
                                     **popen_kwargs)
            else:
                # POSIX: a new session detaches from the controlling terminal.
                subprocess.Popen(command, start_new_session=True,
                                 **popen_kwargs)
        except OSError:
            # The old process must still exit cleanly - the user can relaunch
            # by hand, which is exactly what a failed restart degrades to.
            logging.exception("Relaunch failed; exiting without a new process:")
        logging.info("restart_app: cleanup done, hard-exiting now.")
        self._hard_exit()

    def run(self):
        self.root.mainloop()
        # Flush and close log handlers only after the UI (and quit_app) is done.
        # Closing them inside quit_app produced 'I/O operation on closed file'
        # noise from daemon threads still logging during the shutdown itself.
        logging.shutdown()


if __name__ == "__main__":
    # CLI arguments (--version) were already parsed at the top of the module,
    # before the heavy imports, so reaching this point means "run the app".
    app = PronunciationTrainerGUI()
    app.run()
