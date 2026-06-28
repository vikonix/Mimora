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

import time
import threading
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

import warnings
import logging
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import numpy as np

# Disable Hugging Face hub symlinks warning for a cleaner console output
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Ignore specific deprecation and model warnings from underlying libraries
warnings.filterwarnings("ignore", message="dropout option adds dropout.*")
warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")

from mimora import config, prosody, __version__
from mimora.llm import LLMManager
from mimora.llm_server_ctl import LLMServerController
from mimora.audio_io import KOKORO_SAMPLE_RATE
from mimora.tts import TTSManager, loudness_envelope
from mimora.translator import TranslatorManager
from mimora.recorder import (
    AudioRecorder,
    DEBUG_DUMP_RECORDINGS,
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
        self.root.title(f"Mimora - Pronunciation Trainer v{__version__}")

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
        # one via _new_playback_event() and stop_playback() sets the current.
        self.shutdown_event = threading.Event()
        self.playback_stop_event = threading.Event()

        # Recording is press-to-start / auto-stop (see the recording controls
        # section). space_down only tracks whether the spacebar is physically
        # held, so key-autorepeat does not fire repeated toggles - it is no
        # longer a "hold to record" flag. The actual capture lives in
        # AudioRecorder; all four callbacks fire on the capture thread, so they
        # marshal every UI touch onto the Tk main thread via root.after.
        self.space_down = False
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
        # Kokoro voice the current reference was synthesized with (logged with
        # every analysis sample - the acoustic calibration is voice-specific).
        self.current_voice: str = config.KOKORO_VOICE
        self.reference_audio: Optional[np.ndarray] = None   # 24 kHz Kokoro output
        self.last_user_audio: Optional[np.ndarray] = None   # 16 kHz recorded attempt
        # Last user name written to settings.json; lets on_user_name_changed
        # skip the file write when the field loses focus without an edit.
        self._saved_user_name: str = config.USER_NAME

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
        self.view = TrainerView(self.root, ViewCallbacks(
            on_open_practice_text=self.on_open_practice_text,
            quit_app=self.quit_app,
            on_gui_btn_press=self.on_gui_btn_press,
            on_gui_btn_release=self.on_gui_btn_release,
            on_user_name_changed=self.on_user_name_changed,
            on_length_changed=self.on_length_changed,
            on_translation_language_changed=self.on_translation_language_changed,
            on_voice_changed=self.on_voice_changed,
            on_speed_changed=self.on_speed_changed,
            on_show_face_toggled=self.on_show_face_toggled,
            on_prosody_charts_toggled=self.on_prosody_charts_toggled,
            on_test_reference=self.on_test_reference,
            play_user_recording=self.play_user_recording,
            play_reference=self.play_reference,
            on_generate_phrase=self.on_generate_phrase,
        ))
        self.bind_events()

        # Load all models in the background to keep the UI responsive.
        threading.Thread(target=self.load_components, daemon=True).start()

    def bind_events(self):
        self.root.bind("<KeyPress-space>", self.on_keyboard_press)
        self.root.bind("<KeyRelease-space>", self.on_keyboard_release)
        # Arrow-key shortcuts mirror the four action buttons. Each is gated the
        # same way the button is: ignored while a text field has focus (so arrows
        # still move the caret there) and only fired when the matching button is
        # actually enabled, so a hotkey can never trigger an action the UI is
        # currently disallowing (e.g. replaying into an open mic).
        self.root.bind("<Left>", lambda _: self._hotkey(
            self.view.is_reference_enabled, self.play_reference))
        self.root.bind("<Right>", lambda _: self._hotkey(
            self.view.is_generate_enabled, self.on_generate_phrase))
        self.root.bind("<Up>", lambda _: self._hotkey(
            self.view.is_test_enabled, self.on_test_reference))
        self.root.bind("<Down>", lambda _: self._hotkey(
            self.view.is_user_enabled, self.play_user_recording))
        self.root.bind("<Escape>", lambda _: self.quit_app())
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

            self.root.after(0, self.view.append_system_msg, "Loading Wav2Vec2 (pronunciation, ~1.2 GB on first run)...")
            # Inject app settings into the active engine before it loads any model.
            # The dispatcher builds the engine-specific config from app settings;
            # this is the analyzer's composition root.
            engine.configure()
            engine.load_models()
            logging.info("Pronunciation model loaded (engine=%s).", engine.name())

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
                    self.root.after(0, self.view.update_instruction, "LLM server failed to start. Check the log and restart.")
                    return
                self.root.after(0, self.view.append_system_msg, "LLM server is ready.")
            else:
                self.llm_mgr.init_client()
                if not self.llm_mgr.check_connection():
                    self.root.after(0, self.view.append_error_msg, "Warning: LM Studio is offline. Start it to generate phrases!")

            self.root.after(0, self.view.enter_warming_up)
            self.tts_mgr.warm_up()
            engine.warm_up()
            if config.TRANSLATION_LANGUAGE:
                self.translator_mgr.warm_up()
            logging.info("Models warmed up.")

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

    def on_open_practice_text(self):
        """Pick a practice text file via File → Open Practice Text… (main thread).

        The chosen file is loaded into the source panel right away and the path
        is persisted to settings.json ("practice_text_file"), so the same file
        is loaded again on the next launch.
        """
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Open Practice Text",
            initialdir=os.path.dirname(config.PRACTICE_TEXT_FILE),
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        # Return focus to the window so the spacebar record toggle keeps working.
        self.root.focus_set()
        if not path:
            return  # dialog cancelled

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

    def make_app_ready(self):
        self.app_ready = True
        self.view.enter_app_ready()
        self.view.append_system_msg("Ready. Generate a phrase, listen, then press SPACE to repeat it.")
        # Speak a personal greeting first, then auto-generate the first phrase.
        # The name is read here, on the main thread (Tk is not thread-safe).
        name = self.view.get_user_name()
        threading.Thread(target=self._greet_and_start, args=(name,), daemon=True).start()

    def _greet_and_start(self, name: str):
        """Speak a greeting, then start the first phrase. (Background thread.)

        The greeting uses the same Kokoro voice as the reference phrases. Any
        failure here is non-fatal - the first phrase is generated regardless,
        so a TTS hiccup cannot leave the app stuck without a phrase.
        """
        try:
            greeting = f"Hello {name}, listen and repeat." if name else "Hello, listen and repeat."
            self.root.after(0, self.view.append_system_msg, greeting)
            audio = self.tts_mgr.synthesize(greeting, voice=self._selected_voice())
            if audio.size > 0:
                self._play_with_face(audio, KOKORO_SAMPLE_RATE, self._new_playback_event())
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

    def _persist_setting(self, key: str, value):
        """Save one UI setting to settings.json, reporting failure in the UI."""
        if not config.save_user_setting(key, value):
            self.view.append_error_msg(f"Could not save {key} to settings.json.")

    def on_voice_changed(self, event=None):
        """Regenerate the phrase with the newly chosen voice.

        Re-using the standard generation path is simpler than re-synthesizing the
        current reference, and it also refreshes the analysis reference. If the app
        is busy the change is ignored here and simply applies to the next phrase.
        """
        logging.info(f"Reference voice changed to {self._selected_voice()}.")
        self._persist_setting("voice", self._selected_voice())
        # Return focus to the window so the spacebar record toggle keeps working.
        self.root.focus_set()
        if self.app_ready and not self.is_generating:
            self.on_generate_phrase()

    def _selected_length(self) -> str:
        """Map the length selector label to generate_phrase's mode ('full'/'fragment')."""
        try:
            return "fragment" if self.view.get_length_label() == LENGTH_FEW_WORDS else "full"
        except AttributeError:
            return "full"

    def on_speed_changed(self, event=None):
        """Replay the reference at the newly selected speed so it is heard right away."""
        logging.info(f"Reference speed changed to {self.view.get_speed_label()!r}.")
        self._persist_setting("reference_speed", self._selected_speed())
        # Return focus to the window so the spacebar record toggle keeps working.
        self.root.focus_set()
        # Replay only when the Reference button is also allowed (a phrase is
        # ready and nothing is recording/analyzing).
        if self.view.is_reference_enabled():
            self.play_reference()

    def on_user_name_changed(self, event=None):
        """Persist the user name to settings.json once editing finishes (FocusOut)."""
        name = self.view.get_user_name()
        if name == self._saved_user_name:
            return  # unchanged - don't rewrite the file
        if config.save_user_setting("user_name", name):
            self._saved_user_name = name
            logging.info(f"User name saved: {name!r}.")
        else:
            self.view.append_error_msg("Could not save the user name to settings.json.")

    def on_length_changed(self, event=None):
        """Regenerate the phrase when the desired length changes."""
        logging.info(f"Phrase length changed to {self.view.get_length_label()!r}.")
        self._persist_setting("phrase_length", self._selected_length())
        # The translation panel and selector depend on the length mode (fragments
        # are not translated), so reconcile them before anything else.
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
        # The cached translation belonged to the previous language, so drop it and
        # blank the panel to "-"; the next generated phrase fills it for the new
        # language (translations are applied to the next phrase, like voice/length).
        self.current_translation = ""
        self.view.set_translation("")
        self.view.refresh_translation_ui()
        # Return focus to the window so the spacebar record toggle keeps working
        # (a focused combobox would otherwise capture the spacebar).
        self.root.focus_set()

    def on_prosody_charts_toggled(self):
        """Apply a prosody-chart checkbox change and persist both flags.

        Saving both keys on either toggle keeps this trivially simple; the
        extra write of an unchanged value is harmless.
        """
        self.view.toggle_prosody_charts()
        self._persist_setting("show_pitch_chart", self.view.get_show_pitch())
        self._persist_setting("show_energy_chart", self.view.get_show_energy())

    def on_show_face_toggled(self):
        """Apply the face checkbox (show/hide the panel) and persist it."""
        self.view.toggle_face()
        self._persist_setting("show_face", self.view.get_show_face())

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

        threading.Thread(target=self._generate_and_prompt, args=(source_text,), daemon=True).start()

    def _generate_and_prompt(self, source_text: str):
        """Generate one phrase, synthesize the reference, and play it. (Background thread.)"""
        try:
            # Capture the length and translation language once, up front, so the
            # phrase, its audio, and its translation all stay consistent even if
            # the user changes a selector mid-generation.
            length = self._selected_length()
            language = self._selected_translation_language()
            phrase = self.llm_mgr.generate_phrase(source_text, length=length)
            if not phrase:
                self.root.after(0, self._phrase_generation_failed, "The model returned no phrase. Try again.")
                return

            # Synthesize the reference once; reused for playback and analysis.
            # Capture the voice in a local first so the phrase, audio and voice
            # stored below are guaranteed consistent with each other.
            voice = self._selected_voice()
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
            if DEBUG_DUMP_RECORDINGS:
                dump_record_wav(reference_audio, RECORD_MODEL_FILE, KOKORO_SAMPLE_RATE)
                dump_record_text(phrase, RECORD_PHRASE_FILE)
            # Show the phrase and play the reference for the user to hear
            # (fresh per-playback stop event; see _new_playback_event).
            self.root.after(0, self._show_new_phrase, phrase)
            # Honor the selected reference speed (see play_reference for the
            # lowered-sample-rate slowing approach) instead of always 1.0×.
            effective_sr = int(KOKORO_SAMPLE_RATE * self._selected_speed())
            self._play_with_face(self.reference_audio, effective_sr,
                                 self._new_playback_event())

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

    def _typing_in_text_field(self) -> bool:
        """True when a text-input widget owns focus - spacebar should type, not record."""
        return isinstance(self.root.focus_get(), (tk.Entry, tk.Text))

    def on_keyboard_press(self, event):
        # Holding the spacebar makes Tk fire KeyPress repeatedly (auto-repeat).
        # space_down gates those out so one physical press is one toggle; it is
        # cleared on the matching KeyRelease.
        if event.keysym == "space" and not self.space_down \
                and not self._typing_in_text_field():
            self.space_down = True
            self._toggle_recording()

    def on_keyboard_release(self, event):
        # Only clears the auto-repeat guard; the take keeps recording until it
        # auto-stops on silence or the user presses space/clicks the mic again.
        if event.keysym == "space":
            self.space_down = False

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
        threading.Thread(target=self._finalize_recording, daemon=True).start()

    def _finalize_recording(self):
        """Join the record thread, then run analysis - off the main thread.

        The is_processing_audio slot was claimed by trigger_recording_stop and
        is always released here, whatever happens.
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
            self.analyze_recording()
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
        threading.Thread(target=self._run_reference_test, daemon=True).start()

    def _run_reference_test(self):
        """Play the reference, then analyze it against itself (off the main thread)."""
        try:
            # Play the reference back first (fresh per-playback stop event;
            # see _new_playback_event).
            self.root.after(0, self.view.enter_playing, "Playing reference...")
            self._play_with_face(self.reference_audio, KOKORO_SAMPLE_RATE,
                                 self._new_playback_event())

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
            # Prosody is the engine-agnostic audio layer: compute it here from the
            # same waveforms so the charts work identically across engines.
            result.prosody = prosody.compute_prosody(
                user_audio=self.reference_audio,
                user_sr=KOKORO_SAMPLE_RATE,
                reference_audio=self.reference_audio,
                reference_sr=KOKORO_SAMPLE_RATE,
            )
            self.root.after(0, self.view.show_feedback, result, self.current_phrase,
                            self._has_user_recording())
        except Exception:
            logging.exception("Reference self-test error:")
            self.root.after(0, self.view.append_error_msg, "Reference test failed.")
            self.root.after(0, self._reset_to_retry)
        finally:
            with self.processing_lock:
                self.is_processing_audio = False

    def analyze_recording(self):
        try:
            audio = self.recorder.get_audio()
            if audio is None or len(audio) < config.AUDIO_SAMPLE_RATE * 0.2:
                logging.warning("Captured audio too short or empty.")
                self.root.after(0, self.view.append_error_msg, "Audio is too short. Speak a little longer and try again.")
                self.root.after(0, self._reset_to_retry)
                return

            if DEBUG_DUMP_RECORDINGS:
                dump_record_wav(audio, RECORD_RAW_FILE, config.AUDIO_SAMPLE_RATE)

            audio = normalize_audio(audio)
            self.last_user_audio = audio

            if DEBUG_DUMP_RECORDINGS:
                dump_record_wav(audio, RECORD_NORMALIZED_FILE, config.AUDIO_SAMPLE_RATE)

            if self.current_phrase is None or self.reference_audio is None:
                self.root.after(0, self.view.append_error_msg, "No active phrase to compare against.")
                self.root.after(0, self._reset_to_retry)
                return

            # Play the just-recorded audio back to the user right away, before the
            # (slower) pronunciation analysis runs (fresh per-playback stop event;
            # see _new_playback_event).
            self.root.after(0, self.view.enter_playing, "Playing your recording...")
            self._play_with_face(self.last_user_audio, config.AUDIO_SAMPLE_RATE,
                                 self._new_playback_event())
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

            # Prosody is the engine-agnostic audio layer: compute it here from the
            # same waveforms so the charts work identically across engines.
            result.prosody = prosody.compute_prosody(
                user_audio=audio,
                user_sr=config.AUDIO_SAMPLE_RATE,
                reference_audio=self.reference_audio,
                reference_sr=KOKORO_SAMPLE_RATE,
            )

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
        """Parse the reference-speed selector (e.g. '0.9×') into a float.

        Falls back to normal speed (1.0) if the value is missing or malformed.
        """
        try:
            return float(self.view.get_speed_label().rstrip("×"))
        except (ValueError, AttributeError):
            return 1.0

    def play_reference(self):
        if self.reference_audio is None or self.reference_audio.size == 0:
            return
        # Slowing is done by lowering the effective sample rate: playing 24 kHz
        # audio at, say, 12 kHz (0.5×) makes it twice as long. This is the simple
        # resampling approach - it also shifts the pitch down, no extra deps.
        speed = self._selected_speed()
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
    def quit_app(self):
        if self._closing:
            return  # Escape and the window-close button can both land here
        self._closing = True
        logging.info("Shutting down Mimora...")
        self.shutdown_event.set()
        self.stop_playback()

        # Stop a recording in progress so the capture thread exits its loop and
        # closes the input stream before the process goes away.
        self.recorder.stop()
        self.recorder.join()

        # Kill the local LLM server subprocess now: os._exit (below) does NOT
        # terminate children, so without this the llama_cpp server would leak
        # and keep holding VRAM after we exit.
        self.llm_server.shutdown()

        # Hard-exit immediately on the main thread instead of via root.destroy()
        # + the interpreter's normal finalization. With CUDA + PyTorch loaded,
        # the native CUDA context is torn down while still live and crashes
        # inside the C extensions, surfacing as Windows exit code 0xC0000409
        # (STATUS_STACK_BUFFER_OVERRUN) with no Python traceback.
        #
        # os._exit is NOT enough on Windows: it maps to ExitProcess, which still
        # runs DLL_PROCESS_DETACH for every loaded DLL - and the CUDA runtime's
        # detach is exactly what crashes. TerminateProcess ends the process at
        # the OS level without running any DLL detach handlers, so that crash
        # never runs. The external resources that actually need releasing - the
        # mic stream and the LLM subprocess - were handled just above; logs are
        # flushed here first. os._exit is the fallback for non-Windows.
        logging.info("quit_app: cleanup done, hard-exiting now.")
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

    def run(self):
        self.root.mainloop()
        # Flush and close log handlers only after the UI (and quit_app) is done.
        # Closing them inside quit_app produced 'I/O operation on closed file'
        # noise from daemon threads still logging during the shutdown itself.
        logging.shutdown()


if __name__ == "__main__":
    import argparse

    # Parse CLI args before the heavy GUI/model startup so `--version` returns fast.
    parser = argparse.ArgumentParser(
        prog="mimora", description="Mimora pronunciation trainer.")
    parser.add_argument(
        "--version", action="version", version=f"Mimora {__version__}")
    parser.parse_args()

    app = PronunciationTrainerGUI()
    app.run()
