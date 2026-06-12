import time
import threading
from typing import Optional, List
import os
import sys
import warnings
import logging
import tkinter as tk
import numpy as np

# Disable Hugging Face hub symlinks warning for a cleaner console output
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Ignore specific deprecation and model warnings from underlying libraries
warnings.filterwarnings("ignore", message="dropout option adds dropout.*")
warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")

from echoloop import config
# Whisper STT is disabled: echoloop/stt.py is not imported, so faster-whisper
# is never loaded and no VRAM/start-up time is spent on it. Transcription is
# done by Wav2Vec2 in pronounce/. Re-enable by importing STTManager again.
from echoloop.llm import LLMManager
from echoloop.llm_server_ctl import LLMServerController
from echoloop.tts import TTSManager, KOKORO_SAMPLE_RATE
from echoloop.recorder import (
    AudioRecorder,
    DEBUG_DUMP_RECORDINGS,
    RECORD_MODEL_FILE,
    RECORD_NORMALIZED_FILE,
    RECORD_RAW_FILE,
    dump_record_wav,
    normalize_audio,
)
import pronounce
from echoloop.ui import PronunciationTrainerUI, LENGTH_FEW_WORDS

# Resolved UI color palette (semantic name -> hex), selected by the
# "color_theme" setting in settings.json; see config.py.
THEME = config.THEME

# Configure comprehensive events logging (console + file). force=True replaces
# any handlers auto-installed by logging calls during the imports above (e.g.
# pronounce loads calibration.json at import and logs it), which would otherwise
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

class PronunciationTrainerGUI(PronunciationTrainerUI):
    """Tkinter front-end for the EchoLoop pronunciation trainer.

    Inherits widget construction and rendering from PronunciationTrainerUI
    (ui.py); this class holds the controller logic (threads, analysis, state
    transitions). Microphone capture lives in echoloop/recorder.py and the
    local LLM server lifecycle in echoloop/llm_server_ctl.py.

    Flow per phrase (spec state machine):
        Prompt   -> Kokoro speaks an LLM-generated reference phrase.
        Record   -> user repeats it (shared recording path).
        Analyze  -> pronounce.analyze() runs in a daemon thread.
        Feedback -> score + problem words shown via root.after().
        Loop     -> repeat the same phrase until the score passes the threshold,
                    then the user generates the next phrase.
    """

    def __init__(self):
        logging.info("Starting EchoLoop Pronunciation Trainer...")

        # Core Tkinter setup
        self.root = tk.Tk()
        self.root.title("EchoLoop - Pronunciation Trainer")
        self.root.configure(bg=THEME["bg_main"])

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

        # Push-to-talk state; the actual capture lives in AudioRecorder.
        # Both callbacks fire on the capture thread, so they marshal all UI
        # work onto the Tk main thread via root.after.
        self.space_is_held = False
        self.recorder = AudioRecorder(
            on_max_duration=self._on_record_max_duration,
            on_stream_error=self._on_record_stream_error,
        )

        # Audio processing guard — prevents concurrent analysis runs
        self.is_processing_audio = False
        self.processing_lock = threading.Lock()

        # Application readiness and per-phrase practice state
        self.app_ready = False
        self.is_generating = False
        self._closing = False  # guards quit_app against double invocation
        self.current_phrase: Optional[str] = None
        # Kokoro voice the current reference was synthesized with (logged with
        # every analysis sample — the acoustic calibration is voice-specific).
        self.current_voice: str = config.KOKORO_VOICE
        self.reference_audio: Optional[np.ndarray] = None   # 24 kHz Kokoro output
        self.last_user_audio: Optional[np.ndarray] = None   # 16 kHz recorded attempt
        self.recent_phrases: List[str] = []
        # Last analysis prosody, kept so the canvases can redraw on window resize.
        self._last_prosody: Optional[dict] = None
        # Last user name written to settings.json; lets on_user_name_changed
        # skip the file write when the field loses focus without an edit.
        self._saved_user_name: str = config.USER_NAME

        # Initialize core modular sub-managers
        self.tts_mgr = TTSManager()

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

        self.setup_styles()
        self.build_ui()
        self.bind_events()

        # Load all models in the background to keep the UI responsive.
        threading.Thread(target=self.load_components, daemon=True).start()

    def bind_events(self):
        self.root.bind("<KeyPress-space>", self.on_keyboard_press)
        self.root.bind("<KeyRelease-space>", self.on_keyboard_release)
        self.root.bind("<Escape>", lambda _: self.quit_app())
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

    # ------------------------------------------------------------------
    # Startup: LLM server + model loading
    # ------------------------------------------------------------------
    def load_components(self):
        logging.info("Starting model loading thread...")
        self.root.after(0, self.update_status, "Loading models...", THEME["warn"])
        self.root.after(0, self.append_system_msg, "Loading TTS and pronunciation models...")

        try:
            self.tts_mgr.load_model()
            logging.info("TTS model loaded.")

            self.root.after(0, self.append_system_msg, "Loading Wav2Vec2 (pronunciation, ~1.2 GB on first run)...")
            pronounce.load_models()
            logging.info("Wav2Vec2 model loaded.")

            if self.llm_backend == "local_server":
                model_name = os.path.basename(config.EXTERNAL_MODEL_PATH)
                self.root.after(0, self.append_system_msg, f"Starting LLM server with {model_name}...")
                self.root.after(0, self.update_status, "Starting LLM server...", THEME["warn"])
                if not self.llm_server.start(self.llm_mgr):
                    self.root.after(0, self.append_error_msg, "Error: LLM server failed to start. Check model path and GPU memory.")
                    self.root.after(0, self.update_status, "LLM Server Error", THEME["bad"])
                    self.root.after(0, self.update_instruction, "LLM server failed to start. Check the log and restart.")
                    return
                self.root.after(0, self.append_system_msg, "LLM server is ready.")
            else:
                self.llm_mgr.init_client()
                if not self.llm_mgr.check_connection():
                    self.root.after(0, self.append_error_msg, "Warning: LM Studio is offline. Start it to generate phrases!")

            self.root.after(0, self.update_status, "Warming up models...", THEME["warn"])
            self.tts_mgr.warm_up()
            pronounce.warm_up()
            logging.info("Models warmed up.")

            self.root.after(0, self.load_practice_text)
            self.root.after(0, self.make_app_ready)
            logging.info("EchoLoop initialization complete.")

        except Exception as e:
            logging.exception("Error during initialization thread:")
            self.root.after(0, self.append_error_msg, f"Initialization Error: {e}")
            self.root.after(0, self.update_status, "Initialization Failed", THEME["bad"])

    def load_practice_text(self):
        """Pre-fill the source panel from the practice text file (main thread)."""
        text = ""
        try:
            with open(config.PRACTICE_TEXT_FILE, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            logging.warning(f"Could not read practice text file: {e}")
            text = "Hello and welcome to EchoLoop. Edit this text and click New phrase to begin."

        self.source_text.delete("1.0", tk.END)
        self.source_text.insert("1.0", text.strip())

    def make_app_ready(self):
        self.app_ready = True
        self.draw_mic_button("idle")
        self.update_status("Ready", THEME["ready"])
        self.update_instruction("Edit the text, then click 'New phrase' to begin.")
        self.generate_btn.config(state=tk.NORMAL)
        self.append_system_msg("Ready. Generate a phrase, listen, then hold SPACE to repeat it.")
        # Speak a personal greeting first, then auto-generate the first phrase.
        # The name is read here, on the main thread (Tk is not thread-safe).
        name = self.user_name_var.get().strip()
        threading.Thread(target=self._greet_and_start, args=(name,), daemon=True).start()

    def _greet_and_start(self, name: str):
        """Speak a greeting, then start the first phrase. (Background thread.)

        The greeting uses the same Kokoro voice as the reference phrases. Any
        failure here is non-fatal — the first phrase is generated regardless,
        so a TTS hiccup cannot leave the app stuck without a phrase.
        """
        try:
            greeting = f"Hello {name}, listen and repeat." if name else "Hello, listen and repeat."
            self.root.after(0, self.append_system_msg, greeting)
            audio = self.tts_mgr.synthesize(greeting, voice=self._selected_voice())
            if audio.size > 0:
                self.tts_mgr.play_array(audio, KOKORO_SAMPLE_RATE,
                                        self._new_playback_event(), self.shutdown_event)
        except Exception:
            logging.exception("Greeting error:")
        finally:
            # Auto-generate the first phrase so the user isn't met with an
            # empty phrase card; afterwards generation is driven by the New
            # phrase button. Scheduled via after() — on_generate_phrase
            # touches widgets, so it must run on the Tk main thread.
            self.root.after(0, self.on_generate_phrase)

    # ------------------------------------------------------------------
    # Phrase generation + Prompt phase
    # ------------------------------------------------------------------
    def _selected_voice(self) -> str:
        """Return the currently selected Kokoro voice, falling back to the default."""
        try:
            return self.voice_var.get() or config.KOKORO_VOICE
        except AttributeError:
            return config.KOKORO_VOICE

    def on_voice_changed(self, event=None):
        """Regenerate the phrase with the newly chosen voice.

        Re-using the standard generation path is simpler than re-synthesizing the
        current reference, and it also refreshes the analysis reference. If the app
        is busy the change is ignored here and simply applies to the next phrase.
        """
        logging.info(f"Reference voice changed to {self._selected_voice()}.")
        # Return focus to the window so the spacebar push-to-talk keeps working.
        self.root.focus_set()
        if self.app_ready and not self.is_generating:
            self.on_generate_phrase()

    def _selected_length(self) -> str:
        """Map the length selector label to generate_phrase's mode ('full'/'fragment')."""
        try:
            return "fragment" if self.length_var.get() == LENGTH_FEW_WORDS else "full"
        except AttributeError:
            return "full"

    def on_speed_changed(self, event=None):
        """Replay the reference at the newly selected speed so it is heard right away."""
        logging.info(f"Reference speed changed to {self.playback_speed.get()!r}.")
        # Return focus to the window so the spacebar push-to-talk keeps working.
        self.root.focus_set()
        # Replay only when the Reference button is also allowed (a phrase is
        # ready and nothing is recording/analyzing).
        if str(self.ref_btn["state"]) == str(tk.NORMAL):
            self.play_reference()

    def on_user_name_changed(self, event=None):
        """Persist the user name to settings.json once editing finishes (FocusOut)."""
        name = self.user_name_var.get().strip()
        if name == self._saved_user_name:
            return  # unchanged — don't rewrite the file
        if config.save_user_setting("user_name", name):
            self._saved_user_name = name
            logging.info(f"User name saved: {name!r}.")
        else:
            self.append_error_msg("Could not save the user name to settings.json.")

    def on_length_changed(self, event=None):
        """Regenerate the phrase when the desired length changes."""
        logging.info(f"Phrase length changed to {self.length_var.get()!r}.")
        # Return focus to the window so the spacebar push-to-talk keeps working.
        self.root.focus_set()
        if self.app_ready and not self.is_generating:
            self.on_generate_phrase()

    def on_generate_phrase(self):
        if not self.app_ready or self.is_generating:
            return
        with self.processing_lock:
            if self.is_processing_audio:
                return  # don't generate mid-analysis

        # Read the editable source text on the main thread (Tk is not thread-safe).
        source_text = self.source_text.get("1.0", tk.END).strip()
        if not source_text:
            self.append_error_msg("Please enter some practice text first.")
            return

        self.is_generating = True
        self.current_phrase = None
        self.generate_btn.config(state=tk.DISABLED)
        self.ref_btn.config(state=tk.DISABLED)
        self.user_btn.config(state=tk.DISABLED)
        self.test_btn.config(state=tk.DISABLED)
        self.last_user_audio = None
        self.draw_mic_button("processing")
        self.update_status("Generating phrase (LLM)...", THEME["info"])
        self.update_instruction("Generating a new phrase...")

        threading.Thread(target=self._generate_and_prompt, args=(source_text,), daemon=True).start()

    def _generate_and_prompt(self, source_text: str):
        """Generate one phrase, synthesize the reference, and play it. (Background thread.)"""
        try:
            phrase = self.llm_mgr.generate_phrase(
                source_text, self.recent_phrases, length=self._selected_length())
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
            self.current_voice = voice
            self.reference_audio = reference_audio
            if DEBUG_DUMP_RECORDINGS:
                dump_record_wav(reference_audio, RECORD_MODEL_FILE, KOKORO_SAMPLE_RATE)
            self.recent_phrases.append(phrase)
            if len(self.recent_phrases) > config.PHRASE_GEN_RECENT_MEMORY:
                self.recent_phrases.pop(0)

            # Show the phrase and play the reference for the user to hear
            # (fresh per-playback stop event; see _new_playback_event).
            self.root.after(0, self._show_new_phrase, phrase)
            # Honor the selected reference speed (see play_reference for the
            # lowered-sample-rate slowing approach) instead of always 1.0×.
            effective_sr = int(KOKORO_SAMPLE_RATE * self._selected_speed())
            self.tts_mgr.play_array(self.reference_audio, effective_sr,
                                    self._new_playback_event(), self.shutdown_event)

            self.root.after(0, self._phrase_ready)

        except Exception as e:
            logging.exception("Phrase generation / prompt error:")
            self.root.after(0, self._phrase_generation_failed, f"Error: {e}")

    def _show_new_phrase(self, phrase: str):
        self.phrase_label.config(text=phrase)
        self.append_system_msg(f"New phrase: {phrase}")
        self.update_status("Listen to the reference...", THEME["reference"])
        self.draw_mic_button("speaking")
        self.update_instruction("Listening to the example...")

    def _phrase_ready(self):
        self.is_generating = False
        self.draw_mic_button("idle")
        self.update_status("Your turn", THEME["ready"])
        self.update_instruction("Hold SPACE or click the mic, then repeat the phrase.")
        self.generate_btn.config(state=tk.NORMAL)
        self.ref_btn.config(state=tk.NORMAL)  # reference can be replayed any time now
        self.test_btn.config(state=tk.NORMAL)  # reference self-test available now

    def _phrase_generation_failed(self, message: str):
        self.is_generating = False
        self.append_error_msg(message)
        self.draw_mic_button("idle")
        self.update_status("Ready", THEME["ready"])
        self.update_instruction("Click 'New phrase' to try again.")
        self.generate_btn.config(state=tk.NORMAL)

    # ------------------------------------------------------------------
    # Recording controls (shared recording path)
    # ------------------------------------------------------------------
    def on_gui_btn_press(self):
        if not self.space_is_held:
            self.trigger_recording_start()

    def on_gui_btn_release(self):
        if self.recorder.is_active() and not self.space_is_held:
            self.trigger_recording_stop()

    def _typing_in_text_field(self) -> bool:
        """True when a text-input widget owns focus — spacebar should type, not record."""
        return isinstance(self.root.focus_get(), (tk.Entry, tk.Text))

    def on_keyboard_press(self, event):
        if event.keysym == "space" and not self.space_is_held \
                and not self._typing_in_text_field():
            self.space_is_held = True
            self.trigger_recording_start()

    def on_keyboard_release(self, event):
        if event.keysym == "space" and self.space_is_held:
            self.space_is_held = False
            self.trigger_recording_stop()

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
        # Re-enabled in _show_feedback / _reset_to_retry.
        self.generate_btn.config(state=tk.DISABLED)
        self.ref_btn.config(state=tk.DISABLED)
        self.user_btn.config(state=tk.DISABLED)
        self.test_btn.config(state=tk.DISABLED)
        self.root.after(0, self.draw_mic_button, "recording")
        self.root.after(0, self.update_status, "Recording...", THEME["bad"])
        self.root.after(0, self.update_instruction, "Release when finished speaking.")

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

        self.root.after(0, self.draw_mic_button, "processing")
        self.root.after(0, self.update_status, "Analyzing pronunciation...", THEME["warn"])
        threading.Thread(target=self._finalize_recording, daemon=True).start()

    def _finalize_recording(self):
        """Join the record thread, then run analysis — off the main thread.

        The is_processing_audio slot was claimed by trigger_recording_stop and
        is always released here, whatever happens.
        """
        try:
            if not self.recorder.join():
                # The capture thread is stuck (e.g. the device hangs on close)
                # and its callback may still be appending chunks. Reading them
                # now would race the writer — drop the take.
                self.root.after(0, self.append_error_msg,
                                "Audio device did not stop in time — take discarded, please try again.")
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
        self.root.after(0, self.append_error_msg, "Reached maximum record limit — take cut off.")
        self.root.after(0, self.trigger_recording_stop)

    def _on_record_stream_error(self):
        """The input stream failed (called on the capture thread).

        Re-enable the buttons disabled at recording start (without this a
        failed stream open would leave the whole UI locked), then show the
        error status on top of the reset's default "Ready".
        """
        self.root.after(0, self._reset_to_retry)
        self.root.after(0, self.update_status, "Recording Error", THEME["bad"])

    # ------------------------------------------------------------------
    # Analyze phase
    # ------------------------------------------------------------------
    def on_test_reference(self):
        """Diagnostic: feed the reference audio through analysis instead of a
        recording. Since the reference is compared against itself it should score
        near 100 — a quick way to sanity-check the pipeline without speaking.
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
            self.root.after(0, self.draw_mic_button, "speaking")
            self.root.after(0, self.update_status, "Playing reference...", THEME["reference"])
            self.tts_mgr.play_array(self.reference_audio, KOKORO_SAMPLE_RATE,
                                    self._new_playback_event(), self.shutdown_event)

            # Then run analysis with the reference as both inputs.
            self.root.after(0, self.draw_mic_button, "processing")
            self.root.after(0, self.update_status, "Testing with reference...", THEME["warn"])
            result = pronounce.analyze(
                user_audio=self.reference_audio,
                expected_text=self.current_phrase,
                reference_audio=self.reference_audio,
                user_sr=KOKORO_SAMPLE_RATE,       # reference is Kokoro's 24 kHz output
                reference_sr=KOKORO_SAMPLE_RATE,
                voice=self.current_voice,
            )
            self.root.after(0, self._show_feedback, result)
        except Exception:
            logging.exception("Reference self-test error:")
            self.root.after(0, self.append_error_msg, "Reference test failed.")
            self.root.after(0, self._reset_to_retry)
        finally:
            with self.processing_lock:
                self.is_processing_audio = False

    def analyze_recording(self):
        try:
            audio = self.recorder.get_audio()
            if audio is None or len(audio) < config.AUDIO_SAMPLE_RATE * 0.2:
                logging.warning("Captured audio too short or empty.")
                self.root.after(0, self.append_error_msg, "Audio is too short. Hold the mic longer and try again.")
                self.root.after(0, self._reset_to_retry)
                return

            if DEBUG_DUMP_RECORDINGS:
                dump_record_wav(audio, RECORD_RAW_FILE, config.AUDIO_SAMPLE_RATE)

            audio = normalize_audio(audio)
            self.last_user_audio = audio

            if DEBUG_DUMP_RECORDINGS:
                dump_record_wav(audio, RECORD_NORMALIZED_FILE, config.AUDIO_SAMPLE_RATE)

            if self.current_phrase is None or self.reference_audio is None:
                self.root.after(0, self.append_error_msg, "No active phrase to compare against.")
                self.root.after(0, self._reset_to_retry)
                return

            # Play the just-recorded audio back to the user right away, before the
            # (slower) pronunciation analysis runs (fresh per-playback stop event;
            # see _new_playback_event).
            self.root.after(0, self.draw_mic_button, "speaking")
            self.root.after(0, self.update_status, "Playing your recording...", THEME["reference"])
            self.tts_mgr.play_array(self.last_user_audio, config.AUDIO_SAMPLE_RATE,
                                    self._new_playback_event(), self.shutdown_event)
            self.root.after(0, self.draw_mic_button, "processing")
            self.root.after(0, self.update_status, "Analyzing pronunciation...", THEME["warn"])

            analyze_start = time.perf_counter()
            result = pronounce.analyze(
                user_audio=audio,
                expected_text=self.current_phrase,
                reference_audio=self.reference_audio,
                user_sr=config.AUDIO_SAMPLE_RATE,
                reference_sr=KOKORO_SAMPLE_RATE,
                voice=self.current_voice,
            )
            elapsed_ms = (time.perf_counter() - analyze_start) * 1000
            logging.info(f"Pronunciation analysis done in {elapsed_ms:.0f}ms. Score={result.score}")

            self.root.after(0, self._show_feedback, result)

        except Exception:
            logging.exception("Error in analyze_recording:")
            self.root.after(0, self.append_error_msg, "Analysis error. Please try again.")
            self.root.after(0, self._reset_to_retry)

    def _reset_to_retry(self):
        """Return to a state where the user can record the current phrase again.

        Also re-enables the buttons disabled while recording, according to what
        is actually available (phrase / last recording).
        """
        self.draw_mic_button("idle")
        self.update_status("Ready", THEME["ready"])
        self.generate_btn.config(state=tk.NORMAL)
        if self.current_phrase:
            self.ref_btn.config(state=tk.NORMAL)
            self.test_btn.config(state=tk.NORMAL)
            self.update_instruction("Hold SPACE or click the mic to repeat the phrase.")
        else:
            self.update_instruction("Click 'New phrase' to begin.")
        if self.last_user_audio is not None and self.last_user_audio.size > 0:
            self.user_btn.config(state=tk.NORMAL)

    # ------------------------------------------------------------------
    # Playback (reference / own recording)
    # ------------------------------------------------------------------
    def _selected_speed(self) -> float:
        """Parse the reference-speed selector (e.g. '0.9×') into a float.

        Falls back to normal speed (1.0) if the value is missing or malformed.
        """
        try:
            return float(self.playback_speed.get().rstrip("×"))
        except (ValueError, AttributeError):
            return 1.0

    def play_reference(self):
        if self.reference_audio is None or self.reference_audio.size == 0:
            return
        # Slowing is done by lowering the effective sample rate: playing 24 kHz
        # audio at, say, 12 kHz (0.5×) makes it twice as long. This is the simple
        # resampling approach — it also shifts the pitch down, no extra deps.
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
        playback's event and it stays set — nothing is ever cleared from under
        a still-running playback.
        """
        event = threading.Event()
        self.playback_stop_event = event
        return event

    def _play_async(self, waveform: np.ndarray, sample_rate: int, status: str):
        """Play a waveform in a background thread, stopping any current playback first."""
        self.stop_playback()
        stop_event = self._new_playback_event()
        self.update_status(status, THEME["reference"])

        def _worker():
            self.tts_mgr.play_array(waveform, sample_rate, stop_event, self.shutdown_event)
            self.root.after(0, self._playback_finished, stop_event)

        threading.Thread(target=_worker, daemon=True).start()

    def _playback_finished(self, stop_event: threading.Event):
        """Restore the Ready status unless this playback was stopped/superseded.

        Without the check, the worker of an interrupted playback would
        overwrite the status set by whatever replaced it (e.g. a newer
        playback's "Playing..." line).
        """
        if stop_event is self.playback_stop_event and not stop_event.is_set():
            self.update_status("Ready", THEME["ready"])

    def stop_playback(self):
        self.playback_stop_event.set()
        self.tts_mgr.stop_playback()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def quit_app(self):
        if self._closing:
            return  # Escape and the window-close button can both land here
        self._closing = True
        logging.info("Shutting down EchoLoop...")
        self.shutdown_event.set()
        self.stop_playback()

        # Stop a recording in progress so the capture thread exits its loop and
        # closes the input stream before the process goes away.
        self.recorder.stop()
        self.recorder.join()

        self.llm_server.shutdown()

        self.root.destroy()

    def run(self):
        self.root.mainloop()
        # Flush and close log handlers only after the UI (and quit_app) is done.
        # Closing them inside quit_app produced 'I/O operation on closed file'
        # noise from daemon threads still logging during the shutdown itself.
        logging.shutdown()


if __name__ == "__main__":
    app = PronunciationTrainerGUI()
    app.run()
