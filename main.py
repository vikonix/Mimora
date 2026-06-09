import time
import subprocess
import threading
from typing import Optional, List
import os
import sys
import warnings
import logging
import tkinter as tk
import numpy as np
import sounddevice as sd

# Disable Hugging Face hub symlinks warning for a cleaner console output
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Ignore specific deprecation and model warnings from underlying libraries
warnings.filterwarnings("ignore", message="dropout option adds dropout.*")
warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")

import config
from stt import STTManager, WHISPER_SAMPLE_RATE
from llm import LLMManager
from tts import TTSManager, KOKORO_SAMPLE_RATE
import pronounce
from ui import PronunciationTrainerUI, LENGTH_FEW_WORDS

# Configure comprehensive events logging (console + file)
log_format = "%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(config.LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Technical recording & signal processing parameters
RECORDING_BLOCKSIZE = 0  # 0 → PortAudio picks an optimal block size. A small fixed
                         # size combined with low-latency buffers caused capture
                         # underruns (driver-inserted silence gaps) on Windows MME.

# Signal gain normalization parameters
AUDIO_MIN_PEAK_THRESHOLD = 0.01      # Prevents boosting pure background noise floor during silence
AUDIO_NORMALIZATION_CEILING = 0.9    # Scales the peak target output level directly to 90%

# How long to wait for the recording thread to finish after stopping.
RECORD_THREAD_JOIN_TIMEOUT_SEC = 1.5

# Debug: when True, every analyzed take is written to disk as WAV (raw capture
# before normalization, and the normalized signal) so the captured audio can be
# inspected independently of playback. Set back to False once diagnosed.
DEBUG_DUMP_RECORDINGS = True
DEBUG_DUMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug")


class PronunciationTrainerGUI(PronunciationTrainerUI):
    """Tkinter front-end for the EchoLoop pronunciation trainer.

    Inherits widget construction and rendering from PronunciationTrainerUI
    (ui.py); this class holds the controller logic (audio, threads, analysis).

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
        self.root.configure(bg="#121214")

        # Center the window on the screen. winfo_screen* are valid before the
        # first mainloop iteration, so the window appears centered from the start.
        window_width, window_height = 520, 820
        x = (self.root.winfo_screenwidth() - window_width) // 2
        y = (self.root.winfo_screenheight() - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Thread management events
        self.shutdown_event = threading.Event()
        self.playback_stop_event = threading.Event()

        # Recording state management
        self.is_recording = False
        self.space_is_held = False
        self.record_lock = threading.Lock()
        self.recorded_chunks: List[np.ndarray] = []
        self.record_thread: Optional[threading.Thread] = None
        # Sample rate the microphone is actually captured at. We record at the
        # device's native rate (via WASAPI on Windows) to avoid the driver's
        # low-quality on-the-fly resampling, then downsample to 16 kHz ourselves.
        self.capture_sr: int = WHISPER_SAMPLE_RATE

        # Audio processing guard — prevents concurrent analysis runs
        self.is_processing_audio = False
        self.processing_lock = threading.Lock()

        # Application readiness and per-phrase practice state
        self.app_ready = False
        self.is_generating = False
        self.current_phrase: Optional[str] = None
        self.reference_audio: Optional[np.ndarray] = None   # 24 kHz Kokoro output
        self.last_user_audio: Optional[np.ndarray] = None   # 16 kHz recorded attempt
        self.recent_phrases: List[str] = []
        # Last analysis prosody, kept so the canvases can redraw on window resize.
        self._last_prosody: Optional[dict] = None

        # Initialize core modular sub-managers
        self.stt_mgr = STTManager()
        self.tts_mgr = TTSManager()

        # LLM backend (used only to generate practice phrases)
        self.llm_backend = config.LLM_BACKEND
        self._llm_server_process: Optional[subprocess.Popen] = None
        self._llm_server_log_file = None

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
    def _start_llm_server(self) -> bool:
        """Launch llm_server.py as a subprocess and wait until it responds."""
        model_path = config.EXTERNAL_MODEL_PATH
        if not model_path:
            logging.error("EXTERNAL_MODEL_PATH is empty — cannot start local server.")
            return False

        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "llm_server", "server.py"),
            "--model", model_path,
            "--host", config.LOCAL_SERVER_HOST,
            "--port", str(config.LOCAL_SERVER_PORT),
            "--n-gpu-layers", str(config.EXTERNAL_N_GPU_LAYERS),
            "--n-ctx", str(config.EXTERNAL_N_CTX),
        ]
        log_path = config.LLM_SERVER_LOG_FILE
        logging.info(f"Starting LLM server: {' '.join(cmd)}")
        self._llm_server_log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        self._llm_server_process = subprocess.Popen(
            cmd, stdout=self._llm_server_log_file, stderr=self._llm_server_log_file)

        deadline = time.time() + config.LOCAL_SERVER_STARTUP_TIMEOUT
        self.llm_mgr.init_client(base_url=config.LOCAL_SERVER_URL,
                                 api_key=config.LOCAL_SERVER_API_KEY)
        while time.time() < deadline:
            if self._llm_server_process.poll() is not None:
                logging.error(f"LLM server exited unexpectedly (code {self._llm_server_process.returncode}).")
                return False
            if self.llm_mgr.check_connection(silent=True):
                logging.info("LLM server is ready.")
                return True
            time.sleep(1.0)

        logging.error("LLM server did not become ready within the timeout.")
        return False

    def load_components(self):
        logging.info("Starting model loading thread...")
        self.root.after(0, self.update_status, "Loading models...", "#ffb86c")
        self.root.after(0, self.append_system_msg, "Loading STT, TTS and pronunciation models...")

        try:
            self.stt_mgr.load_model()
            logging.info("STT model loaded.")

            self.tts_mgr.load_model()
            logging.info("TTS model loaded.")

            self.root.after(0, self.append_system_msg, "Loading Wav2Vec2 (pronunciation, ~1.2 GB on first run)...")
            pronounce.load_models()
            logging.info("Wav2Vec2 model loaded.")

            if self.llm_backend == "local_server":
                model_name = os.path.basename(config.EXTERNAL_MODEL_PATH)
                self.root.after(0, self.append_system_msg, f"Starting LLM server with {model_name}...")
                self.root.after(0, self.update_status, "Starting LLM server...", "#ffb86c")
                if not self._start_llm_server():
                    self.root.after(0, self.append_system_msg, "Error: LLM server failed to start. Check model path and GPU memory.")
                    self.root.after(0, self.update_status, "LLM Server Error", "#ff5555")
                    self.root.after(0, self.update_instruction, "LLM server failed to start. Check the log and restart.")
                    return
                self.root.after(0, self.append_system_msg, "LLM server is ready.")
            else:
                self.llm_mgr.init_client()
                if not self.llm_mgr.check_connection():
                    self.root.after(0, self.append_system_msg, "Warning: LM Studio is offline. Start it to generate phrases!")

            self.root.after(0, self.update_status, "Warming up models...", "#ffb86c")
            self.stt_mgr.warm_up()
            self.tts_mgr.warm_up()
            pronounce.warm_up()
            logging.info("Models warmed up.")

            self.root.after(0, self.load_practice_text)
            self.root.after(0, self.make_app_ready)
            logging.info("EchoLoop initialization complete.")

        except Exception as e:
            logging.exception("Error during initialization thread:")
            self.root.after(0, self.append_system_msg, f"Initialization Error: {e}")
            self.root.after(0, self.update_status, "Initialization Failed", "#ff5555")

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
        self.update_status("Ready", "#00e676")
        self.update_instruction("Edit the text, then click 'New phrase' to begin.")
        self.generate_btn.config(state=tk.NORMAL)
        self.append_system_msg("Ready. Generate a phrase, listen, then hold SPACE to repeat it.")
        # Auto-generate the first phrase so the user isn't met with an empty
        # empty phrase card; afterwards generation is driven by the New phrase button.
        self.on_generate_phrase()

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
            self.append_system_msg("Please enter some practice text first.")
            return

        self.is_generating = True
        self.current_phrase = None
        self.generate_btn.config(state=tk.DISABLED)
        self.ref_btn.config(state=tk.DISABLED)
        self.user_btn.config(state=tk.DISABLED)
        self.test_btn.config(state=tk.DISABLED)
        self.last_user_audio = None
        self.draw_mic_button("processing")
        self.update_status("Generating phrase (LLM)...", "#8be9fd")
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
            reference_audio = self.tts_mgr.synthesize(phrase, voice=self._selected_voice())
            if reference_audio.size == 0:
                self.root.after(0, self._phrase_generation_failed, "Could not synthesize the reference audio.")
                return

            self.current_phrase = phrase
            self.reference_audio = reference_audio
            self.recent_phrases.append(phrase)
            if len(self.recent_phrases) > config.PHRASE_GEN_RECENT_MEMORY:
                self.recent_phrases.pop(0)

            # Show the phrase and play the reference for the user to hear.
            self.root.after(0, self._show_new_phrase, phrase)
            self.playback_stop_event.clear()
            self.tts_mgr.play_array(self.reference_audio, KOKORO_SAMPLE_RATE,
                                    self.playback_stop_event, self.shutdown_event)

            self.root.after(0, self._phrase_ready)

        except Exception as e:
            logging.exception("Phrase generation / prompt error:")
            self.root.after(0, self._phrase_generation_failed, f"Error: {e}")

    def _show_new_phrase(self, phrase: str):
        self.phrase_label.config(text=phrase)
        self.append_system_msg(f"New phrase: {phrase}")
        self.update_status("Listen to the reference...", "#ff79c6")
        self.draw_mic_button("speaking")
        self.update_instruction("Listening to the example...")

    def _phrase_ready(self):
        self.is_generating = False
        self.draw_mic_button("idle")
        self.update_status("Your turn", "#00e676")
        self.update_instruction("Hold SPACE or click the mic, then repeat the phrase.")
        self.generate_btn.config(state=tk.NORMAL)
        self.ref_btn.config(state=tk.NORMAL)  # reference can be replayed any time now
        self.test_btn.config(state=tk.NORMAL)  # reference self-test available now

    def _phrase_generation_failed(self, message: str):
        self.is_generating = False
        self.append_system_msg(message)
        self.draw_mic_button("idle")
        self.update_status("Ready", "#00e676")
        self.update_instruction("Click 'New phrase' to try again.")
        self.generate_btn.config(state=tk.NORMAL)

    # ------------------------------------------------------------------
    # Recording controls (shared recording path)
    # ------------------------------------------------------------------
    def on_gui_btn_press(self):
        if not self.space_is_held:
            self.trigger_recording_start()

    def on_gui_btn_release(self):
        with self.record_lock:
            currently_recording = self.is_recording
        if currently_recording and not self.space_is_held:
            self.trigger_recording_stop()

    def on_keyboard_press(self, event):
        if event.keysym == "space" and not self.space_is_held:
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
        with self.record_lock:
            if self.is_recording:
                return
            logging.info("Starting audio recording...")
            self.stop_playback()  # silence any reference playback before recording
            self.is_recording = True
            self.recorded_chunks = []
            self.root.after(0, self.draw_mic_button, "recording")
            self.root.after(0, self.update_status, "Recording...", "#ff5555")
            self.root.after(0, self.update_instruction, "Release when finished speaking.")
            self.record_thread = threading.Thread(target=self.record_loop, daemon=True)
            self.record_thread.start()

    def trigger_recording_stop(self):
        with self.record_lock:
            if not self.is_recording:
                return
            logging.info("Stopping audio recording...")
            self.is_recording = False

        self.root.after(0, self.draw_mic_button, "processing")
        self.root.after(0, self.update_status, "Analyzing pronunciation...", "#ffb86c")
        threading.Thread(target=self._finalize_recording, daemon=True).start()

    def _finalize_recording(self):
        """Join the record thread, then run analysis — off the main thread."""
        if self.record_thread:
            self.record_thread.join(timeout=RECORD_THREAD_JOIN_TIMEOUT_SEC)

        with self.processing_lock:
            if self.is_processing_audio:
                logging.warning("Analysis already running, skipping duplicate.")
                return
            self.is_processing_audio = True

        try:
            self.analyze_recording()
        finally:
            with self.processing_lock:
                self.is_processing_audio = False

    def _select_capture_device(self):
        """Choose the input device and capture sample rate.

        On Windows the default PortAudio host API is MME, which drops samples
        (driver-inserted silence gaps -> clicks). WASAPI is glitch-free, so we
        prefer its default input device and capture at that device's native rate.
        Returns (device_index, sample_rate); falls back to the configured device
        at 16 kHz if WASAPI or its device cannot be resolved.
        """
        # An explicit device override always wins.
        if config.AUDIO_INPUT_DEVICE is not None:
            return config.AUDIO_INPUT_DEVICE, WHISPER_SAMPLE_RATE
        try:
            for api in sd.query_hostapis():
                if "wasapi" not in api["name"].lower():
                    continue
                dev_index = api.get("default_input_device", -1)
                if dev_index is None or dev_index < 0:
                    break
                native_sr = int(round(sd.query_devices(dev_index)["default_samplerate"]))
                logging.info(f"Capturing via WASAPI device #{dev_index} at {native_sr} Hz.")
                return dev_index, native_sr
        except Exception:
            logging.exception("WASAPI device selection failed; using defaults.")
        return config.AUDIO_INPUT_DEVICE, WHISPER_SAMPLE_RATE

    def record_loop(self):
        start_time = time.time()
        logging.info("sd.InputStream thread started.")
        callback_warnings: List[str] = []
        capture_device, self.capture_sr = self._select_capture_device()

        def callback(indata, frames, time_info, status):
            # Runs on PortAudio's realtime audio thread, which has a hard deadline.
            # It must never block, so we take no locks here: list.append is atomic
            # under the GIL, and recorded_chunks is only read after the stream is
            # closed and this thread is joined (see _finalize_recording), so there
            # is no concurrent reader to guard against. Holding record_lock here was
            # the cause of dropped samples (audible clicks/crackle) when the GUI
            # thread held the same lock during start/stop.
            if status:
                callback_warnings.append(str(status))
            self.recorded_chunks.append(indata.copy())

        try:
            with config.AUDIO_LOCK:
                try:
                    sd._terminate()
                    sd._initialize()
                except Exception as init_err:
                    logging.debug(f"PortAudio reinitialization error: {init_err}")

                stream = sd.InputStream(
                        samplerate=self.capture_sr,
                        channels=config.AUDIO_CHANNELS,
                        dtype="float32",
                        blocksize=RECORDING_BLOCKSIZE,
                        # "high" requests the host API's larger, safer buffers to
                        # stop input underruns (the source of the silence-gap clicks).
                        latency="high",
                        device=capture_device,
                        callback=callback,
                )
                stream.start()

            try:
                while True:
                    while callback_warnings:
                        logging.warning(f"Audio input warning: {callback_warnings.pop(0)}")

                    with self.record_lock:
                        still_recording = self.is_recording
                    if not still_recording:
                        break

                    if time.time() - start_time >= config.MAX_RECORD_SECONDS:
                        logging.info("Maximum recording duration reached.")
                        self.root.after(0, self.append_system_msg, "Reached maximum record limit.")
                        with self.record_lock:
                            self.is_recording = False
                        break

                    time.sleep(0.01)
            finally:
                with config.AUDIO_LOCK:
                    try:
                        stream.stop()
                        stream.close()
                    except Exception as close_error:
                        logging.debug(f"Error during sound input stream close: {close_error}")

        except Exception:
            logging.exception("Recording InputStream error:")
            with self.record_lock:
                self.is_recording = False
            self.root.after(0, self.update_status, "Recording Error", "#ff5555")

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(audio))
        logging.info(f"Normalizing audio. Peak signal level: {peak:.4f}")
        if peak < AUDIO_MIN_PEAK_THRESHOLD:
            logging.info("Peak signal is too low (silence). Skipping gain adjustment.")
            return audio.astype(np.float32)
        audio = audio / peak * AUDIO_NORMALIZATION_CEILING
        return np.nan_to_num(audio).astype(np.float32)

    def _dump_debug_wav(self, audio: np.ndarray, label: str, sample_rate: int):
        """Write a mono float32 waveform to debug/<timestamp>_<label>.wav as 16-bit PCM.

        Diagnostic only (guarded by DEBUG_DUMP_RECORDINGS). Lets the raw capture be
        inspected on disk, isolating recording artifacts from the playback path.
        """
        try:
            import wave
            os.makedirs(DEBUG_DUMP_DIR, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(DEBUG_DUMP_DIR, f"{stamp}_{label}.wav")
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            with wave.open(path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm.tobytes())
            logging.info(f"[debug] Dumped {label} capture -> {path} "
                         f"(peak={np.max(np.abs(audio)):.4f}, n={len(audio)})")
        except Exception:
            logging.exception("Failed to dump debug WAV:")

    def get_recorded_audio(self) -> Optional[np.ndarray]:
        with self.record_lock:
            if not self.recorded_chunks:
                return None
            chunks = list(self.recorded_chunks)
            self.recorded_chunks = []
        audio = np.concatenate(chunks, axis=0).flatten().astype(np.float32, copy=False)

        # Audio was captured at the device's native rate; downsample to the 16 kHz
        # the rest of the pipeline (playback, analysis, debug dump) expects.
        if self.capture_sr != WHISPER_SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=self.capture_sr,
                                     target_sr=WHISPER_SAMPLE_RATE)
        return np.ascontiguousarray(audio, dtype=np.float32)

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
        with self.processing_lock:
            if self.is_processing_audio:
                return
            self.is_processing_audio = True

        self.stop_playback()  # silence any current playback first
        threading.Thread(target=self._run_reference_test, daemon=True).start()

    def _run_reference_test(self):
        """Play the reference, then analyze it against itself (off the main thread)."""
        try:
            # Play the reference back first. stop_playback() set the stop event when
            # the test started, so clear it to allow playback here.
            self.playback_stop_event.clear()
            self.root.after(0, self.draw_mic_button, "speaking")
            self.root.after(0, self.update_status, "Playing reference...", "#ff79c6")
            self.tts_mgr.play_array(self.reference_audio, KOKORO_SAMPLE_RATE,
                                    self.playback_stop_event, self.shutdown_event)

            # Then run analysis with the reference as both inputs.
            self.root.after(0, self.draw_mic_button, "processing")
            self.root.after(0, self.update_status, "Testing with reference...", "#ffb86c")
            result = pronounce.analyze(
                user_audio=self.reference_audio,
                expected_text=self.current_phrase,
                reference_audio=self.reference_audio,
                user_sr=KOKORO_SAMPLE_RATE,       # reference is Kokoro's 24 kHz output
                reference_sr=KOKORO_SAMPLE_RATE,
            )
            self.root.after(0, self._show_feedback, result)
        except Exception:
            logging.exception("Reference self-test error:")
            self.root.after(0, self.append_system_msg, "Reference test failed.")
            self.root.after(0, self._reset_to_retry)
        finally:
            with self.processing_lock:
                self.is_processing_audio = False

    def analyze_recording(self):
        try:
            audio = self.get_recorded_audio()
            if audio is None or len(audio) < WHISPER_SAMPLE_RATE * 0.2:
                logging.warning("Captured audio too short or empty.")
                self.root.after(0, self.append_system_msg, "Audio is too short. Hold the mic longer and try again.")
                self.root.after(0, self._reset_to_retry)
                return

            if DEBUG_DUMP_RECORDINGS:
                self._dump_debug_wav(audio, "raw", WHISPER_SAMPLE_RATE)

            audio = self.normalize_audio(audio)
            self.last_user_audio = audio

            if DEBUG_DUMP_RECORDINGS:
                self._dump_debug_wav(audio, "normalized", WHISPER_SAMPLE_RATE)

            if self.current_phrase is None or self.reference_audio is None:
                self.root.after(0, self.append_system_msg, "No active phrase to compare against.")
                self.root.after(0, self._reset_to_retry)
                return

            # Play the just-recorded audio back to the user right away, before the
            # (slower) pronunciation analysis runs. stop_playback() ran when recording
            # started, so the stop event must be cleared to allow playback here.
            self.playback_stop_event.clear()
            self.root.after(0, self.draw_mic_button, "speaking")
            self.root.after(0, self.update_status, "Playing your recording...", "#ff79c6")
            self.tts_mgr.play_array(self.last_user_audio, WHISPER_SAMPLE_RATE,
                                    self.playback_stop_event, self.shutdown_event)
            self.root.after(0, self.draw_mic_button, "processing")
            self.root.after(0, self.update_status, "Analyzing pronunciation...", "#ffb86c")

            analyze_start = time.perf_counter()
            result = pronounce.analyze(
                user_audio=audio,
                expected_text=self.current_phrase,
                reference_audio=self.reference_audio,
                user_sr=WHISPER_SAMPLE_RATE,
                reference_sr=KOKORO_SAMPLE_RATE,
            )
            elapsed_ms = (time.perf_counter() - analyze_start) * 1000
            logging.info(f"Pronunciation analysis done in {elapsed_ms:.0f}ms. Score={result.score}")

            self.root.after(0, self._show_feedback, result)

        except Exception:
            logging.exception("Error in analyze_recording:")
            self.root.after(0, self.append_system_msg, "Analysis error. Please try again.")
            self.root.after(0, self._reset_to_retry)

    def _reset_to_retry(self):
        """Return to a state where the user can record the current phrase again."""
        self.draw_mic_button("idle")
        self.update_status("Ready", "#00e676")
        if self.current_phrase:
            self.update_instruction("Hold SPACE or click the mic to repeat the phrase.")
        else:
            self.update_instruction("Click 'New phrase' to begin.")

    # ------------------------------------------------------------------
    # Playback (reference / own recording)
    # ------------------------------------------------------------------
    def _selected_speed(self) -> float:
        """Parse the reference-speed selector (e.g. '0.75×') into a float.

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
        self._play_async(self.last_user_audio, WHISPER_SAMPLE_RATE, "Playing your recording...")

    def _play_async(self, waveform: np.ndarray, sample_rate: int, status: str):
        """Play a waveform in a background thread, stopping any current playback first."""
        self.stop_playback()
        self.playback_stop_event.clear()
        self.update_status(status, "#ff79c6")

        def _worker():
            self.tts_mgr.play_array(waveform, sample_rate, self.playback_stop_event, self.shutdown_event)
            self.root.after(0, self.update_status, "Ready", "#00e676")

        threading.Thread(target=_worker, daemon=True).start()

    def stop_playback(self):
        self.playback_stop_event.set()
        self.tts_mgr.stop_playback()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def quit_app(self):
        logging.info("Shutting down EchoLoop...")
        self.shutdown_event.set()
        self.stop_playback()

        if self._llm_server_process is not None:
            logging.info("Terminating LLM server subprocess...")
            self._llm_server_process.terminate()
            try:
                self._llm_server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logging.warning("LLM server did not exit cleanly — killing it.")
                self._llm_server_process.kill()

        if self._llm_server_log_file is not None:
            self._llm_server_log_file.close()

        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)
        logging.shutdown()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = PronunciationTrainerGUI()
    app.run()
