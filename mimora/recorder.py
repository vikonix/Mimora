# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Microphone capture and recorded-signal helpers for Mimora.

AudioRecorder owns the capture thread, input device selection and the raw
chunk buffer; the GUI controller only starts/stops takes and collects the
result as a single 16 kHz numpy array. The pure signal helpers
(normalize_audio, dump_record_wav) live here too, so the whole
microphone-side audio path is contained in this module.
"""

import logging
import os
import threading
import time
import wave
from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd

from mimora import config
from mimora.audio_io import reset_portaudio, stream_closed, stream_opened

# Technical recording & signal processing parameters
RECORDING_BLOCKSIZE = 0  # 0 → PortAudio picks an optimal block size. A small fixed
                         # size combined with low-latency buffers caused capture
                         # underruns (driver-inserted silence gaps) on Windows MME.

# Signal gain normalization parameters
AUDIO_MIN_PEAK_THRESHOLD = 0.01      # Prevents boosting pure background noise floor during silence
AUDIO_NORMALIZATION_CEILING = 0.9    # Scales the peak target output level directly to 90%

# How long to wait for the recording thread to finish after stopping.
RECORD_THREAD_JOIN_TIMEOUT_SEC = 1.5

# When True, every take is written to disk as WAV so the audio can be inspected
# independently of playback. Only three fixed files are kept, each overwritten on
# every take (no history): the model's spoken reference, the raw mic capture, and
# the normalized signal. Set to False to disable the dumps entirely.
DEBUG_DUMP_RECORDINGS = True
RECORDS_DIR = str(config.BASE_DIR / "records")

# Fixed file names for the dumped recordings (overwritten each take).
RECORD_MODEL_FILE = "model.wav"        # what the model said (Kokoro reference)
RECORD_RAW_FILE = "raw.wav"            # raw microphone capture
RECORD_NORMALIZED_FILE = "normalized.wav"  # normalized capture
RECORD_PHRASE_FILE = "phrase.txt"      # the text of the spoken phrase


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Scale the waveform so its peak hits the normalization ceiling.

    Near-silent takes are returned unchanged so the background noise floor
    is not boosted into the analysis path.
    """
    peak = np.max(np.abs(audio))
    logging.info(f"Normalizing audio. Peak signal level: {peak:.4f}")
    if peak < AUDIO_MIN_PEAK_THRESHOLD:
        logging.info("Peak signal is too low (silence). Skipping gain adjustment.")
        return audio.astype(np.float32)
    audio = audio / peak * AUDIO_NORMALIZATION_CEILING
    return np.nan_to_num(audio).astype(np.float32)


def dump_record_wav(audio: np.ndarray, file_name: str, sample_rate: int):
    """Write a mono float32 waveform to records/<file_name> as 16-bit PCM.

    Diagnostic only (guarded by DEBUG_DUMP_RECORDINGS). The file name is fixed
    (model.wav / raw.wav / normalized.wav), so each take overwrites the previous
    one and only the latest recording is kept on disk.
    """
    try:
        os.makedirs(RECORDS_DIR, exist_ok=True)
        path = os.path.join(RECORDS_DIR, file_name)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())
        logging.info(f"[record] Saved {file_name} -> {path} "
                     f"(peak={np.max(np.abs(audio)):.4f}, n={len(audio)})")
    except Exception:
        logging.exception("Failed to save record WAV:")


def dump_record_text(text: str, file_name: str):
    """Write the spoken phrase to records/<file_name> as UTF-8 text.

    Companion to dump_record_wav (guarded by DEBUG_DUMP_RECORDINGS): the file
    name is fixed (phrase.txt), so each take overwrites the previous one and
    only the latest phrase is kept on disk, matching the dumped WAV files.
    """
    try:
        os.makedirs(RECORDS_DIR, exist_ok=True)
        path = os.path.join(RECORDS_DIR, file_name)
        with open(path, "w", encoding="utf-8") as text_file:
            text_file.write(text)
        logging.info(f"[record] Saved {file_name} -> {path}")
    except Exception:
        logging.exception("Failed to save record text:")


class AudioRecorder:
    """One-press microphone capture running on its own daemon thread.

    Usage: start() opens the capture thread, stop() asks it to finish, join()
    waits for it, get_audio() returns the take as one 16 kHz float32 array.

    All callbacks are invoked on the capture thread, so GUI callers must
    marshal any widget work onto the Tk main thread themselves (root.after):
        on_max_duration  - the take hit config.MAX_RECORD_SECONDS; the caller
                           is expected to route this through its normal stop
                           path so the take is finalized like a manual stop.
        on_stream_error  - the input stream failed; recording is already
                           flagged off, the caller only restores its UI.
        on_silence_stop  - the speaker fell silent for config.SILENCE_TIMEOUT
                           after having started speaking; like on_max_duration,
                           the caller routes this through its normal stop path
                           so the take is finalized exactly like a manual stop.
        on_level         - periodic live input level (RMS, 0..1) while a take
                           is in progress, throttled to ~20 Hz. Lets the UI show
                           that the mic is hearing, so the auto-stop does not
                           feel like a black box. Best-effort: never blocks the
                           capture loop, exceptions are swallowed.
    """

    # How often (seconds) the live input level is reported via on_level. ~20 Hz
    # is smooth enough for a level indicator without flooding the Tk event queue.
    LEVEL_EMIT_INTERVAL_SEC = 0.05

    def __init__(self, on_max_duration: Callable[[], None],
                 on_stream_error: Callable[[], None],
                 on_silence_stop: Callable[[], None],
                 on_level: Callable[[float], None]):
        self._on_max_duration = on_max_duration
        self._on_stream_error = on_stream_error
        self._on_silence_stop = on_silence_stop
        self._on_level = on_level

        self.is_recording = False
        self.record_lock = threading.Lock()
        self.recorded_chunks: List[np.ndarray] = []
        self.record_thread: Optional[threading.Thread] = None
        # Sample rate the microphone is actually captured at. We record at the
        # device's native rate (via WASAPI on Windows) to avoid the driver's
        # low-quality on-the-fly resampling, then downsample to 16 kHz ourselves.
        self.capture_sr: int = config.AUDIO_SAMPLE_RATE

    def is_active(self) -> bool:
        with self.record_lock:
            return self.is_recording

    def start(self) -> bool:
        """Begin a new take.

        Returns False if a take is already in progress, or if the previous
        take's capture thread has not exited yet (its join timed out): that
        thread's callback may still be appending, and since the callback reads
        ``self.recorded_chunks`` at call time, starting now would mix the stuck
        take's tail into the new take's buffer.
        """
        with self.record_lock:
            if self.is_recording:
                return False
            if self.record_thread is not None and self.record_thread.is_alive():
                logging.warning("Previous record thread is still alive; "
                                "refusing to start a new take.")
                return False
            logging.info("Starting audio recording...")
            self.is_recording = True
            self.recorded_chunks = []
            self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
            self.record_thread.start()
        return True

    def stop(self) -> bool:
        """Ask the capture thread to finish. Returns False if not recording."""
        with self.record_lock:
            if not self.is_recording:
                return False
            logging.info("Stopping audio recording...")
            self.is_recording = False
        return True

    def join(self, timeout: float = RECORD_THREAD_JOIN_TIMEOUT_SEC) -> bool:
        """Wait for the capture thread to finish.

        Returns True once the thread has exited (or never ran). Returns False
        if it is still alive after the timeout - its callback may then still
        be appending chunks, so the buffer must not be read.
        """
        if self.record_thread is None:
            return True
        self.record_thread.join(timeout=timeout)
        if self.record_thread.is_alive():
            logging.warning(f"Record thread still alive after {timeout}s; "
                            "its chunk buffer may still be written to.")
            return False
        return True

    def get_audio(self) -> Optional[np.ndarray]:
        """Return the finished take as a 16 kHz mono float32 array (or None).

        Only call after join() confirmed the capture thread has exited; the
        chunk buffer has no other guard against a still-running writer.
        """
        with self.record_lock:
            if not self.recorded_chunks:
                return None
            chunks = list(self.recorded_chunks)
            self.recorded_chunks = []
        audio = np.concatenate(chunks, axis=0).flatten().astype(np.float32, copy=False)

        # Audio was captured at the device's native rate; downsample to the 16 kHz
        # the rest of the pipeline (playback, analysis, debug dump) expects.
        if self.capture_sr != config.AUDIO_SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=self.capture_sr,
                                     target_sr=config.AUDIO_SAMPLE_RATE)
        return np.ascontiguousarray(audio, dtype=np.float32)

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
            return config.AUDIO_INPUT_DEVICE, config.AUDIO_SAMPLE_RATE
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
        return config.AUDIO_INPUT_DEVICE, config.AUDIO_SAMPLE_RATE

    def _record_loop(self):
        start_time = time.time()
        logging.info("sd.InputStream thread started.")
        callback_warnings: List[str] = []
        capture_device, self.capture_sr = self._select_capture_device()

        def callback(indata, frames, time_info, status):
            # Runs on PortAudio's realtime audio thread, which has a hard deadline.
            # It must never block, so we take no locks here: list.append is atomic
            # under the GIL, and recorded_chunks is only read after the stream is
            # closed and this thread is joined (see join/get_audio), so there is
            # no concurrent reader to guard against. Holding record_lock here was
            # the cause of dropped samples (audible clicks/crackle) when the GUI
            # thread held the same lock during start/stop.
            if status:
                callback_warnings.append(str(status))
            self.recorded_chunks.append(indata.copy())

        try:
            with config.AUDIO_LOCK:
                reset_portaudio()
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
                try:
                    stream.start()
                except Exception:
                    stream.close()  # don't leak the never-started stream
                    raise
                stream_opened()  # counted only once fully started (see finally)

            # Voice-activity state for the silence auto-stop. We measure the
            # level of newly arrived chunks here on the poll thread (never in the
            # realtime callback, which must not do this much work):
            #   processed       - how many chunks we have already measured, so we
            #                     only look at the ones appended since last poll.
            #   speech_started  - lead-in grace: silence is ignored until the
            #                     speaker first crosses the speech threshold, so a
            #                     slow start never clips the take before it begins.
            #   last_voice_time - wall-clock of the most recent above-threshold
            #                     chunk; the take auto-stops once the gap since it
            #                     exceeds config.SILENCE_TIMEOUT.
            #   last_level_emit - throttles on_level to LEVEL_EMIT_INTERVAL_SEC.
            processed = 0
            speech_started = False
            last_voice_time = start_time
            last_level_emit = 0.0
            try:
                while True:
                    while callback_warnings:
                        logging.warning(f"Audio input warning: {callback_warnings.pop(0)}")

                    with self.record_lock:
                        still_recording = self.is_recording
                    if not still_recording:
                        break

                    # Measure the chunks that arrived since the last poll. list
                    # slicing is safe against the appending callback under the GIL
                    # (we simply may not see the very newest chunk yet).
                    total = len(self.recorded_chunks)
                    if total > processed:
                        block = np.concatenate(self.recorded_chunks[processed:total], axis=0)
                        processed = total
                        rms = float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0
                        now = time.time()
                        if rms >= config.SILENCE_THRESHOLD:
                            speech_started = True
                            last_voice_time = now
                        if now - last_level_emit >= self.LEVEL_EMIT_INTERVAL_SEC:
                            last_level_emit = now
                            try:
                                self._on_level(rms)
                            except Exception:
                                logging.exception("on_level callback failed:")
                        # Auto-stop only after the user has actually started
                        # speaking, so the initial lead-in pause is never counted.
                        if speech_started and now - last_voice_time >= config.SILENCE_TIMEOUT:
                            logging.info("Silence timeout reached; auto-stopping take.")
                            # Routed through the caller's normal stop path, exactly
                            # like on_max_duration, so the take is finalized and
                            # analyzed rather than left stuck in recording state.
                            self._on_silence_stop()
                            break

                    if time.time() - start_time >= config.MAX_RECORD_SECONDS:
                        logging.info("Maximum recording duration reached.")
                        # The caller routes this through its normal stop path so
                        # the take is finalized and analyzed exactly like a manual
                        # stop; flipping is_recording directly here used to leave
                        # the take unanalyzed and the UI stuck in recording state.
                        self._on_max_duration()
                        break

                    time.sleep(0.01)
            finally:
                with config.AUDIO_LOCK:
                    try:
                        stream.stop()
                        stream.close()
                    except Exception as close_error:
                        logging.debug(f"Error during sound input stream close: {close_error}")
                    stream_closed()

        except Exception:
            logging.exception("Recording InputStream error:")
            with self.record_lock:
                self.is_recording = False
            self._on_stream_error()
