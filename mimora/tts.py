# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

import logging
import os
import time
from threading import Event, Thread
from typing import Optional
import numpy as np
import sounddevice as sd
from kokoro import KModel, KPipeline
from mimora import config
from mimora.audio_io import (
    WINSOUND_AVAILABLE,
    WINSOUND_LEAD_IN_SECONDS,
    reset_portaudio,
    stream_closed,
    stream_opened,
    uses_winsound,
)
import io
import wave

# winsound is the module actually driving Windows playback below; the
# availability flag and path-selection logic live in mimora.audio_io.
if WINSOUND_AVAILABLE:
    import winsound

# A never-set event used as a default so callers may omit stop/shutdown events.
_NULL_EVENT = Event()

# How long the winsound stop-guard thread keeps watching for a racing stop
# after playback starts (see play_array). The stop/start race window is
# microseconds; 0.2 s covers it with a huge margin without keeping an extra
# thread alive for the whole duration of long clips.
WINSOUND_STOP_GUARD_SECONDS = 0.2

# Language-specific warm-up words to prevent out-of-vocabulary phoneme warnings.
# Keys are Kokoro v1.0 lang_codes; German ("g") and Russian ("r") are omitted
# because Kokoro v1.0 ships no voices for them (there is no pipeline to warm).
KOKORO_WARMUP_WORDS = {
    "a": "Hi.",        # English (American)
    "b": "Hi.",        # English (British)
    "e": "Hola.",      # Spanish
    "f": "Salut.",     # French
    "h": "नमस्ते",     # Hindi
    "i": "Ciao.",      # Italian
    "j": "こんにちは", # Japanese
    "p": "Olá.",       # Portuguese
    "z": "你好",       # Chinese
}


def loudness_envelope(waveform: np.ndarray, sample_rate: int,
                      fps: int = 30, gain: float = 8.0) -> list:
    """Pre-compute a per-frame loudness track for the articulation face.

    Returns openness values in ``[0, 1]`` -- one per animation frame at ``fps``
    -- from the windowed RMS of ``waveform``. The mapping mirrors
    ``FaceWidget.level_from_rms`` (same ``gain``) so a pre-computed track looks
    identical to live RMS feeding. The whole signal is known before playback,
    so the face can be driven by a wall-clock timer instead of a per-frame
    audio callback -- which winsound does not provide on Windows.
    """
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.size == 0:
        return []
    frame = max(1, int(round(sample_rate / fps)))
    pad = (-len(audio)) % frame  # right-pad with silence to a whole frame count
    if pad:
        audio = np.concatenate([audio, np.zeros(pad, dtype=np.float32)])
    frames = audio.reshape(-1, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    return np.clip(rms * gain, 0.0, 1.0).tolist()


class TTSManager:
    def __init__(self):
        self.model = None
        self.pipeline = None

    def stop_playback(self):
        """Immediately interrupts any active winsound playback (Windows only)."""
        if WINSOUND_AVAILABLE:
            winsound.PlaySound(None, 0)

    def playback_lead_in_seconds(self) -> float:
        """Silence play_array prepends before audio, in seconds (0 if none).

        The face animation prepends the same amount of closed-mouth frames so
        the mouth stays shut during the Windows audio-session warm-up and the
        talking animation lines up with the sound.
        """
        return WINSOUND_LEAD_IN_SECONDS if uses_winsound() else 0.0

    def load_model(self):
        """Instantiates Kokoro TTS network into memory."""
        self.model = KModel(repo_id="hexgrad/Kokoro-82M").to(config.DEVICE)
        self.pipeline = KPipeline(lang_code=config.KOKORO_LANG_CODE)
        self._prefetch_voices()

    def _prefetch_voices(self):
        """Download every selectable voice once, while the Hub is still online.

        Kokoro fetches a voice's data lazily on first use. In offline mode that
        lazy download fails, so switching to a voice the user never tried would
        break. We pre-pull all configured voices during the first (online) run;
        on later offline runs they are already cached and this is skipped.
        """
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            return  # offline: anything not already cached can't be fetched anyway
        for voice in config.KOKORO_VOICES:
            try:
                self.pipeline.load_voice(voice)
            except Exception as error:
                logging.debug(f"Could not prefetch Kokoro voice {voice!r}: {error}")

    def warm_up(self):
        """Runs a mock synthesis pass to eliminate initial latency."""
        if self.model is None or self.pipeline is None:
            raise RuntimeError("TTS model not loaded. Call load_model() first.")
        warmup_word = KOKORO_WARMUP_WORDS.get(config.KOKORO_LANG_CODE, "Hi.")
        list(self.pipeline(warmup_word, voice=config.KOKORO_VOICE, model=self.model))

    def synthesize(self, text: str, voice: Optional[str] = None) -> np.ndarray:
        """Synthesize ``text`` and return the raw waveform (mono float32, 24 kHz).

        ``voice`` selects the Kokoro voice; when omitted it falls back to the
        configured default. Any voice of the loaded pipeline's language works
        without reloading the model.

        Returns an empty array if there is nothing to say. The returned array is
        reused both for playback and as the reference signal fed into
        pronunciation analysis, so we synthesize once.
        """
        if self.model is None or self.pipeline is None:
            raise RuntimeError("TTS model not loaded. Call load_model() first.")

        text = " ".join(text.split())
        if not text:
            return np.zeros(0, dtype=np.float32)

        voice = voice or config.KOKORO_VOICE
        generator = self.pipeline(text, voice=voice, model=self.model)

        audio_chunks = []
        for _, _, audio in generator:
            if audio is not None and len(audio) > 0:
                audio_chunks.append(np.asarray(audio, dtype=np.float32))

        if not audio_chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(audio_chunks)

    def play_array(self, waveform: np.ndarray, sample_rate: int,
                   stop_event: Event = _NULL_EVENT, shutdown_event: Event = _NULL_EVENT):
        """Play a pre-synthesized / recorded waveform at the given sample rate.

        Used for the reference phrase (24 kHz Kokoro output) and to replay the
        user's own recording (16 kHz). Blocking; call from a background thread.
        """
        full_audio = np.asarray(waveform, dtype=np.float32)
        if full_audio.size == 0:
            return

        # Normalise the peak to avoid clipping. Applied before the platform
        # branch so playback loudness is identical on the winsound and
        # sounddevice paths (it used to run only for winsound).
        peak = np.max(np.abs(full_audio))
        if peak > 0:
            full_audio = full_audio / peak * 0.9

        try:
            # Windows-only robust winsound implementation (bypasses PortAudio MME
            # error 6). winsound can only target the default output device, so an
            # explicit AUDIO_OUTPUT_DEVICE forces the sounddevice path below -
            # otherwise that config option would be silently ignored on Windows.
            if uses_winsound():
                # No config.AUDIO_LOCK here (unlike the sounddevice branch below
                # and the recorder): that lock guards PortAudio init/teardown, and
                # winsound does not touch PortAudio at all. Playback and recording
                # never overlap anyway - the controller stops playback before it
                # opens the mic (see main.py trigger_recording_start).
                #
                # Prepend silence so the Windows Audio Session can initialize without
                # clipping the first ~150ms. Lead-in scales with the sample rate.
                lead_in = np.zeros(int(sample_rate * WINSOUND_LEAD_IN_SECONDS), dtype=np.float32)
                full_audio = np.concatenate([lead_in, full_audio])

                # Convert float32 (-1.0..1.0) to 16-bit PCM.
                pcm_data = (full_audio * 32767).astype(np.int16)

                wav_io = io.BytesIO()
                with wave.open(wav_io, 'wb') as wav_file:
                    wav_file.setnchannels(1)   # Mono
                    wav_file.setsampwidth(2)    # 16-bit PCM
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(pcm_data.tobytes())
                wav_bytes = wav_io.getvalue()

                if stop_event.is_set() or shutdown_event.is_set():
                    return
                # winsound cannot combine SND_MEMORY with SND_ASYNC (it raises
                # "Cannot play asynchronously from memory"), so playback stays
                # synchronous and a short-lived stop-guard thread closes the
                # stop/start race instead: a stop_playback() landing in the
                # microseconds between the check above and PlaySound taking the
                # audio channel has already fired its PlaySound(None, 0) too
                # early, and the clip would play to the end anyway (possibly
                # into an open microphone). The guard re-checks the stop events
                # for the first WINSOUND_STOP_GUARD_SECONDS of playback and
                # re-issues the interrupt, so such a stop cuts the sound within
                # ~10 ms. ``done`` keeps a guard that outlives this playback
                # from killing a superseding one.
                done = Event()

                def _stop_guard():
                    guard_deadline = time.monotonic() + WINSOUND_STOP_GUARD_SECONDS
                    while time.monotonic() < guard_deadline and not done.is_set():
                        if stop_event.is_set() or shutdown_event.is_set():
                            if not done.is_set():
                                winsound.PlaySound(None, 0)
                            return
                        time.sleep(0.01)

                Thread(target=_stop_guard, daemon=True).start()
                try:
                    # Synchronous; normally interrupted by stop_playback() ->
                    # PlaySound(None, 0).
                    winsound.PlaySound(wav_bytes, winsound.SND_MEMORY | winsound.SND_NODEFAULT)
                finally:
                    done.set()
                return

            # Fallback to sounddevice for non-Windows platforms.
            if full_audio.ndim == 1:
                full_audio = full_audio.reshape(-1, 1)

            with config.AUDIO_LOCK:
                reset_portaudio()
                stream = sd.OutputStream(
                        samplerate=sample_rate,
                        channels=config.AUDIO_CHANNELS,
                        dtype="float32",
                        blocksize=0,
                        latency=config.AUDIO_LATENCY,
                        device=config.AUDIO_OUTPUT_DEVICE,
                )
                try:
                    stream.start()
                except Exception:
                    stream.close()  # don't leak the never-started stream
                    raise
                stream_opened()  # counted only once fully started (see finally)

            try:
                chunk_size = 1024
                for i in range(0, len(full_audio), chunk_size):
                    if stop_event.is_set() or shutdown_event.is_set():
                        return
                    stream.write(full_audio[i:i + chunk_size])
            finally:
                with config.AUDIO_LOCK:
                    try:
                        stream.stop()
                        stream.close()
                    except Exception as close_error:
                        logging.debug(f"Error during sound output stream close: {close_error}")
                    stream_closed()

        except Exception:
            logging.exception("TTS playback error:")
