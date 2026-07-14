# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Text-to-speech: synthesis backends plus the shared playback path.

Two roles live here, split on purpose (tasks/supertonic_tts_backend_task.md):

* **Synthesis backends** - one class per TTS engine, selected by the active
  language variant's data (``config.TTS_BACKEND``, never an ``if language``
  branch). Each backend exposes the same small surface: ``load_model()``,
  ``warm_up()``, ``synthesize(text, voice) -> np.ndarray`` (mono float32 at the
  backend's native rate) and a ``sample_rate`` attribute.
    - ``KokoroBackend``     - Kokoro-82M (torch), 24 kHz. English variants.
    - ``SupertonicBackend`` - Supertonic 3 (ONNX, no torch), 44.1 kHz.
      Spanish variant; weights are OpenRAIL-M licensed and download into
      ``model_cache/supertonic3/`` (env ``SUPERTONIC_CACHE_DIR``, set in
      config.py; pre-fetched by install.py).

* **Playback** - ``TTSManager.play_array`` plays any waveform at any sample
  rate (winsound on Windows, sounddevice elsewhere), so it works unchanged
  with either backend. The slowed reference replay stays playback-side: it
  plays the waveform at a lowered sample rate, which is backend-agnostic.

``TTSManager`` remains the facade main.py composes: it owns the playback path
and delegates synthesis to the selected backend, adding the ``sample_rate``
property main.py uses everywhere it previously imported KOKORO_SAMPLE_RATE.
"""

import logging
import os
import time
from threading import Event, Thread
from typing import Optional
import numpy as np
import sounddevice as sd
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

# Kokoro synthesizes native 24 kHz audio; a property of the model itself, so
# the constant lives next to the backend (moved here from mimora/audio_io.py -
# main.py now reads TTSManager.sample_rate instead of importing a constant).
KOKORO_SAMPLE_RATE = 24_000

# Supertonic 3 synthesizes native 44.1 kHz audio.
SUPERTONIC_SAMPLE_RATE = 44_100

# The warm-up word is language text, so it comes from the active language
# profile (config.TTS_WARMUP, e.g. "Hi." / "Hola.") - never a table in code.


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


class KokoroBackend:
    """Kokoro-82M synthesis (torch), 24 kHz. Used by the English variants."""

    sample_rate = KOKORO_SAMPLE_RATE

    def __init__(self):
        self.model = None
        self.pipeline = None

    def load_model(self):
        """Instantiates Kokoro TTS network into memory."""
        # Imported here, not at module top: only the backend the active
        # variant selects should pull its ML stack into the process.
        from kokoro import KModel, KPipeline
        self.model = KModel(repo_id="hexgrad/Kokoro-82M").to(config.DEVICE)
        self.pipeline = KPipeline(lang_code=config.TTS_LANG_CODE)
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
        for voice in config.TTS_VOICES:
            try:
                self.pipeline.load_voice(voice)
            except Exception as error:
                logging.debug(f"Could not prefetch Kokoro voice {voice!r}: {error}")

    def warm_up(self):
        """Runs a mock synthesis pass to eliminate initial latency.

        The word comes from the language profile (config.TTS_WARMUP): a short
        in-vocabulary word of the practiced language, so the dummy synthesis
        raises no out-of-vocabulary phoneme warnings.
        """
        if self.model is None or self.pipeline is None:
            raise RuntimeError("TTS model not loaded. Call load_model() first.")
        list(self.pipeline(config.TTS_WARMUP, voice=config.TTS_VOICE,
                           model=self.model))

    def synthesize(self, text: str, voice: str) -> np.ndarray:
        """Synthesize *text* with *voice*; mono float32 at 24 kHz.

        The caller (TTSManager) has already normalized the text and resolved
        the voice, so both arguments are non-empty here.
        """
        if self.model is None or self.pipeline is None:
            raise RuntimeError("TTS model not loaded. Call load_model() first.")

        generator = self.pipeline(text, voice=voice, model=self.model)

        audio_chunks = []
        for _, _, audio in generator:
            if audio is not None and len(audio) > 0:
                audio_chunks.append(np.asarray(audio, dtype=np.float32))

        if not audio_chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(audio_chunks)


class SupertonicBackend:
    """Supertonic 3 synthesis (ONNX runtime, no torch), 44.1 kHz.

    Used by the Spanish variant: Kokoro's Spanish is trained on little data
    (audible artifacts), while Supertonic 3 is multilingual by design and
    offers 10 voices (F1..F5, M1..M5) at a fraction of the runtime cost.
    Model weights (~400 MB) are OpenRAIL-M licensed and are therefore never
    bundled: they download on install (install.py) or on the first online run
    into ``model_cache/supertonic3/`` (env SUPERTONIC_CACHE_DIR, set in
    config.py). The download is atomic (temp dir + rename inside the package),
    so a present cache directory is a complete one - later runs are offline.
    """

    sample_rate = SUPERTONIC_SAMPLE_RATE

    def __init__(self):
        self._tts = None
        # Voice-style objects are loaded from the JSONs bundled with the model
        # (a purely local read); cached so each voice is parsed only once.
        self._styles = {}

    def load_model(self):
        """Load the ONNX sessions, downloading the model on the first run."""
        # Imported here, not at module top: only the backend the active
        # variant selects should pull onnxruntime into the process.
        from supertonic import TTS
        # The cache location comes from SUPERTONIC_CACHE_DIR (config.py points
        # it at model_cache/supertonic3/, matching the HF_HOME policy).
        # auto_download only downloads when the ONNX files are missing; once
        # install.py (or a first online run) has fetched them, startup is
        # fully offline.
        self._tts = TTS(model="supertonic-3", auto_download=True)

    def _style(self, voice: str):
        """The cached voice-style object for *voice* (loads it on first use)."""
        style = self._styles.get(voice)
        if style is None:
            style = self._tts.get_voice_style(voice_name=voice)
            self._styles[voice] = style
        return style

    def warm_up(self):
        """Runs a mock synthesis pass to eliminate initial latency.

        The word comes from the language profile (config.TTS_WARMUP), in the
        practiced language - "Hola." for the Spanish variant.
        """
        if self._tts is None:
            raise RuntimeError("TTS model not loaded. Call load_model() first.")
        self.synthesize(config.TTS_WARMUP, config.TTS_VOICE)

    def synthesize(self, text: str, voice: str) -> np.ndarray:
        """Synthesize *text* with *voice*; mono float32 at 44.1 kHz.

        The caller (TTSManager) has already normalized the text and resolved
        the voice, so both arguments are non-empty here. Synthesis always runs
        at speed 1.0 (the package default is 1.05): Mimora's slowed replay is
        a playback-side effect (lowered sample rate), so the synthesized
        waveform itself must be the neutral-speed reference.
        """
        if self._tts is None:
            raise RuntimeError("TTS model not loaded. Call load_model() first.")

        wav, _duration = self._tts.synthesize(
            text=text,
            voice_style=self._style(voice),
            lang=config.TTS_LANG_CODE,
            total_steps=config.TTS_TOTAL_STEPS,
            speed=1.0,
            verbose=False,
        )
        # The package returns shape (1, samples); flatten to the mono float32
        # contract shared with KokoroBackend.
        return np.asarray(wav, dtype=np.float32).reshape(-1)


# Backend registry: the active language variant selects by name
# (config.TTS_BACKEND, validated there against these keys via
# config.TTS_BACKEND_CHOICES). Adding a backend = one class + one entry.
TTS_BACKENDS = {
    "kokoro": KokoroBackend,
    "supertonic": SupertonicBackend,
}


class TTSManager:
    """Facade main.py composes: synthesis (delegated) plus playback (owned)."""

    def __init__(self):
        self._backend = TTS_BACKENDS[config.TTS_BACKEND]()

    @property
    def sample_rate(self) -> int:
        """Native sample rate of the active synthesis backend (Hz)."""
        return self._backend.sample_rate

    def load_model(self):
        """Loads the active backend's synthesis model into memory."""
        self._backend.load_model()

    def warm_up(self):
        """Runs a mock synthesis pass to eliminate initial latency."""
        self._backend.warm_up()

    def synthesize(self, text: str, voice: Optional[str] = None) -> np.ndarray:
        """Synthesize ``text`` and return the raw waveform (mono float32).

        The waveform is at the backend's native rate (``self.sample_rate``).
        ``voice`` selects the backend voice; when omitted it falls back to the
        configured default. Any voice of the active variant works without
        reloading the model.

        Returns an empty array if there is nothing to say. The returned array is
        reused both for playback and as the reference signal fed into
        pronunciation analysis, so we synthesize once. The whitespace collapse
        and the empty-text short-circuit live here so every backend honors the
        same contract.
        """
        text = " ".join(text.split())
        if not text:
            return np.zeros(0, dtype=np.float32)
        voice = voice or config.TTS_VOICE
        return self._backend.synthesize(text, voice)

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

    def play_array(self, waveform: np.ndarray, sample_rate: int,
                   stop_event: Event = _NULL_EVENT, shutdown_event: Event = _NULL_EVENT):
        """Play a pre-synthesized / recorded waveform at the given sample rate.

        Used for the reference phrase (the backend's native output) and to
        replay the user's own recording (16 kHz). Blocking; call from a
        background thread.
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
