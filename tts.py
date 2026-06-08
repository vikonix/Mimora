import logging
from threading import Event
from typing import Optional
import numpy as np
import sounddevice as sd
from kokoro import KModel, KPipeline
import config
import io
import wave

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False

# Technical synthesizer & audio configurations
KOKORO_SAMPLE_RATE = 24_000   # Kokoro synthesizes native 24kHz audio outputs

# A never-set event used as a default so callers may omit stop/shutdown events.
_NULL_EVENT = Event()

# Silence padding prepended to every winsound playback block.
# Windows Audio Session needs ~50-200ms to initialize on the first call,
# which clips the very beginning of the first sentence without this buffer.
WINSOUND_LEAD_IN_SAMPLES = int(KOKORO_SAMPLE_RATE * 0.15)  # 150ms of silence

# Language-specific warm-up words to prevent out-of-vocabulary phoneme warnings
KOKORO_WARMUP_WORDS = {
    "a": "Hi.",        # English (American)
    "b": "Hi.",        # English (British)
    "e": "Hola.",      # Spanish
    "f": "Salut.",     # French
    "g": "Hallo.",     # German
    "h": "नमस्ते",     # Hindi
    "i": "Ciao.",      # Italian
    "j": "こんにちは", # Japanese
    "p": "Olá.",       # Portuguese
    "r": "Привет.",    # Russian
    "z": "你好",       # Chinese
}


class TTSManager:
    def __init__(self):
        self.model = None
        self.pipeline = None

    def stop_playback(self):
        """Immediately interrupts any active winsound playback (Windows only)."""
        if WINSOUND_AVAILABLE:
            winsound.PlaySound(None, 0)

    def load_model(self):
        """Instantiates Kokoro TTS network into memory."""
        self.model = KModel(repo_id="hexgrad/Kokoro-82M").to(config.DEVICE)
        self.pipeline = KPipeline(lang_code=config.KOKORO_LANG_CODE)

    def warm_up(self):
        """Runs a mock synthesis pass to eliminate initial latency."""
        if self.model is None or self.pipeline is None:
            raise RuntimeError("TTS model not loaded. Call load_model() first.")
        warmup_word = KOKORO_WARMUP_WORDS.get(config.KOKORO_LANG_CODE, "Hi.")
        list(self.pipeline(warmup_word, voice=config.KOKORO_VOICE, model=self.model))

    def synthesize(self, text: str, stop_event: Event = _NULL_EVENT,
                   shutdown_event: Event = _NULL_EVENT,
                   voice: Optional[str] = None) -> np.ndarray:
        """Synthesize ``text`` and return the raw waveform (mono float32, 24 kHz).

        ``voice`` selects the Kokoro voice; when omitted it falls back to the
        configured default. Any voice of the loaded pipeline's language works
        without reloading the model.

        Returns an empty array if there is nothing to say or playback was
        interrupted. The returned array is reused both for playback and as the
        reference signal fed into pronunciation analysis, so we synthesize once.
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
            if stop_event.is_set() or shutdown_event.is_set():
                return np.zeros(0, dtype=np.float32)
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

        try:
            # Windows-only robust winsound implementation (bypasses PortAudio MME error 6).
            if WINSOUND_AVAILABLE:
                # Normalise peak to avoid clipping.
                peak = np.max(np.abs(full_audio))
                if peak > 0:
                    full_audio = full_audio / peak * 0.9

                # Prepend silence so the Windows Audio Session can initialize without
                # clipping the first ~150ms. Lead-in scales with the sample rate.
                lead_in = np.zeros(int(sample_rate * 0.15), dtype=np.float32)
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

                # Synchronous; interrupted by stop_playback() -> PlaySound(None, 0).
                if stop_event.is_set() or shutdown_event.is_set():
                    return
                winsound.PlaySound(wav_bytes, winsound.SND_MEMORY | winsound.SND_NODEFAULT)
                return

            # Fallback to sounddevice for non-Windows platforms.
            if full_audio.ndim == 1:
                full_audio = full_audio.reshape(-1, 1)

            # Open the stream with a full PortAudio reset to heal HDMI/NVIDIA driver
            # disconnects caused by CUDA power-state transitions on Windows.
            with config.AUDIO_LOCK:
                try:
                    sd._terminate()
                    sd._initialize()
                except Exception as init_err:
                    logging.debug(f"PortAudio reinitialization error: {init_err}")

                stream = sd.OutputStream(
                        samplerate=sample_rate,
                        channels=config.AUDIO_CHANNELS,
                        dtype="float32",
                        blocksize=0,
                        latency=config.AUDIO_LATENCY,
                        device=config.AUDIO_OUTPUT_DEVICE,
                )
                stream.start()

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

        except Exception:
            logging.exception("TTS playback error:")

    def play_stream(self, text: str, stop_event: Event = _NULL_EVENT,
                    shutdown_event: Event = _NULL_EVENT):
        """Synthesize ``text`` with Kokoro and play it (convenience wrapper)."""
        audio = self.synthesize(text, stop_event, shutdown_event)
        self.play_array(audio, KOKORO_SAMPLE_RATE, stop_event, shutdown_event)
