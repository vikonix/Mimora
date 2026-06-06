import numpy as np
from faster_whisper import WhisperModel
import config


COMPUTE_TYPE = "float16" if config.DEVICE == "cuda" else "int8"

# Technical transcription configuration
WHISPER_SAMPLE_RATE = 16_000   # Whisper architecture requires strict 16kHz audio layouts
WHISPER_VAD_MIN_SPEECH_MS = 250   # Shortest duration considered as valid spoken word segments
WHISPER_VAD_MIN_SILENCE_MS = 500  # Silence gap thickness required before triggering split boundaries
WHISPER_VAD_SPEECH_PAD_MS = 300   # Padding attached around text fragments to avoid chopping words


class STTManager:
    def __init__(self):
        self.model = None

    def load_model(self):
        """Instantiates the Whisper AI engine into memory."""
        self.model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.DEVICE,
            compute_type=COMPUTE_TYPE,
            cpu_threads=config.WHISPER_CPU_THREADS,
            num_workers=1,
        )

    def warm_up(self):
        """Runs a mock inference pass to eliminate initial latency."""
        if self.model is None:
            raise RuntimeError("STT model not loaded. Call load_model() first.")
        dummy_audio = np.zeros(WHISPER_SAMPLE_RATE, dtype=np.float32)
        list(self.model.transcribe(dummy_audio, language=config.TARGET_LANG_CODE, beam_size=config.WHISPER_BEAM_SIZE, vad_filter=True)[0])

    def transcribe(self, audio: np.ndarray) -> str:
        """Passes the audio waveform data into Whisper for text extraction."""
        if self.model is None:
            raise RuntimeError("STT model not loaded. Call load_model() first.")

        segments, _info = self.model.transcribe(
            audio,
            language=config.TARGET_LANG_CODE,
            task="transcribe",
            beam_size=config.WHISPER_BEAM_SIZE,
            vad_filter=True,
            vad_parameters={
                "min_speech_duration_ms": WHISPER_VAD_MIN_SPEECH_MS,
                "min_silence_duration_ms": WHISPER_VAD_MIN_SILENCE_MS,
                "speech_pad_ms": WHISPER_VAD_SPEECH_PAD_MS,
            },
            no_speech_threshold=config.WHISPER_NO_SPEECH_THRESHOLD,
            condition_on_previous_text=False,
            without_timestamps=True,
            temperature=0.0,
            initial_prompt=config.WHISPER_INITIAL_PROMPT,
        )

        text = " ".join(segment.text.strip() for segment in segments).strip()
        return self._clean_transcript(text)

    def _clean_transcript(self, text: str) -> str:
        """Removes trailing structural whitespace and capitalizes the starting character."""
        if not text:
            return ""
        text = " ".join(text.split())
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        return text
