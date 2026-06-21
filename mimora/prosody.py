"""Prosody extraction (pitch & energy contours) — the engine-agnostic audio layer.

Pitch (F0) and energy contours describe *how* something was said (intonation,
stress, rhythm), independent of *which* pronunciation engine scores the words.
So they live here, in a light module computed from the raw user and reference
waveforms in ``main.py`` regardless of the active engine — not inside any single
engine. Both the acoustic (``pronunciation/acoustic/``) and the future phoneme engine show
the exact same two charts because neither computes prosody anymore.

Why a separate module from ``mimora/prosody_utils.py``: that file holds *pure*
arithmetic helpers (``to_semitones`` / ``resample_series``) and is deliberately
kept free of the ML/audio stack so it stays trivially unit-testable. This module
pulls in ``librosa``/``scikit-learn`` for the actual signal analysis. It stays
free of ``torch``/``transformers`` (the heavy engine stack), so computing
prosody never forces the recognition model to load.

The waveform-prep helpers (``_prepare_waveform`` / ``_trim_silence``) mirror the
ones in ``pronunciation/acoustic/speech.py`` on purpose: the prosody must be measured on the
same prepared signal the acoustic engine used, yet ``pronunciation.acoustic`` is meant to stay
application-agnostic and must not import from ``mimora``. The two copies are a
handful of lines of pure numpy/librosa and carry no torch dependency.
"""

import threading
from typing import Any, Dict, List

import numpy as np
import librosa
from sklearn.preprocessing import MinMaxScaler

# F0/energy are analysed at 16 kHz (the recognition sample rate); both user and
# reference waveforms are resampled to this before any contour is taken.
TARGET_SAMPLE_RATE = 16_000
# Silence-trim threshold relative to the peak, in dB. Matches pronunciation/acoustic/speech.py
# so the trimmed signal — and therefore the contours — line up with the engine's.
TRIM_TOP_DB = 30


# =====================================================================
# Waveform preparation (kept torch-free; mirrors pronunciation/acoustic/speech.py)
# =====================================================================
def _prepare_waveform(waveform: np.ndarray, orig_sr: int) -> np.ndarray:
    """Return a 1-D float32 mono waveform resampled to TARGET_SAMPLE_RATE."""
    wav = np.asarray(waveform, dtype=np.float32)

    # Down-mix to mono. torchaudio gives [channels, samples] while soundfile gives
    # [samples, channels], so average along whichever axis is smaller (the channels).
    if wav.ndim > 1:
        wav = wav.mean(axis=int(np.argmin(wav.shape)))

    if orig_sr != TARGET_SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=TARGET_SAMPLE_RATE)

    return np.ascontiguousarray(wav, dtype=np.float32)


def _trim_silence(wav: np.ndarray) -> np.ndarray:
    """Cut leading/trailing silence so pauses don't distort the contours.

    Mirrors the engine's trim so prosody is measured on the same span the
    acoustic comparison used. Keeps the original audio when trimming would leave
    less than 0.1 s (i.e. near-silent input).
    """
    if wav.size == 0:
        return wav
    trimmed, _ = librosa.effects.trim(wav, top_db=TRIM_TOP_DB)
    if trimmed.size < int(0.1 * TARGET_SAMPLE_RATE):
        return wav
    return np.ascontiguousarray(trimmed, dtype=np.float32)


# =====================================================================
# Contour extraction
# =====================================================================
def extract_f0(audio_waveform: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Extract the fundamental frequency (F0) contour; NaNs -> 0.

    fmax must cover the intonation peaks of female reference voices (Kokoro
    af_*/bf_*: median ~200-220 Hz, expressive peaks 300-400 Hz) — anything above
    fmax is marked unvoiced and would be flattened by interpolation.
    """
    f0, _voiced_flag, _voiced_probs = librosa.pyin(audio_waveform, fmin=50, fmax=450, sr=sr)
    return np.nan_to_num(f0)


def extract_energy(audio_waveform: np.ndarray) -> np.ndarray:
    """Extract the RMS energy contour, MinMax-scaled per utterance.

    Scaling each signal to its own full range erases the level difference
    between user and reference on purpose: the capture path peak-normalizes the
    recording anyway, so only the stress/rhythm *shape* is comparable.
    """
    energy = librosa.feature.rms(y=audio_waveform)
    scaler = MinMaxScaler(feature_range=(0, 250))
    return scaler.fit_transform(energy.T).flatten()


def interpolate_f0(f0: np.ndarray) -> np.ndarray:
    """Interpolate missing (zero) F0 values to avoid gaps in the contour."""
    f0 = np.array(f0)
    mask = f0 > 0
    if not mask.any():  # fully unvoiced/silent input -> nothing to interpolate
        return f0
    return np.interp(np.arange(len(f0)), np.where(mask)[0], f0[mask])


# =====================================================================
# Reference prosody cache
# =====================================================================
# A phrase is practised many times against the same Kokoro reference, but the
# reference waveform — and therefore its F0/energy — never changes between
# attempts. Cache the most recent reference (one phrase is practised at a time)
# so repeats skip the pyin pitch tracking on the reference. This mirrors the
# embedding cache in pronunciation/acoustic/speech.py. The lock makes concurrent compute
# calls safe; in the app they are already serialized by the GUI's
# is_processing_audio guard, so it is uncontended.
_reference_cache: Dict[str, Any] = {}
_reference_cache_lock = threading.Lock()


def _reference_prosody(reference_audio: np.ndarray, reference_sr: int) -> Dict[str, np.ndarray]:
    """Return the reference F0 (interpolated) and energy contours, cached."""
    global _reference_cache
    arr = np.asarray(reference_audio)
    key = (reference_sr, arr.shape, hash(arr.tobytes()))
    with _reference_cache_lock:
        if _reference_cache.get("key") != key:
            wav = _trim_silence(_prepare_waveform(arr, reference_sr))
            _reference_cache = {
                "key": key,
                "f0": interpolate_f0(extract_f0(wav, TARGET_SAMPLE_RATE)),
                "energy": extract_energy(wav),
            }
        return _reference_cache


# =====================================================================
# Public entry point
# =====================================================================
def compute_prosody(user_audio: np.ndarray,
                    user_sr: int,
                    reference_audio: np.ndarray,
                    reference_sr: int) -> Dict[str, List[float]]:
    """Compute the four prosody contours the UI overlays (you vs reference).

    Returns a dict with plain Python lists (JSON/Tk-friendly):
    ``{"f0", "energy", "ref_f0", "ref_energy"}``. Pitch contours are
    interpolated over unvoiced gaps; the UI converts them to semitones for
    display. The reference contours are cached across repeats of the same phrase.
    """
    user_wav = _trim_silence(_prepare_waveform(user_audio, user_sr))
    reference = _reference_prosody(reference_audio, reference_sr)

    f0 = interpolate_f0(extract_f0(user_wav, TARGET_SAMPLE_RATE))
    energy = extract_energy(user_wav)

    return {
        "f0": f0.tolist(),
        "energy": energy.tolist(),
        "ref_f0": reference["f0"].tolist(),
        "ref_energy": reference["energy"].tolist(),
    }
