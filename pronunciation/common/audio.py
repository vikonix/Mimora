# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Shared, torch-free waveform helpers for the pronunciation stack.

Both engines (``pronunciation.acoustic`` / ``pronunciation.phoneme``) and the
host's prosody layer (``mimora/prosody.py``) must prepare audio identically:
the score and the prosody contours have to be measured on the same signal.
These helpers used to be mirrored in all three modules on purpose (the engines
must not import from ``mimora``); this module is the single shared copy that
keeps the layering intact - it lives beside ``PronunciationResult`` in
``pronunciation.common``, which everything is already allowed to depend on.

Deliberately free of torch/transformers, and librosa is imported lazily inside
the functions, so importing this module never pulls in the heavy ML stack -
the same constraint ``mimora/prosody.py`` documents for itself.
"""

from __future__ import annotations

import hashlib

import numpy as np

# The Wav2Vec2 recognizers and the prosody analysis expect strictly 16 kHz mono;
# every waveform is resampled to this rate before use.
TARGET_SAMPLE_RATE = 16_000

# Silence-trim threshold relative to the peak, in dB. Shared so the trimmed
# signal - and therefore the scores and the contours - line up across modules.
TRIM_TOP_DB = 30


def prepare_waveform(waveform: np.ndarray, orig_sr: int) -> np.ndarray:
    """Return a 1-D float32 mono waveform resampled to TARGET_SAMPLE_RATE."""
    import librosa

    wav = np.asarray(waveform, dtype=np.float32)

    # Down-mix to mono. torchaudio gives [channels, samples] while soundfile
    # gives [samples, channels], so average along whichever axis is smaller
    # (the channels).
    if wav.ndim > 1:
        wav = wav.mean(axis=int(np.argmin(wav.shape)))

    if orig_sr != TARGET_SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=TARGET_SAMPLE_RATE)

    return np.ascontiguousarray(wav, dtype=np.float32)


def trim_silence(wav: np.ndarray) -> np.ndarray:
    """Cut leading/trailing silence so pauses don't distort scores or contours.

    Matters especially for the user recording: peak normalization in the
    capture path boosts the noise floor of quiet takes, turning silent padding
    into loud noise with no counterpart in the clean TTS reference. Keeps the
    original audio when trimming would leave less than 0.1 s (i.e. near-silent
    input).
    """
    import librosa

    if wav.size == 0:
        return wav
    trimmed, _ = librosa.effects.trim(wav, top_db=TRIM_TOP_DB)
    if trimmed.size < int(0.1 * TARGET_SAMPLE_RATE):
        return wav
    return np.ascontiguousarray(trimmed, dtype=np.float32)


def waveform_digest(waveform: np.ndarray) -> bytes:
    """Stable content digest of a waveform, for cache keys.

    The reference caches (embeddings, recognized phonemes, prosody contours)
    need a content identity for the same waveform across repeated attempts.
    Hashing through a memoryview avoids materializing an intermediate bytes
    copy of the whole waveform (unlike ``hash(arr.tobytes())``), and a real
    SHA-1 digest - unlike Python's 64-bit ``hash()`` - makes cache-key
    collisions a non-concern.
    """
    arr = np.ascontiguousarray(waveform)
    return hashlib.sha1(memoryview(arr).cast("B")).digest()
