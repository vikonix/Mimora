# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Mimora pronunciation analysis engines.

A single top-level package grouping the interchangeable scoring engines and the
type they share:

    pronunciation.acoustic -- Wav2Vec2 embeddings + cosine-DTW (the alternative engine)
    pronunciation.phoneme  -- espeak reference + phoneme ASR + edit distance (the default engine)
    pronunciation.common   -- the engine-neutral PronunciationResult both return,
                              plus the shared torch-free waveform helpers
                              (pronunciation.common.audio)

The host never imports a subpackage directly: ``mimora/engine.py`` selects one by
``config.ENGINE`` and exposes a uniform interface. This package intentionally does
NOT import its subpackages at import time, so importing ``pronunciation`` stays
light and only the selected engine's (heavy) dependencies are loaded.
"""
