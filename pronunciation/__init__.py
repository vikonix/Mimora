"""Mimora pronunciation analysis engines.

A single top-level package grouping the interchangeable scoring engines and the
type they share:

    pronunciation.acoustic -- Wav2Vec2 embeddings + cosine-DTW (the default engine)
    pronunciation.phoneme  -- espeak reference + phoneme ASR + edit distance
    pronunciation.common   -- the engine-neutral PronunciationResult both return

The host never imports a subpackage directly: ``mimora/engine.py`` selects one by
``config.ENGINE`` and exposes a uniform interface. This package intentionally does
NOT import its subpackages at import time, so importing ``pronunciation`` stays
light and only the selected engine's (heavy) dependencies are loaded.
"""
