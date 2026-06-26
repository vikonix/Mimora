# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Pronunciation engine dispatcher.

The host (main.py) drives pronunciation through this one module instead of a
specific engine, so it never knows which backend is active. The backend is chosen
by ``config.ENGINE`` (set in mimora/config.py):

    "acoustic" -> pronunciation.acoustic  (Wav2Vec2 embeddings + cosine-DTW; default)
    "phoneme"  -> pronunciation.phoneme   (espeak reference + phoneme ASR + edit distance)

Both backends expose the same small interface (``configure`` / ``load_models`` /
``warm_up`` / ``analyze``) and return the shared ``pronunciation.common.PronunciationResult``,
so switching is a single config flip. Only the selected backend is imported, so the
inactive engine's (heavy) weights are never loaded -- task §3 requirement #3.

To switch engines, edit ``ENGINE`` in mimora/config.py and restart; the process
binds one backend at startup (the first ``_backend()`` call). Hot-swapping without
a restart is part of the acceptance step (§7) and is not done here.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from mimora import config

# config.ENGINE value -> the package implementing it.
_BACKENDS = {
    "acoustic": "pronunciation.acoustic",
    "phoneme": "pronunciation.phoneme",
}

# The imported backend module, bound lazily on first use so the inactive engine is
# never imported (and never loads its weights).
_module = None


def name() -> str:
    """The active engine name, falling back to 'acoustic' for an unknown value."""
    return config.ENGINE if config.ENGINE in _BACKENDS else "acoustic"


def _backend():
    """Return the active backend module, importing it once on first use."""
    global _module
    if _module is None:
        _module = importlib.import_module(_BACKENDS[name()])
    return _module


def configure(engine_name: str | None = None) -> None:
    """Build an engine's AnalyzerConfig from app settings and inject it.

    Each engine has its own AnalyzerConfig (different fields and model defaults), so
    the per-engine construction lives here -- the one place that knows both the app
    settings and which backend is active. Call once at startup (no args -> the active
    engine) before load_models().

    A calibration CLI passes an explicit ``engine_name`` to configure that specific
    engine regardless of ``config.ENGINE`` (e.g. acoustic/calibrate.py always
    calibrates "acoustic"), so the per-engine field mapping is not duplicated there.
    """
    if engine_name and engine_name in _BACKENDS:
        eng = importlib.import_module(_BACKENDS[engine_name])
        eng_name = engine_name
    else:
        eng = _backend()
        eng_name = name()
    if eng_name == "phoneme":
        cfg = eng.AnalyzerConfig(
            model_name=config.WAV2VEC2_PHONEME_MODEL_NAME,
            device=config.WAV2VEC2_DEVICE,
            espeak_language=config.ESPEAK_LANGUAGE,
            score_threshold=config.PRONUNCIATION_SCORE_THRESHOLD,
            log_dir=Path(config.LOG_DIR),
            user_name=config.USER_NAME,
        )
    else:
        cfg = eng.AnalyzerConfig(
            model_name=config.WAV2VEC2_MODEL_NAME,
            device=config.WAV2VEC2_DEVICE,
            espeak_language=config.ESPEAK_LANGUAGE,
            score_threshold=config.PRONUNCIATION_SCORE_THRESHOLD,
            acoustic_good=config.PRONUNCIATION_ACOUSTIC_GOOD,
            log_dir=Path(config.LOG_DIR),
            user_name=config.USER_NAME,
        )
    eng.configure(cfg)


def load_models() -> None:
    """Load the active engine's models once (call in a background thread)."""
    _backend().load_models()


def warm_up() -> None:
    """Run the active engine's warm-up pass to remove first-call latency."""
    _backend().warm_up()


def analyze(*args, **kwargs):
    """Run the active engine's analysis, returning a ``PronunciationResult``.

    Arguments are passed straight through, so the host calls one ``engine.analyze(...)``
    exactly as it called the engine's ``analyze(...)`` before.
    """
    return _backend().analyze(*args, **kwargs)
