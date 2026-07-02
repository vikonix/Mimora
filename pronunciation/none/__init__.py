# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""No-op pronunciation engine (scoring disabled).

Selected with ``"engine": "none"`` in settings.json, for machines too slow for
the recognizer engines: no Wav2Vec2 model is downloaded or loaded (~1.2 GB RAM
and the load time saved) and ``analyze`` returns instantly. Every take is
"accepted" (``passed=True``) so the practice loop never blocks; the learner
compares the reference and their own recording by ear instead of a score.

Public API mirrors ``pronunciation.phoneme`` / ``pronunciation.acoustic``
exactly (``configure`` / ``load_models`` / ``warm_up`` / ``analyze``) so the
dispatcher (mimora/engine.py) treats all engines the same. The returned
``PronunciationResult`` carries ``scored=False``, which tells the GUI to show a
neutral "scoring off" read-out instead of pretending a verdict was reached.
Prosody is unaffected: the host computes it from the raw waveforms outside any
engine, so the pitch/energy charts keep working in this mode.

This package never touches the GUI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pronunciation.common import PronunciationResult

# Nominal sample rates, kept for signature symmetry with the other engines
# (the audio is never inspected here).
TARGET_SAMPLE_RATE = 16_000
KOKORO_SAMPLE_RATE = 24_000


@dataclass(frozen=True)
class AnalyzerConfig:
    """Accepted for interface symmetry with the other engines; nothing is consumed."""


# Active configuration for this process (unused, kept so configure()/get_config()
# behave like the other engines' and the dispatcher needs no special case).
_active: AnalyzerConfig = AnalyzerConfig()


def configure(cfg: AnalyzerConfig) -> None:
    """Install the analyzer configuration (a no-op beyond storing it)."""
    global _active
    _active = cfg


def get_config() -> AnalyzerConfig:
    """Return the currently active analyzer configuration."""
    return _active


def load_models() -> None:
    """Nothing to load: this engine has no model."""


def warm_up() -> None:
    """Nothing to warm up: this engine performs no inference."""


def analyze(user_audio: np.ndarray,
            expected_text: str,
            reference_audio: Optional[np.ndarray] = None,
            user_sr: int = TARGET_SAMPLE_RATE,
            reference_sr: int = KOKORO_SAMPLE_RATE,
            voice: Optional[str] = None,
            is_reference: bool = False) -> PronunciationResult:
    """Accept the take without scoring it.

    The signature matches the other engines' ``analyze`` so the host calls it
    identically; every argument except ``expected_text`` (logged) is ignored.
    ``passed=True`` keeps the practice loop moving and ``scored=False`` makes
    the GUI render a neutral "scoring off" read-out instead of a verdict.
    """
    logging.info("[none] scoring disabled; take accepted (text=%r, is_ref=%s)",
                 expected_text, is_reference)
    return PronunciationResult(
        score=0.0,
        word_errors=[],
        prosody={},
        transcription="",
        passed=True,
        scored=False,
        feedback="Pronunciation scoring is off - compare the takes by ear.",
    )


__all__ = [
    "analyze",
    "load_models",
    "warm_up",
    "PronunciationResult",
    "AnalyzerConfig",
    "configure",
    "get_config",
]
