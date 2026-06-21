"""Mimora phoneme pronunciation engine (text-only scoring).

The lightweight alternative to the acoustic ``pronounce/`` core: it scores a take
from the phrase **text** (espeak reference phonemes) plus a wav2vec2 phoneme ASR of
the user's audio -- no per-phrase reference recording needed to score.

Public API mirrors ``pronounce/`` so a host can switch between engines through one
shared call: ``analyze`` is the single entry point; ``load_models`` / ``warm_up``
manage the recognizer lifecycle (call them in a background thread at mode startup).
Settings come from this package's own ``AnalyzerConfig``; a host injects values once
at startup with ``configure()``, and the built-in defaults keep it autonomous.

This package never imports from ``prototypes-pronunciation/`` and never touches the
GUI; the result it returns is structurally identical to
``pronounce.PronunciationResult`` so the UI stays engine-neutral.
"""

from .config import AnalyzerConfig, configure, get_config
from .speech import (
    analyze,
    load_models,
    warm_up,
    PronunciationResult,
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
