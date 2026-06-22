# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Mimora pronunciation analysis package.

Reuses the OpenPronounce (MIT) acoustic/phoneme comparison core as a library.
The single entry point is ``analyze``; ``load_models`` / ``warm_up`` manage the
Wav2Vec2 model lifecycle (call them in a background thread at mode startup).

The package is GUI- and application-agnostic: settings come from its own
``AnalyzerConfig``. A host injects values once at startup with ``configure()``;
without it the built-in defaults keep the package fully autonomous.
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
