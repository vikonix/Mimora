# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Configuration for the pronunciation analysis library.

``pronunciation/acoustic/`` is a GUI- and application-agnostic core (adapted from
OpenPronounce). It must not reach back into the host application, so every
tunable setting lives in the small :class:`AnalyzerConfig` dataclass below.

The library ships with working defaults and is fully autonomous: importing and
calling :func:`pronunciation.acoustic.analyze` works without any host. A host application
injects its own values **once at startup** with :func:`configure`, mirroring the
``logging.basicConfig`` pattern -- later analysis simply reads whatever is
active here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalyzerConfig:
    """Settings consumed by the pronunciation analyzer.

    The defaults make the library usable on its own; a host application
    overrides them by building an ``AnalyzerConfig`` and passing it to
    :func:`configure`.
    """

    # Wav2Vec2 weights and the device they run on ("cuda"/"cpu").
    model_name: str = "facebook/wav2vec2-large-960h"
    device: str = "cpu"
    # espeak dialect used to phonemize the reference text ("en-us"/"en-gb").
    espeak_language: str = "en-us"
    # Score (0-100) at or above which a repetition is considered acceptable.
    score_threshold: float = 70.0
    # Pre-calibration acoustic floor: the typical per-step cosine DTW distance
    # of a *good* attempt. A per-user value in calibration.json overrides it.
    acoustic_good: float = 0.20
    # Directory the calibration sample log (acoustic_samples.jsonl) is written to.
    log_dir: Path = Path("logs")
    # Practising user; calibration is per-user ("" when unset).
    user_name: str = ""


# Active configuration for this process. The default keeps the library
# autonomous; a host app replaces it once at startup via configure().
_active: AnalyzerConfig = AnalyzerConfig()


def configure(cfg: AnalyzerConfig) -> None:
    """Install the analyzer configuration for this process.

    Call once at startup, before ``load_models()``/``analyze()``. Subsequent
    analysis reads whatever is active here. Reconfiguring with a different
    ``espeak_language`` is safe: the per-word phonemization cache (keyed by
    word only) is dropped so no phonemes of the old accent survive.
    """
    global _active
    if cfg.espeak_language != _active.espeak_language:
        # Deferred import: speech.py imports this module, so a top-level
        # import here would be circular. By the time configure() is callable
        # the package __init__ has already imported speech anyway.
        from . import speech
        speech._phonemize_word.cache_clear()
    _active = cfg


def get_config() -> AnalyzerConfig:
    """Return the currently active analyzer configuration."""
    return _active
