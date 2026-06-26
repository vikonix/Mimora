# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Configuration for the phoneme pronunciation engine.

``pronunciation/phoneme/`` is the text-only alternative to the acoustic ``pronunciation/acoustic/``
core: it scores a take from the phrase **text** (espeak reference phonemes) and a
phoneme ASR of the user's audio, with no per-phrase reference recording required.

Like ``pronunciation/acoustic/`` it is GUI- and application-agnostic: every tunable lives in
the small :class:`AnalyzerConfig` below. The library ships with working defaults
and is fully autonomous -- importing and calling :func:`pronunciation.phoneme.analyze`
works without any host. A host injects its own values **once at startup** with
:func:`configure`, mirroring the ``logging.basicConfig`` / ``pronunciation.acoustic.configure``
pattern; later analysis simply reads whatever is active here.

Data-derived scoring constants (the GOOD anchor, recall threshold, axis weights,
insertion cap/gate, buckets) are NOT here: they live in the per-language model
calibration next to the engine (``<lang>_model_calibration.json``, committed),
with a machine-local ``calibration.json`` (gitignored) overriding only the GOOD
anchor per user — so they stay shared with the eval/calibration tooling and out
of source.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalyzerConfig:
    """Settings consumed by the phoneme pronunciation analyzer.

    The defaults make the library usable on its own; a host application overrides
    them by building an ``AnalyzerConfig`` and passing it to :func:`configure`.
    """

    # wav2vec2 CTC model that emits espeak-style IPA phonemes. Its phone inventory
    # matches the espeak reference, so there is no inventory mismatch. Heavier than
    # a word ASR only on first download (~1.2 GB).
    model_name: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    # Device the recognizer runs on ("cuda"/"cpu").
    device: str = "cpu"
    # espeak dialect used to phonemize the reference text ("en-us"/"en-gb").
    espeak_language: str = "en-us"
    # Score (0-100) at or above which a repetition is considered acceptable.
    score_threshold: float = 70.0
    # GOOD-anchor mode for the phoneme-quality axis:
    #   "global"  -- the single PHONEME_GOOD anchor from the model calibration,
    #                optionally overridden per user by calibration.json (the
    #                default; the 0-5 bucket cutpoints were calibrated under this
    #                anchor, so production scores and the buckets stay consistent).
    #   "ceiling" -- per-phrase GOOD = the TTS reference's own per-phone distance,
    #                so a flawless read maps to 100 for each phrase (needs the
    #                reference audio, which the host passes in).
    # A missing/empty reference silently falls back to "global" (never fails).
    good_mode: str = "global"
    # Directory engine logs are written to (kept symmetric with pronunciation/acoustic/).
    log_dir: Path = Path("logs")
    # Practising user; reserved for per-user calibration ("" when unset).
    user_name: str = ""


# Active configuration for this process. The default keeps the library
# autonomous; a host app replaces it once at startup via configure().
_active: AnalyzerConfig = AnalyzerConfig()


def configure(cfg: AnalyzerConfig) -> None:
    """Install the analyzer configuration for this process.

    Call once at startup, before ``load_models()``/``analyze()``. Subsequent
    analysis reads whatever is active here.
    """
    global _active
    _active = cfg


def get_config() -> AnalyzerConfig:
    """Return the currently active analyzer configuration."""
    return _active
