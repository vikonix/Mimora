"""Reference engine: the production Wav2Vec2 + cosine-DTW core (``pronounce``).

This wraps ``pronounce.analyze`` behind the common ``Engine`` interface so the
harness can treat it like any other engine. It is the **reference** the lighter
engines are compared against (see ``run_eval.py``): the experiment is whether a
text-only engine can match the verdicts of this audio-vs-audio core on English.

Unlike the light engines, this one needs the **reference recording** of each
phrase (``model.wav`` in every sample folder). The harness always provides it via
``Sample.reference_audio``; engines that don't need it just ignore it.

Status: prototype evaluation tooling. Not wired into the GUI.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import soundfile as sf

# Side-effect import: adds the project root to sys.path (so ``pronounce`` imports)
# and points phonemizer at the bundled espeak-ng. Must precede project imports.
import _bootstrap  # noqa: F401

import pronounce

from eval_core import EngineResult, Sample


def _load_mono(path: str) -> Tuple[np.ndarray, int]:
    """Load ``path`` as float32 mono plus its native sample rate.

    soundfile preserves the file's real rate; ``pronounce.analyze`` resamples
    internally, so handing it the true rate is correct (and avoids guessing).
    Multichannel input is averaged to mono.
    """
    data, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), sample_rate


class ProdEngine:
    """The production ``pronounce`` core, adapted to the ``Engine`` protocol."""

    name = "core_prod"

    def init(self) -> None:
        """Load the Wav2Vec2 weights once (heavy: ~1.2 GB on first run)."""
        pronounce.load_models()

    def parse(self, sample: Sample) -> EngineResult:
        """Score one attempt against its reference recording.

        ``passed`` is taken straight from the core (it applies the configured
        ``score_threshold``), so the reference verdict is exactly what the shipped
        app would report.
        """
        user_audio, user_sr = _load_mono(str(sample.user_audio))
        reference_audio, reference_sr = _load_mono(str(sample.reference_audio))

        result = pronounce.analyze(
            user_audio=user_audio,
            expected_text=sample.text,
            reference_audio=reference_audio,
            user_sr=user_sr,
            reference_sr=reference_sr,
        )

        return EngineResult(
            score=float(result.score),
            passed=bool(result.passed),
            detail=f"asr={result.transcription!r}",
            extra={
                "acoustic_per_step": float(result.acoustic_per_step),
                "acoustic_baseline": float(result.acoustic_baseline),
            },
        )

    def close(self) -> None:
        """Nothing to release; model weights are cached in the ``pronounce`` module."""
        return None
