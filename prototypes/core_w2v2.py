"""Test engine: the lightweight text-only pipeline (espeak -> w2v2 phonemes -> edit distance).

This is the candidate *alternative* to the production core. It needs only the
phrase **text** (the reference phonemes come from espeak-ng), not a reference
recording -- which is what makes adding languages cheap. The harness compares its
scores against ``core_prod`` to see whether it is "not worse" on English.

It is a thin adapter: all the real logic lives in ``allosaurus_pronounce_poc``
(the original spike), which this module reuses unchanged so there is a single
source of truth for the scoring. Backend is fixed to ``w2v2`` (the accurate one;
the allosaurus backend was shown to be too noisy).

Status: prototype evaluation tooling. Not wired into the GUI.
"""

from __future__ import annotations

# Side-effect import: project root on sys.path + espeak registration. First.
import _bootstrap  # noqa: F401

# The spike module holds the pipeline (phonemize, recognize, align, score). It
# sits in this same folder, which is on sys.path when a prototype is run as a
# script, so the bare import resolves.
import allosaurus_pronounce_poc as light

from eval_core import EngineResult, Sample


# Match the production core's default pass threshold (AnalyzerConfig.score_threshold
# = 70) so the verdict-agreement metric compares like with like.
DEFAULT_THRESHOLD = 70.0


class W2V2Engine:
    """espeak reference + wav2vec2 phoneme ASR + feature-weighted edit distance."""

    name = "core_w2v2"

    def __init__(
        self,
        lang: str = "en",
        device: str = "cpu",
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        if lang not in light.LANGUAGES:
            raise ValueError(
                f"unknown language {lang!r}; known: {sorted(light.LANGUAGES)}"
            )
        self.lang = lang
        self.device = device
        self.threshold = threshold
        self._spec = light.LANGUAGES[lang]

    def init(self) -> None:
        """Preload the wav2vec2 phoneme model once (heavy on first run).

        We reuse the spike's cached loader so the model is shared with any other
        caller in the process and is not reloaded per sample.
        """
        light._w2v2_model(self.device)  # populates the lru_cache

    def parse(self, sample: Sample) -> EngineResult:
        """Score the user attempt from text + audio only (reference audio unused)."""
        reference = light.reference_phonemes(sample.text, self._spec.espeak)
        spoken = light.spoken_phonemes(
            str(sample.user_audio), self._spec, backend="w2v2", device=self.device
        )
        result = light.align_and_score(reference, spoken)

        return EngineResult(
            score=float(result.score),
            passed=result.score >= self.threshold,
            detail=f"ref={' '.join(reference)} | spoken={' '.join(spoken)}",
            extra={
                "phoneme_score": float(result.phoneme_score),
                "recall": float(result.recall),
            },
        )

    def close(self) -> None:
        """Nothing to release; the model lives in the spike module's cache."""
        return None
