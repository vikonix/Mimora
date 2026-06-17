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

from typing import Dict, List, Tuple

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
        verbose: bool = False,
    ) -> None:
        if lang not in light.LANGUAGES:
            raise ValueError(
                f"unknown language {lang!r}; known: {sorted(light.LANGUAGES)}"
            )
        self.lang = lang
        self.device = device
        self.threshold = threshold
        self.verbose = verbose          # include the full alignment table in detail
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

        detail = f"ref={' '.join(reference)} | spoken={' '.join(spoken)}"
        if self.verbose:
            detail += "\n" + _alignment_table(result.pairs)

        return EngineResult(
            score=float(result.score),
            passed=result.score >= self.threshold,
            detail=detail,
            extra={
                # Score components plus the raw distances calibration tunes against:
                # PHONEME_GOOD is set relative to per_phone_distance on good reads,
                # and the score is anchored against bad_baseline.
                "phoneme_score": float(result.phoneme_score),
                "recall": float(result.recall),
                "per_phone_distance": float(result.per_phone_distance),
                "bad_baseline": float(result.bad_baseline),
            },
        )

    def config(self) -> Dict[str, object]:
        """Parameters this engine ran with -- logged so each run records its setup."""
        return {
            "model": light.W2V2_PHONEME_MODEL,
            "lang": self.lang,
            "espeak": self._spec.espeak,
            "device": self.device,
            "threshold": self.threshold,
            "phoneme_good": light.PHONEME_GOOD,
            "recall_max_dist": light.RECALL_MAX_DIST,
            "weight_phoneme": light.WEIGHT_PHONEME,
            "weight_word": light.WEIGHT_WORD,
        }

    def close(self) -> None:
        """Nothing to release; the model lives in the spike module's cache."""
        return None


def _alignment_table(pairs: List[Tuple[str, str]]) -> str:
    """Render the phone alignment as ``ref hyp flag`` rows (verbose diagnostics).

    Mirrors the original spike's table: ``~`` marks a near-miss (small feature
    distance), ``*`` an insertion/deletion or a far substitution, blank an exact
    match. Makes systematic inventory mismatches visible at a glance.
    """
    rows = []
    for ref_sym, hyp_sym in pairs:
        if not ref_sym or not hyp_sym:
            flag = "*"                       # insertion / deletion
        else:
            cost = light._substitution_cost(ref_sym, hyp_sym)
            flag = "" if cost == 0 else ("~" if cost < 0.34 else "*")
        rows.append(f"                {ref_sym or '_':<4} {hyp_sym or '_':<4} {flag}")
    return "\n".join(rows)
