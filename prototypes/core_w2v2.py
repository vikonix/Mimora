"""Test engine: the lightweight text-only pipeline (espeak -> w2v2 phonemes -> edit distance).

This is the candidate *alternative* to the production core. It needs only the
phrase **text** (the reference phonemes come from espeak-ng), not a reference
recording -- which is what makes adding languages cheap. The harness compares its
scores against ``core_prod`` to see whether it is "not worse" on English.

It is a thin adapter: all the real logic lives in ``w2v2_pronounce_poc``
(the original spike), which this module reuses unchanged so there is a single
source of truth for the scoring. The recognizer is ``w2v2``; an earlier universal
recognizer was tried and dropped as too noisy.

Status: prototype evaluation tooling. Not wired into the GUI.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# Side-effect import: project root on sys.path + espeak registration. First.
import _bootstrap  # noqa: F401

# The spike module holds the pipeline (phonemize, recognize, align, score). It
# sits in this same folder, which is on sys.path when a prototype is run as a
# script, so the bare import resolves.
import w2v2_pronounce_poc as light

from eval_core import EngineResult, Sample


# Match the production core's default pass threshold (AnalyzerConfig.score_threshold
# = 70) so the verdict-agreement metric compares like with like.
DEFAULT_THRESHOLD = 70.0

# GOOD-anchor modes for the phoneme-quality axis (see W2V2Engine.parse):
#   "global"  -- the single PHONEME_GOOD constant (original behavior).
#   "ceiling" -- variant A: per-phrase GOOD = the TTS reference's own distance, so a
#                flawless read maps to 100 for each phrase (calibration-by-reference).
GOOD_MODES = ("global", "ceiling")


class W2V2Engine:
    """espeak reference + wav2vec2 phoneme ASR + feature-weighted edit distance."""

    name = "core_w2v2"

    def __init__(
        self,
        lang: str = "en",
        device: str = "cpu",
        threshold: float = DEFAULT_THRESHOLD,
        verbose: bool = False,
        good_mode: str = "ceiling",
    ) -> None:
        if lang not in light.LANGUAGES:
            raise ValueError(
                f"unknown language {lang!r}; known: {sorted(light.LANGUAGES)}"
            )
        if good_mode not in GOOD_MODES:
            raise ValueError(f"unknown good_mode {good_mode!r}; known: {GOOD_MODES}")
        self.lang = lang
        self.device = device
        self.threshold = threshold
        self.verbose = verbose          # include the full alignment table in detail
        self.good_mode = good_mode      # "global" PHONEME_GOOD, or per-phrase "ceiling"
        self._spec = light.LANGUAGES[lang]

    def init(self) -> None:
        """Preload the wav2vec2 phoneme model once (heavy on first run).

        We reuse the spike's cached loader so the model is shared with any other
        caller in the process and is not reloaded per sample.
        """
        light._w2v2_model(self.device)  # populates the lru_cache

    def parse(self, sample: Sample) -> EngineResult:
        """Score the user attempt from text + audio.

        The reference *text* always supplies the espeak phonemes. In the default
        ``good_mode="ceiling"`` (variant A) the TTS reference (``model.wav``) is
        recognized once to get this phrase's own per-phone distance, used as the GOOD
        anchor so a flawless read maps to 100 per phrase; that recognition is cached
        by path, so model.wav is decoded only once even when the harness also scores
        it standalone for the ceiling columns. In ``good_mode="global"`` the reference
        *audio* is unused (a pure text-only engine on the global ``PHONEME_GOOD``).
        """
        reference = light.reference_phonemes(sample.text, self._spec.espeak)
        spoken = light.spoken_phonemes(
            str(sample.user_audio), self._spec, backend="w2v2", device=self.device
        )

        good = self._ceiling_good(reference, sample)
        result = light.align_and_score(reference, spoken, good=good)

        detail = f"ref={' '.join(reference)} | spoken={' '.join(spoken)}"
        if self.verbose:
            detail += "\n" + _alignment_table(result.pairs)

        return EngineResult(
            score=float(result.score),
            passed=result.score >= self.threshold,
            detail=detail,
            extra={
                # Score components plus the raw distances calibration tunes against:
                # good_anchor is the GOOD actually used (per-phrase in ceiling mode);
                # the score is anchored between it and bad_baseline.
                "phoneme_score": float(result.phoneme_score),
                "recall": float(result.recall),
                "per_phone_distance": float(result.per_phone_distance),
                "bad_baseline": float(result.bad_baseline),
                "good_anchor": float(result.good),
            },
        )

    def _ceiling_good(self, reference: List[str], sample: Sample) -> Optional[float]:
        """Per-phrase GOOD anchor from the TTS reference, or None for global mode.

        Returns the ``model.wav`` per-phone distance (variant A) when ``good_mode`` is
        "ceiling" and the reference take exists; otherwise None, which leaves the
        scorer on the global ``PHONEME_GOOD``. A missing reference audio silently falls
        back to global, so the engine never fails just because a sample lacks model.wav.
        """
        if self.good_mode != "ceiling" or not sample.reference_audio.is_file():
            return None
        ceiling_spoken = light.spoken_phonemes(
            str(sample.reference_audio), self._spec, backend="w2v2", device=self.device
        )
        if not ceiling_spoken:
            return None
        return light.align_and_score(reference, ceiling_spoken).per_phone_distance

    def config(self) -> Dict[str, object]:
        """Parameters this engine ran with -- logged so each run records its setup."""
        return {
            "model": light.W2V2_PHONEME_MODEL,
            "lang": self.lang,
            "espeak": self._spec.espeak,
            "device": self.device,
            "threshold": self.threshold,
            "good_mode": self.good_mode,
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
