"""Engine-neutral result type shared by the pronunciation engines.

Both ``pronounce/`` (acoustic) and ``pronounce_phoneme/`` (text-only phoneme)
return the *same* :class:`PronunciationResult` so the GUI reads one stable shape
regardless of which engine the dispatcher (``mimora/engine.py``) selected -- this
is the "общий тип результата, который понимает UI" of the productionization task
(§3). Keeping it in its own tiny package (not inside either engine) avoids a
dependency from one engine on the other.

The four fields the GUI strictly requires are ``score``, ``word_errors``,
``prosody`` and ``transcription``. Everything else is engine-specific extra with a
default, so each engine fills only what it has and the GUI safely ignores the rest:
the acoustic engine fills the ``acoustic_*`` fields, the phoneme engine the
``per_phone_distance`` / ``phoneme_score`` / ``recall`` / ``good_anchor`` block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PronunciationResult:
    """Outcome of one pronunciation comparison (engine-neutral).

    Required fields (always filled by every engine):
        score, word_errors, prosody, transcription.
    The rest are extras with defaults; an engine fills only the ones it computes.
    ``prosody`` is left empty by the engine and filled by the host
    (mimora/prosody.py) from the raw waveforms, so prosody charts work identically
    across engines.
    """

    score: float                                  # 0-100 overall pronunciation score
    word_errors: List[Dict[str, Any]]             # per mispronounced word: expected/heard
    prosody: Dict[str, List[float]]               # filled by the host; engine returns {}
    transcription: str                            # what the recognizer produced
    passed: bool = False                          # score >= configured score_threshold
    feedback: str = ""                            # human-readable summary

    # --- Engine-neutral display fields (read by the GUI regardless of engine). ---
    words_with_errors: List[str] = field(default_factory=list)
    expected_phonemes: List[str] = field(default_factory=list)
    transcribed_phonemes: List[str] = field(default_factory=list)
    # word_diff: non-empty iff there are word-level errors; the GUI shows
    # "matches the target" when empty. The acoustic engine puts one
    # {"expected", "heard"} pair per diverging segment; the phoneme engine lists
    # the diverging words.
    word_diff: List[Dict[str, str]] = field(default_factory=list)
    # reference_words: one {"word", "correct"} per target-phrase word, in order;
    # drives the green/red highlight on the "Phrase" line.
    reference_words: List[Dict[str, Any]] = field(default_factory=list)
    # recognized_units: what was recognised, in order, each {"unit", "correct"}.
    # Units are *words* for the acoustic engine and *phonemes* for the phoneme
    # engine; the GUI renders both the same way, so the "Heard" line works for both.
    recognized_units: List[Dict[str, Any]] = field(default_factory=list)

    # --- Acoustic-engine diagnostics (pronounce/); safe to ignore elsewhere. ---
    acoustic_distance: int = 0                     # total Wav2Vec2-embedding DTW distance
    acoustic_per_step: float = 0.0                 # DTW distance per alignment step (scored)
    acoustic_baseline: float = 0.0                 # random-pair distance (per-utterance ceiling)

    # --- Phoneme-engine diagnostics (pronounce_phoneme/); safe to ignore elsewhere. ---
    per_phone_distance: float = 0.0               # observed feature distance per reference phone
    bad_baseline: float = 0.0                     # per-utterance "completely wrong" anchor
    phoneme_score: float = 0.0                    # 0-100 pronunciation-quality component
    recall: float = 0.0                           # 0-1 phoneme-weighted recall of reference phones
    good_anchor: float = 0.0                      # GOOD anchor actually used (per-phrase in ceiling mode)


__all__ = ["PronunciationResult"]
