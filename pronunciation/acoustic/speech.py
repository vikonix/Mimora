# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Pronunciation analysis core for Mimora.

Adapted from OpenPronounce (https://github.com/Halleck45/OpenPronounce), MIT License.
The acoustic / phoneme comparison logic is reused as a library; the original web
front-end and built-in TTS were dropped. In Mimora the reference audio is produced
by the existing Kokoro TTS and passed into ``analyze`` as a NumPy array, so this module
never synthesizes speech itself and never touches the GUI.

Public API:
    load_models()  -- load Wav2Vec2 weights once (call at mode startup, in a thread).
    warm_up()      -- run a dummy pass to remove first-call latency (mirrors tts.py).
    analyze(...)   -- single entry point returning a PronunciationResult.

Design notes:
    * Models load lazily on first use; ``load_models`` only makes that explicit so the
      heavy download/initialisation can happen in a background daemon thread, matching
      the warm-up pattern in ``tts.py``.
    * Settings (model, device, accent, thresholds, log dir, user name) come from
      the library's own ``AnalyzerConfig`` (see acoustic/config.py), read at
      use-time. A host injects its values once with ``acoustic.configure(...)``;
      the defaults keep the package autonomous, so it never imports the host app.
    * Wav2Vec2 needs 16 kHz mono audio. User audio already arrives at 16 kHz from the
      recording path; the Kokoro reference is 24 kHz, so it is resampled here.
"""

import json
import logging
import re
import threading
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import Levenshtein
from fastdtw import fastdtw
from scipy.spatial.distance import cosine
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC
from phonemizer import phonemize

# Settings come from the library's own AnalyzerConfig (see acoustic/config.py),
# never from the host application: a host injects its values once at startup via
# acoustic.configure(). get_config() returns the active configuration.
from .config import get_config

# Shared waveform preparation (single torch-free copy in pronunciation.common).
# The underscored aliases keep this module's established local names (and the
# unit tests that exercise them) intact.
from pronunciation.common.audio import (
    TARGET_SAMPLE_RATE,
    prepare_waveform as _prepare_waveform,
    trim_silence as _trim_silence,
    waveform_digest,
)


# =====================================================================
# Configuration.
#
# All host-tunable settings (model, device, espeak accent, score threshold,
# acoustic floor default, log dir, user name) come from the active
# AnalyzerConfig via get_config(); they are read at use-time so a host app can
# inject them with acoustic.configure() after import but before analysis.
# The constants below are intrinsic to the analyzer and are not host-tunable.
# =====================================================================
# Wav2Vec2 expects strictly 16 kHz mono input: TARGET_SAMPLE_RATE (imported
# above from pronunciation.common.audio).
# Kokoro synthesises at 24 kHz; used as the default reference sample rate.
KOKORO_SAMPLE_RATE = 24_000

# ---------------------------------------------------------------------
# Acoustic score calibration.
#
# The acoustic component compares per-step cosine DTW distance between the two
# Wav2Vec2 embedding sequences. Two anchors map it to 0-100:
#   * floor (the acoustic "good" distance) - typical per-step distance of a
#     *good* attempt by a different speaker (the user vs the TTS voice never
#     reach 0). Configurable default in AnalyzerConfig.acoustic_good; calibrated
#     per voice/microphone by ``python acoustic/calibrate.py``, which writes it
#     to calibration.json. current_acoustic_floor() returns the value in effect.
#   * ceiling - per-step distance when content does not match. Derived
#     automatically per utterance from the random-pair baseline (the mean
#     distance between unaligned frames of the two recordings), so it adapts to
#     each phrase without manual tuning.
# ---------------------------------------------------------------------
ACOUSTIC_BAD_DEFAULT = 0.60      # fixed ceiling when no per-utterance baseline exists
ACOUSTIC_BAD_FRACTION = 0.9      # ceiling = this fraction of the random-pair baseline
ACOUSTIC_MIN_SPAN = 0.05         # minimal floor-to-ceiling span (avoids degenerate scale)

# Persisted calibration (floor override) lives next to this module.
CALIBRATION_FILE = Path(__file__).resolve().parent / "calibration.json"


def samples_file() -> Path:
    """Path of the per-attempt calibration sample log.

    Lives under the configured log directory; ``acoustic/calibrate.py`` reads it
    to recompute the acoustic floor on request.
    """
    return Path(get_config().log_dir) / "acoustic_samples.jsonl"


# The acoustic floor is per practising user: calibration.json maps a user name
# (AnalyzerConfig.user_name, "" when unset) to that user's floor under a "users"
# object:
#   {"users": {"": {"acoustic_good": 0.18, "created": ...}, "valery": {...}}}
# A pre-per-user file had the floor at the top level ({"acoustic_good": ...}); it
# is still honoured as a fallback and migrated into the "" profile on next save.
def _load_calibration() -> float:
    """Return the current user's calibrated acoustic floor, or the default."""
    user_name = get_config().user_name
    default = get_config().acoustic_good
    try:
        if CALIBRATION_FILE.exists():
            data = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
            users = data.get("users") if isinstance(data, dict) else None
            entry = users.get(user_name) if isinstance(users, dict) else None
            if isinstance(entry, dict) and "acoustic_good" in entry:
                value = float(entry["acoustic_good"])
                source = f"user={user_name!r}"
            elif isinstance(data, dict) and "acoustic_good" in data:
                value = float(data["acoustic_good"])  # legacy flat file
                source = "legacy"
            else:
                return default
            logging.info(f"[acoustic] Loaded calibration ({source}): "
                         f"acoustic_good={value:.4f} ({CALIBRATION_FILE})")
            return value
    except Exception:
        logging.exception("Failed to read calibration file; using defaults:")
    return default


# Calibrated acoustic floor in effect, cached after first load so calibration.json
# is read once. Loaded lazily (so configure() can install the user name first) and
# refreshed automatically when the active user changes; None means "not loaded".
_acoustic_good: Optional[float] = None
_acoustic_good_user: Optional[str] = None


def current_acoustic_floor() -> float:
    """Return the active user's calibrated acoustic floor (cached).

    Reloads from calibration.json when the configured user changes, so switching
    users via configure() picks up the right floor without an explicit reset.
    """
    global _acoustic_good, _acoustic_good_user
    user_name = get_config().user_name
    if _acoustic_good is None or _acoustic_good_user != user_name:
        _acoustic_good = _load_calibration()
        _acoustic_good_user = user_name
    return _acoustic_good


# Keys carried over when migrating a legacy flat calibration into a user profile.
_LEGACY_CALIBRATION_KEYS = ("acoustic_good", "created", "samples_used", "voice")


def save_calibration(acoustic_good: float, extra: Optional[Dict[str, Any]] = None) -> None:
    """Persist the current user's acoustic floor and apply it to this process.

    The floor is stored under the configured user name in the ``users`` map,
    leaving other users' calibrations untouched.
    """
    global _acoustic_good, _acoustic_good_user
    user_name = get_config().user_name
    entry: Dict[str, Any] = {
        "acoustic_good": round(float(acoustic_good), 5),
        "created": datetime.now().isoformat(timespec="seconds"),
        "user_name": user_name,
    }
    if extra:
        entry.update(extra)

    # Merge into the existing store so other users keep their floors.
    data: Dict[str, Any] = {}
    if CALIBRATION_FILE.exists():
        try:
            existing = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                data = existing
        except Exception:
            logging.exception("Calibration file unreadable; rewriting it:")
    users = data.get("users")
    if not isinstance(users, dict):
        users = {}
    # Preserve a pre-per-user floor by parking it in the default ("") profile.
    if "acoustic_good" in data and "" not in users:
        users[""] = {k: data[k] for k in _LEGACY_CALIBRATION_KEYS if k in data}
    users[user_name] = entry

    CALIBRATION_FILE.write_text(
        json.dumps({"users": users}, indent=2) + "\n", encoding="utf-8")
    _acoustic_good = float(acoustic_good)
    _acoustic_good_user = user_name
    logging.info(f"[acoustic] Saved calibration user={user_name!r} "
                 f"acoustic_good={acoustic_good:.4f} -> {CALIBRATION_FILE}")

# Lazily-initialised model singletons (loaded once, reused for every analysis).
_processor: Optional[Wav2Vec2Processor] = None
_model: Optional[Wav2Vec2Model] = None          # embeddings; alias of _model_ctc.wav2vec2
_model_ctc: Optional[Wav2Vec2ForCTC] = None     # transcription (what was recognised)
# Guards load_models() so concurrent callers cannot load the weights twice
# (the public API does not require callers to serialize themselves).
_load_lock = threading.Lock()


# =====================================================================
# Result type (the module's contract with the GUI layer)
# =====================================================================
# PronunciationResult is the engine-neutral shared type (pronunciation.common):
# both this acoustic engine and pronunciation.phoneme return the same shape so the
# GUI is engine-agnostic. Re-exported via this package's __init__, so
# ``pronunciation.acoustic.PronunciationResult`` keeps working. This engine fills
# the ``acoustic_*`` fields; the phoneme-specific fields stay at their defaults.
from pronunciation.common import PronunciationResult


# =====================================================================
# Model lifecycle
# =====================================================================
def load_models() -> None:
    """Load Wav2Vec2 weights into memory once. Safe to call repeatedly.

    Heavy (~1.2 GB download on first run); call from a background daemon thread
    at mode startup so the GUI stays responsive.
    """
    global _processor, _model, _model_ctc
    with _load_lock:
        if _model is not None and _model_ctc is not None and _processor is not None:
            return

        from pronunciation.common.compat import allow_torch_load_for_trusted_models

        allow_torch_load_for_trusted_models()
        cfg = get_config()
        _processor = Wav2Vec2Processor.from_pretrained(cfg.model_name)

        _model_ctc = Wav2Vec2ForCTC.from_pretrained(cfg.model_name).to(cfg.device)
        _model_ctc.eval()

        # The CTC checkpoint already contains the full base encoder - reuse it
        # for embeddings instead of loading a second identical copy of the
        # weights (~1.2 GB of RAM/VRAM and a second load from disk).
        _model = _model_ctc.wav2vec2


def warm_up() -> None:
    """Run a short dummy pass through both models to remove first-call latency."""
    load_models()
    dummy = np.zeros(TARGET_SAMPLE_RATE // 2, dtype=np.float32)  # 0.5 s of silence
    extract_embeddings(dummy)
    transcribe(dummy)


def _ensure_loaded() -> None:
    """Guarantee models are available before inference."""
    if _model is None or _model_ctc is None or _processor is None:
        load_models()


# Audio preparation lives in pronunciation.common.audio (_prepare_waveform /
# _trim_silence are imported above), shared with the phoneme engine and the
# host's prosody layer so all of them measure the same prepared signal.


# =====================================================================
# Wav2Vec2 inference
# =====================================================================
def extract_embeddings(audio_waveform: np.ndarray,
                       sampling_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Extract raw Wav2Vec2 embeddings (time, features) for the given audio."""
    _ensure_loaded()

    inputs = _processor(audio_waveform, sampling_rate=sampling_rate,
                        return_tensors="pt", padding=True)

    input_values = inputs.input_values
    if input_values.dim() > 2:  # drop any spurious leading dimension
        input_values = input_values.squeeze(0)
    input_values = input_values.to(get_config().device)

    with torch.no_grad():
        features = _model(input_values).last_hidden_state  # (batch, time, features)

    return features.squeeze(0).cpu().numpy()


def transcribe(audio_waveform: np.ndarray) -> str:
    """Transcribe audio to text using the Wav2Vec2 CTC head."""
    _ensure_loaded()

    inputs = _processor(audio_waveform, sampling_rate=TARGET_SAMPLE_RATE,
                        return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(get_config().device)

    with torch.no_grad():
        logits = _model_ctc(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1).cpu()  # decode on CPU
    return _processor.batch_decode(predicted_ids)[0]


# =====================================================================
# Phoneme / text comparison (reused OpenPronounce core)
# =====================================================================
@lru_cache(maxsize=4096)
def _phonemize_word(word: str) -> tuple:
    """Phonemize one word (each phonemize call spawns espeak, so cache results;
    words repeat both across attempts at a phrase and across phrases).

    The cache key is the word alone even though the result depends on the
    espeak accent: configure() clears this cache whenever the language
    changes, so a stale-accent entry can never be served.
    """
    try:
        return tuple(phonemize(word, language=get_config().espeak_language,
                               backend="espeak",
                               strip=True, preserve_punctuation=False).split())
    except Exception:
        # A broken/missing espeak-ng must not fail the whole analysis, but it
        # must not be silent either: with empty phonemes every word is skipped
        # in compare_transcriptions and the score is quietly distorted.
        logging.exception(f"[acoustic] espeak phonemization failed for "
                          f"{word!r}; trying festival")
        try:
            return tuple(phonemize(word, language="en-us", backend="festival",
                                   strip=True, preserve_punctuation=False).split())
        except Exception:
            logging.exception(f"[acoustic] festival fallback failed for "
                              f"{word!r}; returning no phonemes - the word "
                              f"will be excluded from scoring")
            return ()  # fallback if every backend fails


def get_word_phonemes(text: str) -> List[tuple]:
    """Return ordered (word, phonemes) pairs for each word in the text."""
    # Split on words, ignoring punctuation, to avoid issues like "times,".
    words = re.findall(r"\b[\w']+\b", text)
    return [(word, _phonemize_word(word)) for word in words]


def get_phonemes_with_word_mapping(text: str):
    """Return a list of phonemes and a mapping {phoneme_index: source_word}."""
    phonemes: List[str] = []
    phoneme_to_word: Dict[int, str] = {}

    for word, word_phonemes in get_word_phonemes(text):
        for phoneme in word_phonemes:
            phoneme_to_word[len(phonemes)] = word
            phonemes.append(phoneme)

    return phonemes, phoneme_to_word


def compare_transcriptions(transcription: str, text_reference: str) -> Dict[str, Any]:
    """Compare an ASR transcription against the expected text.

    Identifies per-word pronunciation errors via phoneme alignment and returns
    the distances used by the scoring formula.
    """
    # Normalize both sides identically (lower-case, punctuation stripped).
    # The transcription usually arrives pre-cleaned, but the reference keeps
    # its punctuation - without cleaning it too, every comma/period counted as
    # a guaranteed edit and inflated word_error_rate (noticeable on short phrases).
    transcription_clean = clean_transcription(transcription)
    reference_clean = clean_transcription(text_reference)

    # Edit distance between transcription and reference text. *Character*-level
    # on purpose (named accordingly): it gives partial credit for near-misses,
    # which a word-token distance would count as whole-word errors.
    char_distance = Levenshtein.distance(transcription_clean, reference_clean)

    # Extract phonemes from both versions. The reference keeps its per-word
    # grouping so the word-boundary walk below reuses it instead of running
    # espeak a second time for every word.
    expected_pairs = get_word_phonemes(text_reference)
    expected_phonemes = [p for _word, word_phonemes in expected_pairs for p in word_phonemes]
    transcribed_phonemes, transcribed_map = get_phonemes_with_word_mapping(transcription_clean)

    # Global phoneme distance: *character-level* edit distance over the joined
    # phoneme strings. espeak returns one undelimited phoneme string per word
    # (e.g. "ðə"), so an edit distance over the word-token lists would count a
    # whole word as a single all-or-nothing error; characters approximate
    # individual phonemes and give partial credit for near-misses.
    expected_join = " ".join(expected_phonemes)
    transcribed_join = " ".join(transcribed_phonemes)
    phoneme_distance = Levenshtein.distance(expected_join, transcribed_join)

    errors: List[Dict[str, Any]] = []
    words_with_errors = set()

    # Map each expected phoneme index to the set of transcribed indices it aligns to.
    # This handles 1-to-N and N-to-1 word mappings (e.g. "I'm" -> "I M").
    alignment_map = [set() for _ in range(len(expected_phonemes))]

    opcodes = Levenshtein.opcodes(expected_phonemes, transcribed_phonemes)

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            for k, l in zip(range(i1, i2), range(j1, j2)):
                alignment_map[k].add(l)
        elif tag == 'replace':
            # Map the replaced range proportionally rather than all-to-all,
            # which keeps "Hello how are" from mapping everything to everything.
            len_i = i2 - i1
            len_j = j2 - j1
            for k in range(i1, i2):
                start_j = j1 + int((k - i1) * len_j / len_i)
                end_j = j1 + int((k - i1 + 1) * len_j / len_i)
                if start_j == end_j and len_j > 0:
                    idx = min(start_j, j2 - 1)
                    alignment_map[k].add(idx)
                else:
                    for l in range(start_j, end_j):
                        alignment_map[k].add(l)
        # 'delete' (missing expected phonemes) and 'insert' (extra transcribed
        # phonemes) need no alignment entry here.

    # Walk the reference words in order to recover phoneme boundaries reliably
    # (a flat phoneme-to-word map cannot distinguish adjacent duplicate words).
    current_phoneme_idx = 0

    for word, p_list in expected_pairs:
        if not p_list:
            continue  # word produced no phonemes (e.g. a number/symbol)

        word_indices = range(current_phoneme_idx, current_phoneme_idx + len(p_list))
        current_phoneme_idx += len(p_list)

        # Collect the transcribed phoneme indices this word aligns to.
        matched_trans_indices = set()
        for idx in word_indices:
            if idx < len(alignment_map):
                matched_trans_indices.update(alignment_map[idx])

        if not matched_trans_indices:
            # Word is missing from the transcription entirely.
            errors.append({"position": word_indices.start, "expected": word,
                           "actual": "", "word": word})
            words_with_errors.add(word)
            continue

        sorted_trans_indices = sorted(matched_trans_indices)

        # Reconstruct the actual word(s), de-duplicating while preserving order.
        actual_words: List[str] = []
        seen_words = set()
        for tidx in sorted_trans_indices:
            if tidx in transcribed_map:
                w = transcribed_map[tidx]
                if w not in seen_words:
                    actual_words.append(w)
                    seen_words.add(w)
        actual_text = " ".join(actual_words)

        # Compare the expected vs actual phonemes character by character.
        # Joining into strings first matters: each list element is a whole
        # word's phoneme string, so a list-level edit distance would flag the
        # word on *any* difference and kill the 40% tolerance below.
        expected_str = "".join(expected_phonemes[i] for i in word_indices)
        actual_str = "".join(transcribed_phonemes[i] for i in sorted_trans_indices)

        p_dist = Levenshtein.distance(expected_str, actual_str)

        # Mark a mispronunciation when the phoneme edit distance exceeds 40% of length.
        if p_dist > len(expected_str) * 0.4:
            errors.append({
                "position": word_indices.start,
                "expected": expected_str,           # expected phonemes
                "actual": actual_str,               # actual phonemes
                "word": word,                       # expected word text
                "actual_word": actual_text,         # actual word text (e.g. "I M")
            })
            words_with_errors.add(word)

    # Human-readable feedback summary.
    feedback = "🔊 Feedback on your pronunciation:\n"
    if words_with_errors:
        feedback += "❌ You need to better pronounce these words: " + ", ".join(words_with_errors) + "\n"
    else:
        feedback += "✅ Your pronunciation is excellent! 🎉\n"

    return {
        "char_distance": char_distance,
        "reference_length": len(reference_clean),
        "phoneme_distance": phoneme_distance,
        "phoneme_length": len(expected_join),
        "errors": errors,
        "feedback": feedback,
        "transcribe": transcription,
        "expected_phonemes": expected_phonemes,
        "transcribed_phonemes": transcribed_phonemes,
        "words_with_errors": list(words_with_errors),
    }


def word_level_diff(transcription: str, text_reference: str) -> List[Dict[str, str]]:
    """Align recognised words against the target phrase, returning only mismatches.

    Lets the GUI present concrete "expected -> heard" pairs instead of the raw
    ASR string. Both sides are cleaned identically (lower-case, punctuation
    stripped) and split on whitespace, then aligned with a word-token edit
    distance via ``Levenshtein.opcodes`` (the same primitive the phoneme
    comparison uses, which accepts token lists):

        * substitution -> {"expected": "time", "heard": "times"}
        * deletion     -> {"expected": "the",  "heard": ""}      (word dropped)
        * insertion    -> {"expected": "",      "heard": "uh"}    (extra word)

    'equal' segments are skipped, so an empty list means the recognition
    matched the target word for word.
    """
    expected_words = clean_transcription(text_reference).split()
    heard_words = clean_transcription(transcription).split()

    diffs: List[Dict[str, str]] = []
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(expected_words, heard_words):
        if tag == "equal":
            continue
        diffs.append({
            "expected": " ".join(expected_words[i1:i2]),
            "heard": " ".join(heard_words[j1:j2]),
        })
    return diffs


def heard_word_tags(transcription: str, text_reference: str) -> List[Dict[str, Any]]:
    """Tag each recognised word by whether it matched the target phrase.

    Returns one entry per heard word, in spoken order::

        [{"word": "hullo", "correct": False}, {"word": "i", "correct": True}, ...]

    A word is ``correct`` when it falls in an 'equal' segment of the same
    word-token alignment used by :func:`word_level_diff`. Lets the GUI colour
    correctly-recognised words on the raw ASR line.
    """
    expected_words = clean_transcription(text_reference).split()
    heard_words = clean_transcription(transcription).split()

    tags = [{"word": w, "correct": False} for w in heard_words]
    for tag, _i1, _i2, j1, j2 in Levenshtein.opcodes(expected_words, heard_words):
        if tag == "equal":
            for j in range(j1, j2):
                tags[j]["correct"] = True
    return tags


def reference_word_tags(text_reference: str,
                        words_with_errors: List[str]) -> List[Dict[str, Any]]:
    """Tag each word of the target phrase as correctly pronounced or not.

    Returns one entry per phrase word, in order, preserving the original token
    (case and punctuation) for display::

        [{"word": "Hello,", "correct": True}, {"word": "world", "correct": False}]

    A word is incorrect when its normalised form appears in ``words_with_errors``.
    The normalisation mirrors the GUI's old inline check (lower-case, strip edge
    punctuation) so highlighting is unchanged. Engine-neutral: the phoneme engine
    will build the same shape by mapping phone errors back to words.
    """
    error_words = {w.lower() for w in words_with_errors}
    tags: List[Dict[str, Any]] = []
    for token in text_reference.split():
        clean = token.lower().strip(".,!?;:\"")
        tags.append({"word": token, "correct": clean not in error_words})
    return tags


def _random_pair_baseline(emb_a: np.ndarray, emb_b: np.ndarray,
                          n_pairs: int = 2000, seed: int = 0) -> float:
    """Mean cosine distance between randomly paired frames of two embeddings.

    Approximates the per-step DTW distance of *unrelated* content, giving each
    utterance its own automatic "completely wrong" ceiling for the acoustic score.
    """
    if len(emb_a) == 0 or len(emb_b) == 0:
        # Degenerate input (near-empty audio): no frames to sample from, so
        # fall back to the fixed ceiling instead of crashing the analysis.
        return ACOUSTIC_BAD_DEFAULT
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(emb_a), n_pairs)
    j = rng.integers(0, len(emb_b), n_pairs)
    a, b = emb_a[i], emb_b[j]
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9
    return float(np.mean(1.0 - num / den))


def acoustic_bad_for(baseline: float, acoustic_good: Optional[float] = None) -> float:
    """Per-utterance acoustic ceiling derived from the random-pair baseline."""
    good = current_acoustic_floor() if acoustic_good is None else acoustic_good
    return max(ACOUSTIC_BAD_FRACTION * baseline, good + ACOUSTIC_MIN_SPAN)


def compute_pronunciation_score(acoustic_per_step: float,
                                phoneme_error_rate: float,
                                word_error_rate: float,
                                acoustic_bad: Optional[float] = None,
                                acoustic_good: Optional[float] = None) -> float:
    """Combine the three normalized components into a 0-100 score.

    Args:
        acoustic_per_step: cosine DTW distance per alignment step (length-invariant).
        phoneme_error_rate: phoneme edit distance / expected phoneme count.
        word_error_rate: character edit distance / reference text length.
        acoustic_bad: per-utterance ceiling (see ``acoustic_bad_for``); falls back
            to a fixed default when no baseline is available.
        acoustic_good: floor override; defaults to the calibrated floor in effect
            (see ``current_acoustic_floor``).
    """
    good = current_acoustic_floor() if acoustic_good is None else acoustic_good
    bad = ACOUSTIC_BAD_DEFAULT if acoustic_bad is None else acoustic_bad
    bad = max(bad, good + ACOUSTIC_MIN_SPAN)

    dtw_score = 100.0 * min(1.0, max(0.0, 1.0 - (acoustic_per_step - good) / (bad - good)))
    phoneme_score = 100.0 * min(1.0, max(0.0, 1.0 - phoneme_error_rate))
    word_score = 100.0 * min(1.0, max(0.0, 1.0 - word_error_rate))

    # Weighting: acoustic DTW 40%, phonemes 30%, words 30%.
    final_score = 0.4 * dtw_score + 0.3 * phoneme_score + 0.3 * word_score

    final_score = min(100.0, max(0.0, final_score))
    return round(final_score, 2)


# Prosody (pitch & energy contours) is no longer owned by this engine. It moved
# to the engine-agnostic ``mimora/prosody.py`` and is computed in ``main.py`` from
# the raw user/reference waveforms, so the same charts work for every engine.
# ``analyze`` returns an empty ``prosody`` dict; the host fills it in.


def clean_transcription(text: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace in a transcription.

    Only ``[a-z' ]`` survives: digits and non-ASCII letters are dropped along
    with punctuation ("2" vs the ASR's "two", "mañana" -> "maana"), which
    inflates the word error rate when the expected text contains them. The
    English ASR checkpoint cannot emit such characters anyway, so the loss is
    logged (not fixed) - it makes a surprising score on such a phrase
    traceable in the log.
    """
    text = text.lower().strip()
    lost = sorted(set(re.findall(r"[0-9]|[^\x00-\x7f]", text)))
    if lost:
        logging.info("[acoustic] clean_transcription dropped %r from %r - "
                     "digits/non-ASCII are outside the scoring alphabet",
                     "".join(lost), text)
    text = re.sub(r"[^a-zA-Z' ]+", "", text)
    return " ".join(text.split()).strip()


# =====================================================================
# Reference feature cache
# =====================================================================
# A phrase is repeated many times against the same Kokoro reference, but the
# reference-side waveform and embeddings never change between attempts. Cache the
# most recent reference (one phrase is practiced at a time) so repeats skip the
# Wav2Vec2 pass on the reference. (Reference prosody is cached separately in
# mimora/prosody.py, which now owns the F0/energy contours.) The lock makes
# concurrent analyze() calls safe; in the app they are already serialized by the
# GUI's is_processing_audio guard, so it is uncontended.
_reference_cache: Dict[str, Any] = {}
_reference_cache_lock = threading.Lock()


def _reference_features(reference_audio: np.ndarray, reference_sr: int) -> Dict[str, Any]:
    """Return the prepared waveform and embeddings of the reference."""
    global _reference_cache
    arr = np.asarray(reference_audio)
    key = (reference_sr, arr.shape, waveform_digest(arr))
    with _reference_cache_lock:
        if _reference_cache.get("key") != key:
            wav = _trim_silence(_prepare_waveform(arr, reference_sr))
            _reference_cache = {
                "key": key,
                "wav": wav,
                "embeddings": extract_embeddings(wav),
            }
        return _reference_cache


# =====================================================================
# Single entry point
# =====================================================================
def _append_calibration_sample(record: Dict[str, Any]) -> None:
    """Append one analysis record to the calibration sample log (best effort).

    The file feeds ``acoustic/calibrate.py``; a write failure must never break
    the analysis itself.
    """
    try:
        path = samples_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        logging.exception("Failed to append calibration sample:")


def analyze(user_audio: np.ndarray,
            expected_text: str,
            reference_audio: Optional[np.ndarray] = None,
            user_sr: int = TARGET_SAMPLE_RATE,
            reference_sr: int = KOKORO_SAMPLE_RATE,
            voice: Optional[str] = None,
            is_reference: bool = False) -> PronunciationResult:
    """Compare a user's spoken attempt against the expected phrase.

    Args:
        user_audio: user's recorded waveform (1-D float32; from the recording path).
        expected_text: the reference phrase the user was asked to repeat.
        reference_audio: Kokoro-synthesised reference waveform for the same
            phrase. Optional only to mirror the phoneme/none signature
            ("Public API mirrors acoustic/ exactly"): THIS engine cannot score
            without it - the acoustic component is a comparison against the
            reference - so None raises a ValueError instead of the TypeError a
            positional-only signature used to produce.
        user_sr: sample rate of ``user_audio`` (recording path is 16 kHz).
        reference_sr: sample rate of ``reference_audio`` (Kokoro is 24 kHz).
        voice: Kokoro voice the reference was synthesized with. Only recorded in
            the calibration sample log - the acoustic floor is voice-specific,
            so calibrate.py needs to know which voice produced each sample.
        is_reference: marks the reference self-test (reference compared with
            itself). Accepted so the dispatcher can call every engine with the
            same signature; this engine ignores it - its calibrate.py already
            excludes self-tests by their near-zero acoustic distance
            (SELF_TEST_ACOUSTIC), so no flag is needed here.

    Returns:
        PronunciationResult with score, per-word errors, prosody and transcription.
    """
    if reference_audio is None:
        raise ValueError(
            "the acoustic engine requires reference_audio: its score is a "
            "comparison against the reference recording")
    _ensure_loaded()

    # Trim silent padding: user takes have button-press pauses (with the noise
    # floor boosted by peak normalization), the TTS reference has almost none.
    user_wav = _trim_silence(_prepare_waveform(user_audio, user_sr))
    reference = _reference_features(reference_audio, reference_sr)

    # Acoustic similarity: cosine DTW between the two embedding sequences,
    # normalized by the alignment path length so it does not grow with phrase
    # duration. Cosine (vs euclidean) also ignores embedding magnitude, which
    # drifts with loudness/voice rather than pronunciation.
    emb_user = extract_embeddings(user_wav)
    emb_reference = reference["embeddings"]
    acoustic_total, path = fastdtw(emb_user, emb_reference, dist=cosine)
    acoustic_per_step = float(acoustic_total) / max(1, len(path))
    acoustic_baseline = _random_pair_baseline(emb_user, emb_reference)
    acoustic_bad = acoustic_bad_for(acoustic_baseline)

    # Transcription + per-word phoneme comparison.
    transcription = clean_transcription(transcribe(user_wav))
    differences = compare_transcriptions(transcription, expected_text)

    phoneme_length = max(1, differences["phoneme_length"])
    phoneme_error_rate = differences["phoneme_distance"] / phoneme_length
    reference_length = max(1, differences["reference_length"])
    # The rate keeps its historical "word_error_rate" name: it is part of the
    # scoring API and the calibration-log schema read by calibrate.py.
    word_error_rate = differences["char_distance"] / reference_length

    score = compute_pronunciation_score(
        acoustic_per_step,
        phoneme_error_rate,
        word_error_rate,
        acoustic_bad=acoustic_bad,
    )

    # Calibration log: raw components in one greppable line. ``calibrate.py``
    # consumes the structured copy appended to the samples file below.
    logging.info(
        "[acoustic] score=%.1f | acoustic/step=%.4f (good=%.3f bad=%.3f baseline=%.4f) | "
        "phonemes=%d/%d (err=%.2f) | chars_lev=%d/%d (err=%.2f) | voice=%s | asr=%r",
        score, acoustic_per_step, current_acoustic_floor(), acoustic_bad, acoustic_baseline,
        differences["phoneme_distance"], phoneme_length, phoneme_error_rate,
        differences["char_distance"], reference_length, word_error_rate,
        voice, transcription,
    )
    _append_calibration_sample({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "text": expected_text,
        "asr": transcription,
        # Practising user (AnalyzerConfig.user_name, "" when unset). The acoustic
        # floor is per-user, so calibrate.py filters the log by this field.
        "user_name": get_config().user_name,
        "voice": voice,
        "acoustic_per_step": round(acoustic_per_step, 5),
        "acoustic_baseline": round(acoustic_baseline, 5),
        "phoneme_distance": int(differences["phoneme_distance"]),
        "phoneme_length": int(phoneme_length),
        # Renamed from "word_distance" (it is character-level); calibrate.py
        # only reads the *_error_rate fields, so old sample lines stay usable.
        "char_distance": int(differences["char_distance"]),
        "reference_length": int(reference_length),
        "phoneme_error_rate": round(phoneme_error_rate, 4),
        "word_error_rate": round(word_error_rate, 4),
        "score": score,
    })

    # Prosody is the engine-agnostic audio layer (mimora/prosody.py), computed by
    # the host from the raw waveforms. The engine returns it empty; the host fills
    # it before handing the result to the UI.
    return PronunciationResult(
        score=score,
        word_errors=differences["errors"],
        prosody={},
        transcription=transcription,
        passed=score >= get_config().score_threshold,
        feedback=differences["feedback"],
        acoustic_distance=int(acoustic_total),
        acoustic_per_step=acoustic_per_step,
        acoustic_baseline=acoustic_baseline,
        words_with_errors=differences["words_with_errors"],
        expected_phonemes=differences["expected_phonemes"],
        transcribed_phonemes=differences["transcribed_phonemes"],
        word_diff=word_level_diff(transcription, expected_text),
        # Engine-neutral display fields. The acoustic engine's "units" are words:
        # recognized_units mirrors heard_word_tags with the neutral "unit" key.
        reference_words=reference_word_tags(expected_text, differences["words_with_errors"]),
        recognized_units=[{"unit": t["word"], "correct": t["correct"]}
                          for t in heard_word_tags(transcription, expected_text)],
        # The acoustic engine has no per-phone breakdown, so it cannot rank weak
        # phonemes; passed explicitly (empty) to keep the result API uniform
        # with the phoneme engine. The GUI falls back to its "Heard" line here.
        weak_phonemes=[],
    )
