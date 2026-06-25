# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Phoneme pronunciation engine for Mimora (text-only scoring).

Pipeline::

    text --espeak-ng--> reference IPA phonemes -+
                                                +-- feature-weighted edit distance
    user audio --w2v2 phoneme ASR--> phonemes --+        |
                                                         v
                                          score (0-100) + per-word / per-phone tags

Unlike the acoustic core it needs no per-phrase reference *recording* to score --
the reference phonemes come from espeak. The reference audio is still accepted and,
in the optional ``good_mode="ceiling"``, recognized once to anchor a flawless read
to 100 per phrase; the default ``good_mode="global"`` scores against the single
calibrated PHONEME_GOOD (matching how the 0-5 buckets were fit).

Public API mirrors ``pronounce/`` exactly so the dispatcher (task stage B) can
treat both engines the same:
    load_models()  -- load the wav2vec2 phoneme weights once (call in a thread).
    warm_up()      -- dummy pass to remove first-call latency.
    analyze(...)   -- single entry point returning a PronunciationResult.

Implementation notes:
    * The recognizer works on in-memory waveforms, not file paths, so only the
      model is cached.
    * panphon's bundled data is UTF-8; on a cp1252 Windows process its load raises
      UnicodeDecodeError. Instead of an unconditional ``pathlib.Path.open``
      monkey-patch we build the table normally and only fall back to a narrowly
      scoped UTF-8 default on that error (never triggered when the app runs in
      UTF-8 mode -- set PYTHONUTF8=1 at launch).
    * Scoring constants (GOOD anchor, recall threshold, axis weights, insertion
      cap/gate) load from a JSON config next to this module;
      ``config.AnalyzerConfig`` holds only host settings.

This module never touches the GUI.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Settings come from the library's own AnalyzerConfig (see config.py), never from
# the host application: a host injects its values once at startup via configure().
from .config import get_config

# Engine-neutral result type shared with the acoustic engine, so the GUI reads one
# stable shape regardless of the active engine (task §3). This engine fills the
# phoneme-specific fields (per_phone_distance / phoneme_score / recall / good_anchor);
# the acoustic_* fields stay at their defaults.
from pronunciation.common import PronunciationResult


# =====================================================================
# Constants and config.
# =====================================================================
# The wav2vec2 recognizer expects strictly 16 kHz mono input.
TARGET_SAMPLE_RATE = 16_000
# Kokoro synthesises at 24 kHz; used as the default reference sample rate.
KOKORO_SAMPLE_RATE = 24_000

# Tunable scoring constants live in a JSON config next to this module. A
# missing/malformed file degrades silently to the literal defaults so scoring
# never breaks. Keys prefixed with ``_`` (``_meta``) are informational and
# ignored by the ``.get`` lookups.
_CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration.json"


def _load_calibration() -> dict:
    """Return the config dict, or ``{}`` if the file is absent/malformed."""
    try:
        with _CALIBRATION_PATH.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


_CALIB = _load_calibration()

# --- Phoneme-quality axis anchors (see _score_from_distance / _bad_baseline). ---
PHONEME_GOOD = _CALIB.get("phoneme_good", 0.0)      # per-phone distance scored 100
BAD_MIN_SPAN = 0.10                                 # keep bad strictly above good
BAD_BASELINE_DEFAULT = 0.5                          # ceiling when a sequence is empty
BAD_SHRINK_PHONES = _CALIB.get("bad_shrink_phones", 12)   # short-phrase widening strength
BAD_CEILING = _CALIB.get("bad_ceiling", 0.40)             # conservative "wrong" anchor

# --- Insertion cap and confidence gate (recognizer-hallucination defenses). ---
INSERTION_CAP_PER_PHONE = _CALIB.get("insertion_cap_per_phone", 0.25)
INSERTION_CONF_MIN = _CALIB.get("insertion_conf_min", 0.0)   # tau; 0 == argmax baseline
INSERTION_CONF_AGG = _CALIB.get("insertion_conf_agg", "max")  # "max" | "mean"

# --- Recall axis and final blend (mirrors the core's quality/word split). ---
RECALL_MAX_DIST = _CALIB.get("recall_max_dist", 0.13)   # per-phone dist counted as recalled
WEIGHT_PHONEME = _CALIB.get("weight_phoneme", 0.7)
WEIGHT_WORD = _CALIB.get("weight_word", 0.3)
# A reference word is "correct" when at least this fraction of its phones are
# recalled. Retained for back-compat; the per-word highlight now uses the mean
# word distance (WORD_*_FRAC below), which agrees with the overall score.
WORD_RECALL_MIN = _CALIB.get("word_recall_min", 0.5)
# Three-level per-word highlight cutoffs, as a fraction of the [good, bad] window
# (0 == flawless, 1 == completely wrong): <= good_frac is "good" (green), >=
# bad_frac is "bad" (red), in between is "ok" (acceptable / light grey).
WORD_GOOD_FRAC = _CALIB.get("word_good_frac", 0.33)
WORD_BAD_FRAC = _CALIB.get("word_bad_frac", 0.66)

# --- 0-5 bucketization (task §4). Coarsens the noisy raw 0-100 score onto a
# human-calibrated 0-5 grade so good speech reliably reads 4-5. Cutpoints and the
# bucket->percent bands live in calibration.json; an empty/absent block disables
# bucketing (bucket stays -1 and scoring falls back to the raw threshold). ---
_BUCKETS = _CALIB.get("buckets", {})
# Ascending score thresholds; bucket = how many a score clears (0..len).
BUCKET_CUTPOINTS = [float(c) for c in _BUCKETS.get("cutpoints", [])]
# bucket (as str) -> [lo, hi] user-facing percent band; the displayed percent is
# the band midpoint (flat per bucket, so engine noise within a bucket is hidden).
BUCKET_TO_PERCENT = _CALIB.get("bucket_to_percent", {})
# A take "passes" at or above this bucket (good speech is bucket 4-5; task §7).
PASS_BUCKET = _CALIB.get("pass_bucket", 4)

# Where save_calibration() / phoneme/calibrate.py read and write the anchor.
CALIBRATION_FILE = _CALIBRATION_PATH


# =====================================================================
# Sample log + calibration write-back (task §5.3; mirrors the acoustic engine).
# analyze() appends one record per take to logs/phoneme_samples.jsonl; the offline
# phoneme/calibrate.py re-anchors PHONEME_GOOD from the reference self-tests there.
# Kept separate from the acoustic pronounce_samples.jsonl so the two engines' logs
# never mix.
# =====================================================================
def samples_file() -> Path:
    """Path of the phoneme sample log (under the host's configured log dir)."""
    return Path(get_config().log_dir) / "phoneme_samples.jsonl"


def _append_sample(record: Dict[str, Any]) -> None:
    """Append one analysis record to the sample log (best effort).

    A write failure must never break the analysis itself, so every error is logged
    and swallowed (mirrors the acoustic engine's _append_calibration_sample).
    """
    try:
        path = samples_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        logging.exception("[phoneme] failed to append sample")


def current_phoneme_good() -> float:
    """The GOOD anchor in effect (the calibrated PHONEME_GOOD)."""
    return PHONEME_GOOD


def save_calibration(phoneme_good: float, extra: Optional[Dict[str, Any]] = None) -> None:
    """Write a new PHONEME_GOOD into calibration.json, preserving the other keys.

    Used by phoneme/calibrate.py. Only ``phoneme_good`` and a small ``_meta`` block
    are updated; buckets / bucket_to_percent / gates are left untouched. The new
    value is picked up on the next process start (module-load time).
    """
    data = _load_calibration()
    data["phoneme_good"] = round(float(phoneme_good), 6)
    meta = data.get("_meta", {}) if isinstance(data.get("_meta"), dict) else {}
    meta.update({
        "phoneme_good_recalibrated_at": datetime.now().isoformat(timespec="seconds"),
        **(extra or {}),
    })
    data["_meta"] = meta
    CALIBRATION_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logging.info("[phoneme] wrote phoneme_good=%.6f -> %s", phoneme_good, CALIBRATION_FILE)


# =====================================================================
# Model lifecycle.
# =====================================================================
_processor = None
_model = None
_load_lock = threading.Lock()


def load_models() -> None:
    """Load the wav2vec2 phoneme weights into memory once. Safe to call repeatedly.

    Heavy (~1.2 GB download on first run); call from a background daemon thread at
    mode startup so the GUI stays responsive (mirrors pronounce.load_models).
    """
    global _processor, _model
    with _load_lock:
        if _model is not None and _processor is not None:
            return
        from transformers import AutoModelForCTC, AutoProcessor

        cfg = get_config()
        _processor = AutoProcessor.from_pretrained(cfg.model_name)
        _model = AutoModelForCTC.from_pretrained(cfg.model_name).to(cfg.device).eval()


def warm_up() -> None:
    """Run dummy passes to remove first-call latency (recognizer + panphon + espeak)."""
    load_models()
    cfg = get_config()
    dummy = np.zeros(TARGET_SAMPLE_RATE // 2, dtype=np.float32)  # 0.5 s of silence
    try:
        _spoken_from_wav(dummy, cfg.device)
    except Exception:
        logging.exception("[phoneme] recognizer warm-up failed")
    try:
        _feature_table()                                   # build panphon table once
        reference_phonemes("warm up", cfg.espeak_language)  # spawn espeak once
    except Exception:
        logging.exception("[phoneme] scoring warm-up failed")


def _ensure_loaded() -> None:
    """Guarantee the recognizer is available before inference."""
    if _model is None or _processor is None:
        load_models()


# =====================================================================
# espeak registration (autonomy: mirrors how Kokoro/misaki registers espeak-ng).
# =====================================================================
_espeak_registered = False


def _ensure_espeak() -> None:
    """Point phonemizer at the bundled espeak-ng once (no system install needed).

    In the full app importing Kokoro registers espeak-ng as a side effect; this
    defensive call keeps the module usable on its own. Any failure falls through to
    a system-installed espeak, matching the acoustic core's assumption.
    """
    global _espeak_registered
    if _espeak_registered:
        return
    try:
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    except Exception:
        pass  # rely on a system espeak / host-side registration
    _espeak_registered = True


# =====================================================================
# Audio preparation.
# =====================================================================
def _prepare_waveform(waveform: np.ndarray, orig_sr: int) -> np.ndarray:
    """Return a 1-D float32 mono waveform resampled to TARGET_SAMPLE_RATE."""
    import librosa

    wav = np.asarray(waveform, dtype=np.float32)
    if wav.ndim > 1:
        # Down-mix to mono along the channel axis (the smaller dimension).
        wav = wav.mean(axis=int(np.argmin(wav.shape)))
    if orig_sr != TARGET_SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=TARGET_SAMPLE_RATE)
    return np.ascontiguousarray(wav, dtype=np.float32)


# =====================================================================
# Step 2 -- spoken phonemes from audio via the wav2vec2 phoneme recognizer.
# Works on an in-memory 16 kHz waveform (no file path), so there is no path-keyed
# cache to leak on temporary files; only the model is cached (see load_models).
# =====================================================================
def _recognize_argmax(wav16: np.ndarray, device: str) -> str:
    """Greedy CTC decode -> space-separated espeak/IPA phonemes."""
    import torch

    _ensure_loaded()
    inputs = _processor(wav16, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        logits = _model(inputs.input_values.to(device)).logits
    predicted_ids = logits.argmax(dim=-1)
    return _processor.batch_decode(predicted_ids)[0]


def _aggregate_conf(frame_confs: List[float]) -> float:
    """Pool a CTC token's per-frame posteriors into one confidence ("max"/"mean")."""
    if not frame_confs:
        return 0.0
    if INSERTION_CONF_AGG == "mean":
        return sum(frame_confs) / len(frame_confs)
    return max(frame_confs)


def _recognize_with_conf(wav16: np.ndarray, device: str) -> List[Tuple[str, float]]:
    """Greedy CTC decode that also returns each phone's posterior confidence.

    Reproduces ``batch_decode``'s CTC collapse (group consecutive-equal frame ids,
    drop the blank and word-delimiter) so the surviving tokens are exactly the
    argmax phones, each tagged with the pooled posterior of its frame run. Used by
    the confidence gate to drop low-confidence hallucinated insertions
    before scoring; only taken when INSERTION_CONF_MIN > 0.
    """
    import torch

    _ensure_loaded()
    inputs = _processor(wav16, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        logits = _model(inputs.input_values.to(device)).logits
    probs = logits.softmax(dim=-1)[0]                 # (T, vocab)
    frame_ids = probs.argmax(dim=-1).tolist()         # (T,) greedy CTC path
    frame_conf = probs.max(dim=-1).values.tolist()    # (T,) posterior of the chosen id

    tokenizer = _processor.tokenizer
    blank_id = tokenizer.pad_token_id                 # CTC blank == pad for wav2vec2
    delimiter = getattr(tokenizer, "word_delimiter_token", None)

    tokens_conf: List[Tuple[str, float]] = []
    n = len(frame_ids)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and frame_ids[j + 1] == frame_ids[i]:
            j += 1
        token_id = frame_ids[i]
        if token_id != blank_id:
            token = tokenizer.convert_ids_to_tokens(token_id)
            if token and token != delimiter:
                tokens_conf.append((token, _aggregate_conf(frame_conf[i : j + 1])))
        i = j + 1
    return tokens_conf


def _spoken_from_wav(wav16: np.ndarray, device: str) -> List[str]:
    """Recognize the phonemes the speaker produced, normalized and folded.

    Confidence gating drops low-confidence phones before normalization
    when tau (INSERTION_CONF_MIN) is set; tau == 0 takes the plain argmax path, so
    the two are byte-identical when nothing is dropped.
    """
    if INSERTION_CONF_MIN > 0:
        kept = [tok for tok, conf in _recognize_with_conf(wav16, device)
                if conf >= INSERTION_CONF_MIN]
        raw = " ".join(kept)
    else:
        raw = _recognize_argmax(wav16, device)
    return _normalize_phones(_tokenize_ipa(raw))


# =====================================================================
# Step 1 -- reference phonemes from text (espeak-ng via phonemizer).
# =====================================================================
def reference_phonemes(text: str, espeak_lang: str) -> List[str]:
    """Phonemize ``text`` into a flat list of IPA phoneme symbols (per-phone separated)."""
    return [p for word in reference_word_phonemes(text, espeak_lang) for p in word]


def reference_word_phonemes(text: str, espeak_lang: str) -> List[List[str]]:
    """Phonemize ``text`` into one phone group **per whitespace token**, in order.

    Returns exactly one group per ``text.split()`` token (an empty list for a token
    that yields no phones, e.g. pure punctuation), so the groups align 1:1 with the
    display tokens. This is what lets the scorer map phone errors back to the right
    word for the GUI highlight (task §6).

    Earlier this phonemized the whole sentence and split on espeak's own word
    boundaries, which desynced from ``text.split()`` whenever espeak split or dropped
    a token (numbers -> several words, punctuation-only tokens), mis-colouring every
    word after the divergence. Phonemizing each token separately (one batched espeak
    call) keeps the boundaries the GUI uses. Flatten with
    ``[p for w in groups for p in w]`` to recover ``reference_phonemes``'s sequence.
    """
    tokens = text.split()
    if not tokens:
        return []

    from phonemizer import phonemize
    from phonemizer.separator import Separator

    _ensure_espeak()
    # A list input phonemizes each token independently in a single backend call;
    # word="" because each item is already one token (no internal boundary needed).
    ipa_list = phonemize(
        tokens,
        language=espeak_lang,
        backend="espeak",
        strip=True,
        with_stress=False,
        preserve_punctuation=False,
        separator=Separator(phone=" ", word="", syllable=""),
    )
    # Some phonemizer versions return a bare str for a length-1 list; normalize.
    if isinstance(ipa_list, str):
        ipa_list = [ipa_list]
    return [_normalize_phones(_tokenize_ipa(ipa)) for ipa in ipa_list]


# =====================================================================
# Phone tokenization, diacritic stripping and inventory fold.
# =====================================================================
def _tokenize_ipa(ipa: str) -> List[str]:
    """Split an IPA string into phone symbols (whitespace-separated, else per char)."""
    ipa = ipa.strip()
    if " " in ipa or "\n" in ipa:
        return [tok for tok in ipa.split() if tok]
    return [ch for ch in ipa if not ch.isspace()]


# Suprasegmental diacritics one side marks and the other does not (espeak vs the
# w2v2 recognizer). Written as code points so no bare combining marks appear here.
_DIACRITIC_CODEPOINTS = (
    0x02D0, 0x02D1, 0x02B0, 0x02C0, 0x02C8, 0x02CC,
    0x0329, 0x030D, 0x032F, 0x0361, 0x035C,
)
_STRIP_DIACRITICS = dict.fromkeys(_DIACRITIC_CODEPOINTS)

# Inventory fold: the espeak reference (en-us) and the recognizer emit the same
# sounds under different IPA conventions; this canonicalizes both sides so they
# align. Spanish-safe (Spanish espeak already emits these cardinal symbols, so the
# table is near-identity there).
_PHONE_FOLD = {
    "ɹ": "r", "ɾ": "r", "ɻ": "r",     # rhotic approximant / flap / retroflex -> r
    "æ": "a", "ɐ": "a",               # near-open front / near-open central -> a
    "ᵻ": "ɪ", "ɨ": "ɪ",              # reduced / central high vowel -> ɪ
    "ɚ": "ə", "ɝ": "ə",              # r-colored schwa -> plain schwa
    "oʊ": "o", "əʊ": "o",            # GA / RP "goat" diphthong -> o
}


def _normalize_phones(tokens: List[str]) -> List[str]:
    """Drop suprasegmental diacritics and fold the inventory so both sides align."""
    cleaned = (tok.translate(_STRIP_DIACRITICS) for tok in tokens)
    folded = (_PHONE_FOLD.get(tok, tok) for tok in cleaned)
    return [tok for tok in folded if tok]


# =====================================================================
# Articulatory feature distance (panphon). Imported lazily and cached so the
# module imports without panphon and the feature table is built only once.
# =====================================================================
@lru_cache(maxsize=1)
def _feature_table():
    """Build the panphon feature table once (UTF-8 safe on cp1252 Windows)."""
    import panphon

    try:
        return panphon.FeatureTable()
    except UnicodeDecodeError:
        # panphon opens its bundled UTF-8 data via pathlib.Path.open without an
        # encoding, which fails under a cp1252 default. Retry once with a narrowly
        # scoped UTF-8 default, restored immediately afterwards. Running the app in
        # UTF-8 mode (PYTHONUTF8=1 at launch) makes this branch never execute.
        import pathlib

        original_open = pathlib.Path.open

        def _utf8_open(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
            if "b" not in mode and encoding is None:
                encoding = "utf-8"
            return original_open(self, mode, buffering, encoding, errors, newline)

        pathlib.Path.open = _utf8_open
        try:
            return panphon.FeatureTable()
        finally:
            pathlib.Path.open = original_open


@lru_cache(maxsize=4096)
def _phone_vector(phone: str):
    """Numeric articulatory feature vector for one phone, or None if unknown."""
    vectors = _feature_table().word_to_vector_list(phone, numeric=True)
    return tuple(vectors[0]) if vectors else None


@lru_cache(maxsize=8192)
def _substitution_cost(a: str, b: str) -> float:
    """Feature distance in [0, 1] between two phones (0 == identical; 1 == unknown)."""
    if a == b:
        return 0.0
    va, vb = _phone_vector(a), _phone_vector(b)
    if va is None or vb is None or len(va) != len(vb):
        return 1.0
    differing = sum(1 for x, y in zip(va, vb) if x != y)
    return differing / len(va)


# =====================================================================
# Step 3 -- feature-weighted edit-distance alignment and scoring.
# =====================================================================
_OP_SUB, _OP_DEL, _OP_INS = "sub", "del", "ins"


def _edit_alignment(reference: List[str],
                    spoken: List[str]) -> Tuple[List[Tuple[str, str]], float]:
    """Feature-weighted edit-distance alignment over phone tokens.

    Returns the aligned ``(reference, spoken)`` pairs ("" marks an inserted or
    deleted phone) and the total fractional distance. Rolled by hand because
    ``Levenshtein`` works on characters, not lists of multi-character IPA tokens,
    and we need per-phone feature substitution costs.
    """
    n, m = len(reference), len(spoken)
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    back: List[List[Optional[str]]] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = float(i)
        back[i][0] = _OP_DEL
    for j in range(1, m + 1):
        cost[0][j] = float(j)
        back[0][j] = _OP_INS
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitute = cost[i - 1][j - 1] + _substitution_cost(reference[i - 1], spoken[j - 1])
            delete = cost[i - 1][j] + 1.0
            insert = cost[i][j - 1] + 1.0
            best = min(substitute, delete, insert)
            cost[i][j] = best
            back[i][j] = _OP_SUB if best == substitute else (_OP_DEL if best == delete else _OP_INS)

    pairs: List[Tuple[str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op == _OP_SUB:
            pairs.append((reference[i - 1], spoken[j - 1]))
            i, j = i - 1, j - 1
        elif op == _OP_DEL:
            pairs.append((reference[i - 1], ""))
            i -= 1
        else:
            pairs.append(("", spoken[j - 1]))
            j -= 1
    pairs.reverse()
    return pairs, cost[n][m]


def _capped_per_phone_distance(pairs: List[Tuple[str, str]],
                               distance: float, n_reference: int) -> float:
    """Per-phone distance with the insertion contribution capped per reference phone.

    Insertions scale with the *spoken* length, so a hallucinating recognizer can
    inflate the distance without bound and floor a correct read. We cap the
    insertion part at ``INSERTION_CAP_PER_PHONE * n_reference`` and leave
    substitutions/deletions (real errors) intact.
    """
    if INSERTION_CAP_PER_PHONE <= 0 or n_reference <= 0:
        return distance / n_reference if n_reference else 0.0
    insertion_cost = float(sum(1 for ref_sym, _ in pairs if not ref_sym))
    cap = INSERTION_CAP_PER_PHONE * n_reference
    capped = (distance - insertion_cost) + min(insertion_cost, cap)
    return capped / n_reference


def _bad_baseline(reference: List[str], spoken: List[str]) -> float:
    """Per-utterance "completely wrong" anchor: mean feature distance over all pairs."""
    if not reference or not spoken:
        return BAD_BASELINE_DEFAULT
    total = sum(_substitution_cost(r, s) for r in reference for s in spoken)
    observed = total / (len(reference) * len(spoken))
    return _widen_bad_for_length(observed, len(reference))


def _widen_bad_for_length(observed_bad: float, n_reference: int) -> float:
    """Raise a short phrase's ``bad`` anchor toward a conservative ceiling.

    Short references have a noisy, often-too-low ``bad``, which floors correct-but-
    accented speech. We blend toward ``BAD_CEILING`` with a length-dependent trust
    weight; ``max(observed, ceiling)`` guarantees this only ever *widens* the window
    (genuine garbage still floors). BAD_SHRINK_PHONES == 0 disables it.
    """
    if BAD_SHRINK_PHONES <= 0:
        return observed_bad
    trust = n_reference / (n_reference + BAD_SHRINK_PHONES)
    return trust * observed_bad + (1.0 - trust) * max(observed_bad, BAD_CEILING)


def _score_from_distance(per_phone_distance: float, bad: float, good: float) -> float:
    """Map a per-phone distance onto 0-100 against the [good, bad] window."""
    span = max(bad - good, BAD_MIN_SPAN)
    accuracy = 1.0 - (per_phone_distance - good) / span
    return round(max(0.0, min(1.0, accuracy)) * 100.0, 1)


def _phoneme_recall(pairs: List[Tuple[str, str]]) -> float:
    """Fraction of reference phones actually produced, read off the alignment."""
    recalled = 0
    total = 0
    for ref_sym, hyp_sym in pairs:
        if not ref_sym:                       # insertion: no reference phone here
            continue
        total += 1
        if hyp_sym and _substitution_cost(ref_sym, hyp_sym) < RECALL_MAX_DIST:
            recalled += 1
    return recalled / total if total else 0.0


@dataclass
class ScoreResult:
    """Two-axis score plus the alignment and the raw per-phone distances."""

    score: float
    pairs: List[Tuple[str, str]]
    per_phone_distance: float
    bad_baseline: float
    phoneme_score: float
    recall: float
    good: float


def align_and_score(reference: List[str], spoken: List[str],
                    good: Optional[float] = None) -> ScoreResult:
    """Align ``spoken`` against ``reference`` and score on two axes (quality + recall).

    ``good`` overrides the GOOD anchor of the phoneme axis: ``None`` uses the global
    ``PHONEME_GOOD``; a supplied value (ceiling mode) is the reference's own per-phone
    distance, so a flawless read maps to 100 for that phrase.
    """
    if not reference:
        raise ValueError("empty reference phoneme sequence")

    pairs, distance = _edit_alignment(reference, spoken)
    per_phone_distance = _capped_per_phone_distance(pairs, distance, len(reference))
    bad = _bad_baseline(reference, spoken)
    good_anchor = PHONEME_GOOD if good is None else good
    phoneme_score = _score_from_distance(per_phone_distance, bad, good_anchor)
    recall = _phoneme_recall(pairs)
    score = round(WEIGHT_PHONEME * phoneme_score + WEIGHT_WORD * recall * 100.0, 1)
    return ScoreResult(
        score=score,
        pairs=pairs,
        per_phone_distance=per_phone_distance,
        bad_baseline=bad,
        phoneme_score=phoneme_score,
        recall=recall,
        good=good_anchor,
    )


# =====================================================================
# Reference recognition cache (ceiling-mode GOOD anchor + retry reuse).
# The same reference take is scored on every retry of a phrase; recognizing it once
# mirrors pronounce's _reference_features. Keyed by content hash + sample rate; the
# cache is tiny and self-clearing, so it never grows on a long session.
# =====================================================================
_ref_cache: Dict[Tuple[int, int], List[str]] = {}


def _recognize_reference(reference_audio: np.ndarray, reference_sr: int,
                         device: str) -> List[str]:
    """Recognized phonemes of the reference take, cached by content + sample rate."""
    key = (hash(np.asarray(reference_audio, dtype=np.float32).tobytes()), reference_sr)
    cached = _ref_cache.get(key)
    if cached is not None:
        return cached
    spoken = _spoken_from_wav(_prepare_waveform(reference_audio, reference_sr), device)
    if len(_ref_cache) >= 8:
        _ref_cache.clear()
    _ref_cache[key] = spoken
    return spoken


# =====================================================================
# Phone-error -> word mapping (for the GUI's word-level highlight, task §6).
# =====================================================================
def _word_recall(groups: List[List[str]],
                 pairs: List[Tuple[str, str]]
                 ) -> Tuple[List[int], List[List[str]], List[float]]:
    """Per reference word: recalled-phone count, heard phones, and mean distance.

    ``groups`` is the per-word phone grouping; ``pairs`` is the alignment over the
    flattened reference (same order), so we can attribute each non-insertion pair to
    its word by walking a phone->word index table.

    ``word_dist[wi]`` is the mean articulatory distance over that word's reference
    phones: a substituted phone contributes its feature cost, a *deleted* phone
    (not spoken) contributes the maximum 1.0. This is the same axis the overall
    score uses, so the per-word highlight agrees with the bucket -- a "dirty" or
    swapped word is far from the reference and reads as such, unlike the lenient
    50%-recall flag it replaces (task §6).
    """
    word_of: List[int] = []
    for wi, group in enumerate(groups):
        word_of.extend([wi] * len(group))
    n_ref = len(word_of)

    recalled = [0] * len(groups)
    heard: List[List[str]] = [[] for _ in groups]
    dist_sum = [0.0] * len(groups)            # summed phone distance per word
    ref_idx = 0
    for ref_sym, hyp_sym in pairs:
        if not ref_sym:                       # insertion: no reference phone
            continue
        if ref_idx < n_ref:
            wi = word_of[ref_idx]
            if hyp_sym:
                heard[wi].append(hyp_sym)
                cost = _substitution_cost(ref_sym, hyp_sym)
                if cost < RECALL_MAX_DIST:
                    recalled[wi] += 1
            else:
                cost = 1.0                    # deletion: phone not produced at all
            dist_sum[wi] += cost
        ref_idx += 1

    word_dist = [
        dist_sum[wi] / len(groups[wi]) if groups[wi] else 0.0
        for wi in range(len(groups))
    ]
    return recalled, heard, word_dist


def _word_level(word_avg: float, good: float, bad: float) -> str:
    """Classify a word's mean phone distance as "good" / "ok" / "bad".

    Placed on the same [good, bad] window the phoneme-quality score uses, so the
    three-level highlight tracks the bucket: ``frac`` is the word's position from a
    flawless read (0) to "completely wrong" (1). Cutoffs come from calibration.json
    (``word_good_frac`` / ``word_bad_frac``); the defaults split the window roughly
    in thirds.
    """
    span = max(bad - good, BAD_MIN_SPAN)
    frac = (word_avg - good) / span
    if frac <= WORD_GOOD_FRAC:
        return "good"
    if frac >= WORD_BAD_FRAC:
        return "bad"
    return "ok"


def _reference_word_tags(tokens: List[str], levels: List[str]) -> List[Dict[str, Any]]:
    """One {"word", "level", "correct"} per target token (case/punctuation kept).

    ``level`` drives the GUI's three-level colour (good/ok/bad); ``correct`` is kept
    for any consumer still reading the boolean (a word is "correct" unless it is bad).
    """
    tags: List[Dict[str, Any]] = []
    for i, token in enumerate(tokens):
        level = levels[i] if i < len(levels) else "good"
        tags.append({"word": token, "level": level, "correct": level != "bad"})
    return tags


# =====================================================================
# 0-5 bucketization (task §4): raw 0-100 score -> human-calibrated grade + percent.
# =====================================================================
def _score_to_bucket(score: float) -> int:
    """Coarse 0-5 grade: how many ascending cutpoints the score clears (0..5).

    Returns -1 when no cutpoints are configured, signalling the caller to keep the
    raw 0-100 score (and its threshold) instead of a bucket.
    """
    if not BUCKET_CUTPOINTS:
        return -1
    return int(sum(1 for c in BUCKET_CUTPOINTS if score >= c))


def _bucket_to_percent(bucket: int, fallback: float) -> float:
    """User-facing percent for a bucket: the midpoint of its [lo, hi] band.

    The midpoint keeps every take in a bucket on one number (flat), so within-bucket
    engine noise is hidden (task §4.3). Falls back to ``fallback`` (the raw score)
    when the bucket has no configured band.
    """
    band = BUCKET_TO_PERCENT.get(str(bucket))
    if not band or len(band) != 2:
        return fallback
    lo, hi = band
    return round((float(lo) + float(hi)) / 2.0, 1)


# =====================================================================
# Entry point.
# =====================================================================
def analyze(user_audio: np.ndarray,
            expected_text: str,
            reference_audio: Optional[np.ndarray] = None,
            user_sr: int = TARGET_SAMPLE_RATE,
            reference_sr: int = KOKORO_SAMPLE_RATE,
            voice: Optional[str] = None,
            is_reference: bool = False) -> PronunciationResult:
    """Compare a user's spoken attempt against the expected phrase, phoneme-level.

    Args:
        user_audio: user's recorded waveform (1-D float32).
        expected_text: the reference phrase the user was asked to repeat.
        reference_audio: Kokoro-synthesised reference waveform. Used in ceiling mode
            to anchor a flawless read to 100 per phrase; optional (falls back to the
            global GOOD anchor when absent).
        user_sr: sample rate of ``user_audio`` (recording path is 16 kHz).
        reference_sr: sample rate of ``reference_audio`` (Kokoro is 24 kHz).
        voice: Kokoro voice the reference was synthesized with (recorded for logs).
        is_reference: marks the reference self-test. Accepted now so the
            signature is stable; honest scoring is unchanged here.

    Returns:
        PronunciationResult with score, per-word/per-phone tags and transcription.
        ``prosody`` is left empty; the host fills it from the raw waveforms.
    """
    cfg = get_config()
    _ensure_loaded()

    # Reference phonemes from text (per-word groups -> flat sequence).
    groups = reference_word_phonemes(expected_text, cfg.espeak_language)
    reference = [p for group in groups for p in group]
    if not reference:
        raise ValueError(f"espeak produced no phonemes for: {expected_text!r}")

    spoken = _spoken_from_wav(_prepare_waveform(user_audio, user_sr), cfg.device)

    # Ceiling-mode GOOD anchor: the reference take's own per-phone distance, so a
    # flawless read maps to 100 for this phrase regardless of per-phrase quirks.
    good = _ceiling_good(reference, reference_audio, reference_sr, cfg)
    result = align_and_score(reference, spoken, good=good)

    # Map phone errors back to whole words for the GUI highlight. The per-word mean
    # distance is classified on the SAME [good, bad] window the score uses, so the
    # three-level colour (good/ok/bad) tracks the bucket rather than a lenient recall.
    tokens = expected_text.split()
    _recalled, heard, word_dist = _word_recall(groups, result.pairs)
    levels = [
        _word_level(word_dist[wi], result.good, result.bad_baseline)
        for wi in range(len(groups))
    ]
    # A word is an error (red, "you need to pronounce X") only when "bad"; "ok"
    # words are acceptable (light grey) and stay out of the error list.
    words_with_errors = [
        tokens[wi] for wi in range(min(len(groups), len(tokens))) if levels[wi] == "bad"
    ]
    reference_words = _reference_word_tags(tokens, levels)
    word_errors = [
        {
            "word": tokens[wi] if wi < len(tokens) else "",
            "expected": groups[wi],
            "heard": heard[wi],
            "level": levels[wi],
            "distance": round(word_dist[wi], 4),
        }
        for wi in range(len(groups))
        if levels[wi] == "bad"
    ]
    word_diff = [{"expected": w} for w in words_with_errors]

    # "Heard" line: recognized phonemes in order, each flagged correct/incorrect.
    recognized_units = [
        {
            "unit": hyp_sym,
            "correct": bool(ref_sym) and _substitution_cost(ref_sym, hyp_sym) < RECALL_MAX_DIST,
        }
        for ref_sym, hyp_sym in result.pairs
        if hyp_sym
    ]

    transcription = " ".join(spoken)

    # Coarsen the raw 0-100 score to a 0-5 bucket (task §4). When buckets are
    # configured, "passed" and the user percent come from the bucket; otherwise the
    # engine degrades to the raw score and its threshold (no calibration -> no crash).
    bucket = _score_to_bucket(result.score)
    if bucket >= 0:
        passed = bucket >= PASS_BUCKET
        user_percent = _bucket_to_percent(bucket, result.score)
    else:
        passed = result.score >= cfg.score_threshold
        user_percent = result.score
    feedback = _build_feedback(result.score, passed, words_with_errors)

    logging.info(
        "[phoneme] score=%.1f -> bucket=%d (%.0f%%) | (phoneme=%.1f recall=%.2f) | "
        "dist/phone=%.4f (good=%.3f bad=%.3f) | ref=%d spoken=%d | is_ref=%s | voice=%s | "
        "bad_words=%s | ref_ipa=%r | asr_ipa=%r",
        result.score, bucket, user_percent, result.phoneme_score, result.recall,
        result.per_phone_distance, result.good, result.bad_baseline,
        len(reference), len(spoken), is_reference, voice,
        words_with_errors, " ".join(reference), transcription,
    )

    # Calibration / analysis sample log (best effort; mirrors the acoustic engine's
    # pronounce_samples.jsonl). phoneme/calibrate.py re-anchors PHONEME_GOOD from the
    # good *real* attempts here (it drops the is_reference self-tests, whose ~0
    # distance would wreck the anchor); the per-word block and the two phoneme
    # strings make a low score easy to inspect after the fact.
    _append_sample({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "text": expected_text,
        "user_name": cfg.user_name,
        "voice": voice,
        "is_reference": bool(is_reference),
        "score": result.score,
        "bucket": bucket,
        "user_percent": user_percent,
        "passed": passed,
        "phoneme_score": round(result.phoneme_score, 1),
        "recall": round(result.recall, 4),
        "per_phone_distance": round(result.per_phone_distance, 5),
        "bad_baseline": round(result.bad_baseline, 5),
        "good_anchor": round(result.good, 5),
        "n_reference": len(reference),
        "n_spoken": len(spoken),
        "reference_phonemes": reference,
        "spoken_phonemes": spoken,
        "words": [
            {"word": tokens[wi] if wi < len(tokens) else "",
             "level": levels[wi],
             "distance": round(word_dist[wi], 4),
             "expected": groups[wi],
             "heard": heard[wi]}
            for wi in range(len(groups))
        ],
    })

    return PronunciationResult(
        score=result.score,
        word_errors=word_errors,
        prosody={},
        transcription=transcription,
        passed=passed,
        feedback=feedback,
        words_with_errors=words_with_errors,
        expected_phonemes=reference,
        transcribed_phonemes=spoken,
        word_diff=word_diff,
        reference_words=reference_words,
        recognized_units=recognized_units,
        bucket=bucket,
        user_percent=user_percent,
        per_phone_distance=result.per_phone_distance,
        bad_baseline=result.bad_baseline,
        phoneme_score=result.phoneme_score,
        recall=result.recall,
        good_anchor=result.good,
    )


def _ceiling_good(reference: List[str], reference_audio: Optional[np.ndarray],
                  reference_sr: int, cfg) -> Optional[float]:
    """Per-phrase GOOD anchor from the reference take, or None for global mode.

    Returns the reference's per-phone distance (ceiling mode) when a reference take
    is available; otherwise None, leaving the scorer on the global PHONEME_GOOD. A
    missing reference never fails the analysis.
    """
    if cfg.good_mode != "ceiling" or reference_audio is None:
        return None
    ref_array = np.asarray(reference_audio, dtype=np.float32)
    if ref_array.size == 0:
        return None
    ceiling_spoken = _recognize_reference(ref_array, reference_sr, cfg.device)
    if not ceiling_spoken:
        return None
    return align_and_score(reference, ceiling_spoken).per_phone_distance


def _build_feedback(score: float, passed: bool, words_with_errors: List[str]) -> str:
    """Short human-readable summary mirroring the acoustic engine's style."""
    lines = [f"Score: {score:.0f}/100 " + ("(passed)" if passed else "(try again)")]
    if words_with_errors:
        lines.append("❌ You need to better pronounce these words: "
                     + ", ".join(words_with_errors))
    elif passed:
        lines.append("✅ Great pronunciation!")
    return "\n".join(lines)
