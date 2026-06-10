"""Pronunciation analysis core for EchoLoop.

Adapted from OpenPronounce (https://github.com/Halleck45/OpenPronounce), MIT License.
The acoustic / phoneme comparison logic is reused as a library; the original web
front-end and built-in TTS were dropped. In EchoLoop the reference audio is produced
by the existing Kokoro TTS and passed into ``analyze`` as a NumPy array, so this module
never synthesizes speech itself and never touches the GUI.

Public API:
    load_models()  -- load Wav2Vec2 weights once (call at mode startup, in a thread).
    warm_up()      -- run a dummy pass to remove first-call latency (mirrors stt/tts).
    analyze(...)   -- single entry point returning a PronunciationResult.

Design notes:
    * Models load lazily on first use; ``load_models`` only makes that explicit so the
      heavy download/initialisation can happen in a background daemon thread, matching
      the warm-up pattern in ``stt.py`` / ``tts.py``.
    * Device follows ``config.DEVICE``; it can be overridden with an optional
      ``config.WAV2VEC2_DEVICE`` value without editing this file.
    * Wav2Vec2 needs 16 kHz mono audio. User audio already arrives at 16 kHz from the
      recording path; the Kokoro reference is 24 kHz, so it is resampled here.
"""

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import librosa
import Levenshtein
from fastdtw import fastdtw
from scipy.spatial.distance import cosine, euclidean
from sklearn.preprocessing import MinMaxScaler
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC
from phonemizer import phonemize

# Allow autonomous use (e.g. running the test directly from inside pronounce/):
# make the project root importable so ``import config`` resolves.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import config


# =====================================================================
# Configuration (read from config.py with safe fallbacks so this module
# stays usable even before config.py gains pronunciation-specific keys)
# =====================================================================
# Wav2Vec2 expects strictly 16 kHz mono input.
TARGET_SAMPLE_RATE = 16_000
# Kokoro synthesises at 24 kHz; used as the default reference sample rate.
KOKORO_SAMPLE_RATE = 24_000

MODEL_NAME = getattr(config, "WAV2VEC2_MODEL_NAME", "facebook/wav2vec2-large-960h")
# Default device follows config.DEVICE (cuda/cpu); overridable via config.WAV2VEC2_DEVICE.
DEVICE = getattr(config, "WAV2VEC2_DEVICE", config.DEVICE)
# Score (0-100) at or above which a repetition is considered acceptable.
SCORE_THRESHOLD = getattr(config, "PRONUNCE_SCORE_THRESHOLD", 70.0)
# espeak dialect ("en-us"/"en-gb") used to phonemize the reference text. Follows
# the accent selected in config so British speech is scored against British
# phonemes (and American against American), not always en-us.
ESPEAK_LANGUAGE = getattr(config, "ESPEAK_LANGUAGE", "en-us")

# ---------------------------------------------------------------------
# Acoustic score calibration.
#
# The acoustic component compares per-step cosine DTW distance between the two
# Wav2Vec2 embedding sequences. Two anchors map it to 0-100:
#   * floor (ACOUSTIC_GOOD)  — typical per-step distance of a *good* attempt by a
#     different speaker (the user vs the TTS voice never reach 0). Calibrated
#     per voice/microphone; ``python pronounce/calibrate.py`` writes it to
#     calibration.json from collected session samples.
#   * ceiling — per-step distance when content does not match. Derived
#     automatically per utterance from the random-pair baseline (the mean
#     distance between unaligned frames of the two recordings), so it adapts to
#     each phrase without manual tuning.
# ---------------------------------------------------------------------
ACOUSTIC_GOOD_DEFAULT = float(getattr(config, "PRONUNCE_ACOUSTIC_GOOD", 0.20))
ACOUSTIC_BAD_DEFAULT = 0.60      # fixed ceiling when no per-utterance baseline exists
ACOUSTIC_BAD_FRACTION = 0.9      # ceiling = this fraction of the random-pair baseline
ACOUSTIC_MIN_SPAN = 0.05         # minimal floor-to-ceiling span (avoids degenerate scale)
TRIM_TOP_DB = 30                 # silence trim threshold relative to peak, in dB

# Persisted calibration (floor override) and the per-attempt sample log used by
# pronounce/calibrate.py to compute that override on request.
CALIBRATION_FILE = Path(__file__).resolve().parent / "calibration.json"
SAMPLES_FILE = Path(getattr(config, "LOG_DIR", _ROOT / "logs")) / "pronounce_samples.jsonl"


def _load_calibration() -> float:
    """Return the calibrated acoustic floor, or the default when not calibrated."""
    try:
        if CALIBRATION_FILE.exists():
            data = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
            value = float(data["acoustic_good"])
            logging.info(f"[pronounce] Loaded calibration: acoustic_good={value:.4f} "
                         f"({CALIBRATION_FILE})")
            return value
    except Exception:
        logging.exception("Failed to read calibration file; using defaults:")
    return ACOUSTIC_GOOD_DEFAULT


ACOUSTIC_GOOD = _load_calibration()


def save_calibration(acoustic_good: float, extra: Optional[Dict[str, Any]] = None) -> None:
    """Persist a new acoustic floor (and apply it to the running process)."""
    global ACOUSTIC_GOOD
    payload: Dict[str, Any] = {
        "acoustic_good": round(float(acoustic_good), 5),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    CALIBRATION_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    ACOUSTIC_GOOD = float(acoustic_good)
    logging.info(f"[pronounce] Saved calibration acoustic_good={acoustic_good:.4f} "
                 f"-> {CALIBRATION_FILE}")

# Lazily-initialised model singletons (loaded once, reused for every analysis).
_processor: Optional[Wav2Vec2Processor] = None
_model: Optional[Wav2Vec2Model] = None          # embeddings (acoustic similarity)
_model_ctc: Optional[Wav2Vec2ForCTC] = None     # transcription (what was recognised)


# =====================================================================
# Result type (the module's contract with the GUI layer)
# =====================================================================
@dataclass
class PronunciationResult:
    """Outcome of one pronunciation comparison.

    The four fields required by the spec are ``score``, ``word_errors``,
    ``prosody`` and ``transcription``; the rest are extras useful for richer
    feedback (e.g. plotting contours) and are safe for the GUI to ignore.
    """

    score: float                                  # 0-100 overall pronunciation score
    word_errors: List[Dict[str, Any]]             # per-word expected/actual phonemes
    prosody: Dict[str, List[float]]               # {"f0": [...], "energy": [...]}
    transcription: str                            # what the ASR recognised
    passed: bool = False                          # score >= SCORE_THRESHOLD
    feedback: str = ""                            # human-readable summary
    acoustic_distance: int = 0                    # total Wav2Vec2-embedding DTW distance
    acoustic_per_step: float = 0.0                # DTW distance per alignment step (scored)
    acoustic_baseline: float = 0.0                # random-pair distance (per-utterance ceiling)
    words_with_errors: List[str] = field(default_factory=list)
    expected_phonemes: List[str] = field(default_factory=list)
    transcribed_phonemes: List[str] = field(default_factory=list)


# =====================================================================
# Model lifecycle
# =====================================================================
def load_models() -> None:
    """Load Wav2Vec2 weights into memory once. Safe to call repeatedly.

    Heavy (~1.2 GB download on first run); call from a background daemon thread
    at mode startup so the GUI stays responsive.
    """
    global _processor, _model, _model_ctc
    if _model is not None and _model_ctc is not None and _processor is not None:
        return

    _processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

    _model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(DEVICE)
    _model.eval()

    _model_ctc = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(DEVICE)
    _model_ctc.eval()


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


# =====================================================================
# Audio preparation
# =====================================================================
def _prepare_waveform(waveform: np.ndarray, orig_sr: int) -> np.ndarray:
    """Return a 1-D float32 mono waveform resampled to TARGET_SAMPLE_RATE."""
    wav = np.asarray(waveform, dtype=np.float32)

    # Down-mix to mono. torchaudio gives [channels, samples] while soundfile gives
    # [samples, channels], so average along whichever axis is smaller (the channels).
    if wav.ndim > 1:
        wav = wav.mean(axis=int(np.argmin(wav.shape)))

    if orig_sr != TARGET_SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=TARGET_SAMPLE_RATE)

    return np.ascontiguousarray(wav, dtype=np.float32)


def _trim_silence(wav: np.ndarray) -> np.ndarray:
    """Cut leading/trailing silence so pauses don't inflate the DTW distance.

    Matters especially for the user recording: peak normalization in the capture
    path boosts the noise floor of quiet takes, turning silent padding into loud
    noise that has no counterpart in the clean TTS reference. Keeps the original
    audio when trimming would leave less than 0.1 s (i.e. near-silent input).
    """
    if wav.size == 0:
        return wav
    trimmed, _ = librosa.effects.trim(wav, top_db=TRIM_TOP_DB)
    if trimmed.size < int(0.1 * TARGET_SAMPLE_RATE):
        return wav
    return np.ascontiguousarray(trimmed, dtype=np.float32)


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
    input_values = input_values.to(DEVICE)

    with torch.no_grad():
        features = _model(input_values).last_hidden_state  # (batch, time, features)

    return features.squeeze(0).cpu().numpy()


def transcribe(audio_waveform: np.ndarray) -> str:
    """Transcribe audio to text using the Wav2Vec2 CTC head."""
    _ensure_loaded()

    inputs = _processor(audio_waveform, sampling_rate=TARGET_SAMPLE_RATE,
                        return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(DEVICE)

    with torch.no_grad():
        logits = _model_ctc(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1).cpu()  # decode on CPU
    return _processor.batch_decode(predicted_ids)[0]


# =====================================================================
# Phoneme / text comparison (reused OpenPronounce core)
# =====================================================================
def get_phonemes_with_word_mapping(text: str):
    """Return a list of phonemes and a mapping {phoneme_index: source_word}."""
    # Split on words, ignoring punctuation, to avoid issues like "times,".
    words = re.findall(r"\b[\w']+\b", text)

    phonemes: List[str] = []
    phoneme_to_word: Dict[int, str] = {}

    for word in words:
        try:
            word_phonemes = phonemize(word, language=ESPEAK_LANGUAGE, backend="espeak",
                                      strip=True, preserve_punctuation=False).split()
        except Exception:
            try:
                word_phonemes = phonemize(word, language="en-us", backend="festival",
                                          strip=True, preserve_punctuation=False).split()
            except Exception:
                word_phonemes = []  # fallback if every backend fails

        for phoneme in word_phonemes:
            phoneme_to_word[len(phonemes)] = word
            phonemes.append(phoneme)

    return phonemes, phoneme_to_word


def get_phoneme_embeddings(phoneme_seq: str) -> np.ndarray:
    """Convert a phoneme string into a numerical column vector (ord per char)."""
    return np.array([ord(p) for p in phoneme_seq]).reshape(-1, 1)


def compare_transcriptions(transcription: str, text_reference: str) -> Dict[str, Any]:
    """Compare an ASR transcription against the expected text.

    Identifies per-word pronunciation errors via phoneme alignment and returns
    distances used by the scoring formula plus contours for optional plotting.
    """
    transcription_clean = transcription.lower().strip()
    reference_clean = text_reference.lower().strip()

    # Edit distance between transcription and reference text (global word-level signal).
    word_distance = Levenshtein.distance(transcription_clean, reference_clean)

    # Extract phonemes from both versions.
    expected_phonemes, expected_map = get_phonemes_with_word_mapping(text_reference)
    transcribed_phonemes, transcribed_map = get_phonemes_with_word_mapping(transcription_clean)

    # Global phoneme distance: edit distance over the phoneme sequences. Unlike
    # the old DTW over character codes, this counts actual phoneme substitutions/
    # insertions/deletions and can be normalized by the expected length.
    distance = Levenshtein.distance(expected_phonemes, transcribed_phonemes)

    # Numerical phoneme sequences, used below to build comparable contours.
    expected_seq = get_phoneme_embeddings(" ".join(expected_phonemes))
    transcribed_seq = get_phoneme_embeddings(" ".join(transcribed_phonemes))

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
    # (the flat phoneme_to_word map cannot distinguish adjacent duplicate words).
    expected_words_list = re.findall(r"\b[\w']+\b", text_reference)
    current_phoneme_idx = 0

    for word in expected_words_list:
        # Re-phonemize the word to learn how many phonemes it owns.
        try:
            p_list = phonemize(word, language=ESPEAK_LANGUAGE, backend="espeak",
                               strip=True, preserve_punctuation=False).split()
        except Exception:
            try:
                p_list = phonemize(word, language="en-us", backend="festival",
                                   strip=True, preserve_punctuation=False).split()
            except Exception:
                p_list = []

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

        # Compare the expected vs actual phoneme segments.
        expected_seg = [expected_phonemes[i] for i in word_indices]
        actual_seg = [transcribed_phonemes[i] for i in sorted_trans_indices]

        p_dist = Levenshtein.distance(expected_seg, actual_seg)

        # Mark a mispronunciation when the phoneme edit distance exceeds 40% of length.
        if p_dist > len(expected_seg) * 0.4:
            errors.append({
                "position": word_indices.start,
                "expected": "".join(expected_seg),  # expected phonemes
                "actual": "".join(actual_seg),      # actual phonemes
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

    # DTW-align the numeric phoneme vectors so the two contours share a length.
    expected_vector, transcribed_vector = align_sequences_dtw(
        expected_seq.tolist(), transcribed_seq.tolist())

    return {
        "word_distance": word_distance,
        "reference_length": len(reference_clean),
        "phoneme_distance": distance,
        "errors": errors,
        "feedback": feedback,
        "transcribe": transcription,
        "expected_vector": expected_vector.astype(float).tolist(),
        "transcribed_vector": transcribed_vector.astype(float).tolist(),
        "expected_phonemes": expected_phonemes,
        "transcribed_phonemes": transcribed_phonemes,
        "words_with_errors": list(words_with_errors),
    }


def align_sequences_dtw(seq1, seq2):
    """Align two numeric sequences with DTW and return equal-length arrays.

    Lets contours of different speeds/lengths be compared directly.
    """
    _distance, path = fastdtw(seq1, seq2, dist=euclidean)

    aligned_seq1 = []
    aligned_seq2 = []
    for i, j in path:
        aligned_seq1.append(seq1[i][0])  # preserve the first (only) dimension
        aligned_seq2.append(seq2[j][0])

    return np.array(aligned_seq1), np.array(aligned_seq2)


def _random_pair_baseline(emb_a: np.ndarray, emb_b: np.ndarray,
                          n_pairs: int = 2000, seed: int = 0) -> float:
    """Mean cosine distance between randomly paired frames of two embeddings.

    Approximates the per-step DTW distance of *unrelated* content, giving each
    utterance its own automatic "completely wrong" ceiling for the acoustic score.
    """
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(emb_a), n_pairs)
    j = rng.integers(0, len(emb_b), n_pairs)
    a, b = emb_a[i], emb_b[j]
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9
    return float(np.mean(1.0 - num / den))


def acoustic_bad_for(baseline: float, acoustic_good: Optional[float] = None) -> float:
    """Per-utterance acoustic ceiling derived from the random-pair baseline."""
    good = ACOUSTIC_GOOD if acoustic_good is None else acoustic_good
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
        acoustic_good: floor override; defaults to the calibrated ACOUSTIC_GOOD.
    """
    good = ACOUSTIC_GOOD if acoustic_good is None else acoustic_good
    bad = ACOUSTIC_BAD_DEFAULT if acoustic_bad is None else acoustic_bad
    bad = max(bad, good + ACOUSTIC_MIN_SPAN)

    dtw_score = 100.0 * min(1.0, max(0.0, 1.0 - (acoustic_per_step - good) / (bad - good)))
    phoneme_score = 100.0 * min(1.0, max(0.0, 1.0 - phoneme_error_rate))
    word_score = 100.0 * min(1.0, max(0.0, 1.0 - word_error_rate))

    # Weighting: acoustic DTW 40%, phonemes 30%, words 30%.
    final_score = 0.4 * dtw_score + 0.3 * phoneme_score + 0.3 * word_score

    final_score = min(100.0, max(0.0, final_score))
    return round(final_score, 2)


# =====================================================================
# Prosody (pitch & energy contours)
# =====================================================================
def extract_f0(audio_waveform: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Extract the fundamental frequency (F0) contour; NaNs -> 0."""
    f0, _voiced_flag, _voiced_probs = librosa.pyin(audio_waveform, fmin=50, fmax=300, sr=sr)
    return np.nan_to_num(f0)


def extract_energy(audio_waveform: np.ndarray) -> np.ndarray:
    """Extract and scale the RMS energy contour to roughly match the F0 range."""
    energy = librosa.feature.rms(y=audio_waveform)
    scaler = MinMaxScaler(feature_range=(0, 250))  # 0-250 to align with F0 scale
    return scaler.fit_transform(energy.T).flatten()


def interpolate_f0(f0: np.ndarray) -> np.ndarray:
    """Interpolate missing (zero) F0 values to avoid gaps in the contour."""
    f0 = np.array(f0)
    mask = f0 > 0
    if not mask.any():  # fully unvoiced/silent input -> nothing to interpolate
        return f0
    return np.interp(np.arange(len(f0)), np.where(mask)[0], f0[mask])


def clean_transcription(text: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace in a transcription."""
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z' ]+", "", text)
    return " ".join(text.split()).strip()


# =====================================================================
# Single entry point
# =====================================================================
def _append_calibration_sample(record: Dict[str, Any]) -> None:
    """Append one analysis record to the calibration sample log (best effort).

    The file feeds ``pronounce/calibrate.py``; a write failure must never break
    the analysis itself.
    """
    try:
        SAMPLES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with SAMPLES_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        logging.exception("Failed to append calibration sample:")


def analyze(user_audio: np.ndarray,
            expected_text: str,
            reference_audio: np.ndarray,
            user_sr: int = TARGET_SAMPLE_RATE,
            reference_sr: int = KOKORO_SAMPLE_RATE) -> PronunciationResult:
    """Compare a user's spoken attempt against the expected phrase.

    Args:
        user_audio: user's recorded waveform (1-D float32; from the recording path).
        expected_text: the reference phrase the user was asked to repeat.
        reference_audio: Kokoro-synthesised reference waveform for the same phrase.
        user_sr: sample rate of ``user_audio`` (recording path is 16 kHz).
        reference_sr: sample rate of ``reference_audio`` (Kokoro is 24 kHz).

    Returns:
        PronunciationResult with score, per-word errors, prosody and transcription.
    """
    _ensure_loaded()

    # Trim silent padding: user takes have button-press pauses (with the noise
    # floor boosted by peak normalization), the TTS reference has almost none.
    user_wav = _trim_silence(_prepare_waveform(user_audio, user_sr))
    reference_wav = _trim_silence(_prepare_waveform(reference_audio, reference_sr))

    # Acoustic similarity: cosine DTW between the two embedding sequences,
    # normalized by the alignment path length so it does not grow with phrase
    # duration. Cosine (vs euclidean) also ignores embedding magnitude, which
    # drifts with loudness/voice rather than pronunciation.
    emb_user = extract_embeddings(user_wav)
    emb_reference = extract_embeddings(reference_wav)
    acoustic_total, path = fastdtw(emb_user, emb_reference, dist=cosine)
    acoustic_per_step = float(acoustic_total) / max(1, len(path))
    acoustic_baseline = _random_pair_baseline(emb_user, emb_reference)
    acoustic_bad = acoustic_bad_for(acoustic_baseline)

    # Transcription + per-word phoneme comparison.
    transcription = clean_transcription(transcribe(user_wav))
    differences = compare_transcriptions(transcription, expected_text)

    phoneme_count = max(1, len(differences["expected_phonemes"]))
    phoneme_error_rate = differences["phoneme_distance"] / phoneme_count
    reference_length = max(1, differences["reference_length"])
    word_error_rate = differences["word_distance"] / reference_length

    score = compute_pronunciation_score(
        acoustic_per_step,
        phoneme_error_rate,
        word_error_rate,
        acoustic_bad=acoustic_bad,
    )

    # Calibration log: raw components in one greppable line. ``calibrate.py``
    # consumes the structured copy appended to SAMPLES_FILE below.
    logging.info(
        "[pronounce] score=%.1f | acoustic/step=%.4f (good=%.3f bad=%.3f baseline=%.4f) | "
        "phonemes=%d/%d (err=%.2f) | words_lev=%d/%d (err=%.2f) | asr=%r",
        score, acoustic_per_step, ACOUSTIC_GOOD, acoustic_bad, acoustic_baseline,
        differences["phoneme_distance"], phoneme_count, phoneme_error_rate,
        differences["word_distance"], reference_length, word_error_rate,
        transcription,
    )
    _append_calibration_sample({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "text": expected_text,
        "asr": transcription,
        "acoustic_per_step": round(acoustic_per_step, 5),
        "acoustic_baseline": round(acoustic_baseline, 5),
        "phoneme_distance": int(differences["phoneme_distance"]),
        "phoneme_count": int(phoneme_count),
        "word_distance": int(differences["word_distance"]),
        "reference_length": int(reference_length),
        "phoneme_error_rate": round(phoneme_error_rate, 4),
        "word_error_rate": round(word_error_rate, 4),
        "score": score,
    })

    # Prosody contours from the user's audio, plus the reference for overlay so
    # the GUI can show "you vs reference" pitch and energy on the same axes.
    energy = extract_energy(user_wav)
    f0 = interpolate_f0(extract_f0(user_wav, TARGET_SAMPLE_RATE))
    ref_energy = extract_energy(reference_wav)
    ref_f0 = interpolate_f0(extract_f0(reference_wav, TARGET_SAMPLE_RATE))

    return PronunciationResult(
        score=score,
        word_errors=differences["errors"],
        prosody={
            "f0": f0.tolist(), "energy": energy.tolist(),
            "ref_f0": ref_f0.tolist(), "ref_energy": ref_energy.tolist(),
        },
        transcription=transcription,
        passed=score >= SCORE_THRESHOLD,
        feedback=differences["feedback"],
        acoustic_distance=int(acoustic_total),
        acoustic_per_step=acoustic_per_step,
        acoustic_baseline=acoustic_baseline,
        words_with_errors=differences["words_with_errors"],
        expected_phonemes=differences["expected_phonemes"],
        transcribed_phonemes=differences["transcribed_phonemes"],
    )
