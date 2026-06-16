"""Prototype: language-agnostic pronunciation scoring (espeak-ng -> phoneme ASR -> edit distance).

Goal of this spike
------------------
Show the *simplest* path to add a new language (Spanish first) to Mimora without
training anything and without heavy hardware. The production ``pronounce`` core
compares Wav2Vec2 *embeddings* with DTW; here we deliberately try the lighter,
fully language-parametrized alternative:

    text --espeak-ng--> reference IPA phonemes -+
                                                +-- edit distance --> score + diff
    user audio --phoneme ASR--> spoken phonemes +

Recognizer backends (``--asr``)
-------------------------------
An English baseline comparison (see ``wav2vec2_compare_poc.py``) showed the
universal Allosaurus recognizer is too noisy to be useful (~16/100 where the
production core scores ~95). So the default backend is now a stronger one:

* ``w2v2`` (default): ``facebook/wav2vec2-xlsr-53-espeak-cv-ft``, a wav2vec2 CTC
  model that emits espeak-style IPA. More accurate, and its phone inventory
  matches the espeak reference (no inventory mismatch). Needs only project deps
  plus panphon (used for scoring; see ``requirements.txt``).
* ``allosaurus``: the original universal recognizer, kept for comparison.

Scoring uses an articulatory feature distance (panphon), so near-misses like the
rhotic ``r``/``ɹ`` cost little while unrelated swaps cost ~1 -- language-general,
so it works for Spanish unchanged.

Both keep the pipeline **text-only** -- the reference phonemes come from espeak,
so no per-phrase TTS audio is needed (unlike the production core).

Why this shape
--------------
* espeak-ng already ships with Mimora (via ``phonemizer-fork``) and covers 100+
  languages, so the grapheme->phoneme step is a one-line language switch.
* The w2v2 phoneme model runs on CPU or a small GPU; for "general feedback" a
  couple of seconds of latency on a low-end laptop is acceptable.
* Token-level edit distance (Levenshtein / Needleman-Wunsch) over the two phoneme
  sequences gives both a phoneme-error-rate score *and* a per-phoneme alignment.
  This is the standard tool for discrete symbol sequences and matches how the
  production ``pronounce`` core already compares phonemes (it uses Levenshtein,
  not DTW; DTW there is only for the continuous Wav2Vec2 embeddings).

Adding a language later == adding a row to ``LANGUAGES`` below. Nothing else.

Status: throwaway prototype. Not wired into the GUI, not optimized. It is meant
as a starting point to evaluate Allosaurus quality on real recordings before
deciding whether fine-tuning (on a CUDA GPU) is needed at all.

Tip: validate this pipeline in English first (``--lang en``) against the trusted
``pronounce`` core (see ``wav2vec2_compare_poc.py``); once it tracks that
baseline, trust it for Spanish.

Run
---
    # Install panphon (scoring); the w2v2 backend itself needs no extra packages:
    pip install -r prototypes/requirements.txt   # panphon (+ allosaurus, optional)
    # espeak-ng needs no system install: _bootstrap registers the bundled
    # espeakng_loader library, same as the main app does via Kokoro.

    # No arguments: w2v2 backend on the English sample in records/.
    python prototypes/allosaurus_pronounce_poc.py

    # On GPU, or with the Allosaurus backend, or your own audio/text:
    python prototypes/allosaurus_pronounce_poc.py --device cuda
    python prototypes/allosaurus_pronounce_poc.py --asr allosaurus
    python prototypes/allosaurus_pronounce_poc.py path/to/user.wav --text "..." --lang es
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

# Side-effect import: adds the project root to sys.path and exposes PROJECT_ROOT.
import _bootstrap

from phonemizer import phonemize  # provided by phonemizer-fork (project dep)
from phonemizer.separator import Separator


# Default sample data so the prototype runs with no arguments. records/ holds a
# normalized user take, the model (reference) audio and the phrase that was read.
RECORDS_DIR = _bootstrap.PROJECT_ROOT / "records"
DEFAULT_AUDIO = RECORDS_DIR / "normalized.wav"
DEFAULT_PHRASE_FILE = RECORDS_DIR / "phrase.txt"


# ---------------------------------------------------------------------------
# Language table. This is the ONLY place that needs a new entry per language.
#   espeak : voice id passed to espeak-ng (grapheme -> phoneme).
#   allosaurus : ISO 639-3 lang_id passed to Allosaurus (audio -> phoneme).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LanguageSpec:
    espeak: str
    allosaurus: str


LANGUAGES = {
    "es": LanguageSpec(espeak="es", allosaurus="spa"),          # European Spanish
    "es-419": LanguageSpec(espeak="es-419", allosaurus="spa"),  # Latin-American
    "en": LanguageSpec(espeak="en-us", allosaurus="eng"),       # for parity checks
}


# ---------------------------------------------------------------------------
# Step 1 -- reference phonemes from text (espeak-ng via phonemizer).
# ---------------------------------------------------------------------------
def reference_phonemes(text: str, espeak_lang: str) -> List[str]:
    """Phonemize the target ``text`` into a flat list of IPA phoneme symbols.

    Crucially we ask espeak for a **per-phone** separator. Its default only
    separates *words* by spaces (phones within a word are glued together), which
    would leave the reference at a coarser granularity than Allosaurus's
    per-phone output and make the alignment meaningless.
    """
    ipa = phonemize(
        text,
        language=espeak_lang,
        backend="espeak",
        strip=True,
        with_stress=False,
        preserve_punctuation=False,
        # Separate every phone with a space; drop word boundaries (newline, so it
        # never clashes with the phone space). We score at the phoneme level only.
        separator=Separator(phone=" ", word="\n", syllable=""),
    )
    return _normalize_phones(_tokenize_ipa(ipa))


# ---------------------------------------------------------------------------
# Step 2 -- spoken phonemes from audio. Two interchangeable recognizer backends:
#   "w2v2"       : facebook/wav2vec2-xlsr-53-espeak-cv-ft -- a wav2vec2 CTC model
#                  that emits espeak-style IPA. Much more accurate than Allosaurus
#                  here, and its phone inventory matches the espeak reference, so
#                  there is no inventory mismatch. Uses only project deps
#                  (transformers / torch / librosa). Recommended default.
#   "allosaurus" : universal phone recognizer (needs `pip install allosaurus`).
#                  Kept for comparison; in practice too noisy for English.
# Both keep the pipeline text-only: no per-phrase TTS reference is needed.
# ---------------------------------------------------------------------------
RECOGNIZERS = ("w2v2", "allosaurus")
W2V2_PHONEME_MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"


def spoken_phonemes(
    wav_path: str, spec: LanguageSpec, backend: str = "w2v2", device: str = "cpu"
) -> List[str]:
    """Recognize the phonemes the speaker actually produced, via ``backend``."""
    if backend == "w2v2":
        raw = _recognize_w2v2(wav_path, device)
    elif backend == "allosaurus":
        raw = _recognize_allosaurus(wav_path, spec.allosaurus)
    else:
        raise ValueError(f"unknown recognizer backend: {backend!r}")
    return _normalize_phones(_tokenize_ipa(raw))


@lru_cache(maxsize=2)
def _w2v2_model(device: str):
    """Load the wav2vec2 phoneme model once per device (heavy: ~1.2 GB first run)."""
    from transformers import AutoModelForCTC, AutoProcessor

    processor = AutoProcessor.from_pretrained(W2V2_PHONEME_MODEL)
    model = AutoModelForCTC.from_pretrained(W2V2_PHONEME_MODEL).to(device).eval()
    return processor, model


def _recognize_w2v2(wav_path: str, device: str) -> str:
    """Greedy CTC decode -> space-separated espeak/IPA phonemes."""
    import librosa
    import torch

    processor, model = _w2v2_model(device)
    # The model needs 16 kHz mono; librosa resamples whatever the file is.
    audio, _ = librosa.load(wav_path, sr=16_000, mono=True)
    inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    predicted_ids = logits.argmax(dim=-1)
    return processor.batch_decode(predicted_ids)[0]


def _recognize_allosaurus(wav_path: str, lang_id: str) -> str:
    """Universal phone recognizer; allosaurus is imported lazily, only when used."""
    return _allosaurus_model().recognize(wav_path, lang_id=lang_id)


@lru_cache(maxsize=1)
def _allosaurus_model():
    """Load the Allosaurus recognizer once (cached on disk after first download)."""
    from allosaurus.app import read_recognizer

    return read_recognizer()


# ---------------------------------------------------------------------------
# Step 3 -- align the two phoneme sequences and derive a coarse score.
# ---------------------------------------------------------------------------
@dataclass
class ScoreResult:
    score: float                       # 0-100, higher is better
    pairs: List[Tuple[str, str]]       # aligned (reference, spoken); "" == gap


def align_and_score(reference: List[str], spoken: List[str]) -> ScoreResult:
    """Align ``spoken`` against ``reference`` by edit distance and score it.

    The score is ``1 - feature_error_rate``, where the rate is the weighted edit
    distance divided by the reference length, mapped to 0-100.

    Substitutions cost the **articulatory feature distance** (0..1) between the two
    phones via panphon, so a notational/near-miss like the rhotic ``r``/``ɹ`` or a
    rounded/unrounded vowel costs little, while an unrelated swap costs ~1.
    Insertions and deletions cost a full 1 (a phone wholly missing or extra). This
    is language-independent, so it works for Spanish unchanged.
    """
    if not reference:
        raise ValueError("empty reference phoneme sequence")

    pairs, distance = _edit_alignment(reference, spoken)

    # Feature error rate relative to the reference length, mapped to 0-100.
    accuracy = 1.0 - (distance / len(reference))
    score = round(max(0.0, accuracy) * 100.0, 1)
    return ScoreResult(score=score, pairs=pairs)


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------
def _tokenize_ipa(ipa: str) -> List[str]:
    """Split an IPA string into phone symbols.

    Both espeak (with the per-phone separator above) and Allosaurus separate
    phones with whitespace; when they don't, we fall back to one Unicode
    character per phone, a reasonable approximation for a prototype.
    """
    ipa = ipa.strip()
    if " " in ipa or "\n" in ipa:
        return [tok for tok in ipa.split() if tok]
    return [ch for ch in ipa if not ch.isspace()]


# Suprasegmental diacritics that one recognizer marks and the other does not
# (espeak vs Allosaurus). Stripping them removes spurious mismatches such as an
# aspirated vs plain stop, or a long vs short vowel. Written as \u escapes so no
# bare combining marks appear in the source. Code points: 02D0 long, 02D1
# half-long, 02B0 aspiration, 02C0 glottalization, 02C8/02CC primary/secondary
# stress, 0329/030D syllabic, 032F non-syllabic, 0361/035C tie bars. This is a
# coarse prototype heuristic; it does NOT reconcile genuine inventory differences
# (e.g. a diphthong written as one token vs two), which a feature-based distance
# (panphon) would handle later.
_DIACRITIC_CODEPOINTS = (
    0x02D0, 0x02D1, 0x02B0, 0x02C0, 0x02C8, 0x02CC,
    0x0329, 0x030D, 0x032F, 0x0361, 0x035C,
)
_STRIP_DIACRITICS = dict.fromkeys(_DIACRITIC_CODEPOINTS)


def _normalize_phones(tokens: List[str]) -> List[str]:
    """Drop suprasegmental diacritics so the two inventories line up better."""
    cleaned = (tok.translate(_STRIP_DIACRITICS) for tok in tokens)
    return [tok for tok in cleaned if tok]


# ---------------------------------------------------------------------------
# Articulatory feature distance (panphon). Imported lazily and cached so the
# module imports without panphon and the feature table is built only once.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _feature_table():
    """Build the panphon feature table once.

    panphon opens its bundled UTF-8 data files without specifying an encoding, so
    on Windows (cp1252 default) construction raises ``UnicodeDecodeError``. We
    default ``pathlib.Path.open`` to UTF-8 just for the duration of construction --
    panphon only opens its own data files here, so the scope is safe -- instead of
    re-exec'ing the interpreter (which detaches the console on Windows).
    """
    import pathlib

    import panphon

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
    """Feature distance in [0, 1] between two phones (0 == identical).

    The cost is the fraction of articulatory features that differ. Symbols panphon
    cannot parse fall back to a full mismatch (1.0).
    """
    if a == b:
        return 0.0
    va, vb = _phone_vector(a), _phone_vector(b)
    if va is None or vb is None or len(va) != len(vb):
        return 1.0
    differing = sum(1 for x, y in zip(va, vb) if x != y)
    return differing / len(va)


# Backtrace operations recorded by the DP.
_OP_SUB, _OP_DEL, _OP_INS = "sub", "del", "ins"


def _edit_alignment(
    reference: List[str], spoken: List[str]
) -> Tuple[List[Tuple[str, str]], float]:
    """Feature-weighted edit-distance alignment over phone tokens.

    Returns the aligned ``(reference, spoken)`` pairs ("" marks an inserted or
    deleted phone) and the total (fractional) distance. We roll our own DP because
    ``Levenshtein`` operates on characters, not on lists of multi-character IPA
    tokens, and we need per-phone feature substitution costs. A backpointer matrix
    is kept so the fractional costs reconstruct the path without float-equality
    comparisons during backtrace.
    """
    n, m = len(reference), len(spoken)

    # cost[i][j] = best distance between reference[:i] and spoken[:j].
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = float(i)       # delete the first i reference phones
        back[i][0] = _OP_DEL
    for j in range(1, m + 1):
        cost[0][j] = float(j)       # insert the first j spoken phones
        back[0][j] = _OP_INS
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitute = cost[i - 1][j - 1] + _substitution_cost(reference[i - 1], spoken[j - 1])
            delete = cost[i - 1][j] + 1.0    # phone expected but not spoken
            insert = cost[i][j - 1] + 1.0    # extra phone spoken
            best = min(substitute, delete, insert)
            cost[i][j] = best
            back[i][j] = _OP_SUB if best == substitute else (_OP_DEL if best == delete else _OP_INS)

    # Backtrace from the bottom-right corner using the recorded operations.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio",
        nargs="?",
        default=str(DEFAULT_AUDIO),
        help=f"user's recording, 16 kHz mono wav (default: {DEFAULT_AUDIO.name} in records/)",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="reference text the user read (default: contents of records/phrase.txt)",
    )
    parser.add_argument(
        "--lang",
        default="en",
        choices=sorted(LANGUAGES),
        help="language key (default: en, matching the sample data in records/)",
    )
    parser.add_argument(
        "--asr",
        default="w2v2",
        choices=RECOGNIZERS,
        help="recognizer backend for the spoken audio (default: w2v2)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="device for the w2v2 backend (default: cpu)",
    )
    args = parser.parse_args()

    # Fall back to the phrase that ships with the sample recording.
    text = args.text if args.text is not None else DEFAULT_PHRASE_FILE.read_text(
        encoding="utf-8"
    ).strip()

    spec = LANGUAGES[args.lang]
    reference = reference_phonemes(text, spec.espeak)
    spoken = spoken_phonemes(args.audio, spec, backend=args.asr, device=args.device)
    result = align_and_score(reference, spoken)

    print(f"language   : {args.lang}  (asr={args.asr})")
    print(f"reference  : {' '.join(reference)}")
    print(f"spoken     : {' '.join(spoken)}")
    print(f"score      : {result.score} / 100")
    print("alignment  : ref | spoken   (~ = near, * = mismatch)")
    for ref_sym, hyp_sym in result.pairs:
        if not ref_sym or not hyp_sym:
            flag = "  *"                       # insertion / deletion
        else:
            cost = _substitution_cost(ref_sym, hyp_sym)
            flag = "" if cost == 0 else ("  ~" if cost < 0.34 else "  *")
        print(f"             {ref_sym or '_':<4} | {hyp_sym or '_':<4}{flag}")


if __name__ == "__main__":
    main()
