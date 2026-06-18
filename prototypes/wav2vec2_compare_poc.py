"""Prototype: run the *existing* Wav2Vec2 core and (optionally) compare it
side-by-side with the lighter Allosaurus pipeline on the same recording.

Why this exists
---------------
``allosaurus_pronounce_poc.py`` proposes a lighter mechanism (espeak -> Allosaurus
-> phoneme edit distance). Before adopting it we want to see how it scores relative to the
production core in ``pronounce/`` (Wav2Vec2 embeddings + cosine-DTW + CTC ASR).
This script runs that real core on a recording and, with ``--compare``, prints
both scores together.

Key difference between the two mechanisms (worth keeping in mind)
-----------------------------------------------------------------
* ``pronounce.analyze`` compares the user's audio against a **reference audio**
  of the same phrase (normally Kokoro TTS). So it needs *two* recordings.
* The Allosaurus pipeline only needs the **text**: the reference phonemes come
  from espeak. So it needs *one* recording.

That asymmetry is exactly why the Allosaurus route is attractive for adding
languages cheaply — there is no reference-audio/TTS dependency per language.

Compare in English first (this matters for fairness)
----------------------------------------------------
The ``pronounce`` core is strongest and *calibrated* on English: the shipped
model is English (``facebook/wav2vec2-large-960h``, espeak ``en-us``) and the
acoustic floor is tuned against an English TTS voice. So the honest experiment is
to run **both** pipelines on the same *English* recordings, where the Wav2Vec2
core is a trusted reference, and check whether the Allosaurus pipeline reproduces
its good/bad verdicts. Only once the lighter route tracks that baseline does it
make sense to trust it on Spanish. Hence ``--lang`` and the defaults below are
English; switch to Spanish (``--model``/``--espeak``/``--lang es``,
``--device cuda`` on a CUDA GPU) only after the English check looks good.

Status: throwaway prototype, mirrors allosaurus_pronounce_poc.py. Not GUI-wired.

Run
---
    # Step 1 — fair baseline on the bundled English sample (records/), no args:
    python prototypes/wav2vec2_compare_poc.py --compare

    # Step 2 — only after step 1 looks good: move the core to Spanish.
    python prototypes/wav2vec2_compare_poc.py user.wav reference.wav \\
        --text "hola, ¿cómo estás?" --lang es \\
        --model facebook/wav2vec2-large-xlsr-53-spanish --espeak es \\
        --device cuda --compare
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional, Tuple

import numpy as np
import soundfile as sf  # already a project dependency

# Side-effect import: adds the project root to sys.path and exposes PROJECT_ROOT.
import _bootstrap

import pronounce
from pronounce import AnalyzerConfig


# Default sample data so the prototype runs with no arguments: a normalized user
# take, the model (reference) audio, and the phrase that was read — all in records/.
RECORDS_DIR = _bootstrap.PROJECT_ROOT / "records"
DEFAULT_AUDIO = RECORDS_DIR / "normalized.wav"
DEFAULT_REFERENCE = RECORDS_DIR / "model.wav"
DEFAULT_PHRASE_FILE = RECORDS_DIR / "phrase.txt"


def _load_mono(path: str) -> Tuple[np.ndarray, int]:
    """Load ``path`` as float32 mono and its native sample rate.

    soundfile preserves the original rate; ``pronounce`` resamples internally,
    so we just hand it the true rate instead of guessing.
    """
    data, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim == 2:  # stereo/multichannel -> average to mono
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), sample_rate


def _maybe_configure(
    model: Optional[str], espeak: Optional[str], device: str
) -> None:
    """Override the pronounce defaults only for the fields the user supplied.

    Built from the current defaults so unspecified fields keep their values; a
    frozen ``AnalyzerConfig`` is replaced wholesale via ``configure``.
    """
    if model is None and espeak is None and device == "cpu":
        return  # nothing to change; keep library defaults

    current = pronounce.get_config()
    pronounce.configure(
        AnalyzerConfig(
            model_name=model or current.model_name,
            device=device,
            espeak_language=espeak or current.espeak_language,
            score_threshold=current.score_threshold,
            acoustic_good=current.acoustic_good,
            log_dir=current.log_dir,
            user_name=current.user_name,
        )
    )


def run_wav2vec2(
    user_path: str, reference_path: str, text: str
) -> "pronounce.PronunciationResult":
    """Run the production Wav2Vec2 + DTW core on one (user, reference) pair."""
    user_audio, user_sr = _load_mono(user_path)
    reference_audio, reference_sr = _load_mono(reference_path)
    return pronounce.analyze(
        user_audio=user_audio,
        expected_text=text,
        reference_audio=reference_audio,
        user_sr=user_sr,
        reference_sr=reference_sr,
    )


def _print_wav2vec2(result: "pronounce.PronunciationResult") -> None:
    logging.info("=" * 60)  # visually separate runs in the appended log
    logging.info("=== Wav2Vec2 core (pronounce.analyze) ===")
    logging.info("score          : %s / 100  (passed=%s)", result.score, result.passed)
    logging.info("transcription  : %r", result.transcription)
    logging.info("acoustic/step  : %.4f (baseline=%.4f)",
                 result.acoustic_per_step, result.acoustic_baseline)
    if result.words_with_errors:
        logging.info("words w/ errors: %s", ", ".join(result.words_with_errors))
    logging.info("feedback       : %s", result.feedback)
    logging.info("=" * 60)  # visually separate runs in the appended log
    logging.info("")


def _print_light_pipeline(audio: str, text: str, lang: str, asr: str, device: str) -> None:
    """Run the sibling light pipeline (espeak -> phoneme ASR -> edit distance).

    Imported lazily so the Wav2Vec2-only path never loads it. The sibling module
    sits in this same folder (on sys.path when run as a script).
    """
    import w2v2_pronounce_poc as poc

    spec = poc.LANGUAGES[lang]
    reference_words = poc.reference_word_phonemes(text, spec.espeak)
    reference = [phone for word in reference_words for phone in word]
    spoken = poc.spoken_phonemes(audio, spec, backend=asr, device=device)
    result = poc.align_and_score(reference, spoken, reference_words)

    logging.info("=== Light pipeline (espeak -> %s -> edit distance) ===", asr)
    logging.info("score          : %s / 100  (phonemes %s, words %.0f%%)",
                 result.score, result.phoneme_score, result.word_recall * 100)
    logging.info("reference IPA  : %s", " ".join(reference))
    logging.info("spoken IPA     : %s", " ".join(spoken))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio",
        nargs="?",
        default=str(DEFAULT_AUDIO),
        help=f"user's recording, wav (default: {DEFAULT_AUDIO.name} in records/)",
    )
    parser.add_argument(
        "reference",
        nargs="?",
        default=str(DEFAULT_REFERENCE),
        help=f"reference recording of the same phrase, wav (default: {DEFAULT_REFERENCE.name})",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="the phrase being read (default: contents of records/phrase.txt)",
    )
    parser.add_argument("--model", default=None, help="override Wav2Vec2 model name")
    parser.add_argument("--espeak", default=None, help="override espeak voice (e.g. es)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--lang",
        default="en",
        help="language key for the --compare light-pipeline run (default: en, the fair baseline)",
    )
    parser.add_argument(
        "--asr",
        default="w2v2",
        help="recognizer backend for the --compare light pipeline (w2v2 or allosaurus)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="also run the light pipeline (espeak -> phoneme ASR -> edit distance)",
    )
    args = parser.parse_args()

    # Tee all output below to the screen and append a dated copy to prototype.log.
    _bootstrap.setup_logging()

    # Fall back to the phrase that ships with the sample recording.
    text = args.text if args.text is not None else DEFAULT_PHRASE_FILE.read_text(
        encoding="utf-8"
    ).strip()

    _maybe_configure(args.model, args.espeak, args.device)
    _print_wav2vec2(run_wav2vec2(args.audio, args.reference, text))

    if args.compare:
        _print_light_pipeline(args.audio, text, args.lang, args.asr, args.device)


if __name__ == "__main__":
    main()
