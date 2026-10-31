"""Generate pronunciation/phoneme/phone_features.json from panphon.

The phoneme engine needs one articulatory feature vector for each phone token
(speech._phone_vector). This script gets the vectors from panphon once, on the
computer of a developer, so that the runtime does not need panphon
(prototypes-pronunciation/tasks/replace-panphon.md).

The inventory of tokens is every token that the scorer can get:

1. Every token of the recognizer vocabulary (vocab.json of the wav2vec2
   phoneme model, from the local Hugging Face cache).
2. The espeak-ng output for each espeak language of the language profiles, on
   the most frequent words of the language (wordfreq) and on the words of the
   practice text and the fallback text of the profile.

All tokens go through the same _tokenize_ipa and _normalize_phones as in the
runtime. The script imports them from speech.py and does not copy them, so a
change of the normalization also changes the inventory.

For each token the file keeps exactly what panphon gives to the runtime now:
the first vector of word_to_vector_list(token, numeric=True), as a string with
one character for each feature ("+" for 1, "-" for -1, "0" for 0), or null
when panphon gives no vector.

Requirements: the environment of the clone, tools/requirements-dev.txt, and
the recognizer in the model cache (the first run of Mimora downloads it). Run
it from any directory, in UTF-8 mode: panphon reads its data files without an
encoding, which fails with the cp1252 default of Windows.

    python -X utf8 tools/gen_phone_features.py

The output is deterministic: the same input gives the same file. Only the
"generated" date in "_meta" changes from day to day.
"""

import hashlib
import json
import locale
import os
import sys
from datetime import date
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# The code of the clone, also when the script is started from another
# directory or the package is not installed in editable mode.
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_FILE = REPO_ROOT / "pronunciation" / "phoneme" / "phone_features.json"

# Number of most frequent words of each language that go through espeak-ng.
# Decided 2026-10-31 (replace-panphon.md, 4.1). Rare phone combinations that
# this misses get the cost 1.0 and a warning in main.log.
WORDS_PER_LANGUAGE = 20_000

_SIGN_OF_VALUE = {1: "+", -1: "-", 0: "0"}


def _require_utf8_mode() -> None:
    """Stop with a clear message when panphon cannot read its data files."""
    if sys.flags.utf8_mode or locale.getencoding().lower() in ("utf-8", "utf8"):
        return
    sys.exit("Run this script in UTF-8 mode: python -X utf8 tools/gen_phone_features.py")


def _point_hf_hub_at_model_cache() -> None:
    """Make huggingface_hub read the same cache as Mimora.

    Must run before the first import of huggingface_hub, which reads HF_HOME
    at import. An HF_HOME that is already set wins, as in the application.
    """
    from mimora import paths

    os.environ.setdefault("HF_HOME", str(paths.model_cache_dir()))


def _recognizer_vocabulary():
    """Return (model id, sha256 of vocab.json, sorted vocabulary tokens)."""
    from huggingface_hub import hf_hub_download

    from mimora import models_info

    repo_id = models_info.WAV2VEC2_PHONEME.repo_id
    try:
        vocab_path = Path(hf_hub_download(repo_id, "vocab.json", local_files_only=True))
    except Exception as exc:
        sys.exit(f"vocab.json of {repo_id} is not in the model cache ({exc}). "
                 "Start Mimora once to download the recognizer.")
    raw = vocab_path.read_bytes()
    vocabulary = json.loads(raw.decode("utf-8"))
    return repo_id, hashlib.sha256(raw).hexdigest(), sorted(vocabulary)


def _tokens_from_vocabulary(vocabulary):
    """Normalized tokens that the recognizer side can produce.

    The decoded transcript separates tokens with spaces, so a token normally
    reaches the scorer whole. A transcript of one token has no space, and then
    _tokenize_ipa splits it into characters, so both forms are added. Special
    tokens (<pad>, <unk>, "|") are kept too: the table must give for them the
    same answer as panphon, whatever that answer is.
    """
    from pronunciation.phoneme import speech

    tokens = set()
    for vocab_token in vocabulary:
        tokens.update(speech._normalize_phones([vocab_token]))
        tokens.update(speech._normalize_phones(speech._tokenize_ipa(vocab_token)))
    return tokens


def _words_of_language(profile, wordfreq_language):
    """Frequent words of the language plus the words of the profile texts."""
    import wordfreq

    from mimora import config

    words = list(wordfreq.top_n_list(wordfreq_language, WORDS_PER_LANGUAGE))
    practice_file = config.SHIPPED_DIR / profile["practice_text_file"]
    words.extend(practice_file.read_text(encoding="utf-8").split())
    words.extend(profile["practice_text_fallback"].split())
    return words


def _tokens_from_espeak():
    """Return (sorted espeak languages, normalized tokens of their output)."""
    from mimora import config
    from pronunciation.phoneme import speech

    espeak_languages = set()
    tokens = set()
    for profile in config.LANGUAGE_PROFILES.values():
        for variant in profile["variants"].values():
            espeak_language = variant["espeak_language"]
            if espeak_language in espeak_languages:
                continue
            espeak_languages.add(espeak_language)
            # The same rule as config.PHRASE_GEN_WORDFREQ_LANG.
            wordfreq_language = espeak_language.split("-")[0]
            words = _words_of_language(profile, wordfreq_language)
            print(f"espeak {espeak_language}: {len(words)} words", flush=True)
            # The same call as for a reference phrase, so the tokens are
            # exactly the ones that the scorer gets.
            for group in speech.reference_word_phonemes(" ".join(words), espeak_language):
                tokens.update(group)
    return sorted(espeak_languages), tokens


def _espeak_version() -> str:
    """Version of the espeak-ng library that phonemizer uses."""
    from phonemizer.backend import EspeakBackend

    from pronunciation.common.espeak import ensure_espeak

    ensure_espeak()
    return ".".join(str(part) for part in EspeakBackend.version())


def _feature_strings(tokens):
    """Return (feature names, {token: feature string or None})."""
    import panphon

    table = panphon.FeatureTable()
    names = list(table.names)
    strings = {}
    for token in sorted(tokens):
        vectors = table.word_to_vector_list(token, numeric=True)
        if not vectors:
            strings[token] = None
            continue
        vector = vectors[0]
        if len(vector) != len(names):
            sys.exit(f"panphon gave {len(vector)} values for {token!r}, "
                     f"but has {len(names)} feature names")
        strings[token] = "".join(_SIGN_OF_VALUE[value] for value in vector)
    return names, strings


def main() -> int:
    _require_utf8_mode()
    _point_hf_hub_at_model_cache()

    repo_id, vocab_sha256, vocabulary = _recognizer_vocabulary()
    recognizer_tokens = _tokens_from_vocabulary(vocabulary)
    espeak_languages, espeak_tokens = _tokens_from_espeak()
    names, strings = _feature_strings(recognizer_tokens | espeak_tokens)

    document = {
        "_meta": {
            "description": ("Articulatory feature vectors of phone tokens for the "
                            "phoneme engine (pronunciation/phoneme/speech.py)."),
            "source": "panphon (MIT license), https://github.com/dmort27/panphon",
            "generator": "tools/gen_phone_features.py",
            "generated": date.today().isoformat(),
            "panphon": metadata.version("panphon"),
            "espeakng_loader": metadata.version("espeakng-loader"),
            "espeak_ng": _espeak_version(),
            "recognizer": repo_id,
            "recognizer_vocab_sha256": vocab_sha256,
            "espeak_languages": espeak_languages,
            "words_per_language": WORDS_PER_LANGUAGE,
            "encoding": "one character per feature: '+' = 1, '-' = -1, '0' = 0",
        },
        "features": names,
        # The raw vocabulary, so that a test can normalize it with the current
        # code and find a token that the table does not cover.
        "recognizer_vocab": vocabulary,
        "phones": strings,
    }
    text = json.dumps(document, ensure_ascii=False, indent=1) + "\n"
    OUTPUT_FILE.write_text(text, encoding="utf-8", newline="\n")

    unknown = sum(1 for value in strings.values() if value is None)
    print(f"{len(strings)} tokens ({len(recognizer_tokens)} from the recognizer, "
          f"{len(espeak_tokens)} from espeak), {unknown} without a vector")
    print(f"Written: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
