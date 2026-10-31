"""Record the alignment results of the phoneme engine for the golden test.

The articulatory feature vectors move from panphon to a generated table
(prototypes-pronunciation/tasks/replace-panphon.md). The scores must stay the
same to the bit. This script runs on the code BEFORE that change, with
panphon, and writes the results that tests/test_phoneme.py compares with
after the change.

For fixed phrases in English and Spanish it makes the reference phones with
espeak-ng, makes "spoken" variants with substitutions, deletions and
insertions by fixed rules, and records the alignment pairs and the distance
of speech._edit_alignment. The alignment depends only on the substitution
cost, which is the part that changes. The final score also depends on the
calibration of the language, so it is not recorded here.

The cases keep the phone sequences, so the test needs no espeak-ng and no
audio. Requirements: the environment of the clone and
tools/requirements-dev.txt (panphon). Run it from any directory:

    python tools/capture_golden_scores.py

Output: tests/data/phoneme_alignment_golden.json. The file is the reference
that the test trusts, so the script does not replace an existing file. To
record it again on purpose (for example after a change of the alignment
code), add --force.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_FILE = REPO_ROOT / "tests" / "data" / "phoneme_alignment_golden.json"

# (espeak language, phrase). One English phrase is phonemized by both English
# variants, so the two transcriptions are covered.
PHRASES = [
    ("en-us", "The quick brown fox jumps over the lazy dog."),
    ("en-gb", "The quick brown fox jumps over the lazy dog."),
    ("en-us", "She sells seashells by the seashore on Thursday."),
    ("en-us", "Thirty three thousand people think that this is through."),
    ("es", "El perro de San Roque no tiene rabo."),
    ("es", "Me gustaría reservar una mesa para dos personas."),
]

# A Spanish phrase phonemized as English: a strong accent, with many
# substitutions between phones of two languages.
ACCENT_CASES = [
    ("es", "en-us", "Me gustaría reservar una mesa para dos personas."),
]

INSERTED_PHONE = "ə"


def _substituted(phones):
    """Replace every fourth phone (from the second) with the phone after it."""
    result = list(phones)
    for index in range(1, len(result) - 1, 4):
        result[index] = phones[index + 1]
    return result


def _deleted(phones):
    """Drop every fifth phone (from the third)."""
    return [phone for index, phone in enumerate(phones) if index % 5 != 2]


def _inserted(phones):
    """Add a schwa after every fourth phone."""
    result = []
    for index, phone in enumerate(phones):
        result.append(phone)
        if index % 4 == 3:
            result.append(INSERTED_PHONE)
    return result


def _mixed(phones):
    """All three changes together."""
    return _inserted(_deleted(_substituted(phones)))


VARIANTS = [
    ("exact", list),
    ("substituted", _substituted),
    ("deleted", _deleted),
    ("inserted", _inserted),
    ("mixed", _mixed),
]


def _case(name, reference, spoken):
    """One golden case: the input and what the alignment gives for it."""
    from pronunciation.phoneme import speech

    pairs, distance = speech._edit_alignment(reference, spoken)
    return {
        "name": name,
        "reference": reference,
        "spoken": spoken,
        "pairs": [list(pair) for pair in pairs],
        # json writes a float as its shortest exact repr, so the value reads
        # back to the same bits.
        "distance": distance,
    }


def main() -> int:
    if OUTPUT_FILE.exists() and "--force" not in sys.argv[1:]:
        sys.exit(f"{OUTPUT_FILE} exists. Add --force to replace it.")

    from pronunciation.phoneme import speech

    # The reference must come from the panphon cost. After the change the
    # runtime reads the table, and recording from it would compare the table
    # with itself.
    if not hasattr(speech, "_feature_table"):
        sys.exit("speech.py no longer uses panphon. Record the golden file from "
                 "the code before the change.")
    try:
        import panphon  # noqa: F401
    except ImportError:
        sys.exit("panphon is not installed: pip install -r tools/requirements-dev.txt")

    cases = []
    for espeak_language, phrase in PHRASES:
        reference = speech.reference_phonemes(phrase, espeak_language)
        for variant_name, change in VARIANTS:
            name = f"{espeak_language} | {phrase} | {variant_name}"
            cases.append(_case(name, reference, change(reference)))
    for reference_language, spoken_language, phrase in ACCENT_CASES:
        reference = speech.reference_phonemes(phrase, reference_language)
        spoken = speech.reference_phonemes(phrase, spoken_language)
        name = f"{reference_language} read as {spoken_language} | {phrase}"
        cases.append(_case(name, reference, spoken))

    document = {
        "_meta": {
            "description": ("Alignment results of speech._edit_alignment, recorded "
                            "with panphon before the move to phone_features.json."),
            "generator": "tools/capture_golden_scores.py",
        },
        "cases": cases,
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, ensure_ascii=False, indent=1) + "\n"
    OUTPUT_FILE.write_text(text, encoding="utf-8", newline="\n")
    print(f"{len(cases)} cases written: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
