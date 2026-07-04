# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Static IPA phoneme -> example word table for the "WORK ON" badges.

The pronunciation (phoneme) engine reports the few reference phones a take got
worst as espeak-style IPA symbols (``result.weak_phonemes``). The hero card
renders them as ``/x/`` badges; this table backs their tooltip ("as in 'put'")
and the example word spoken on click.

Kept as plain data in ``mimora/`` (not in the view) so the mapping can evolve -
and be reused or tested - without touching UI code. The keys follow the
espeak-ng ``en-us`` inventory the engine phonemizes into; ``example_for`` also
tolerates the stress and length marks espeak attaches to a symbol, so a lookup
succeeds whether the engine hands over ``iː``, ``ˈiː`` or ``i``.
"""

from __future__ import annotations

from typing import Optional

# IPA symbol -> a short, common word that clearly contains the sound. The word
# is spoken (synthesized) on a badge click and shown in the "as in '...'"
# tooltip, so each is picked to be unambiguous and easy to hear.
PHONEME_EXAMPLES: dict[str, str] = {
    # --- Monophthong vowels ---
    "i": "see",
    "iː": "see",
    "ɪ": "sit",
    "e": "bed",
    "ɛ": "bed",
    # The engine folds its phone inventory before scoring (see _PHONE_FOLD in
    # pronunciation/phoneme/speech.py), so the badge symbol is the *folded* one:
    # æ/ɐ are emitted as "a" and oʊ/əʊ as "o". Both the pre-fold and folded keys
    # are kept, so a lookup works whichever symbol reaches here.
    "a": "cat",
    "æ": "cat",
    "ɐ": "cat",
    "ɑ": "father",
    "ɑː": "father",
    "ɒ": "hot",
    "ɔ": "thought",
    "ɔː": "thought",
    "ʊ": "put",
    "u": "food",
    "uː": "food",
    "ʌ": "cup",
    "ə": "about",
    "ɜ": "bird",
    "ɜː": "bird",
    "ɝ": "bird",
    "ɚ": "butter",
    # --- Diphthongs ---
    "eɪ": "day",
    "aɪ": "my",
    "ɔɪ": "boy",
    "aʊ": "now",
    "o": "go",
    "oʊ": "go",
    "əʊ": "go",
    "ɪə": "here",
    "eə": "hair",
    "ʊə": "tour",
    # --- Plosives ---
    "p": "pen",
    "b": "bad",
    "t": "tea",
    "d": "did",
    "k": "cat",
    "ɡ": "go",
    "g": "go",
    # --- Affricates ---
    "tʃ": "chair",
    "dʒ": "jump",
    # --- Fricatives ---
    "f": "five",
    "v": "van",
    "θ": "think",
    "ð": "this",
    "s": "see",
    "z": "zoo",
    "ʃ": "she",
    "ʒ": "vision",
    "h": "hat",
    # --- Nasals ---
    "m": "man",
    "n": "no",
    "ŋ": "sing",
    # --- Approximants ---
    "l": "leg",
    "r": "red",
    "ɹ": "red",
    "j": "yes",
    "w": "wet",
}

# Marks espeak may attach to a symbol that do not change which example applies:
# primary/secondary stress and (for the length-insensitive fallback) the length
# mark. Stripped in order by ``example_for`` until a key matches.
_STRESS_MARKS = "ˈˌ"
_LENGTH_MARK = "ː"


def example_for(phoneme: str) -> Optional[str]:
    """Example word for an IPA ``phoneme``, or ``None`` when unknown.

    Tries the symbol as given, then with stress marks removed, then also without
    the length mark - so ``ˈiː`` and ``i`` both resolve to the same example. An
    unknown symbol returns ``None`` (the caller shows the badge without an
    example rather than inventing one).
    """
    if not phoneme:
        return None
    candidate = phoneme.strip()
    if candidate in PHONEME_EXAMPLES:
        return PHONEME_EXAMPLES[candidate]
    stripped = candidate.strip(_STRESS_MARKS)
    if stripped in PHONEME_EXAMPLES:
        return PHONEME_EXAMPLES[stripped]
    no_length = stripped.replace(_LENGTH_MARK, "")
    return PHONEME_EXAMPLES.get(no_length)


__all__ = ["PHONEME_EXAMPLES", "example_for"]
