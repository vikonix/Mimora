# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Spanish practice-language profile (variant: castilian).

Pure data consumed by mimora/config.py (assembled into LANGUAGE_PROFILES).
See the module docstring in mimora/languages/__init__.py and the profile-shape
comment above the LANGUAGE_PROFILES assembly in config.py.
"""

PROFILE = {
    "display_name": "Spanish",
    "flores_code": "spa_Latn",
    # Only Peninsular (Castilian) Spanish is defined; a Latin-American
    # variant (es-419) would be a second entry under "variants".
    "default_variant": "castilian",
    # The acoustic engine is English-only ASR, so Spanish omits it: only the
    # text-only phoneme engine and the no-op none engine are offered. The
    # phoneme engine runs experimental until a Spanish model calibration
    # (es_model_calibration.json) is committed - see engine_experimental().
    "engines": ("phoneme", "none"),
    "practice_text_file": "texts/practice_text_es.txt",
    # Phrase-generation prompts: the instructions stay in English (they steer
    # the model), only the TARGET language named inside each string is
    # Spanish - mirroring the english entry above.
    "phrase_gen": {
        "system": (
            "You generate short Spanish sentences for pronunciation practice. "
            "Reply with exactly ONE natural spoken sentence of 4 to 8 words, easy to read aloud. "
            "Use only plain words and a single final period - no quotation marks, no numbering, "
            "no extra commentary. Output ONLY the sentence itself, with nothing before or after it: "
            "do not add a lead-in such as 'Here's a sentence' or 'Sure', and never put a colon before "
            "the sentence. Write in natural Peninsular (Castilian) Spanish, using the appropriate "
            "accents (á, é, í, ó, ú, ñ) and inverted opening marks (¿ ¡) where they belong. "
            "Base the sentence on the topic and vocabulary of the text the user provides."
        ),
        "fragment_system": (
            "You generate very short Spanish fragments for pronunciation practice. "
            "Reply with exactly ONE natural fragment of 2 to 4 words that is NOT a complete "
            "sentence - such as a sentence opening, a prepositional phrase, or the start of a "
            "question. Examples: dame eso; sobre la mesa; de dónde eres. "
            "Use only plain lowercase words - no final period, no quotation marks, no numbering, "
            "no extra commentary. Output ONLY the fragment itself, with nothing before or after it: "
            "do not add a lead-in such as 'Here's a fragment' or 'Sure', and never put a colon before "
            "the fragment. Write in natural Peninsular (Castilian) Spanish, using the appropriate "
            "accents (á, é, í, ó, ú, ñ) where they belong. "
            "Base the fragment on the topic and vocabulary of the text the user provides."
        ),
        "full_ask": (
            "Give me ONE short Spanish sentence to practice pronunciation, "
            "based on this text."
        ),
        "fragment_ask": (
            "Give me ONE short Spanish fragment of 2 to 4 words (NOT a complete "
            "sentence) to practice pronunciation, based on this text."
        ),
    },
    "preview_phrase": "¡Hola! Así es como sueno. Vamos a practicar juntos.",
    "translator_warmup": "Hola.",
    # Throwaway word spoken by the TTS warm-up pass (mimora/tts.py warm_up),
    # in the practiced language.
    "tts_warmup": "Hola.",
    "greeting_named": "¡Hola, {name}! Escucha y repite.",
    "greeting_anonymous": "¡Hola! Escucha y repite.",
    # The button name stays English - the UI language is English by design.
    "practice_text_fallback": (
        "Hola y bienvenido a Mimora. Edita este texto y pulsa "
        "New phrase para empezar."
    ),
    "variants": {
        # Spanish runs the Supertonic 3 backend (mimora/tts.py): Kokoro's
        # Spanish is trained on little data (audible artifacts, 3 voices),
        # while Supertonic is multilingual by design - 10 clean voices at
        # 44.1 kHz (decision of 2026-07-14, see
        # tasks/supertonic_tts_backend_task.md). The swap is safe for scoring:
        # Spanish uses the phoneme engine, whose reference is espeak text -
        # the synthesized audio is only played to the user.
        "castilian": {
            "tts_backend": "supertonic",
            "tts_lang_code": "es",
            "espeak_language": "es",
            "default_voice": "F1",
            "voices": ["F1", "F2", "F3", "F4", "F5",
                       "M1", "M2", "M3", "M4", "M5"],
            # Supertonic quality/speed knob (5..12); 8 matched the listening
            # tests. See config.TTS_TOTAL_STEPS.
            "total_steps": 8,
        },
    },
}
