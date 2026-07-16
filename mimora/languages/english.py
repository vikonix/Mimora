# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""English practice-language profile (variants: american / british).

Pure data consumed by mimora/config.py (assembled into LANGUAGE_PROFILES).
See the module docstring in mimora/languages/__init__.py and the profile-shape
comment above the LANGUAGE_PROFILES assembly in config.py.
"""

PROFILE = {
    "display_name": "English",
    "flores_code": "eng_Latn",
    "default_variant": "american",
    "engines": ("phoneme", "acoustic", "none"),
    "practice_text_file": "texts/practice_text.txt",
    # Phrase-generation prompts (mimora/llm.py). The instructions stay in
    # English - they steer the model - while the TARGET language of the
    # generated phrase is named inside each string, so a new language ships
    # its own prompts here rather than a code branch. "system" is the full
    # sentence prompt, "fragment_system" the 2-4 word fragment prompt;
    # "full_ask"/"fragment_ask" are the per-request user asks. "system" is a
    # str.format template: {min_words}/{max_words} come from the active
    # proficiency level (see "levels" below), so the length instruction never
    # contradicts the level hint appended after it.
    "phrase_gen": {
        "system": (
            "You generate short English sentences for pronunciation practice. "
            "Reply with exactly ONE natural spoken sentence of {min_words} to {max_words} words, "
            "easy to read aloud. "
            "Use only plain words and a single final period - no quotation marks, no numbering, "
            "no extra commentary. Output ONLY the sentence itself, with nothing before or after it: "
            "do not add a lead-in such as 'Here's a sentence' or 'Sure', and never put a colon before "
            "the sentence. Base the sentence on the topic and vocabulary of the text the user provides."
        ),
        "fragment_system": (
            "You generate very short English fragments for pronunciation practice. "
            "Reply with exactly ONE natural fragment of 2 to 4 words that is NOT a complete "
            "sentence - such as a sentence opening, a prepositional phrase, or the start of a "
            "question. Examples: give me; on the table; where are you from. "
            "Use only plain lowercase words - no final period, no quotation marks, no numbering, "
            "no extra commentary. Output ONLY the fragment itself, with nothing before or after it: "
            "do not add a lead-in such as 'Here's a fragment' or 'Sure', and never put a colon before "
            "the fragment. Base the fragment on the topic and vocabulary of the text the user provides."
        ),
        "full_ask": (
            "Give me ONE short English sentence to practice pronunciation, "
            "based on this text."
        ),
        "fragment_ask": (
            "Give me ONE short English fragment of 2 to 4 words (NOT a complete "
            "sentence) to practice pronunciation, based on this text."
        ),
        # Proficiency levels 0..5 (~pre-A1..C1), selected by settings.json
        # "phrase_gen_level" (config.PHRASE_GEN_LEVEL). Each entry:
        #   vocab_hint   - vocabulary constraint appended to the system prompt
        #                  (full sentences AND fragments);
        #   grammar_hint - grammar constraint appended for full sentences only
        #                  (fragments are not sentences, so tense hints would
        #                  only confuse the model);
        #   words        - (min, max) word count for full sentences; fills the
        #                  {min_words}/{max_words} placeholders in "system"
        #                  and bounds the post-generation validator;
        #   min_zipf     - wordfreq Zipf floor for the validator and the
        #                  focus-word filter (None = no vocabulary floor).
        # Concrete wording on purpose: a small model follows explicit
        # constraints far better than abstract labels like "CEFR B1".
        # Starting values - tune from logs/phrase_level_samples.jsonl
        # (see tasks/phrase_level_task.md).
        "levels": (
            {
                "vocab_hint": ("Use only the simplest everyday English words "
                               "that a complete beginner knows."),
                "grammar_hint": ("Use the present tense only, with a simple "
                                 "subject-verb structure."),
                "words": (3, 5),
                "min_zipf": 4.8,
            },
            {
                "vocab_hint": ("Use only very common everyday words a "
                               "beginner knows."),
                "grammar_hint": ("Use the simple present tense or a simple "
                                 "command."),
                "words": (3, 6),
                "min_zipf": 4.5,
            },
            {
                "vocab_hint": ("Use common everyday vocabulary an elementary "
                               "learner knows."),
                "grammar_hint": ("Simple present, simple past or 'going to' "
                                 "future are all fine."),
                "words": (4, 7),
                "min_zipf": 4.0,
            },
            {
                "vocab_hint": "Use ordinary everyday vocabulary.",
                "grammar_hint": ("Any common tense is fine; keep the "
                                 "structure simple."),
                "words": (4, 8),
                "min_zipf": 3.7,
            },
            {
                "vocab_hint": ("You may use some less common words and "
                               "phrasal verbs."),
                "grammar_hint": ("Varied structures are welcome, including "
                                 "conditionals and comparisons."),
                "words": (5, 9),
                "min_zipf": 3.3,
            },
            {
                "vocab_hint": ("Use rich natural vocabulary, including "
                               "idioms and less common words."),
                "grammar_hint": ("Any natural structure is fine, including "
                                 "complex sentences."),
                "words": (5, 10),
                "min_zipf": None,
            },
        ),
    },
    # Spoken by the voice-preview button (settings_window.py); short and
    # phonetically varied, in the language being practiced.
    "preview_phrase": "Hello! This is how I sound. Let's practice together.",
    # Throwaway text to prime the NLLB translator's source tokenizer
    # (mimora/translator.py warm_up); any short phrase in the source language.
    "translator_warmup": "Hello.",
    # Throwaway word spoken by the TTS warm-up pass (mimora/tts.py warm_up):
    # a short in-vocabulary word of the practiced language, so the dummy
    # synthesis raises no out-of-vocabulary warnings.
    "tts_warmup": "Hi.",
    # Startup greeting spoken once the app is ready (main.py
    # _greet_and_start), in the practiced language. The named form carries
    # a {name} placeholder; the anonymous form is used when no user name is
    # set (a "{name}"-less template avoids a dangling "Hola, !").
    "greeting_named": "Hello {name}, listen and repeat.",
    "greeting_anonymous": "Hello, listen and repeat.",
    # Shown in the source panel when the practice-text file cannot be read
    # (main.py _load_practice_text), in the practiced language. The button
    # name stays English - the UI language is English by design.
    "practice_text_fallback": (
        "Hello and welcome to Mimora. Edit this text and click "
        "New phrase to begin."
    ),
    "variants": {
        "american": {
            "kokoro_lang_code": "a",
            "espeak_language": "en-us",
            "default_voice": "af_heart",
            "voices": [
                "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
                "am_adam", "am_michael", "am_echo", "am_eric", "am_liam",
            ],
        },
        "british": {
            "kokoro_lang_code": "b",
            "espeak_language": "en-gb",
            "default_voice": "bf_emma",
            "voices": [
                "bf_emma", "bf_alice", "bf_isabella", "bf_lily",
                "bm_george", "bm_daniel", "bm_fable", "bm_lewis",
            ],
        },
    },
}
