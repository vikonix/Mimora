# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Per-language profile data for the practice languages.

Each module here exposes a single ``PROFILE`` dict, pure data with no imports
and no side effects; ``mimora/config.py`` assembles them into
``LANGUAGE_PROFILES`` and derives every per-run constant from there. Adding a
language = add a module here plus an engine calibration, then register it in
that assembly. No ``if language == ...`` branch anywhere.

Top-level keys:

``display_name``
    Shown in the window title and the settings window.
``flores_code``
    FLORES-200 source code for the NLLB translator, e.g. ``"eng_Latn"``.
``default_variant``
    Used when settings.json names none.
``engines``
    Pronunciation engines available for this language. A language property:
    the acoustic engine is English-only ASR, so other profiles omit it (see
    ``config.available_engines``).
``practice_text_file``
    Default source text, relative to ``paths.shipped_root()``.
``variants``
    Display key -> TTS/espeak wiring. Each names its synthesis backend
    (``tts_backend``, default ``"kokoro"``) plus that backend's language code
    and voices: Kokoro uses ``kokoro_lang_code`` and voices like ``af_heart``
    (``af_``/``bf_`` female, ``am_``/``bm_`` male; ``a`` American, ``b``
    British), Supertonic uses ``tts_lang_code`` (ISO), voices ``F1..F5`` /
    ``M1..M5`` and an optional ``total_steps`` quality knob. ``espeak_language``
    must match the variant, or pronunciation is scored against phonemes of a
    different dialect.

The rest is language-specific text rather than wiring - phrase-generation
prompts and levels, greeting, preview and warm-up phrases - and is documented
inline where it is written, in ``english.py``.
"""
