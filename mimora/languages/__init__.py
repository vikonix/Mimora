# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Per-language profile data for the practice languages.

Each module in this package exposes a single ``PROFILE`` dict describing one
practice language (display name, FLORES-200 code, engines, variants, phrase
prompts, greetings, preview/warm-up text, default practice text). These are
pure data with no imports or side effects; ``mimora/config.py`` assembles them
into ``LANGUAGE_PROFILES`` and derives every per-run constant from there.

Adding a language = add a module here plus an engine calibration, then register
it in the ``LANGUAGE_PROFILES`` assembly in config.py. No ``if language == ...``
branch anywhere.
"""
