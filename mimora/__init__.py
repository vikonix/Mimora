# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Mimora application package: configuration, LLM/TTS/translation managers and UI.

Pronunciation analysis lives in the separate top-level ``pronunciation`` package
(subpackages ``acoustic`` / ``phoneme`` / ``common``, dispatched by ``mimora/engine.py``);
``main.py`` in the project root wires everything together.
"""

# Single source of truth for the application version (SemVer MAJOR.MINOR.PATCH).
# pyproject.toml reads this value dynamically; runtime code imports it from here.
__version__ = "1.0.0"
