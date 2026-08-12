# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Mimora application package: configuration, LLM/TTS/translation managers and UI.

Pronunciation analysis lives in the separate top-level ``pronunciation`` package
(subpackages ``acoustic`` / ``phoneme`` / ``common``, dispatched by ``mimora/engine.py``);
``main.py`` in the project root wires everything together.
"""

# Single source of truth for the application version (SemVer MAJOR.MINOR.PATCH,
# with an optional PEP 440 pre-release suffix while a release is being tested).
# pyproject.toml reads this value dynamically; runtime code imports it from here.
#
# Every re-upload costs a version number: PyPI accepts a filename once and for
# all. A pre-release is still what a plain `pip install mimora` resolves to
# while no stable 1.1.0 exists, and can be yanked if it turns out wrong.
__version__ = "1.1.0rc6"
