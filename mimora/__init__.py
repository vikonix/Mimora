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
# all. A pre-release can be yanked if it turns out wrong. While a stable
# release exists, a plain `pip install mimora` ignores pre-releases: testers
# need an exact pin or `--pre` (`--prerelease allow` for uv).
__version__ = "1.2.0rc2"
