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
# The minor bump past the 1.0.0 tagged on GitHub is what this release actually
# is: an installable package (console script, package data, dependencies in
# pyproject.toml, user data in the OS directory) with first-run downloads and
# the LLM in the official llama-server binary. The rc1 is there to hold the PyPI
# name without spending the 1.1.0 filename, which the index will never accept
# twice - pip and uv skip pre-releases unless asked, so nobody gets it by
# accident.
__version__ = "1.1.0rc1"
