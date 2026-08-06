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
# the LLM in the official llama-server binary.
#
# Still a release candidate, and rc2 rather than rc1 because rc1 is already on
# PyPI: the index accepts a filename once and for all, so every re-upload costs
# a number. rc1 is the one that hangs on a missing spaCy model, which is what
# this candidate fixes.
#
# Note what a pre-release does NOT buy here. The resolver rule is "pre-releases
# are not selected WHEN A STABLE RELEASE EXISTS", and none does, so plain
# `pip install mimora` and `uv tool install mimora` do take this. It stops being
# true the moment 1.1.0 ships; until then an rc that turns out wrong can be
# yanked, which leaves it installable by exact pin but out of the resolver's
# reach.
__version__ = "1.1.0rc4"
