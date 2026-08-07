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
# Still a release candidate, and the number keeps climbing because PyPI accepts
# a filename once and for all: every re-upload costs one. rc1 is the one that
# hangs on a missing spaCy model.
#
# rc5 exists because rc4 could not download anything through the first-run
# window at all. The progress stand-in handed to huggingface_hub had no
# class-level get_lock, and snapshot_download passes tqdm_class straight to
# tqdm's thread_map, which asks the CLASS for its lock before it builds a
# single bar - so every hub component of every plan died there
# (mimora/first_run_download.py, make_tqdm_class). That it survived four
# candidates is the part worth remembering: the machines that tested them all
# had the models already, because a maintainer's tree is seeded by install.py,
# which passes no stand-in and keeps tqdm's own bars. A first run can only be
# tested against an empty cache, and MIMORA_HOME pointing at an empty directory
# makes one out of any machine (see mimora/paths.py).
#
# rc5 also stops the default installation reaching the network at all: NLLB is
# required for offline mode only when a translation language is selected, and
# turning translation on restarts into that same first-run window instead of
# downloading 2.5 GB silently on a worker thread.
#
# Note what a pre-release does NOT buy here. The resolver rule is "pre-releases
# are not selected WHEN A STABLE RELEASE EXISTS", and none does, so plain
# `pip install mimora` and `uv tool install mimora` do take this. It stops being
# true the moment 1.1.0 ships; until then an rc that turns out wrong can be
# yanked, which leaves it installable by exact pin but out of the resolver's
# reach.
__version__ = "1.1.0rc5"
