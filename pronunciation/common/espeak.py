# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Point phonemizer at the bundled espeak-ng, for both pronunciation engines.

Why this is not a system dependency any more
--------------------------------------------
``phonemizer`` talks to espeak-ng through a shared library, not through the
``espeak-ng`` executable, and ``espeakng_loader`` ships that library together
with its data directory as an ordinary wheel. Registering it removes the one
step of the install that pip could not perform. A system install stays a valid
setup - it is what a standalone install of one of these subpackages without
``espeakng_loader`` falls back to - but it is no longer required.

Why the engines register it themselves
--------------------------------------
``misaki/espeak.py`` (inside Kokoro) does the same thing as an *import side
effect*, and for a long time that was the only registration in the process.
It cannot be relied on: Mimora's TTS backends import lazily, so a run that
never synthesises through Kokoro (Spanish on Supertonic) never imports misaki,
and the engines are then left with whatever the system happens to provide.
This module is that registration, done on purpose and shared, so the two
engines do not carry a copy each.

Layering
--------
``pronunciation/*`` never imports ``mimora``, so the only imports here are
``espeakng_loader`` and ``phonemizer`` - and both happen *inside* the
functions, which keeps importing this module free of third-party code (the
same constraint ``pronunciation.common.audio`` documents for librosa). A
missing ``espeakng_loader`` is a soft failure: it arrives transitively with
kokoro/misaki in the full application, but a standalone install of one
subpackage may not have it, and there a system espeak-ng is the answer.

Both values or neither
----------------------
``set_library()`` and ``set_data_path()`` are always set together. Setting only
the library looks like it works and then dies inside the C library with an
access violation rather than an exception: ``EspeakAPI`` copies the library to
a temporary directory and loads the copy, passing the data path separately, so
the data directory is never found "next to" the loaded library.

Diagnostics
-----------
The registration used to swallow every exception silently, which made a
failure indistinguishable from success: the process carried on with a system
espeak-ng if one existed, and quietly scored against a different transcription
if one did not. Hence the INFO/WARNING pair - the same idea as the "spaCy
pipeline ... resolves from" line in ``mimora/app.py``.

Also runnable, which is what ``install.py`` step 5 uses::

    python -m pronunciation.common.espeak

That form matters because the answer belongs to the *target* environment: an
installer that imported this module itself would report on the interpreter it
happens to run under.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

# The outcome of the single registration attempt: None until it has been made,
# then True/False for good, so repeat calls from the analysis path cost nothing
# and answer consistently. Registration is idempotent, so a race between two
# threads can at worst do the same assignment twice.
_bundled_registered: Optional[bool] = None


def ensure_espeak() -> bool:
    """Register the bundled espeak-ng with phonemizer once. Never raises.

    Returns True when the bundled library was registered, False when it was
    not - in which case phonemizer falls back to ``PHONEMIZER_ESPEAK_LIBRARY``
    or to a system install, either of which may still work. Idempotent: only
    the first call does anything, and later calls repeat its outcome.
    """
    global _bundled_registered
    if _bundled_registered is not None:
        return _bundled_registered

    log = logging.getLogger(__name__)
    try:
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        library = espeakng_loader.get_library_path()
        # get_data_path() raises when the directory is absent, so ask for it
        # BEFORE touching the wrapper: a half-registration (library set, data
        # path not) is the failure mode described in this module's docstring.
        data_path = espeakng_loader.get_data_path()

        EspeakWrapper.set_library(library)
        EspeakWrapper.set_data_path(data_path)
    except Exception as exc:
        # Not fatal, and not silent either: without the bundled library the
        # engines run on whatever espeak-ng the system provides, whose
        # transcription may differ from the one the scoring was calibrated
        # against - or on nothing at all, and then every word is dropped from
        # the comparison.
        log.warning(
            "Bundled espeak-ng could not be registered (%s: %s); falling back "
            "to PHONEMIZER_ESPEAK_LIBRARY or a system espeak-ng. Scores may "
            "differ from the calibrated ones, and phonemization fails "
            "outright if neither is present.",
            type(exc).__name__, exc)
        _bundled_registered = False
        return False

    # set_library() does not check the path (it is a plain assignment), so this
    # line states what phonemizer will now try to load, not that loading
    # succeeded. The real failure surfaces later, when the first backend is
    # built.
    log.info("espeak-ng resolves from %s (data %s).", library, data_path)
    _bundled_registered = True
    return True


def resolved_library() -> Optional[str]:
    """The espeak library phonemizer would use, or None if it finds none.

    Answers through phonemizer's own lookup, so it reflects all three levels of
    its precedence rule - the registration above, ``PHONEMIZER_ESPEAK_LIBRARY``
    and the system search - rather than only the first. This is the question a
    consumer of the engines actually has; the presence of an ``espeak-ng``
    *executable* on PATH is a different one, and on Windows the two disagree
    (the official installer writes ``libespeak-ng.dll``, which phonemizer's
    system search does not look for).
    """
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        return str(EspeakWrapper.library())
    except Exception:
        # RuntimeError when nothing is found, ImportError when phonemizer is
        # not installed at all. Both mean the same thing to the caller.
        return None


def main() -> int:
    """Report which espeak-ng this environment would use. Exit 0 if any.

    The output is split on purpose, because install.py reads it: **stdout is
    the resolved library path and nothing else** (empty when none was found),
    while the explanation goes to stderr. That way the caller needs no parsing
    rule, and a human running the command still sees the whole story.
    """
    bundled = ensure_espeak()
    library = resolved_library()

    print(f"Bundled espeak-ng: {'registered' if bundled else 'NOT available'}",
          file=sys.stderr)
    if library is None:
        print("phonemizer finds no espeak-ng library at all. Install "
              "espeakng-loader (pip), or a system espeak-ng and point "
              "PHONEMIZER_ESPEAK_LIBRARY and PHONEMIZER_ESPEAK_DATA_PATH at "
              "it.", file=sys.stderr)
        return 1

    print(f"Resolves to      : {library}", file=sys.stderr)
    print(library)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via install.py
    raise SystemExit(main())
