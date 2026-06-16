"""Shared path bootstrap for prototype scripts.

Prototypes live in ``prototypes/`` but often need to reuse the real project code
(``pronounce``, ``mimora``, ...). They are throwaway experiments, so we do NOT
want to turn this folder into a package or fiddle with editable installs. Instead
each prototype starts with::

    import _bootstrap  # noqa: F401  (adds the project root to sys.path)

After that, normal project imports work regardless of the current working
directory::

    from pronounce import analyze
    from pronounce.speech import phonemize  # etc.

It also exposes ``PROJECT_ROOT`` so prototypes can build default paths (e.g. to
``records/``) independent of where they are launched from::

    import _bootstrap
    default_audio = _bootstrap.PROJECT_ROOT / "records" / "normalized.wav"

Finally, it points ``phonemizer`` at the bundled espeak-ng on import (see
``_register_espeak``), so prototypes using espeak work without a system install.

All side effects happen on import; there is nothing to call.
"""

import sys
from pathlib import Path

# prototypes/_bootstrap.py -> project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    # Prepend so project packages win over any same-named site package.
    sys.path.insert(0, str(PROJECT_ROOT))


def _register_espeak() -> None:
    """Point phonemizer at the espeak-ng library bundled in the venv.

    In the full app, importing Kokoro/misaki registers the espeak-ng library as a
    side effect (see ``misaki/espeak.py``). The prototypes don't import Kokoro, so
    without this ``phonemizer`` raises "espeak not installed on your system" on
    Windows, where there is no system espeak binary on PATH. We reuse the same
    ``espeakng_loader`` wheel and registration the project already relies on.
    """
    try:
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
    except ImportError:
        # espeakng_loader absent: fall back to a system-installed espeak, if any.
        return
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())


_register_espeak()
