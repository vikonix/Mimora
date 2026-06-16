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

Path and espeak setup happen on import. The one thing to call explicitly is
``setup_logging()`` (from a prototype's ``main()``), which tees all output to the
screen and appends a dated copy to ``prototype.log``.
"""

import logging
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


# Single, shared log file for all prototypes; lives next to the scripts and is
# only ever appended to, so successive runs accumulate for later analysis.
LOG_FILE = Path(__file__).resolve().parent / "prototype.log"


def setup_logging() -> logging.Logger:
    """Tee all output to the screen *and* append a dated copy to ``prototype.log``.

    Call once at the start of a prototype's ``main()``. Idempotent: a prototype may
    import a sibling that also calls this, so repeated calls must not stack
    duplicate handlers (which would double every line). The file is opened in
    append mode and never truncated; each line there is timestamped for easy
    analysis, while the console stays clean (message only). Handlers are attached
    to the root logger so library logs (e.g. ``pronounce.analyze``) are captured
    alongside the prototype's own ``logging.info`` output.
    """
    root = logging.getLogger()
    if getattr(setup_logging, "_configured", False):
        return root
    root.setLevel(logging.INFO)

    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))

    root.addHandler(file_handler)
    root.addHandler(console)
    setup_logging._configured = True
    return root
