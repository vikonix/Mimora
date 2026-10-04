# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Command line entry point: everything that must happen before the app loads.

This module is what ``[project.scripts]`` points at, and it exists because of
one ordering constraint. Importing ``mimora.app`` pulls in torch, transformers
and Kokoro, which takes many seconds on a slow machine, while ``--version``
should answer instantly - so the import has to happen AFTER argument parsing,
which means the parsing cannot live in the module being imported.

The same constraint applies to ``bootstrap.early_init()``: the UTF-8 console
setup and the library warning filters only take effect if they run before the
libraries are imported.

**Only the standard library may be imported at module level here** (plus
``mimora.bootstrap`` and ``mimora.splash``, which are stdlib-only themselves,
and ``mimora.__init__``, which defines nothing but ``__version__``). Anything heavier would defeat the
purpose of the module.

Three launch forms reach :func:`main`, and all three behave identically:

* ``mimora`` - the console script, which calls it directly;
* ``python -m mimora`` - via ``mimora/__main__.py``;
* ``python main.py`` - via the shim in the project root.

``--detect-hardware`` lives here for a reason of the same shape: an installed
tool's environment is not reachable with ``python -m``, so a maintenance
command that only existed as a module could not be run by the people who need
it most. The console script is the only entry an installed package puts on PATH.
"""

import argparse
import shutil
import sys

from mimora import __version__, bootstrap, splash

# The two native pieces `install.py` checks before the first launch
# (step_check_tkinter, step_check_portaudio). A wheel install runs neither
# check, so what follows is all that stands between a user and a bare
# traceback. Package names differ per distribution: Debian and Ubuntu split Tk
# off as python3-tk, Fedora calls it python3-tkinter, Arch just tk.
_INSTALL_COMMAND = {
    "apt-get": "sudo apt install",
    "dnf": "sudo dnf install",
    "pacman": "sudo pacman -S",
}
_TKINTER_PACKAGE = {"apt-get": "python3-tk", "dnf": "python3-tkinter", "pacman": "tk"}
_PORTAUDIO_PACKAGE = {"apt-get": "libportaudio2", "dnf": "portaudio", "pacman": "portaudio"}


def _package_hint(packages: dict[str, str]) -> str:
    """Return the install command for the first package manager found.

    Best effort by design: an unknown distribution gets prose instead of a
    command, which is still better than the ModuleNotFoundError it replaces.
    """
    for manager, package in packages.items():
        if shutil.which(manager):
            return f"    {_INSTALL_COMMAND[manager]} {package}"
    return "    (install it with your distribution's package manager)"


def _missing_tkinter_message() -> str:
    """Explain a failed `import tkinter` and name the package that fixes it."""
    if sys.platform == "darwin":
        # Homebrew Python omits Tcl/Tk, and its formula is per minor version.
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        hint = f"    brew install python-tk@{version}"
    else:
        hint = _package_hint(_TKINTER_PACKAGE)
    return (
        "Mimora needs tkinter, the Tk GUI toolkit.\n"
        "It belongs to the standard library, but it is packaged apart from the\n"
        "interpreter and is not on PyPI, so no installer can add it:\n"
        f"{hint}\n"
        "It lands in the interpreter's own stdlib, so nothing has to be\n"
        "reinstalled afterwards - just start Mimora again."
    )


def _missing_portaudio_message() -> str:
    """Explain a failed sounddevice import and name the package that fixes it."""
    return (
        "Mimora needs PortAudio, the native library behind recording and\n"
        "playback. The Linux wheels of sounddevice do not carry it:\n"
        f"{_package_hint(_PORTAUDIO_PACKAGE)}\n"
        "Then start Mimora again."
    )


def _native_hint_for(exc: BaseException) -> str | None:
    """Return advice for the two native pieces a wheel install cannot supply.

    None for anything else, which is what keeps a real bug's traceback intact.
    Kept apart from main() so that the mapping can be asked directly, without
    reproducing a failing import.
    """
    if (isinstance(exc, ImportError) and exc.name == "tkinter"
            and sys.platform != "win32"):
        # Windows Python installers bundle Tcl/Tk, so a missing tkinter there
        # means a broken interpreter, not a missing system package.
        return _missing_tkinter_message()
    if isinstance(exc, OSError) and "PortAudio" in str(exc):
        return _missing_portaudio_message()
    return None


def main() -> None:
    """Parse the arguments, then hand over to the application."""
    # Before the heavy imports, not after: see the module docstring.
    bootstrap.early_init()

    parser = argparse.ArgumentParser(
        prog="mimora", description="Mimora pronunciation trainer.")
    parser.add_argument(
        "--version", action="version", version=f"Mimora {__version__}")
    parser.add_argument(
        "--detect-hardware", action="store_true",
        help="probe this machine, rewrite config/hardware_config.json and "
             "exit (run it after changing the installed PyTorch build)")
    # Spelled as bootstrap.APPEND_LOG_FLAG ("--append-log"), because
    # lifecycle.spawn_replacement() has to produce the same string and one
    # constant is cheaper than two that must agree. argparse derives the
    # destination from it as usual, so args.append_log below is unaffected.
    #
    # Listed rather than suppressed although the restart is what normally
    # passes it: a hidden flag surprises the next reader of --help, and
    # continuing a log by hand across two launches is a fair use of it.
    parser.add_argument(
        bootstrap.APPEND_LOG_FLAG, action="store_true",
        help="append to logs/main.log instead of truncating it (added "
             "automatically when the app restarts itself, so one session's "
             "log survives the restart)")
    args = parser.parse_args()  # --version exits inside this call

    if args.detect_hardware:
        # Here rather than only as `python -m mimora.detect_hardware`, because
        # in an installed tool there is no interpreter that can run that: uv
        # puts this package's console scripts on PATH and nothing else, and the
        # `python` a user does have is some other environment, which would
        # rewrite some other hardware_config.json. The advice this refreshes -
        # detect_hardware.warn_if_gpu_unused - has to name a command that
        # exists where it is printed.
        #
        # Imported inside the branch for the same reason as the application
        # below: --version must not pay for anything it does not print.
        from mimora import detect_hardware
        raise SystemExit(detect_hardware.main())

    # Printed before the import rather than after: the import below is the
    # slow part, so this is the first sign of life the user gets. flush=True
    # defeats stdout buffering when the output is redirected.
    print("starting ...", flush=True)
    # The same sign of life for a launch without a console. After the two
    # branches above, which answer at once and need no window.
    splash.show()

    # Deliberately a function-local import. At module level it would run
    # before parse_args() above and make --version pay for the whole
    # application load.
    #
    # The two failures caught here are the native pieces no wheel can supply
    # and no installer checks on this path: tkinter, missing from app.py, and
    # PortAudio, missing from sounddevice under mimora.tts. Both would
    # otherwise end in a traceback that names the module but not the cure.
    # Everything else keeps its traceback, because everything else is a bug.
    try:
        from mimora import app
    except (ImportError, OSError) as exc:
        # Closed before the message below, not left to the exit handler: the
        # splash stays on top of other windows.
        splash.close()
        hint = _native_hint_for(exc)
        if hint is None:
            raise
        raise SystemExit(hint) from exc

    app.run(append_log=args.append_log)
