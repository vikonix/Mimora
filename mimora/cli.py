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
``mimora.bootstrap``, which is stdlib-only itself, and ``mimora.__init__``,
which defines nothing but ``__version__``). Anything heavier would defeat the
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

from mimora import __version__, bootstrap


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

    # Deliberately a function-local import. At module level it would run
    # before parse_args() above and make --version pay for the whole
    # application load.
    from mimora import app

    app.run(append_log=args.append_log)
