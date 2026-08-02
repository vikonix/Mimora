# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

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
    parser.parse_args()  # --version exits inside this call

    # Printed before the import rather than after: the import below is the
    # slow part, so this is the first sign of life the user gets. flush=True
    # defeats stdout buffering when the output is redirected.
    print("starting ...", flush=True)

    # Deliberately a function-local import. At module level it would run
    # before parse_args() above and make --version pay for the whole
    # application load.
    from mimora import app

    app.run()
