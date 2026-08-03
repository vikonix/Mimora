# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Launcher for a source checkout: ``python main.py``.

The application itself lives in ``mimora/app.py`` and the argument handling in
``mimora/cli.py``; this file is the third way into the same ``cli.main()``,
alongside the ``mimora`` console script and ``python -m mimora``.

It stays in the project root, and stays this thin, for two separate reasons.
Thin, because the module used to BE the application: as a package module named
``main`` it would have claimed that name in site-packages for every package in
the environment, which is why the code moved to ``mimora/app.py``. In the root,
because ``python main.py`` is what the README, AGENTS.md, ``install.py`` and
``run_mimora.bat`` all tell people to run, and the file is not part of the
wheel anyway - package discovery only collects ``mimora*`` and
``pronunciation*``.

Nothing may be imported here beyond the line below. Anything heavier would run
before ``cli.main()`` parses the arguments, which is the ordering the split
exists to protect.
"""

from mimora.cli import main

if __name__ == "__main__":
    main()
