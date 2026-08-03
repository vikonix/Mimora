# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""``python -m mimora`` support: a shim, with no logic of its own.

The entry point in pyproject.toml names ``mimora.cli:main`` rather than
anything here on purpose. Under ``python -m mimora`` this file is executed as
the module ``__main__``, so code living here would exist twice in a process
that also imported ``mimora.__main__`` under its real name - two module
objects, two copies of everything at module level. Keeping the logic in
``cli.py`` makes that impossible instead of merely unlikely.

Why keep this form at all when the console script exists: it is the way in
when the script is not reachable. The interpreter's ``Scripts`` directory is
routinely missing from PATH on Windows, and a virtual environment moved to
another directory leaves script shims pointing at an interpreter that is no
longer there. ``python -m mimora`` works in both cases.
"""

from mimora.cli import main

main()
