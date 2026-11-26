# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Open a folder in the file manager of the operating system.

Tk-free and stdlib-only, so the settings window and the tests import it
without any widget machinery.
"""

import os
import subprocess
import sys
from pathlib import Path


def open_folder(path: Path) -> None:
    """Show *path* in Explorer, Finder or the Linux file manager.

    The folder is created first: a missing folder is not an error the user
    can do anything about, and the file manager would only report it.
    Returns as soon as the file manager is started, without waiting for it.
    Raises OSError when no file manager can be started. On Linux a failure
    inside xdg-open after it has started is not seen here.
    """
    path.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        # The desktop-neutral way to open a folder on Linux. Without
        # xdg-utils, Popen raises FileNotFoundError, which is an OSError.
        subprocess.Popen(["xdg-open", str(path)])
