# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Startup splash: a small window that is visible while the application loads.

Importing ``mimora.app`` pulls in torch, transformers and Kokoro, which takes
many seconds and, on a computer with little memory, minutes. The windowless
launcher (``mimora-gui``) has no console, so without this window the user sees
nothing at all between the click and the main window.

The window runs in **its own process**, and that is the design, not a detail.
The main thread of the application is busy with the import and cannot serve a
window: one created there stops answering and Windows marks it "Not
responding". A second process also leaves the startup order in ``cli.py``
untouched.

The same file is both sides: :func:`show` and :func:`close` are called by the
application, and ``python splash.py`` is the child that draws the window.
A mouse click on the window closes it, which is also how a person who started
this file by hand gets rid of it.

**Stdlib-only at module level**, and tkinter is imported only in the child:
``cli.py`` imports this module before it has parsed its arguments.
"""

from __future__ import annotations

import atexit
import subprocess
import sys
from typing import Optional

_POLL_INTERVAL_MS = 100

_process: Optional[subprocess.Popen] = None


def show() -> None:
    """Start the splash process. Does nothing when it already runs.

    A failure to start is ignored: the splash is a courtesy, and the
    application must start without it.
    """
    global _process
    if _process is not None or not sys.executable:
        return
    try:
        _process = subprocess.Popen(
            # -P keeps this directory off sys.path in the child, so that a
            # module here cannot replace a standard library module of the same
            # name.
            [sys.executable, "-P", __file__],
            # The pipe is how the child learns that this process is gone. It
            # closes on every exit, lifecycle.hard_exit() included, which
            # skips the atexit handler below.
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # Without the flag the windowless launcher opens a console window
            # for the child.
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except OSError:
        return
    atexit.register(close)


def close() -> None:
    """Close the splash. Safe to call more than once, and when none is shown."""
    global _process
    process, _process = _process, None
    if process is None:
        return
    try:
        process.stdin.close()
        process.terminate()
    except OSError:
        pass  # The child has already gone.


def _run_window() -> None:
    """The child process: show the window until the parent goes or says so."""
    import threading
    import tkinter as tk
    from tkinter import ttk

    parent_gone = threading.Event()

    def wait_for_parent() -> None:
        # read() returns only at end of file, which is the parent closing the
        # pipe or leaving.
        try:
            sys.stdin.read()
        except (OSError, ValueError):
            pass
        parent_gone.set()

    # No stdin at all (pythonw without a valid handle): only a click closes
    # the window then.
    if sys.stdin is not None:
        threading.Thread(target=wait_for_parent, daemon=True).start()

    root = tk.Tk()
    # No title bar: the window cannot be moved or closed, so it must not look
    # as if it could.
    root.overrideredirect(True)
    root.attributes("-topmost", True)

    frame = tk.Frame(root, padx=40, pady=28, highlightthickness=1,
                     highlightbackground="#888888")
    frame.pack()
    tk.Label(frame, text="Mimora", font=("Segoe UI", 20, "bold")).pack()
    tk.Label(frame, text="Starting...", font=("Segoe UI", 10)).pack(pady=(4, 14))
    progress = ttk.Progressbar(frame, mode="indeterminate", length=260)
    progress.pack()
    progress.start(12)
    # A click closes the window: it is on top of all others and has no close
    # button. Not limited to a start from a terminal, because that cannot be
    # detected reliably - an IDE gives the child a pipe, exactly as the
    # application does. bind_all, because the click lands on a label or on
    # the bar, not on the window itself.
    root.bind_all("<Button-1>", lambda event: root.destroy())

    # Centre of the screen. The size is known only after the layout is done.
    root.update_idletasks()
    left = (root.winfo_screenwidth() - root.winfo_width()) // 2
    top = (root.winfo_screenheight() - root.winfo_height()) // 2
    root.geometry(f"+{left}+{top}")

    def poll() -> None:
        if parent_gone.is_set():
            root.destroy()
        else:
            root.after(_POLL_INTERVAL_MS, poll)

    # No lifetime limit: on a computer with little memory the load takes
    # minutes, and a window that goes away before the main window exists
    # leaves the user with nothing on the screen again.
    root.after(_POLL_INTERVAL_MS, poll)
    root.mainloop()


if __name__ == "__main__":
    _run_window()
