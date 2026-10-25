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
from pathlib import Path
from typing import Optional

_POLL_INTERVAL_MS = 100

# The same value as window_icon.WM_CLASS, which this module cannot import at
# module level. On Linux the splash is a window of its own on the taskbar;
# with the default class "Tk" it is not matched to the Mimora menu entry.
_WM_CLASS = "Mimora"

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


def _remove_title_bar(root) -> None:
    """Show *root* without a title bar, window buttons or frame.

    ``overrideredirect`` does this on Windows and X11 when it is set on the new
    root. On macOS (Tk 9) set that way it leaves the title bar with its
    buttons: there it takes effect only on a window that is withdrawn at that
    moment, so the root is withdrawn first and shown again after. The window
    style "plain" of ``::tk::unsupported::MacWindowStyle`` did not remove the
    title bar on that Tk either.
    """
    if root.tk.call("tk", "windowingsystem") == "aqua":
        root.withdraw()
        root.overrideredirect(True)
        root.deiconify()
        return
    root.overrideredirect(True)


def _set_icon(root) -> None:
    """Give the splash process the Mimora icon, except on Windows.

    The splash is a process of its own, so macOS shows it in the Dock and a
    Linux desktop on its taskbar; without an icon it is an empty square there.
    On Windows a window without a title bar has no taskbar button, so an icon
    would change nothing there.

    The import is here and not at the top: this file must stay stdlib-only at
    module level (see the module docstring), and a run of this file from a
    tree where ``mimora`` cannot be imported must still show the window.

    The child runs with ``-P``, so in a clone whose environment holds only the
    requirements (what install.py sets up) ``mimora`` is not importable: the
    project root is on ``sys.path`` of the application only because main.py
    sits there. It is added here, at the END of ``sys.path``: an installed
    package still wins, and nothing in the project root can take the place of
    a standard library module.
    """
    if sys.platform == "win32":
        return
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)
    try:
        from mimora import window_icon
    except ImportError:
        return
    window_icon.apply_window_icon(root)


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

    root = tk.Tk(className=_WM_CLASS)
    # Named, so that a title bar shown anyway does not read "tk".
    root.title("Mimora")
    # No title bar: the window cannot be moved or closed, so it must not look
    # as if it could.
    _remove_title_bar(root)
    root.attributes("-topmost", True)
    _set_icon(root)

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
