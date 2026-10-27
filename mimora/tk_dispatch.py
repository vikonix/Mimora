# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Hand work from background threads to the Tk main thread.

Background threads must not call into Tk at all, ``root.after`` included.
With Tcl 8.6 tkinter forwards such a call to the main thread. Python 3.12
decides whether to forward by reading ``tcl_platform(threaded)``, which Tcl 9
no longer defines (Tcl 9 is always threaded), so with Tcl 9 the call runs on
the calling thread. The timer it creates then belongs to that thread and its
callback never runs: the window stays on "Starting" while the log says the
models are loaded. The Python that uv installs ships Tcl 9.

So a worker only puts the call in a thread-safe queue, and the main thread
runs what is queued on a timer of its own.
"""

import logging
import queue
from typing import Any, Callable

# How often the main thread runs the queued calls. Short, because the
# talking mouth is started through this queue just before the audio it
# follows; an empty queue costs next to nothing.
POLL_MS = 15


class MainThreadDispatcher:
    """Runs calls queued by any thread on the Tk main thread, in order."""

    def __init__(self, root) -> None:
        """Start the drain timer. Must be called on the Tk main thread."""
        self._root = root
        self._calls: "queue.SimpleQueue[tuple[Callable[..., Any], tuple]]" = (
            queue.SimpleQueue())
        self._root.after(POLL_MS, self._drain)

    def call(self, func: Callable[..., Any], *args: Any) -> None:
        """Run ``func(*args)`` on the Tk main thread. Safe from any thread."""
        self._calls.put((func, args))

    def _drain(self) -> None:
        """Run every queued call, then schedule the next drain. (Tk thread.)"""
        try:
            while True:
                try:
                    func, args = self._calls.get_nowait()
                except queue.Empty:
                    break
                # One failing call must not stop the others or the timer:
                # without the timer every later update is lost and the window
                # stops changing without any message.
                try:
                    func(*args)
                except Exception:
                    logging.exception("UI call %r failed.", func)
        finally:
            self._root.after(POLL_MS, self._drain)
