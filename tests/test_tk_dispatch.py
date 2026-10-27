# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Tests for mimora/tk_dispatch.py.

The dispatcher is tested with a stand-in for the Tk root that records the
timers instead of running them, so no window opens. The bug it exists for
(a root.after from a worker thread is lost under Tcl 9) needs the Python
that uv installs and cannot be reproduced here; the last test class guards
against new worker code calling root.after again.
"""

import ast
import threading
import unittest
from pathlib import Path

from mimora import tk_dispatch
from mimora.tk_dispatch import MainThreadDispatcher

_PACKAGE = Path(tk_dispatch.__file__).resolve().parent


class FakeRoot:
    """Records after() timers; run_timers() plays the role of the Tk loop."""

    def __init__(self):
        self.timers = []

    def after(self, delay_ms, func, *args):
        self.timers.append((delay_ms, func, args))

    def run_timers(self):
        due, self.timers = self.timers, []
        for _delay_ms, func, args in due:
            func(*args)


class DispatcherTests(unittest.TestCase):
    def setUp(self):
        self.root = FakeRoot()
        self.dispatcher = MainThreadDispatcher(self.root)

    def test_construction_starts_the_drain_timer(self):
        self.assertEqual(len(self.root.timers), 1)
        self.assertEqual(self.root.timers[0][0], tk_dispatch.POLL_MS)

    def test_call_does_not_run_on_the_calling_thread(self):
        ran = []
        self.dispatcher.call(ran.append, 1)
        self.assertEqual(ran, [])

    def test_calls_from_a_worker_run_in_order_on_the_drain(self):
        ran = []

        def worker():
            for number in range(5):
                self.dispatcher.call(ran.append, number)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        self.root.run_timers()

        self.assertEqual(ran, [0, 1, 2, 3, 4])

    def test_drain_schedules_the_next_drain(self):
        self.root.run_timers()
        self.assertEqual(len(self.root.timers), 1)

    def test_a_failing_call_does_not_stop_the_others_or_the_timer(self):
        ran = []

        def fail():
            raise ValueError("broken")

        self.dispatcher.call(fail)
        self.dispatcher.call(ran.append, "after the failure")
        with self.assertLogs(level="ERROR"):
            self.root.run_timers()

        self.assertEqual(ran, ["after the failure"])
        self.assertEqual(len(self.root.timers), 1)


def _after_calls(node):
    """Line numbers of every ``<something>.after(...)`` call inside node."""
    return [call.lineno for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "after"]


class NoAfterFromWorkersTests(unittest.TestCase):
    # Methods of PronunciationTrainerGUI that run on a background thread,
    # whole or in part. Every UI update from them goes through
    # self.main_thread.call.
    WORKER_METHODS = {
        "load_components", "_greet_and_start", "_preview_voice_worker",
        "_generate_and_prompt", "_translate_into_panel", "_finalize_recording",
        "_on_record_max_duration", "_on_record_silence_stop",
        "_on_record_level", "_on_record_stream_error", "_run_reference_test",
        "analyze_recording",
    }

    def test_worker_methods_of_app_do_not_call_after(self):
        tree = ast.parse((_PACKAGE / "app.py").read_text(encoding="utf-8"))
        found = {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.FunctionDef)
                    and node.name in self.WORKER_METHODS):
                found[node.name] = _after_calls(node)

        self.assertEqual(set(found), self.WORKER_METHODS,
                         "a listed worker method was renamed or removed")
        self.assertEqual({name: lines for name, lines in found.items() if lines},
                         {})

    def test_playback_does_not_call_after(self):
        # Its play paths run on worker threads.
        tree = ast.parse((_PACKAGE / "playback.py").read_text(encoding="utf-8"))
        self.assertEqual(_after_calls(tree), [])


if __name__ == "__main__":
    unittest.main()
