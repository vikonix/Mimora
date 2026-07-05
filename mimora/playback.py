# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Playback lifecycle: per-playback stop events and the talking-mouth track.

Extracted from the main controller so the whole "current playback" state
machine lives in one place: installing a fresh stop event per playback,
stopping or superseding a running one, and driving the FaceWidget's loudness
track in sync with the audio. The controller composes a PlaybackController
with the Tk root (for ``after``), the TrainerView facade, the TTSManager and
the app-wide shutdown event; workers receive stop events as plain arguments
and never touch the shared state directly.

Threading contract (unchanged from the original main.py code):
  * ``new_event()``, ``stop()`` and ``play_async()`` run on the Tk main
    thread only.
  * ``play_with_face()`` blocks for the playback duration and is safe to
    call from a background thread: widgets are touched only via root.after.
  * ``finished()`` must run on the Tk main thread - workers schedule it with
    ``root.after(0, playback.finished, stop_event)``.
"""

import threading

import numpy as np

from mimora.tts import loudness_envelope


class PlaybackController:
    """Per-playback stop events plus the loudness-track face animation.

    The internal "current playback" reference always points at the *current*
    playback's stop event; each new playback installs a fresh one via
    ``new_event()`` (always on the Tk main thread) and ``stop()`` sets the
    current one. The view is driven only through its facade methods
    (``face_*``, ``playing_status``, ``restore_ready_status``), marshalled
    onto the Tk thread where needed.
    """

    def __init__(self, root, view, tts_mgr, shutdown_event: threading.Event):
        self.root = root
        self.view = view
        self.tts_mgr = tts_mgr
        self.shutdown_event = shutdown_event
        # Pre-installed dummy event so stop() is safe before the first playback.
        self._current_event = threading.Event()

    # ------------------------------------------------------------------
    # Stop-event lifecycle
    # ------------------------------------------------------------------
    def new_event(self) -> threading.Event:
        """Install a fresh stop event for a new playback and return it.

        Every playback gets its own event. The previous shared event needed a
        set()-then-clear() dance: an old playback blocked inside a chunk write
        could miss the brief set() entirely and keep playing alongside the new
        one. With per-playback events, stop() sets the current playback's
        event and it stays set - nothing is ever cleared from under a
        still-running playback.

        Must be called on the Tk main thread: it replaces the shared "current
        playback" reference that stop() (also main thread) reads, so
        installing it from a worker would race the stop. Workers receive the
        event as an argument instead of creating it themselves. A stop issued
        between installing the event and the actual start of playback is not
        lost: play_array checks the event before and during playback.
        """
        event = threading.Event()
        self._current_event = event
        return event

    def stop(self):
        """Stop the current playback and rest the face. (Tk main thread.)"""
        self._current_event.set()
        self.tts_mgr.stop_playback()
        # Close the mouth at once on an interrupt; a track left running would
        # keep flapping with no sound. A superseding playback calls this before
        # starting its own track, so the order (rest -> new track) is correct.
        self.view.face_rest()

    def finished(self, stop_event: threading.Event):
        """Restore the Ready status unless this playback was stopped/superseded.

        Without the check, the worker of an interrupted playback would
        overwrite the status set by whatever replaced it (e.g. a newer
        playback's "Playing..." line). (Tk thread.)
        """
        if stop_event is self._current_event and not stop_event.is_set():
            self.view.restore_ready_status()

    # ------------------------------------------------------------------
    # Play paths
    # ------------------------------------------------------------------
    def play_async(self, waveform: np.ndarray, sample_rate: int, status: str):
        """Play a waveform in a background thread, stopping any current playback first."""
        self.stop()
        stop_event = self.new_event()
        self.view.playing_status(status)

        def _worker():
            self.play_with_face(waveform, sample_rate, stop_event)
            self.root.after(0, self.finished, stop_event)

        threading.Thread(target=_worker, daemon=True).start()

    def play_with_face(self, waveform: np.ndarray, sample_rate: int,
                       stop_event: threading.Event):
        """play_array, with the talking mouth driven from its loudness envelope.

        winsound plays the whole buffer with no per-frame callback, so the mouth
        cannot follow live amplitude on Windows. Instead the envelope is
        pre-computed (the signal is fully known up front) and the face advances
        it on its own wall-clock timer, kept in sync by matching the playback
        lead-in. Safe to call from a background thread: the widget is touched
        only via root.after. Blocks for the playback duration, like play_array.
        """
        self.root.after(0, self._start_face_track, waveform, sample_rate)
        try:
            self.tts_mgr.play_array(waveform, sample_rate, stop_event,
                                    self.shutdown_event)
        finally:
            self.root.after(0, self._rest_face_if_current, stop_event)

    # ------------------------------------------------------------------
    # Articulation face (talking mouth driven from the loudness envelope)
    # ------------------------------------------------------------------
    def _start_face_track(self, waveform: np.ndarray, sample_rate: int):
        """Build the loudness track and hand it to the face. (Tk thread.)"""
        fps = self.view.face_fps()
        if fps is None or waveform is None or getattr(waveform, "size", 0) == 0:
            return
        levels = loudness_envelope(waveform, sample_rate, fps=fps)
        # Keep the mouth shut during any playback lead-in silence (Windows
        # audio-session warm-up) so the animation lines up with the sound.
        lead_frames = int(round(self.tts_mgr.playback_lead_in_seconds() * fps))
        if lead_frames:
            levels = [0.0] * lead_frames + levels
        self.view.face_play_levels(levels, fps=fps)

    def _rest_face_if_current(self, stop_event: threading.Event):
        """Close the mouth, unless a newer playback has already taken over.

        Guards against an interrupted playback's cleanup clobbering the mouth
        track of the playback that superseded it (same reasoning as
        finished()). (Tk thread.)
        """
        if stop_event is self._current_event:
            self.view.face_rest()
