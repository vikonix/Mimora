# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Shared audio-device infrastructure for Mimora.

This module owns the PortAudio/winsound plumbing that the microphone path
(``recorder.py``) and the speaker path (``tts.py``) both need but neither
should own. Keeping it here lets both modules depend on this neutral layer
instead of on each other: previously ``recorder`` imported ``reset_portaudio``
straight from ``tts``, which made the capture side depend on the synthesis
side for purely device-level code.

The lock and pipeline sample rate that coordinate the two streams live in
``config`` (``AUDIO_LOCK``, ``AUDIO_SAMPLE_RATE``) alongside the rest of the
``AUDIO_*`` device settings; this module holds only the behaviour (reset,
path selection) and the constants tied to that behaviour.
"""

import logging
import sys

import sounddevice as sd

from mimora import config

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False

# NOTE: KOKORO_SAMPLE_RATE moved to mimora/tts.py (next to KokoroBackend).
# The synthesis rate is a property of the active TTS backend now - callers
# read TTSManager.sample_rate instead of a module constant (see
# tasks/supertonic_tts_backend_task.md).

# Silence padding prepended to every winsound playback block.
# Windows Audio Session needs ~50-200ms to initialize on the first call,
# which clips the very beginning of the first sentence without this buffer.
WINSOUND_LEAD_IN_SECONDS = 0.15  # 150ms of silence


# Number of PortAudio streams currently open in this process. Guarded by
# config.AUDIO_LOCK: every stream open/close and every reset_portaudio() call
# happens while holding it, so a plain int is safe (no separate lock needed).
# reset_portaudio() consults it to skip the reset while any stream is open:
# the reset invalidates every PortAudio stream in the process, so running it
# then would corrupt a live stream (e.g. recording starting while a playback
# thread is still inside stream.write on the sounddevice path).
_open_streams = 0


def stream_opened() -> None:
    """Record that a PortAudio stream was opened. Call under config.AUDIO_LOCK."""
    global _open_streams
    _open_streams += 1


def stream_closed() -> None:
    """Record that a PortAudio stream was closed. Call under config.AUDIO_LOCK."""
    global _open_streams
    _open_streams = max(0, _open_streams - 1)


def reset_portaudio():
    """Fully reinitialize PortAudio before opening a stream.

    Heals HDMI/NVIDIA device disconnects caused by CUDA power-state transitions
    on Windows. sd._terminate/_initialize are *not* public sounddevice API
    (present in 0.4.x/0.5.x); if an upgrade removes them this degrades to a
    logged no-op and the reset can simply be dropped. Call only while holding
    config.AUDIO_LOCK.

    The reset invalidates every existing PortAudio stream in the process, so it
    is skipped while any stream is still open (see stream_opened/stream_closed);
    an open stream also proves PortAudio is currently healthy, so the heal is
    not needed then.

    Windows-only: the disconnect it heals is a Windows/NVIDIA phenomenon, and
    on macOS repeated Pa_Terminate/Pa_Initialize cycles are known to leave
    CoreAudio in a state where a later stream.stop() hangs forever (record
    thread stuck after "Stopping audio recording...").
    """
    if sys.platform != "win32":
        logging.debug("PortAudio reset skipped: non-Windows platform.")
        return
    if _open_streams > 0:
        logging.debug("PortAudio reset skipped: %d stream(s) still open.",
                      _open_streams)
        return
    try:
        sd._terminate()
        sd._initialize()
    except Exception as init_err:
        logging.debug(f"PortAudio reinitialization error: {init_err}")


def uses_winsound() -> bool:
    """True when play_array will take the blocking winsound path.

    winsound can only target the default output device, so an explicit
    AUDIO_OUTPUT_DEVICE forces the sounddevice path even on Windows. Kept as a
    single source of truth so the playback path and the mouth-animation lead-in
    (see TTSManager.playback_lead_in_seconds) never disagree.
    """
    return WINSOUND_AVAILABLE and config.AUDIO_OUTPUT_DEVICE is None
