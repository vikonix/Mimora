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

import sounddevice as sd

from mimora import config

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False

# Kokoro synthesizes native 24 kHz audio output. Lives here (not in tts.py)
# because it also sizes the winsound lead-in below and is imported by callers
# that only care about the device rate, not the synthesizer.
KOKORO_SAMPLE_RATE = 24_000

# Silence padding prepended to every winsound playback block.
# Windows Audio Session needs ~50-200ms to initialize on the first call,
# which clips the very beginning of the first sentence without this buffer.
WINSOUND_LEAD_IN_SECONDS = 0.15  # 150ms of silence


def reset_portaudio():
    """Fully reinitialize PortAudio before opening a stream.

    Heals HDMI/NVIDIA device disconnects caused by CUDA power-state transitions
    on Windows. sd._terminate/_initialize are *not* public sounddevice API
    (present in 0.4.x/0.5.x); if an upgrade removes them this degrades to a
    logged no-op and the reset can simply be dropped. Call only while holding
    config.AUDIO_LOCK and with no other stream open — the reset invalidates
    every existing PortAudio stream in the process.
    """
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
