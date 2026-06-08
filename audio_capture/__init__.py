"""Standalone microphone capture module for EchoLoop.

Currently used by the mic_test CLI to debug capture in isolation. Once a clean
capture configuration is found, the main app can import Recorder/CaptureConfig
from here instead of duplicating the InputStream logic in main.py.
"""
from .recorder import (CaptureConfig, Recorder, list_input_devices,
                       save_wav, dropout_report)

__all__ = [
    "CaptureConfig",
    "Recorder",
    "list_input_devices",
    "save_wav",
    "dropout_report",
]
