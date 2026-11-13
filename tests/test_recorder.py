# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for the capture rate choice in mimora/recorder.py.

sounddevice is replaced with a mock inside the module, so no audio device is
opened. Importing mimora.recorder still imports the real sounddevice (through
mimora.audio_io), so PortAudio must be present the way the app needs it.

Run: python -m unittest tests.test_recorder
"""

import unittest
from unittest import mock

from mimora import config, recorder


def fake_sd(accepts_configured_rate=True, native_rate=44100.0, hostapis=()):
    """A sounddevice stand-in for capture_rate() and _select_capture_device()."""
    sd = mock.Mock()
    if not accepts_configured_rate:
        sd.check_input_settings.side_effect = Exception("Invalid sample rate")
    if native_rate is None:
        sd.query_devices.side_effect = Exception("no such device")
    else:
        sd.query_devices.return_value = {"default_samplerate": native_rate}
    sd.query_hostapis.return_value = list(hostapis)
    return sd


def make_recorder():
    return recorder.AudioRecorder(on_max_duration=lambda: None,
                                  on_stream_error=lambda: None,
                                  on_silence_stop=lambda: None,
                                  on_level=lambda level: None)


class CaptureRateTests(unittest.TestCase):
    """The configured rate when the device takes it, the native one when not."""

    def test_a_device_that_accepts_the_configured_rate_keeps_it(self):
        # Every machine that records today must stay on the same path.
        sd = fake_sd(accepts_configured_rate=True)
        with mock.patch.object(recorder, "sd", sd):
            self.assertEqual(recorder.capture_rate("hw:1,0"), config.AUDIO_SAMPLE_RATE)
        sd.query_devices.assert_not_called()

    def test_a_device_that_refuses_it_is_opened_at_its_native_rate(self):
        # The case of GitHub issue #3: a USB microphone on ALSA hw: refuses
        # 16 kHz. 44100 and not 48000, so that a fixed second choice fails.
        sd = fake_sd(accepts_configured_rate=False, native_rate=44100.0)
        with mock.patch.object(recorder, "sd", sd):
            self.assertEqual(recorder.capture_rate("hw:1,0"), 44100)
        sd.query_devices.assert_called_once_with("hw:1,0", "input")

    def test_an_unreadable_native_rate_falls_back_to_the_configured_one(self):
        # The stream then reports the error itself, as before the change.
        sd = fake_sd(accepts_configured_rate=False, native_rate=None)
        with mock.patch.object(recorder, "sd", sd), \
                self.assertLogs(level="ERROR"):
            self.assertEqual(recorder.capture_rate(None), config.AUDIO_SAMPLE_RATE)


class SelectCaptureDeviceTests(unittest.TestCase):
    """Both non-WASAPI branches take their rate from capture_rate()."""

    def test_an_explicit_device_gets_its_native_rate_when_16_khz_is_refused(self):
        sd = fake_sd(accepts_configured_rate=False, native_rate=48000.0)
        with mock.patch.object(recorder, "sd", sd), \
                mock.patch.object(config, "AUDIO_INPUT_DEVICE", "hw:1,0"):
            self.assertEqual(make_recorder()._select_capture_device(), ("hw:1,0", 48000))

    def test_without_wasapi_the_default_device_gets_its_native_rate(self):
        sd = fake_sd(accepts_configured_rate=False, native_rate=44100.0,
                     hostapis=[{"name": "ALSA", "default_input_device": 0}])
        with mock.patch.object(recorder, "sd", sd), \
                mock.patch.object(config, "AUDIO_INPUT_DEVICE", None):
            self.assertEqual(make_recorder()._select_capture_device(), (None, 44100))
        sd.query_devices.assert_called_once_with(None, "input")

    def test_wasapi_keeps_the_native_rate_of_its_default_device(self):
        # The Windows path is unchanged: no check of the configured rate.
        sd = fake_sd(native_rate=48000.0,
                     hostapis=[{"name": "Windows WASAPI", "default_input_device": 3}])
        with mock.patch.object(recorder, "sd", sd), \
                mock.patch.object(config, "AUDIO_INPUT_DEVICE", None):
            self.assertEqual(make_recorder()._select_capture_device(), (3, 48000))
        sd.check_input_settings.assert_not_called()


if __name__ == "__main__":
    unittest.main()
