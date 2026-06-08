"""Simple CLI to debug EchoLoop microphone capture in isolation.

Examples (run from the project root, with the venv active):
    python -m audio_capture.mic_test --list
    python -m audio_capture.mic_test
    python -m audio_capture.mic_test --device 5 --rate 48000
    python -m audio_capture.mic_test --device 5 --rate 48000 --latency low

Flow:
    1. Run the command.
    2. Press Enter, then say your phrase.
    3. Press Enter again to stop.
The take is saved as a WAV next to this file and a dropout report is printed,
so you can immediately tell whether a given device/config captures cleanly.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

# Allow both "python -m audio_capture.mic_test" and "python mic_test.py".
try:
    from .recorder import (CaptureConfig, Recorder, list_input_devices,
                           save_wav, dropout_report)
except ImportError:  # run directly from inside the directory
    from recorder import (CaptureConfig, Recorder, list_input_devices,
                          save_wav, dropout_report)


def _parse_latency(value: str):
    """Latency may be the words 'low'/'high' or a number of seconds."""
    try:
        return float(value)
    except ValueError:
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description="EchoLoop microphone capture test")
    parser.add_argument("--list", action="store_true",
                        help="list input devices and exit")
    parser.add_argument("--device", type=int, default=None,
                        help="input device index (see --list); default = system default")
    parser.add_argument("--rate", type=int, default=None,
                        help="sample rate in Hz; default = the device's native rate")
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--blocksize", type=int, default=0,
                        help="frames per block; 0 lets PortAudio choose")
    parser.add_argument("--latency", type=_parse_latency, default="high",
                        help='"low", "high" or a number of seconds')
    parser.add_argument("--out", type=str, default=None,
                        help="output WAV filename")
    args = parser.parse_args()

    if args.list:
        list_input_devices()
        return

    cfg = CaptureConfig(device=args.device, samplerate=args.rate,
                        channels=args.channels, blocksize=args.blocksize,
                        latency=args.latency)
    recorder = Recorder(cfg)

    device_label = args.device if args.device is not None else "default"
    print(f"Device: {device_label}   rate={recorder.samplerate} Hz   "
          f"blocksize={cfg.blocksize}   latency={cfg.latency}")

    input("Press Enter, then say your phrase... ")
    recorder.start()
    print("Recording — press Enter again to stop.")
    input()
    audio = recorder.stop()

    out_name = args.out or f"mic_test_{time.strftime('%Y%m%d_%H%M%S')}.wav"
    out_path = Path(out_name)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_name
    save_wav(str(out_path), audio, recorder.samplerate)

    print("Saved :", out_path)
    print("Report:", dropout_report(audio, recorder.samplerate))
    if recorder.status_warnings:
        unique = ", ".join(sorted(set(recorder.status_warnings)))
        print(f"PortAudio status warnings ({len(recorder.status_warnings)}): {unique}")
    else:
        print("PortAudio status warnings: none")


if __name__ == "__main__":
    main()
