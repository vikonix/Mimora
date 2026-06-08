"""Standalone microphone capture for EchoLoop — isolated from the GUI/model app.

Purpose: debug raw microphone capture on its own. There is deliberately no
normalization, no resampling and no model code here, so what you save is exactly
what the device delivered. Any clicks or dropouts in the saved file therefore
come from the capture path alone (device + host API + buffer settings).
"""
from __future__ import annotations

import wave
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import sounddevice as sd


@dataclass
class CaptureConfig:
    """All the knobs that affect capture quality, in one place."""

    device: Optional[int] = None            # None -> PortAudio default input device
    samplerate: Optional[int] = None        # None -> the device's native rate
    channels: int = 1
    blocksize: int = 0                       # 0 -> PortAudio picks an optimal size
    latency: Union[str, float] = "high"      # "low" / "high" or seconds
    dtype: str = "float32"


def list_input_devices() -> None:
    """Print every input-capable device with its host API and native sample rate.

    Use this first: the click/dropout problem is often a bad default endpoint
    (e.g. a 16 kHz "communications" or Bluetooth hands-free device). Pick the
    real physical microphone (usually 44100 or 48000 Hz) and test it explicitly.
    """
    hostapis = sd.query_hostapis()
    default_in = sd.default.device[0]
    print(f"{'idx':>3}  {'ch':>2}  {'native SR':>9}  host API / name")
    print("-" * 64)
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] < 1:
            continue
        api = hostapis[dev["hostapi"]]["name"]
        marker = " <- default" if idx == default_in else ""
        print(f"{idx:>3}  {dev['max_input_channels']:>2}  "
              f"{int(round(dev['default_samplerate'])):>9}  "
              f"{api} / {dev['name']}{marker}")


def _resolve_samplerate(cfg: CaptureConfig) -> int:
    if cfg.samplerate is not None:
        return int(cfg.samplerate)
    info = sd.query_devices(cfg.device, "input")
    return int(round(info["default_samplerate"]))


class Recorder:
    """Push-to-stop recorder: ``start()``, then ``stop()`` returns the waveform.

    The PortAudio callback runs on a realtime thread and must never block, so it
    only appends each block to a list (``list.append`` is atomic under the GIL).
    The buffer is read in ``stop()`` after the stream is closed, so the callback
    is no longer running and no lock is needed.
    """

    def __init__(self, cfg: CaptureConfig):
        self.cfg = cfg
        self.samplerate = _resolve_samplerate(cfg)
        self._chunks: List[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self.status_warnings: List[str] = []

    def _callback(self, indata, frames, time_info, status):
        if status:
            self.status_warnings.append(str(status))
        self._chunks.append(indata.copy())

    def start(self) -> None:
        self._chunks = []
        self.status_warnings = []
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.cfg.channels,
            dtype=self.cfg.dtype,
            blocksize=self.cfg.blocksize,
            latency=self.cfg.latency,
            device=self.cfg.device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._chunks, axis=0).flatten().astype(np.float32, copy=False)


def save_wav(path: str, audio: np.ndarray, samplerate: int) -> None:
    """Write a mono waveform to a 16-bit PCM WAV file, without normalization."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(samplerate)
        wav_file.writeframes(pcm.tobytes())


def dropout_report(audio: np.ndarray, samplerate: int) -> str:
    """Summarize likely capture problems: exact-zero gaps, clipping, signal level.

    Exact-zero runs are the tell-tale of capture dropouts: a real microphone
    signal is never digitally zero, so any run of zeros is silence inserted by
    the driver during an underrun — and each gap edge is an audible click.
    """
    if audio.size == 0:
        return "empty recording"

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    clip_pct = float(np.mean(np.abs(audio) >= 0.999) * 100)

    is_zero = audio == 0.0
    zero_pct = float(is_zero.mean() * 100)

    # Longest run of consecutive exact zeros (the widest dropout gap).
    longest = run = 0
    for v in is_zero:
        run = run + 1 if v else 0
        longest = max(longest, run)

    return (f"dur={len(audio) / samplerate:.2f}s  peak={peak:.3f}  rms={rms:.4f}  "
            f"clip={clip_pct:.2f}%  zero-samples={zero_pct:.2f}%  "
            f"longest-gap={longest} samples ({longest / samplerate * 1000:.1f} ms)")
