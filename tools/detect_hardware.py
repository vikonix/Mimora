# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Hardware detection for Mimora.

Standalone tool: probes the machine (RAM, CPU, GPU/VRAM, audio devices) and
writes config/hardware_config.json with two sections:

  "hardware" - raw facts about the machine, for diagnostics;
  "config"   - ready-to-use parameter values (EXTERNAL_N_GPU_LAYERS etc.)
               picked from the detected hardware. The main app will read
               these instead of the hard-coded defaults in config.py.

Run it manually whenever the hardware changes:

    python tools/detect_hardware.py

It only relies on packages the project already uses (torch, sounddevice) plus
the stdlib-only mimora.llama_server_fetch; each probe degrades gracefully if
its package is missing or broken, and any such problem is recorded in the
"warnings" list of the output file.
"""

import ctypes
import json
import logging
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The tool lives in tools/ but its output is a config artifact, so it is written
# to the project's config/ directory (read by mimora/config.py), not next to the
# script.
OUTPUT_FILE = PROJECT_ROOT / "config" / "hardware_config.json"

# A timestamped record of each run is kept in the project-wide logs/ directory
# (the same one config.py uses for main.log), alongside the human-friendly
# console print()s. The log file is the place to look when diagnosing why a
# given machine was detected the way it was - it captures the warnings too.
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "hwdetect.log"

logger = logging.getLogger("hwdetect")


def _setup_logging() -> None:
    """Attach a file handler writing to logs/hwdetect.log (overwritten per run).

    Kept independent of the console output: the terminal stays concise while the
    log file preserves a timestamped, complete record for later inspection.
    """
    LOG_DIR.mkdir(exist_ok=True)
    handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False

# llama-3.2-3b-instruct has 28 transformer layers; -1 below means "offload all".
MODEL_TOTAL_LAYERS = 28


# =====================================================================
# RAM / CPU
# =====================================================================

def detect_ram_gb(warnings: list) -> float | None:
    """Total physical RAM in GiB."""
    if platform.system() == "Windows":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.ullTotalPhys / 1024**3, 1)
        warnings.append("GlobalMemoryStatusEx failed; RAM size unknown")
        return None

    # Linux/macOS fallback (sysconf is absent on Windows only).
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        warnings.append("Could not determine RAM size on this platform")
        return None


def detect_cpu_features(warnings: list) -> dict:
    """SIMD CPU features relevant to llama.cpp, read from numpy (already a dep).

    Diagnostics only, but the first thing to look at when the pinned
    llama-server build dies on startup: a binary compiled for an instruction
    set the CPU lacks crashes outright (0xC000001D on Windows) instead of
    degrading, and AVX512 is the usual culprit. numpy exposes a CPUID-based
    feature map, the cleanest cross-platform source without a new dependency
    or fragile Windows API calls.
    """
    features: dict = {}
    try:
        import numpy as np
    except ImportError:
        warnings.append("numpy is not installed; CPU features unknown")
        return features
    try:
        # numpy >= 2 moved the internal module to np._core.
        try:
            umath = np._core._multiarray_umath
        except AttributeError:
            umath = np.core._multiarray_umath
        raw = umath.__cpu_features__
        for key in ("AVX", "AVX2", "AVX512F", "FMA3", "F16C"):
            features[key] = bool(raw.get(key, False))
    except Exception as exc:  # noqa: BLE001 - any probe failure is non-fatal
        warnings.append(f"CPU feature probe failed: {exc}")
    return features


# =====================================================================
# GPU
# =====================================================================

def detect_gpu(warnings: list) -> dict:
    """GPU presence, name, VRAM, and which backends can actually use it.

    Two independent consumers, probed separately:

    - llama-server runs as its own process with its own CUDA runtime, fully
      independent of torch. ``llama_gpu_offload`` reflects the installed
      build's own device probe (None when there is no install to ask -
      physical GPU presence is the fallback signal).
    - torch (used only by Wav2Vec2 in pronunciation/acoustic/) reports CUDA via
      ``torch_cuda``. A CPU-only torch build is normal here and only means
      pronunciation analysis runs on CPU; it says nothing about the LLM.

    Physical presence/name/VRAM come from nvidia-smi first (works regardless
    of torch build), then torch.cuda. Non-NVIDIA adapters (AMD/Intel) are
    listed by name only, for diagnostics.
    """
    gpu = {
        "present": False,
        "name": None,
        "vram_gb": None,
        "torch_cuda": False,
        "llama_gpu_offload": None,
        "device_count": 0,
        "all_adapters": _list_video_adapters(),
    }

    smi = _query_nvidia_smi(warnings)
    if smi:
        gpu.update(present=True, name=smi["name"], vram_gb=smi["vram_gb"],
                   device_count=1)

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu.update(
                present=True,
                name=props.name,
                vram_gb=round(props.total_memory / 1024**3, 1),
                torch_cuda=True,
                device_count=torch.cuda.device_count(),
            )
        elif gpu["present"]:
            warnings.append(
                f"'{gpu['name']}' is present but torch has no CUDA (CPU-only "
                "build) - pronunciation analysis (Wav2Vec2) will run on CPU; "
                "the LLM is unaffected, llama-server carries its own CUDA "
                "runtime"
            )
    except ImportError:
        warnings.append("torch is not installed; Wav2Vec2 device defaults to CPU")
    except Exception as exc:  # noqa: BLE001 - a broken CUDA runtime must not kill detection
        warnings.append(f"torch CUDA probe failed: {exc}")

    # Probed last, once presence is settled: the probe stays quiet on machines
    # that have no GPU at all, where "the LLM cannot use the GPU" is not news.
    gpu["llama_gpu_offload"] = _probe_llama_offload(warnings, gpu["present"])

    return gpu


def _probe_llama_offload(warnings: list, gpu_present: bool) -> bool | None:
    """Whether the installed llama-server binary can offload to the GPU.

    Asks the binary itself with `--list-devices` (loads no model, sub-second)
    and matches the answer against the device pattern of the variant it was
    installed as. That comparison is the point: a CUDA build whose cudart DLLs
    are missing or of the wrong major version starts fine, still logs
    "offloaded N/N layers to GPU", and simply runs about three times slower on
    the CPU - see tasks/llama-cpp.md, phase 0. The same probe guards every app
    start in mimora/llm_server_ctl.py (log_compute_devices).

    Only the project's own install in bin/llama/ is probed. A llama-server the
    user manages themselves carries no record of which backend it was built
    for, so there is nothing to compare its device list against; it yields None
    like an absent install does.

    Returns True/False, or None when the question cannot be answered - callers
    then fall back to physical GPU presence, which is what build_config does.
    *gpu_present* only decides whether a negative answer is worth a warning: on
    a machine without a GPU "the LLM cannot use the GPU" is not news, and
    build_config zeroes the LLM's VRAM budget on absence anyway.
    """
    # detect_hardware.py runs as a script, so sys.path starts at tools/ and the
    # package next door is not importable without this. llama_server_fetch is
    # stdlib-only and side-effect-free on import, which is why it is safe to
    # pull in from a tool that may run before the requirements are installed.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from mimora import llama_server_fetch
    except ImportError as exc:
        warnings.append(f"mimora.llama_server_fetch is not importable ({exc}); "
                        "GPU offload capability unknown")
        return None

    exe = llama_server_fetch.installed_exe()
    if exe is None:
        if gpu_present:
            warnings.append(
                "llama-server is not installed in bin/llama (run "
                "'python -m mimora.llama_server_fetch'); GPU offload "
                "capability unknown")
        return None

    variant_name = llama_server_fetch.installed_variant(exe)
    if variant_name is None:
        # installed_exe() already required a readable stamp, so the only way
        # here is a stamp naming a variant this version of the module dropped.
        warnings.append(f"{exe} was installed as a variant this build does not "
                        "know; GPU offload capability unknown")
        return None

    pattern = llama_server_fetch.VARIANTS[variant_name].device_pattern
    if pattern is None:
        # A CPU build cannot offload, full stop - the same verdict the old
        # llama-cpp-python probe returned for a CPU-only wheel.
        if gpu_present:
            warnings.append(
                f"the installed llama-server is the '{variant_name}' build - "
                "the LLM cannot use the GPU; reinstall it with "
                "'python -m mimora.llama_server_fetch --force' to pick up the "
                "CUDA build")
        return False

    try:
        devices = llama_server_fetch.list_devices(exe)
    except llama_server_fetch.LlamaServerFetchError as exc:
        warnings.append(f"llama-server --list-devices failed: {exc}")
        return None

    if re.search(pattern, devices):
        return True
    warnings.append(
        f"the '{variant_name}' llama-server build lists no matching device, so "
        f"it would silently run on the CPU (about three times slower); check "
        f"that the cudart DLLs sit next to {exe.name} and that their major "
        f"version matches the build. --list-devices said:\n{devices.strip()}")
    return False


def _query_nvidia_smi(warnings: list) -> dict | None:
    """First GPU reported by nvidia-smi, or None if the tool is absent/fails."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        name, mem_mib = out.stdout.strip().splitlines()[0].rsplit(",", 1)
        return {"name": name.strip(), "vram_gb": round(int(mem_mib) / 1024, 1)}
    except ValueError:
        warnings.append(f"Could not parse nvidia-smi output: {out.stdout!r}")
        return None


def _list_video_adapters() -> list[str]:
    """Names of all video adapters (Windows WMI); empty list elsewhere/on failure."""
    if platform.system() != "Windows":
        return []
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_VideoController).Name"],
            capture_output=True, text=True, timeout=20,
        )
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


# =====================================================================
# Audio
# =====================================================================

def detect_audio(warnings: list) -> dict:
    """Input/output audio devices as seen by sounddevice (PortAudio)."""
    audio = {"input_devices": [], "output_devices": []}
    try:
        import sounddevice as sd
    except ImportError:
        warnings.append("sounddevice is not installed; audio devices unknown")
        return audio

    try:
        default_in, default_out = sd.default.device
        for index, dev in enumerate(sd.query_devices()):
            entry = {
                "index": index,
                "name": dev["name"],
                "hostapi": sd.query_hostapis(dev["hostapi"])["name"],
                "default_samplerate": dev["default_samplerate"],
            }
            if dev["max_input_channels"] > 0:
                audio["input_devices"].append(
                    {**entry, "channels": dev["max_input_channels"],
                     "default": index == default_in}
                )
            if dev["max_output_channels"] > 0:
                audio["output_devices"].append(
                    {**entry, "channels": dev["max_output_channels"],
                     "default": index == default_out}
                )
    except Exception as exc:  # noqa: BLE001 - PortAudio errors must not kill detection
        warnings.append(f"Audio device query failed: {exc}")

    if not audio["input_devices"]:
        warnings.append("No audio input device (microphone) found")
    if not audio["output_devices"]:
        warnings.append("No audio output device (speakers) found")
    return audio


# =====================================================================
# Parameter selection
# =====================================================================

def build_config(hardware: dict) -> dict:
    """Pick concrete app parameters from the detected hardware.

    The names match the constants in config.py so the app can apply them
    directly. LLM parameters (N_GPU_LAYERS, N_CTX) follow the physical GPU and
    the installed llama-server build's own device probe - NOT torch, which is a
    separate stack used only by Wav2Vec2. Threshold rationale: the GGUF model
    weighs ~2 GB at Q4_K_M and Kokoro/Wav2Vec2 also claim VRAM when they run on
    the GPU, so full offload plus GPU-side Wav2Vec2 needs a comfortable margin.
    """
    gpu = hardware["gpu"]

    # LLM side: usable unless llama-server explicitly reported no usable device
    # (None = nothing to ask, assume a present GPU is usable).
    llm_vram = gpu["vram_gb"] or 0
    if not gpu["present"] or gpu["llama_gpu_offload"] is False:
        llm_vram = 0

    if llm_vram >= 6:
        n_gpu_layers = -1  # all MODEL_TOTAL_LAYERS layers
    elif llm_vram >= 4:
        n_gpu_layers = 20
    elif llm_vram >= 3:
        n_gpu_layers = 12
    elif llm_vram >= 2:
        n_gpu_layers = 8
    else:
        n_gpu_layers = 0

    # torch side (pronunciation/acoustic/Wav2Vec2): needs a CUDA-enabled torch build, plus
    # VRAM headroom so it does not fight the LLM for the same card.
    torch_vram = (gpu["vram_gb"] or 0) if gpu["torch_cuda"] else 0

    return {
        "DEVICE": "cuda" if gpu["torch_cuda"] else "cpu",
        "EXTERNAL_N_GPU_LAYERS": n_gpu_layers,
        "EXTERNAL_N_CTX": 4096 if llm_vram >= 8 else 2048,
        "WAV2VEC2_DEVICE": "cuda" if torch_vram >= 6 else "cpu",
        # null = system default device, which is the right choice on most
        # machines; the indices of all devices are listed under "hardware".
        "AUDIO_INPUT_DEVICE": None,
        "AUDIO_OUTPUT_DEVICE": None,
    }


# =====================================================================
# Main
# =====================================================================

def main() -> int:
    _setup_logging()
    warnings: list[str] = []

    print("Detecting hardware...")
    logger.info("Detecting hardware...")
    hardware = {
        "platform": f"{platform.system()} {platform.release()}",
        "ram_total_gb": detect_ram_gb(warnings),
        "cpu_cores": os.cpu_count(),
        "cpu_features": detect_cpu_features(warnings),
        "gpu": detect_gpu(warnings),
        "audio": detect_audio(warnings),
    }

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hardware": hardware,
        "config": build_config(hardware),
        "warnings": warnings,
    }

    OUTPUT_FILE.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    gpu = hardware["gpu"]
    ram_line = f"RAM: {hardware['ram_total_gb']} GB, CPU cores: {hardware['cpu_cores']}"
    print(f"  {ram_line}")
    logger.info(ram_line)

    feats = hardware["cpu_features"]
    if feats:
        enabled = [name for name, present in feats.items() if present]
        cpu_feat_line = "CPU features: " + (", ".join(enabled) if enabled else "none")
        print(f"  {cpu_feat_line}")
        logger.info(cpu_feat_line)

    if gpu["present"]:
        llama_state = {True: "yes", False: "NO", None: "unknown"}[gpu["llama_gpu_offload"]]
        gpu_line = (f"GPU: {gpu['name']} ({gpu['vram_gb']} GB VRAM, "
                    f"llama-server offload: {llama_state}, torch CUDA: "
                    f"{'yes' if gpu['torch_cuda'] else 'no'})")
    else:
        gpu_line = "GPU: none detected"
    print(f"  {gpu_line}")
    logger.info(gpu_line)

    audio_line = (f"Audio: {len(hardware['audio']['input_devices'])} input / "
                  f"{len(hardware['audio']['output_devices'])} output device(s)")
    print(f"  {audio_line}")
    logger.info(audio_line)

    print(f"  Config: {json.dumps(result['config'])}")
    logger.info("Config: %s", json.dumps(result["config"]))

    for w in warnings:
        print(f"  WARNING: {w}")
        logger.warning(w)

    print(f"\nWritten to {OUTPUT_FILE}")
    logger.info("Written to %s", OUTPUT_FILE)
    logger.info("Log written to %s", LOG_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
