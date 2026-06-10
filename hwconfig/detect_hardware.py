"""Hardware detection for EchoLoop.

Standalone tool: probes the machine (RAM, CPU, GPU/VRAM, audio devices) and
writes hwconfig/hardware_config.json with two sections:

  "hardware" — raw facts about the machine, for diagnostics;
  "config"   — ready-to-use parameter values (EXTERNAL_N_GPU_LAYERS etc.)
               picked from the detected hardware. The main app will read
               these instead of the hard-coded defaults in config.py.

Run it manually whenever the hardware changes:

    python hwconfig/detect_hardware.py

It only relies on packages the project already uses (torch, sounddevice);
each probe degrades gracefully if its package is missing or broken, and any
such problem is recorded in the "warnings" list of the output file.
"""

import ctypes
import json
import os
import platform
import site
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_FILE = Path(__file__).parent / "hardware_config.json"

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


# =====================================================================
# GPU
# =====================================================================

def detect_gpu(warnings: list) -> dict:
    """GPU presence, name, VRAM, and which backends can actually use it.

    Two independent consumers, probed separately:

    - llama-cpp-python ships its own CUDA runtime, fully independent of torch.
      ``llama_gpu_offload`` reflects its own capability probe (None when the
      probe is unavailable — physical GPU presence is the fallback signal).
    - torch (used only by Wav2Vec2 in pronounce/) reports CUDA via
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
        "llama_gpu_offload": _probe_llama_offload(warnings),
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
                "build) — pronunciation analysis (Wav2Vec2) will run on CPU; "
                "the LLM is unaffected, it uses llama-cpp's own CUDA"
            )
    except ImportError:
        warnings.append("torch is not installed; Wav2Vec2 device defaults to CPU")
    except Exception as exc:  # noqa: BLE001 — a broken CUDA runtime must not kill detection
        warnings.append(f"torch CUDA probe failed: {exc}")

    return gpu


def _register_nvidia_dll_dirs() -> None:
    """Make CUDA runtime DLLs from pip-installed nvidia-* packages findable.

    The CUDA wheel of llama-cpp-python needs cudart64_12.dll / cublas64_12.dll
    but only searches its own lib/ dir and CUDA_PATH. With no system CUDA
    Toolkit those DLLs come from the nvidia-cuda-runtime-cu12 /
    nvidia-cublas-cu12 pip packages, which place them under
    site-packages/nvidia/*/bin — register those dirs before importing
    llama_cpp. No-op on non-Windows or when the packages are absent.
    (llm_server/server.py carries the same helper.)
    """
    if sys.platform != "win32":
        return
    for site_dir in site.getsitepackages():
        for bin_dir in Path(site_dir).glob("nvidia/*/bin"):
            os.add_dll_directory(str(bin_dir))


def _probe_llama_offload(warnings: list) -> bool | None:
    """Whether the installed llama-cpp-python build can offload to GPU.

    Returns None when it cannot be determined (package missing or too old to
    expose the probe) — callers should then fall back to physical GPU presence.
    """
    _register_nvidia_dll_dirs()
    try:
        import llama_cpp
    except ImportError:
        warnings.append("llama-cpp-python is not installed; "
                        "GPU offload capability unknown")
        return None
    except Exception as exc:  # noqa: BLE001 — native DLL load errors
        warnings.append(f"llama_cpp import failed: {exc}")
        return None

    probe = getattr(llama_cpp, "llama_supports_gpu_offload", None)
    if probe is None:
        return None
    try:
        supported = bool(probe())
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"llama_supports_gpu_offload() failed: {exc}")
        return None
    if not supported:
        warnings.append(
            "llama-cpp-python is a CPU-only build — the LLM cannot use the "
            "GPU; reinstall it with CUDA support to enable offload"
        )
    return supported


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
    except Exception as exc:  # noqa: BLE001 — PortAudio errors must not kill detection
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
    llama-cpp's own offload capability — NOT torch, which is a separate stack
    used only by Wav2Vec2. Threshold rationale: the GGUF model weighs ~2 GB at
    Q4_K_M and Kokoro/Wav2Vec2 also claim VRAM when they run on the GPU, so
    full offload plus GPU-side Wav2Vec2 needs a comfortable margin.
    """
    gpu = hardware["gpu"]
    cores = hardware["cpu_cores"] or 4

    # LLM side: usable unless llama-cpp explicitly reports a CPU-only build
    # (None = probe unavailable, assume a present GPU is usable).
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

    # torch side (pronounce/Wav2Vec2): needs a CUDA-enabled torch build, plus
    # VRAM headroom so it does not fight the LLM for the same card.
    torch_vram = (gpu["vram_gb"] or 0) if gpu["torch_cuda"] else 0

    return {
        "DEVICE": "cuda" if gpu["torch_cuda"] else "cpu",
        "EXTERNAL_N_GPU_LAYERS": n_gpu_layers,
        "EXTERNAL_N_CTX": 4096 if llm_vram >= 8 else 2048,
        "WAV2VEC2_DEVICE": "cuda" if torch_vram >= 6 else "cpu",
        "WHISPER_CPU_THREADS": max(2, min(8, cores // 2)),
        # null = system default device, which is the right choice on most
        # machines; the indices of all devices are listed under "hardware".
        "AUDIO_INPUT_DEVICE": None,
        "AUDIO_OUTPUT_DEVICE": None,
    }


# =====================================================================
# Main
# =====================================================================

def main() -> int:
    warnings: list[str] = []

    print("Detecting hardware...")
    hardware = {
        "platform": f"{platform.system()} {platform.release()}",
        "ram_total_gb": detect_ram_gb(warnings),
        "cpu_cores": os.cpu_count(),
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
    print(f"  RAM:   {hardware['ram_total_gb']} GB, CPU cores: {hardware['cpu_cores']}")
    if gpu["present"]:
        llama_state = {True: "yes", False: "NO", None: "unknown"}[gpu["llama_gpu_offload"]]
        print(f"  GPU:   {gpu['name']} ({gpu['vram_gb']} GB VRAM, "
              f"llama offload: {llama_state}, torch CUDA: "
              f"{'yes' if gpu['torch_cuda'] else 'no'})")
    else:
        print("  GPU:   none detected")
    print(f"  Audio: {len(hardware['audio']['input_devices'])} input / "
          f"{len(hardware['audio']['output_devices'])} output device(s)")
    print(f"  Config: {json.dumps(result['config'])}")
    for w in warnings:
        print(f"  WARNING: {w}")
    print(f"\nWritten to {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
