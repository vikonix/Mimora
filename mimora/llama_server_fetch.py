# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Download and install the official llama-server binary into bin/llama/.

Standalone helper for the default "llama-server" LLM backend: it fetches a
PINNED llama.cpp release from GitHub, checks the sha256 of every asset,
unpacks them into bin/llama/, and confirms that the installed binary really
runs with the backend it advertises.

Callers: install.py (its LLM-stack step), the app's first-run window
(mimora/first_run_download.py) when the binary is missing, and - for the read-
only parts - mimora/config.py, mimora/llm_server_ctl.py and
mimora/detect_hardware.py.

Run it directly:

    python -m mimora.llama_server_fetch              # auto-detect the variant
    python -m mimora.llama_server_fetch --list       # show known variants
    python -m mimora.llama_server_fetch --variant win-cpu-x64 --force
    python -m mimora.llama_server_fetch --dry-run    # print the plan only

Design notes
------------
* Standard library only, and no side effects at import time. The module is
  meant to be callable from install.py *before* the project requirements are
  installed, and later from the running app, so it must not pull in requests,
  tqdm or huggingface_hub.
* The release tag is pinned in RELEASE_TAG and every asset carries the sha256
  published on the release page. Moving to a newer llama.cpp build is a
  deliberate commit that updates both, never a silent "latest".
* CUDA correctness is verified rather than assumed. A llama.cpp CUDA build
  whose cudart/cublas DLLs are missing or of the wrong major version falls
  back to CPU *silently*: it still logs "offloaded N/N layers to GPU" and the
  app keeps working, just about three times slower. `--list-devices` is the
  cheap explicit check that catches it.
* A GPU variant may name a `fallback` variant, and only a failed device check
  triggers it. That is how Linux copes with a GPU backend whose availability
  cannot be established before the download (Vulkan needs a loader, an ICD and
  a device the driver publishes through them). The substitution is logged with
  the reason and recorded in the stamp, so it stays the opposite of the silent
  CPU fallback above: visible, and visible afterwards.
* The install is staged: assets are unpacked into bin/llama.new/ and only
  swapped onto bin/llama/ once every archive is verified, so an interrupted
  run cannot leave a half-written installation behind. The stamp file that
  marks the install as complete is written last, after verification, so a
  binary that fails its checks is always re-installed on the next run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NamedTuple, Optional

if __package__ in (None, ""):
    # Executed as a plain script rather than with -m: that form puts THIS
    # directory on sys.path instead of the project root, so the "import mimora"
    # below would not resolve. Same shim, and the same reason, as in
    # model_fetch and gguf_fetch.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mimora import paths

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pinned release
# ---------------------------------------------------------------------------

# llama.cpp build this module installs. Verified manually on Windows + RTX 3090
# (2026-07): chat template, readiness behaviour, sampling defaults, generation
# speed and VRAM use were all compared against the previous backend before the
# pin was chosen. Bumping it means updating every sha256 below from the release
# page as well.
RELEASE_TAG = "b10099"

DOWNLOAD_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{asset}"
RELEASE_PAGE_URL = "https://github.com/ggml-org/llama.cpp/releases/tag/{tag}"


class Asset(NamedTuple):
    """One archive of a release.

    ``name`` is a template because the llama-* archives embed the release tag
    in their filename while the cudart-* archives do not.

    size_mb is the download size in decimal MB (bytes / 1_000_000), needed
    before the download starts: the first-run dialog has to name the volume
    before the user agrees to it, and the progress bar needs a denominator that
    is known in advance so its percentage cannot jump backwards when it moves on
    to the next file. It sits here, next to the name and the checksum, because
    all three describe the same file of the same pinned release: bumping
    RELEASE_TAG rewrites them together, and a size kept in another module would
    not fail loudly when forgotten, it would quietly mis-scale the bar. Snap
    the values with ``python tools/measure_model_sizes.py``.

    At download time Content-Length is the truth and this is only the plan; the
    two are not reconciled on the fly, because re-scaling a running bar makes
    its percentage jump (see _download).
    """

    name: str
    sha256: str
    size_mb: int


class Variant(NamedTuple):
    """A platform/backend build of llama.cpp.

    min_driver_cuda: lowest CUDA version the NVIDIA driver must report for this
        build to be usable (None for builds that need no CUDA at all).
    device_pattern: regex that must match `llama-server --list-devices` output
        after the install, or None when the build has no GPU backend to check.
    backend: the compute backend this build was compiled with ("CUDA",
        "Vulkan", "Metal", "CPU"), stated rather than inferred. Every message that has
        to name it used to derive it some other way - from the platform, from
        the variant's name prefix - and each of those derivations was a guess
        that a new row in this table could silently invalidate. "CPU" holds
        exactly when device_pattern is None.
    fallback: variant to install instead when this one passes its download and
        --version checks but fails the device check, or None to make that
        failure final. It is a property of the build rather than of whoever
        picked it: the same missing GPU backend has the same remedy whether the
        variant came from select_variant() or from an explicit --variant.
    min_macos: lowest macOS version that can load this build, or None when the
        question does not arise (every non-Apple row). It is a property of the
        published asset, not of macOS: it is the `minos` field of the binary's
        LC_BUILD_VERSION load command, which llama.cpp's CI stamps from
        CMAKE_OSX_DEPLOYMENT_TARGET when the workflow sets one and from the
        runner's own macOS version when it does not. Hence its place here,
        beside the sha256 and the size: all three describe the same file of the
        same pinned release, and bumping RELEASE_TAG has to re-snap them
        together. Read it with
        `llvm-objdump --macho --private-headers llama-server`.

        Only select_variant() consults it, exactly like min_driver_cuda: an
        explicit --variant still installs what it was asked for and lets the
        binary's own failure speak.
    """

    assets: tuple[Asset, ...]
    min_driver_cuda: Optional[tuple[int, int]]
    device_pattern: Optional[str]
    backend: str
    fallback: Optional[str] = None
    min_macos: Optional[tuple[int, int]] = None


# Known builds: Windows x64, Linux x64 and macOS (Apple Silicon + Intel).
#
# Bumping RELEASE_TAG means re-snapping every size_mb below with
# `python tools/measure_model_sizes.py`, in the same commit as the checksums. A
# wrong size does not break the install, it only makes the first-run dialog and
# the progress bar lie, which is exactly the kind of error nothing else catches.
VARIANTS: dict[str, Variant] = {
    "win-cuda-12.4-x64": Variant(
        assets=(
            Asset("llama-{tag}-bin-win-cuda-12.4-x64.zip",
                  "02dc3cb4a1a336cb91c53c41522cfd994cb307c5c881f442be80c6ff54443330",
                  size_mb=250),  # measured 2026-07-28
            # The CUDA runtime that ships with the release. Its major version
            # MUST match the build (cuda-12.4 loads cudart64_12.dll,
            # cublas64_12.dll, cublasLt64_12.dll) - a cudart from another major
            # version is exactly what caused the silent CPU fallback that this
            # module's device check exists to catch.
            # Note that it is the LARGER of the two archives: most of what a CUDA
            # install downloads is NVIDIA's runtime, not llama.cpp.
            Asset("cudart-llama-bin-win-cuda-12.4-x64.zip",
                  "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
                  size_mb=391),  # measured 2026-07-28
        ),
        min_driver_cuda=(12, 4),
        device_pattern=r"\bCUDA\d+\b",
        backend="CUDA",
    ),
    "win-cpu-x64": Variant(
        assets=(
            Asset("llama-{tag}-bin-win-cpu-x64.zip",
                  "b1db6ea811e564f4bffe6c5eb699a025bb14b16b4f641404cb95189fe3f550b1",
                  size_mb=18),  # measured 2026-07-28
        ),
        min_driver_cuda=None,
        device_pattern=None,
        backend="CPU",
    ),
    # There is no CUDA build for Linux to mirror win-cuda-*: llama.cpp ships
    # CUDA binaries for Windows only (checked against b10099, whose linux/x64
    # assets are cpu, vulkan, rocm, sycl and openvino). So on an NVIDIA card
    # under Linux the GPU path is the Vulkan build, and min_driver_cuda stays
    # None because the CUDA version of the driver says nothing about it.
    #
    # Whether Vulkan will find the GPU cannot be settled before the download:
    # it needs a loader AND an ICD manifest AND a device the driver exposes
    # through them, and under WSL2 the NVIDIA driver publishes no Vulkan ICD at
    # all (verified on driver 610.43.02: /usr/lib/wsl/lib has CUDA, NVML and
    # NVENC, no ICD). Hence the fallback: try Vulkan, keep CPU as the answer
    # when the device check comes back empty.
    "linux-vulkan-x64": Variant(
        assets=(
            Asset("llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz",
                  "b4ac074f3b2309653b951ac1757e7b9520cee4765e2f295f65bf00c44ac71560",
                  size_mb=32),  # measured 2026-07-30
        ),
        min_driver_cuda=None,
        device_pattern=r"\bVulkan\d+\b",
        backend="Vulkan",
        fallback="linux-cpu-x64",
    ),
    # Reached automatically as the Vulkan variant's fallback, and directly with
    # `--variant linux-cpu-x64`. Its own fallback is None: there is nothing
    # below it, and a CPU build has no device check to fail.
    "linux-cpu-x64": Variant(
        assets=(
            Asset("llama-{tag}-bin-ubuntu-x64.tar.gz",
                  "1a0046e5ef3ca546402c10940f1f9d76e97a22e8cf31d9ecaa1ddccbb88be0a3",
                  size_mb=16),  # measured 2026-07-30
        ),
        min_driver_cuda=None,
        device_pattern=None,
        backend="CPU",
    ),
    # macOS, one build per architecture and no choice to make within either.
    # Both assets are .tar.gz, which is what preserves the executable bit (see
    # _extract), and both are a single archive: there is no separate runtime to
    # merge in the way the Windows CUDA build needs cudart.
    #
    # min_macos is high because llama.cpp's workflow passes no
    # CMAKE_OSX_DEPLOYMENT_TARGET for this asset and builds it on whatever macOS
    # runner is current, so clang stamps the runner's own version as the
    # minimum. Verified 2026-07-31 by reading LC_BUILD_VERSION out of the
    # downloaded binary: minos 26.0, sdk 26.5. Nothing here can lower it, so an
    # Apple Silicon Mac on macOS 14 or 15 has no usable build in this release
    # and select_variant() says so before the download.
    #
    # device_pattern is the one thing here still UNVERIFIED: no Apple Silicon
    # machine has run this build yet, and the device name was read off the
    # sources (ggml-metal.cpp defines GGML_METAL_NAME "MTL", ggml-metal-device.m
    # formats each device as "MTL%d"), so --list-devices is *expected* to print
    # "MTL0: Apple M...". If a real Mac prints something else, this is the field
    # that has to change.
    "macos-metal-arm64": Variant(
        assets=(
            Asset("llama-{tag}-bin-macos-arm64.tar.gz",
                  "b63bb144fc1855b028984e6680b16532b7fbc2e8eb4002ce0314f61c15549263",
                  size_mb=11),  # measured 2026-07-31
        ),
        min_driver_cuda=None,
        device_pattern=r"\bMTL\d+\b",
        backend="Metal",
        # No fallback on purpose: there is no CPU-only arm64 asset to descend
        # to, and a Mac with a GPU quietly running on its CPU is precisely the
        # outcome the device check exists to refuse.
        min_macos=(26, 0),
    ),
    # Intel Macs are CPU-only by construction, not by our choice: llama.cpp
    # builds this asset with -DGGML_METAL=OFF because its Intel CI runners have
    # no GPU to test against. So there is no GPU variant to prefer over this one
    # and nothing for a device check to look for.
    #
    # Two limits of the build are the release's, and only the first can be
    # checked before the download:
    # * It is compiled with -DCMAKE_OSX_DEPLOYMENT_TARGET=13.3, which
    #   LC_BUILD_VERSION confirms (minos 13.3, verified 2026-07-31) - hence
    #   min_macos below.
    # * ggml's GGML_NATIVE defaults to ON and the workflow does not turn it
    #   off, so the asset carries -march=native for the CI runner's CPU. An
    #   older Intel CPU dies on the first instruction it lacks, and nothing in
    #   the Mach-O header says which instructions those are, so that one can
    #   only be found by running the binary - see _probe's signal branch.
    "macos-cpu-x64": Variant(
        assets=(
            Asset("llama-{tag}-bin-macos-x64.tar.gz",
                  "211cb8c436eb4dba356345da505309a6c3e472de20284444b93517cfb14f3279",
                  size_mb=11),  # measured 2026-07-31
        ),
        min_driver_cuda=None,
        device_pattern=None,
        backend="CPU",
        min_macos=(13, 3),
    ),
}

# Auto-selection order for Windows x64: the first variant whose requirements
# the machine satisfies wins, so GPU builds must come before the CPU fallback.
WINDOWS_X64_PREFERENCE = ("win-cuda-12.4-x64", "win-cpu-x64")

# Linux x64 needs no such order: the Vulkan build requires no CUDA, so a
# preference list would return it on the first iteration and never reach a CPU
# entry anyway. The choice is a single name and the descent to CPU happens
# after the device check, through Variant.fallback.
LINUX_X64_DEFAULT = "linux-vulkan-x64"

# macOS needs no preference list either, for a different reason: each Mac
# architecture has exactly one pinned build, so the architecture IS the choice.
MACOS_ARM64_DEFAULT = "macos-metal-arm64"
MACOS_X64_DEFAULT = "macos-cpu-x64"


def variant_size_mb(variant_name: str) -> int:
    """Download size of a whole variant: the sum over its assets.

    A variant has no size of its own on purpose. The CUDA build is two archives
    that are fetched one after another, and the progress callback reports per
    file, so per-asset numbers are what the bar actually needs; the total is
    derived from them and therefore cannot drift away from them.
    """
    return sum(asset.size_mb for asset in VARIANTS[variant_name].assets)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# paths.py is stdlib-only, which keeps this module importable by install.py
# before the requirements exist - the property the whole file is built around.
INSTALL_DIR = paths.llama_dir()
# Written only after a successful install AND verification; its presence with a
# matching tag/variant is what "already installed" means.
STAMP_NAME = "installed.json"
EXE_NAME = "llama-server.exe" if sys.platform == "win32" else "llama-server"

# 1 MiB read chunks: large enough that the loop overhead disappears next to the
# network, small enough that progress stays smooth on the largest asset (the
# cudart archive, 391 MB - see VARIANTS).
_CHUNK_SIZE = 1024 * 1024
_HTTP_TIMEOUT_SEC = 60
# --list-devices and --version load no model, so they return in well under a
# second; the timeout only guards against a binary that hangs on a bad DLL.
_PROBE_TIMEOUT_SEC = 60

ProgressFn = Callable[[str, int, Optional[int]], None]


class LlamaServerFetchError(RuntimeError):
    """Any failure that leaves llama-server not installed and ready."""


class UnsupportedPlatformError(LlamaServerFetchError):
    """No build is pinned for this OS/architecture yet."""


class DeviceCheckError(LlamaServerFetchError):
    """The installed build runs, but its GPU backend brought up no device.

    Separate from the base class so the fallback in ensure_llama_server can
    catch exactly this and nothing else. A download that fails, a checksum that
    does not match or an archive with an unexpected layout says nothing about
    which build suits the machine, so retrying those with another variant would
    only turn one clear error into two.
    """


# ---------------------------------------------------------------------------
# Variant selection
# ---------------------------------------------------------------------------

def detect_driver_cuda() -> Optional[tuple[int, int]]:
    """Max CUDA version the installed NVIDIA driver supports, via nvidia-smi.

    None means no NVIDIA GPU, no driver, or an smi layout we cannot parse.
    Deliberately mirrors detect_gpu() in install.py instead of importing it:
    install.py must stay a standalone script that runs before this package is
    importable, and the check needs no third-party package either way.
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        log.info("nvidia-smi not found - treating this machine as CPU-only.")
        return None
    try:
        result = subprocess.run([smi], capture_output=True, text=True,
                                timeout=15, **_no_window())
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.info("nvidia-smi failed (%s) - treating this machine as CPU-only.", exc)
        return None

    if result.returncode != 0:
        # A driver/library mismatch makes nvidia-smi exit non-zero with an
        # empty stdout; without this branch that looks identical to "parsed
        # nothing" and hides a real problem with the driver install.
        log.info("nvidia-smi exited with code %d: %s",
                 result.returncode, (result.stderr or "").strip() or "(no output)")
        return None

    # The driver's max supported CUDA sits in the smi header. Two layouts are
    # in the wild and both must parse:
    #   "| NVIDIA-SMI 581.15   Driver Version: 581.15   CUDA Version: 13.0  |"
    #   "| NVIDIA-SMI 610.62   KMD Version: 610.62   CUDA UMD Version: 13.3  |"
    # The 610 drivers renamed the fields (KMD/UMD = kernel/user mode driver);
    # the UMD number is the successor of the old CUDA Version field. `-q`
    # output pads before the colon, hence the loose separator.
    match = re.search(r"CUDA(?:\s+UMD)?\s+Version\s*:\s*(\d+)\.(\d+)",
                      result.stdout)
    if not match:
        # Print what we did get: the usual causes are a header layout we have
        # not seen and a literal "CUDA Version: N/A", and only the raw text
        # tells them apart.
        head = "\n".join(result.stdout.strip().splitlines()[:3]) or "(no output)"
        log.info("nvidia-smi printed no parsable CUDA version - assuming the "
                 "newest build still works (re-run with --variant to "
                 "override). First lines were:\n%s", head)
        return None
    version = (int(match.group(1)), int(match.group(2)))
    log.info("Driver CUDA: %d.%d", *version)
    return version


def is_rosetta() -> bool:
    """True when this x86_64 process is being translated on Apple Silicon.

    platform.machine() answers for the interpreter, not for the machine: an
    x86_64 Python under Rosetta 2 reports x86_64 on an M-series Mac, and
    picking the x64 build from that would install a CPU-only binary on a machine
    with a usable GPU - the silent slow path this module exists to prevent.
    Nothing forces the two to agree, because llama-server is a separate process:
    the arm64 build runs natively there whatever this interpreter was built for.

    sysctl.proc_translated does not exist on Intel Macs, where sysctl exits
    non-zero; anything other than a clean "1" therefore means "not translated".
    """
    if sys.platform != "darwin":
        return False
    try:
        result = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"],
                                capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.info("Could not read sysctl.proc_translated (%s) - assuming this "
                 "process is not translated.", exc)
        return False
    return result.stdout.strip() == "1"


def detect_macos_version() -> Optional[tuple[int, int]]:
    """(major, minor) of the running macOS, or None when it cannot be read.

    None is "do not know", not "old": it lets the choice through rather than
    refusing a machine on a reading this module failed to take.

    One quirk is worth knowing and does no harm here. macOS 11+ reports itself
    as "10.16" to a process linked against a pre-11 SDK, so a very old Python
    could under-report its host. The error is always downwards, which turns a
    misread into a refusal to install rather than into an install that cannot
    run - the safe direction, and the reason this is not worked around.
    """
    release = platform.mac_ver()[0]
    match = re.match(r"(\d+)(?:\.(\d+))?", release)
    if not match:
        log.info("platform.mac_ver() returned %r - cannot tell which macOS "
                 "this is.", release)
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def _macos_variant(name: str) -> str:
    """Return *name*, unless this Mac is older than the build can load.

    Refusing here rather than after the download is what makes the difference
    visible to the caller: install.py records an UnsupportedPlatformError as a
    manual step and carries on, while the same machine discovering the same fact
    from dyld three seconds later fails the whole install step.
    """
    required = VARIANTS[name].min_macos
    current = detect_macos_version()
    if required is None or current is None or current >= required:
        return name
    raise UnsupportedPlatformError(
        f"{name} needs macOS {required[0]}.{required[1]} or newer and this Mac "
        f"runs {current[0]}.{current[1]}, so the pinned llama.cpp release "
        f"{RELEASE_TAG} has no build it can load. The minimum is the release's, "
        f"not Mimora's: llama.cpp stamps it into the binary at build time. "
        f"Build llama.cpp on this machine and point llama_server_path at the "
        f"result, or switch llm_backend to lm-studio. Passing --variant "
        f"{name} installs it anyway if you want to see it fail.")


def select_variant() -> str:
    """Pick the best variant for this machine.

    Raises UnsupportedPlatformError when no build is pinned for the platform, or
    when the pinned one cannot run on it (macOS too old for the release).
    """
    machine = platform.machine().lower()
    is_x64 = machine in ("amd64", "x86_64")

    if sys.platform == "darwin":
        # Architecture decides and there is nothing else to weigh (see
        # MACOS_ARM64_DEFAULT), but platform.machine() alone cannot name it -
        # Rosetta 2 makes an Apple Silicon Mac report x86_64.
        if machine == "arm64":
            return _macos_variant(MACOS_ARM64_DEFAULT)
        if is_x64:
            if is_rosetta():
                log.info("This Python runs under Rosetta 2, so the machine is "
                         "Apple Silicon - installing %s rather than the Intel "
                         "build.", MACOS_ARM64_DEFAULT)
                return _macos_variant(MACOS_ARM64_DEFAULT)
            return _macos_variant(MACOS_X64_DEFAULT)
        # Anything else on macOS (a 32-bit or PowerPC interpreter) falls
        # through to the error below: no such asset is published.

    if sys.platform.startswith("linux") and is_x64:
        # Nothing to weigh here (see LINUX_X64_DEFAULT), and nothing cheap to
        # probe either: whether the Vulkan build finds a device depends on the
        # loader and the ICD, neither of which is visible from this process
        # without loading them. So the answer is "the GPU build", and the
        # post-install device check plus its fallback settle the rest.
        return LINUX_X64_DEFAULT

    if sys.platform != "win32" or not is_x64:
        raise UnsupportedPlatformError(
            f"No pinned llama.cpp build for {sys.platform}/{platform.machine()} "
            f"yet (Windows x64, Linux x64 and macOS arm64/x64 only so far). "
            f"Download a build from {RELEASE_PAGE_URL.format(tag=RELEASE_TAG)} "
            f"and point llama_server_path at it, or add the variant to "
            f"VARIANTS.")

    driver_cuda = detect_driver_cuda()
    for name in WINDOWS_X64_PREFERENCE:
        required = VARIANTS[name].min_driver_cuda
        if required is None:
            return name
        # An NVIDIA card whose driver version we could not parse still gets the
        # CUDA build: it is the likely-correct choice, and the post-install
        # device check below turns a wrong guess into a clear error rather than
        # a silent CPU fallback.
        if driver_cuda is None:
            if shutil.which("nvidia-smi"):
                log.info("Driver CUDA unknown but an NVIDIA driver is present - "
                         "trying %s.", name)
                return name
            continue
        if driver_cuda >= required:
            return name
        log.info("Driver CUDA %d.%d is older than %s needs (%d.%d) - skipping it.",
                 *driver_cuda, name, *required)

    # WINDOWS_X64_PREFERENCE always ends with a CPU build, so this is
    # unreachable unless that list is edited badly.
    raise UnsupportedPlatformError("No usable variant for this machine.")


# ---------------------------------------------------------------------------
# Install state
# ---------------------------------------------------------------------------

def installed_exe(dest: Path = INSTALL_DIR) -> Optional[Path]:
    """Path of a complete, stamped installation, or None.

    "Complete" means the stamp file is present (so the install finished and
    passed verification) and the binary it describes still exists.
    """
    stamp = read_stamp(dest)
    if stamp is None:
        return None
    exe = dest / EXE_NAME
    return exe if exe.is_file() else None


def read_stamp(dest: Path = INSTALL_DIR) -> Optional[dict]:
    """Contents of the install stamp, or None when absent/unreadable."""
    try:
        with open(dest / STAMP_NAME, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        # A corrupted stamp is treated as "not installed": the reinstall that
        # follows is cheap compared to running an unidentified binary.
        return None


def is_current(dest: Path, tag: str, variant: str) -> bool:
    """True when dest holds exactly this tag+variant and the binary is there.

    Comparing the variant as well as the tag matters: switching a machine from
    the CPU build to the CUDA build keeps the same tag, and only the variant
    tells the two apart.
    """
    stamp = read_stamp(dest)
    if stamp is None:
        return False
    return (stamp.get("tag") == tag
            and stamp.get("variant") == variant
            and (dest / EXE_NAME).is_file())


def _write_stamp(dest: Path, tag: str, variant: str,
                 assets: tuple[Asset, ...]) -> None:
    payload = {
        "tag": tag,
        "variant": variant,
        "assets": {asset.name.format(tag=tag): asset.sha256 for asset in assets},
        "installed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(dest / STAMP_NAME, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


# ---------------------------------------------------------------------------
# Download and unpack
# ---------------------------------------------------------------------------

def _no_window() -> dict:
    """subprocess kwargs that suppress the console window flash on Windows.

    The app will call this module from a Tk GUI, where every child process
    would otherwise pop up a black console window for a moment.
    """
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def _human(size: Optional[int]) -> str:
    """Render a byte count as decimal MB, the unit Asset.size_mb is stated in.

    Decimal on purpose (bytes / 1_000_000, not / 1024**2). This function labels
    its output "MB", and the same label appears next to the planned sizes in
    VARIANTS and in models_info; rendering MiB under it would make a finished
    391 MB download report "372.9 MB" and leave the user comparing two numbers
    that disagree for no visible reason.
    """
    if size is None:
        return "unknown size"
    return f"{size / 1_000_000:.1f} MB"


def _download(url: str, target: Path, expected_sha256: str,
              progress: Optional[ProgressFn]) -> None:
    """Fetch one asset and fail unless its sha256 matches the pinned value.

    The hash is computed while streaming, so the archive is never fully held in
    memory and a corrupted download (a proxy that rewrites binaries, a truncated
    transfer) is caught before anything is unpacked.
    """
    name = target.name
    # A plain python-urllib user agent is enough for GitHub, but some corporate
    # proxies reject unknown clients outright; an explicit one is friendlier to
    # whoever has to read the proxy log.
    request = urllib.request.Request(
        url, headers={"User-Agent": "mimora-llama-server-fetch/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SEC) as response:
            length = response.headers.get("Content-Length")
            total = int(length) if length and length.isdigit() else None
            digest = hashlib.sha256()
            downloaded = 0
            with open(target, "wb") as handle:
                while True:
                    chunk = response.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(name, downloaded, total)
    except (urllib.error.URLError, OSError) as exc:
        raise LlamaServerFetchError(
            f"Download of {name} failed: {exc}. The release is at "
            f"{RELEASE_PAGE_URL.format(tag=RELEASE_TAG)} if you need to fetch "
            f"it manually (proxy, firewall, or offline machine).") from exc

    actual = digest.hexdigest()
    if actual != expected_sha256:
        # Keep nothing questionable on disk: the next run must start clean.
        target.unlink(missing_ok=True)
        raise LlamaServerFetchError(
            f"Checksum mismatch for {name}: expected {expected_sha256}, "
            f"got {actual}. The download was corrupted or the pinned release "
            f"assets were changed upstream.")


def _extract(archive: Path, into: Path) -> None:
    """Unpack one asset archive into a directory.

    Two formats, because the release ships two: the Windows assets are .zip and
    the Linux and macOS ones are .tar.gz.

    The executable bit is why the branches are not interchangeable.
    zipfile.extractall does NOT apply a member's unix permissions (they live in
    the high half of external_attr and it ignores them), so a POSIX build
    delivered as .zip would unpack a llama-server nobody can run, and the
    failure would surface as a bare PermissionError from _probe rather than as
    anything about unpacking. tarfile does preserve the mode, so the .tar.gz
    branch needs no chmod - and a future POSIX variant published as .zip would.
    """
    root = into.resolve()
    name = archive.name.lower()

    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                # Guard against archive members that escape the target
                # directory. ZipFile.extract already strips absolute paths and
                # '..', but an explicit check keeps that guarantee visible and
                # local.
                if not (root / member.filename).resolve().is_relative_to(root):
                    raise LlamaServerFetchError(
                        f"Refusing to unpack {archive.name}: member "
                        f"{member.filename!r} points outside the target "
                        f"directory.")
            zf.extractall(into)
        return

    if name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tf:
            for member in tf.getmembers():
                if not (root / member.name).resolve().is_relative_to(root):
                    raise LlamaServerFetchError(
                        f"Refusing to unpack {archive.name}: member "
                        f"{member.name!r} points outside the target directory.")
            # filter="data" drops device nodes, setuid/setgid bits and links
            # that escape the destination, while keeping the executable bit -
            # the one mode this unpacking actually depends on. It becomes the
            # default in 3.14 and is passed explicitly so the behaviour does
            # not change with the interpreter.
            tf.extractall(into, filter="data")
        return

    raise LlamaServerFetchError(
        f"Don't know how to unpack {archive.name} (only .zip and .tar.gz are "
        f"supported).")


def _payload_root(extracted: Path) -> Path:
    """Directory inside an unpacked archive whose contents we actually want.

    llama.cpp archives keep the server binary and every library it needs side
    by side, but where that directory sits varies: flat at the root in some
    Windows releases, one folder down in others, under build/bin/ in the Linux
    tarballs and under llama-<tag>/ in the macOS ones. Anchoring on the binary
    makes the unpacking survive all four; the cudart archive has no binary and
    falls through to the single-directory / flat-root cases.
    """
    binaries = sorted(extracted.rglob(EXE_NAME))
    if binaries:
        return binaries[0].parent
    entries = list(extracted.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extracted


# ---------------------------------------------------------------------------
# Post-install verification
# ---------------------------------------------------------------------------

def _probe(exe: Path, args: list[str]) -> str:
    """Run the binary with a cheap flag and return its combined output.

    llama-server prints diagnostics to stderr, so both streams are merged
    before parsing.
    """
    try:
        result = subprocess.run(
            [str(exe), *args], capture_output=True, text=True,
            timeout=_PROBE_TIMEOUT_SEC, **_no_window())
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LlamaServerFetchError(
            f"Could not run {exe.name} {' '.join(args)}: {exc}. "
            f"{_start_failure_hint()}") from exc

    # A negative return code on POSIX means the process was killed by a signal,
    # which is not the same failure as "would not start" and does not raise
    # above: subprocess.run reports it as an ordinary result with empty output.
    # Without this branch it would surface from verify_build as "printed no
    # recognisable version" over an empty string, which says nothing. The case
    # that makes it worth naming is SIGILL (-4) from an Intel Mac older than the
    # CPU the release was compiled for - see the macos-cpu-x64 comment.
    if sys.platform != "win32" and result.returncode < 0:
        raise LlamaServerFetchError(
            f"{exe.name} {' '.join(args)} was killed by signal "
            f"{-result.returncode} before it could answer. "
            f"{_start_failure_hint()}")
    return (result.stdout or "") + (result.stderr or "")


def _start_failure_hint() -> str:
    """Why an unpacked binary might refuse to run, per platform.

    A binary that unpacked fine but will not start almost always misses
    something the release assumes about the machine, and what that is differs
    per platform, so the hint has to as well.
    """
    if sys.platform == "win32":
        return ("On Windows this usually means a missing Visual C++ runtime "
                "(https://aka.ms/vs/17/release/vc_redist.x64.exe).")
    if sys.platform == "darwin":
        # select_variant() already refuses a Mac older than Variant.min_macos,
        # so reaching this on a version grounds means an explicit --variant. The
        # CPU one it cannot catch: no header states which instructions the build
        # needs.
        return ("On macOS this usually means the release does not fit this "
                "Mac. Either the system is older than the build's minimum "
                "(dyld reports a binary 'built for macOS ... newer than "
                "running OS'), or - on Intel - the CPU is older than the one "
                "llama.cpp's CI compiled for, which stops the process with an "
                "illegal instruction. Neither is fixable from here: build "
                "llama.cpp on this machine and point llama_server_path at the "
                "result.")
    return ("On Linux this usually means a system library the release links "
            "against is missing: libcurl.so.4 (package libcurl4t64) or "
            "libgomp.so.1 (package libgomp1).")


def verify_build(exe: Path, tag: str) -> int:
    """Check that the binary runs and is the build we pinned.

    Returns the build number reported by `--version` (llama-server prints
    "version: 10099 (1a064ab09)"). A mismatch means the wrong binary ended up
    at this path, which matters because the command line the app builds uses
    flags that older builds do not have.
    """
    output = _probe(exe, ["--version"])
    match = re.search(r"version:\s*(\d+)", output)
    if not match:
        raise LlamaServerFetchError(
            f"{exe.name} --version printed no recognisable version:\n{output.strip()}")
    build = int(match.group(1))
    expected = int(tag.lstrip("b"))
    if build != expected:
        raise LlamaServerFetchError(
            f"{exe.name} reports build {build}, expected {expected} ({tag}). "
            f"Another llama-server is installed at {exe.parent}.")
    log.info("Binary check: build %d (%s).", build, tag)
    return build


def list_devices(exe: Path) -> str:
    """Raw `--list-devices` output of *exe*.

    Split out of verify_devices because the running app wants the same answer
    without the pass/fail verdict: llama-server's own log shows neither the
    buffer names nor the selected devices at its default verbosity, so this is
    the only cheap evidence of which backend actually came up. Loads no model.
    """
    return _probe(exe, ["--list-devices"])


def installed_variant(exe: Path, dest: Path = INSTALL_DIR) -> Optional[str]:
    """Variant *exe* was installed as, or None when it is not our install.

    Lets a caller tell "the CUDA build we put there" from "some llama-server
    the user manages themselves": only for the former is there a documented
    expectation about which devices must show up.
    """
    stamp = read_stamp(dest)
    if stamp is None:
        return None
    try:
        if Path(exe).resolve() != (dest / EXE_NAME).resolve():
            return None
    except OSError:
        return None
    variant = stamp.get("variant")
    return variant if variant in VARIANTS else None


def verify_devices(exe: Path, variant_name: str) -> None:
    """Check that the GPU backend the variant promises actually came up.

    This is the whole point of the module's verification step: a CUDA build
    with missing or mismatched cudart DLLs starts fine, reports
    "offloaded N/N layers to GPU", and runs about three times slower on the
    CPU without a single error message. --list-devices is the honest answer -
    it loads no model and takes well under a second.
    """
    variant = VARIANTS[variant_name]
    output = list_devices(exe)
    if variant.device_pattern is None:
        log.info("Device check: CPU build, nothing to verify.")
        return
    if not re.search(variant.device_pattern, output):
        # The remedy differs per backend, and this message is the only place
        # the user is told what to look at, so it names the backend's own
        # failure mode rather than a generic "no device".
        if variant.backend == "Vulkan":
            remedy = ("To get the GPU build instead, install the Vulkan loader "
                      "(package libvulkan1) and check that the driver "
                      "publishes an ICD in /usr/share/vulkan/icd.d.")
        elif variant.backend == "CUDA":
            remedy = (f"Check that the cudart DLLs sit next to {exe.name} and "
                      f"that their major version matches the build.")
        elif variant.backend == "Metal":
            # Metal ships with macOS, so unlike Vulkan there is nothing to
            # install: an empty device list means the binary is not seeing the
            # GPU it was built for.
            remedy = ("Metal is part of macOS, so there is nothing to install. "
                      "An empty list here means the process is not reaching "
                      "this Mac's GPU - check that it is not running under "
                      "Rosetta 2, and that the machine really is Apple "
                      "Silicon.")
        else:
            remedy = (f"Check that the {variant.backend} runtime this build "
                      f"needs is installed and reachable from {exe.name}.")
        raise DeviceCheckError(
            f"{variant_name} was installed but no matching device appeared in "
            f"--list-devices, so llama-server would silently run on the CPU "
            f"(about three times slower). Output was:\n{output.strip()}\n"
            f"{remedy}")
    log.info("Device check: GPU backend is up.\n%s", output.strip())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_llama_server(*, variant: Optional[str] = None,
                        dest: Path = INSTALL_DIR,
                        tag: str = RELEASE_TAG,
                        force: bool = False,
                        verify: bool = True,
                        progress: Optional[ProgressFn] = None) -> Path:
    """Make sure llama-server of the pinned build sits in ``dest``; return it.

    Downloads and unpacks only when the stamp does not already describe this
    tag+variant (or when ``force`` is set), so calling it on every app start is
    cheap. Raises LlamaServerFetchError on any failure; on success the returned
    path is a binary that has been run at least once and reported the expected
    build and backend.

    A variant whose GPU backend brings up no device is retried with its
    ``fallback`` (see Variant), which is how a machine with no usable Vulkan
    still ends up with a working CPU build instead of a failed install. Only
    the device check is retried, and only downwards: every other failure is
    raised as it happens, and the chain is finite because a variant is never
    tried twice.
    """
    variant_name = variant or select_variant()
    tried: list[str] = []
    while True:
        if variant_name not in VARIANTS:
            raise LlamaServerFetchError(
                f"Unknown variant {variant_name!r}. "
                f"Known: {', '.join(sorted(VARIANTS))}.")
        try:
            return _install_variant(variant_name, dest=dest, tag=tag,
                                    force=force, verify=verify,
                                    progress=progress)
        except DeviceCheckError as exc:
            fallback = VARIANTS[variant_name].fallback
            if fallback is None or fallback in tried:
                raise
            # Logged, not swallowed: the whole point of the device check is
            # that a CPU run must never be a surprise, so the reason and the
            # substitution both have to reach the user. The install that just
            # failed left dest unstamped, so the next attempt reinstalls over
            # it without needing force.
            log.info("%s\nFalling back to %s.", exc, fallback)
            tried.append(variant_name)
            variant_name = fallback


def _install_variant(variant_name: str, *,
                     dest: Path,
                     tag: str,
                     force: bool,
                     verify: bool,
                     progress: Optional[ProgressFn]) -> Path:
    """Download, unpack and verify exactly one variant. No fallback here.

    Split out of ensure_llama_server so that the retry loop above stays a loop
    over variants and this stays a single install: mixing the two made it hard
    to see which failures are final.
    """
    spec = VARIANTS[variant_name]
    exe = dest / EXE_NAME

    if not force and is_current(dest, tag, variant_name):
        log.info("llama-server %s (%s) is already installed in %s.",
                 tag, variant_name, dest)
        return exe

    existing = read_stamp(dest)
    if existing:
        log.info("Replacing installed %s (%s) with %s (%s).",
                 existing.get("tag"), existing.get("variant"), tag, variant_name)

    # Everything is built in a staging directory and swapped in at the end, so
    # an interrupted or failed run never leaves a partial install behind.
    staging = dest.with_name(dest.name + ".new")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    # Archives go to a temp directory and are dropped afterwards: together they
    # are over half a gigabyte, and a re-run is rare enough not to be worth
    # keeping them around.
    try:
        with tempfile.TemporaryDirectory(prefix="llama-server-") as tmp:
            tmp_dir = Path(tmp)
            for asset in spec.assets:
                name = asset.name.format(tag=tag)
                url = DOWNLOAD_URL.format(tag=tag, asset=name)
                archive = tmp_dir / name
                log.info("Downloading %s ...", name)
                _download(url, archive, asset.sha256, progress)
                log.info("Unpacking %s ...", name)
                unpacked = tmp_dir / (name + ".d")
                unpacked.mkdir()
                _extract(archive, unpacked)
                # dirs_exist_ok: the cudart archive merges its DLLs into the
                # same directory the build archive just filled.
                shutil.copytree(_payload_root(unpacked), staging,
                                dirs_exist_ok=True)
                archive.unlink(missing_ok=True)

        if not (staging / EXE_NAME).is_file():
            raise LlamaServerFetchError(
                f"{EXE_NAME} was not found in the unpacked release - the asset "
                f"layout of {tag} differs from what this module expects.")
    except BaseException:
        # Any failure here (network, checksum, bad layout, Ctrl-C) must not
        # leave a half-filled staging directory for the next run to trip over.
        shutil.rmtree(staging, ignore_errors=True)
        raise

    _swap(staging, dest)

    if verify:
        # Verification runs against the final location: on Windows the binary
        # resolves its DLLs from its own directory, so probing the staging copy
        # would not prove anything about the installed one.
        verify_build(exe, tag)
        verify_devices(exe, variant_name)

    # Written last: an installation that failed verification stays unstamped
    # and is therefore reinstalled on the next call instead of being trusted.
    _write_stamp(dest, tag, variant_name, spec.assets)
    log.info("llama-server %s (%s) installed in %s.", tag, variant_name, dest)
    return exe


def _swap(staging: Path, dest: Path) -> None:
    """Replace ``dest`` with ``staging`` (staging is left alone on failure)."""
    if dest.exists():
        try:
            shutil.rmtree(dest)
        except OSError as exc:
            raise LlamaServerFetchError(
                f"Could not remove the existing install at {dest}: {exc}. A "
                f"running llama-server holds its files open on Windows - stop "
                f"it and re-run. The new build is ready in {staging}.") from exc
    try:
        staging.replace(dest)
    except OSError as exc:
        raise LlamaServerFetchError(
            f"Could not move {staging} onto {dest}: {exc}") from exc


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _cli_progress() -> ProgressFn:
    """Progress reporter: a live percentage on a TTY, decade steps otherwise.

    Redirected output (a log file, a CI job) gets one line per 10 percent
    instead of thousands of carriage returns.
    """
    interactive = sys.stderr.isatty()
    state = {"name": "", "next_step": 0}

    def report(name: str, done: int, total: Optional[int]) -> None:
        if state["name"] != name:
            state["name"] = name
            state["next_step"] = 0
            if interactive:
                sys.stderr.write("\n")
        if total:
            percent = done * 100 // total
            if interactive:
                sys.stderr.write(
                    f"\r    {name}: {percent:3d}%  "
                    f"({_human(done)} / {_human(total)})")
                sys.stderr.flush()
                if done >= total:
                    sys.stderr.write("\n")
            elif percent >= state["next_step"]:
                print(f"    {name}: {percent}% ({_human(done)} / {_human(total)})")
                state["next_step"] = percent - percent % 10 + 10
        elif interactive:
            sys.stderr.write(f"\r    {name}: {_human(done)}")
            sys.stderr.flush()

    return report


def _print_plan(variant_name: str, dest: Path, tag: str) -> None:
    spec = VARIANTS[variant_name]
    print(f"Variant : {variant_name}")
    print(f"Release : {tag}  ({RELEASE_PAGE_URL.format(tag=tag)})")
    print(f"Target  : {dest}")
    if spec.min_macos:
        # Printed for the same reason the download size is: it is a condition
        # on the plan the user is about to agree to, and it is the one that
        # decides whether the binary will start at all.
        print(f"Requires: macOS {spec.min_macos[0]}.{spec.min_macos[1]} or newer")
    print(f"Download: {variant_size_mb(variant_name)} MB")
    print("Assets  :")
    for asset in spec.assets:
        print(f"    {asset.name.format(tag=tag)}  ({asset.size_mb} MB)")
        print(f"        {DOWNLOAD_URL.format(tag=tag, asset=asset.name.format(tag=tag))}")
        print(f"        sha256 {asset.sha256}")
    if spec.fallback:
        print(f"Fallback: {spec.fallback} "
              f"({variant_size_mb(spec.fallback)} MB), installed instead if "
              f"the device check finds no GPU")
    stamp = read_stamp(dest)
    if stamp:
        print(f"Installed: {stamp.get('tag')} ({stamp.get('variant')}), "
              f"{stamp.get('installed_at')}")
    else:
        print("Installed: nothing (or an incomplete install)")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the pinned llama-server build into bin/llama/.")
    parser.add_argument("--variant", choices=sorted(VARIANTS),
                        help="build to install (default: auto-detect)")
    parser.add_argument("--dest", type=Path, default=INSTALL_DIR,
                        help=f"install directory (default: {INSTALL_DIR})")
    parser.add_argument("--force", action="store_true",
                        help="reinstall even when the pinned build is present")
    parser.add_argument("--skip-verify", action="store_true",
                        help="do not run --version / --list-devices afterwards")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be downloaded and exit")
    parser.add_argument("--list", action="store_true",
                        help="list the known variants and exit")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    # stream=stdout, not the default stderr, matching model_fetch and
    # gguf_fetch: this CLI also print()s reports to stdout, and two streams with
    # different buffering come out in whatever order the OS feels like - the
    # "Driver CUDA: x.y" line select_variant() logs before the plan ended up
    # printed after it.
    #
    # It also puts the download log on the same stream as the non-interactive
    # progress report, which _cli_progress print()s to stdout; those two used to
    # be split across stdout and stderr, which is worst exactly when the output
    # is redirected to a file. Interactive progress stays on stderr on purpose
    # (a terminal flushes both promptly, and \r updates do not belong in a pipe).
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stdout)

    if args.list:
        print(f"Pinned release: {RELEASE_TAG}")
        for name, spec in sorted(VARIANTS.items()):
            # The backend is stated first because it is the thing being
            # chosen; the driver requirement is a condition on it, and only
            # the CUDA builds have one. Deriving the whole label from
            # min_driver_cuda alone used to describe the Vulkan build as
            # having "no GPU requirement", which is true of the driver
            # version and false of everything the user meant by it.
            facts = [f"{spec.backend} backend"]
            if spec.min_driver_cuda:
                facts.append("driver CUDA >= "
                             f"{spec.min_driver_cuda[0]}.{spec.min_driver_cuda[1]}")
            if spec.min_macos:
                facts.append(f"macOS >= {spec.min_macos[0]}.{spec.min_macos[1]}")
            if spec.fallback:
                facts.append(f"falls back to {spec.fallback}")
            facts.append(f"{len(spec.assets)} asset(s)")
            facts.append(f"{variant_size_mb(name)} MB")
            print(f"  {name}  ({', '.join(facts)})")
        return 0

    try:
        variant_name = args.variant or select_variant()
        if args.dry_run:
            _print_plan(variant_name, args.dest, RELEASE_TAG)
            return 0
        ensure_llama_server(variant=variant_name, dest=args.dest,
                            force=args.force, verify=not args.skip_verify,
                            progress=_cli_progress())
    except LlamaServerFetchError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
