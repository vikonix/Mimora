# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Download and install the official llama-server binary into bin/llama/.

Standalone helper for the upcoming "llama-server" LLM backend: it fetches a
PINNED llama.cpp release from GitHub, checks the sha256 of every asset,
unpacks them into bin/llama/, and confirms that the installed binary really
runs with the backend it advertises.

Run it directly (nothing else in the project calls it yet):

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
  cheap explicit check that catches it (see tasks/llama-cpp.md, phase 0).
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
import tempfile
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NamedTuple, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pinned release
# ---------------------------------------------------------------------------

# llama.cpp build this module installs. Verified manually on Windows + RTX 3090
# (tasks/llama-cpp.md, "Результаты фазы 0"). Bumping it means updating every
# sha256 below from the release page as well.
RELEASE_TAG = "b10099"

DOWNLOAD_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{asset}"
RELEASE_PAGE_URL = "https://github.com/ggml-org/llama.cpp/releases/tag/{tag}"


class Asset(NamedTuple):
    """One archive of a release.

    ``name`` is a template because the llama-* archives embed the release tag
    in their filename while the cudart-* archives do not.
    """

    name: str
    sha256: str


class Variant(NamedTuple):
    """A platform/backend build of llama.cpp.

    min_driver_cuda: lowest CUDA version the NVIDIA driver must report for this
        build to be usable (None for builds that need no CUDA at all).
    device_pattern: regex that must match `llama-server --list-devices` output
        after the install, or None when the build has no GPU backend to check.
    """

    assets: tuple[Asset, ...]
    min_driver_cuda: Optional[tuple[int, int]]
    device_pattern: Optional[str]


# Known builds. Windows x64 only for now; macOS/Linux entries are one line each
# and get added when there is a machine to verify them on.
VARIANTS: dict[str, Variant] = {
    "win-cuda-12.4-x64": Variant(
        assets=(
            Asset("llama-{tag}-bin-win-cuda-12.4-x64.zip",
                  "02dc3cb4a1a336cb91c53c41522cfd994cb307c5c881f442be80c6ff54443330"),
            # The CUDA runtime that ships with the release. Its major version
            # MUST match the build (cuda-12.4 loads cudart64_12.dll,
            # cublas64_12.dll, cublasLt64_12.dll) - a cudart from another major
            # version is exactly what caused the silent CPU fallback in phase 0.
            Asset("cudart-llama-bin-win-cuda-12.4-x64.zip",
                  "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"),
        ),
        min_driver_cuda=(12, 4),
        device_pattern=r"\bCUDA\d+\b",
    ),
    "win-cpu-x64": Variant(
        assets=(
            Asset("llama-{tag}-bin-win-cpu-x64.zip",
                  "b1db6ea811e564f4bffe6c5eb699a025bb14b16b4f641404cb95189fe3f550b1"),
        ),
        min_driver_cuda=None,
        device_pattern=None,
    ),
}

# Auto-selection order for Windows x64: the first variant whose requirements
# the machine satisfies wins, so GPU builds must come before the CPU fallback.
WINDOWS_X64_PREFERENCE = ("win-cuda-12.4-x64", "win-cpu-x64")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
INSTALL_DIR = BASE_DIR / "bin" / "llama"
# Written only after a successful install AND verification; its presence with a
# matching tag/variant is what "already installed" means.
STAMP_NAME = "installed.json"
EXE_NAME = "llama-server.exe" if sys.platform == "win32" else "llama-server"

# 1 MiB read chunks: large enough that the loop overhead disappears next to the
# network, small enough that progress stays smooth on a 373 MB archive.
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


def select_variant() -> str:
    """Pick the best variant for this machine.

    Raises UnsupportedPlatformError when no build is pinned for the platform,
    which is the honest answer until the macOS/Linux entries are verified.
    """
    machine = platform.machine().lower()
    if sys.platform != "win32" or machine not in ("amd64", "x86_64"):
        raise UnsupportedPlatformError(
            f"No pinned llama.cpp build for {sys.platform}/{platform.machine()} "
            f"yet (Windows x64 only so far). Download a build from "
            f"{RELEASE_PAGE_URL.format(tag=RELEASE_TAG)} and point "
            f"llama_server_path at it, or add the variant to VARIANTS.")

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
    if size is None:
        return "unknown size"
    return f"{size / (1024 * 1024):.1f} MB"


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

    Only .zip is handled: the pinned variants are Windows-only. macOS and Linux
    releases ship .tar.gz, so this grows a branch when those variants land.
    """
    if archive.suffix.lower() != ".zip":
        raise LlamaServerFetchError(
            f"Don't know how to unpack {archive.name} (only .zip is supported "
            f"while the variant table is Windows-only).")
    root = into.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            # Guard against archive members that escape the target directory.
            # ZipFile.extract already strips absolute paths and '..', but an
            # explicit check keeps that guarantee visible and local.
            if not (root / member.filename).resolve().is_relative_to(root):
                raise LlamaServerFetchError(
                    f"Refusing to unpack {archive.name}: member "
                    f"{member.filename!r} points outside the target directory.")
        zf.extractall(into)


def _payload_root(extracted: Path) -> Path:
    """Directory inside an unpacked archive whose contents we actually want.

    llama.cpp Windows archives keep llama-server.exe and every DLL it needs
    side by side, but the layout has moved between releases (flat root in some,
    a single nested folder in others). Anchoring on the binary makes the
    unpacking survive that; the cudart archive has no binary and falls through
    to the single-directory / flat-root cases.
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
            f"Could not run {exe.name} {' '.join(args)}: {exc}. On Windows this "
            f"usually means a missing Visual C++ runtime "
            f"(https://aka.ms/vs/17/release/vc_redist.x64.exe).") from exc
    return (result.stdout or "") + (result.stderr or "")


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
        raise LlamaServerFetchError(
            f"{variant_name} was installed but no matching device appeared in "
            f"--list-devices, so llama-server would silently run on the CPU "
            f"(about three times slower). Output was:\n{output.strip()}\n"
            f"Check that the cudart DLLs sit next to {exe.name} and that their "
            f"major version matches the build.")
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
    """
    variant_name = variant or select_variant()
    if variant_name not in VARIANTS:
        raise LlamaServerFetchError(
            f"Unknown variant {variant_name!r}. Known: {', '.join(sorted(VARIANTS))}.")
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
    print("Assets  :")
    for asset in spec.assets:
        print(f"    {asset.name.format(tag=tag)}")
        print(f"        {DOWNLOAD_URL.format(tag=tag, asset=asset.name.format(tag=tag))}")
        print(f"        sha256 {asset.sha256}")
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
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.list:
        print(f"Pinned release: {RELEASE_TAG}")
        for name, spec in sorted(VARIANTS.items()):
            need = (f"driver CUDA >= {spec.min_driver_cuda[0]}.{spec.min_driver_cuda[1]}"
                    if spec.min_driver_cuda else "no GPU requirement")
            print(f"  {name}  ({need}, {len(spec.assets)} asset(s))")
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
