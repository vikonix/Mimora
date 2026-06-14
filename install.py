#!/usr/bin/env python3
"""EchoLoop installer.

A standalone, idempotent setup helper that walks through everything needed to
run EchoLoop on a fresh machine:

  1. Verify the Python version.
  2. Detect an NVIDIA GPU / CUDA version (via nvidia-smi, no extra packages).
  3. (GPU only) install torch/torchaudio and llama-cpp-python as CUDA builds.
  4. pip install the project requirements (root file pulls in the subprojects).
  5. Check for the native espeak-ng binary (optionally help install it).
  6. Pre-download the Hugging Face models into model_cache/.
  7. Download the GGUF chat model into models/.
  8. Run hwconfig/detect_hardware.py to write hardware_config.json.

Design notes
------------
* Every step prints exactly what it will do (including the precise command)
  and asks for confirmation before running. Answer Y to run, n to abort the
  whole installer, or s to skip just that step. Use --yes to auto-confirm and
  --dry-run to print the steps without executing anything.
* If a step's target is already installed/present, the installer does NOT
  silently redo it: it says so and asks reinstall vs. skip (defaulting to
  skip). Under --yes such steps are skipped unless --reinstall is also given.
* GPU detection deliberately relies only on `nvidia-smi`, because
  detect_hardware.py imports torch/llama-cpp (which may not be installed yet) —
  a classic bootstrap chicken-and-egg. detect_hardware.py is run at the very
  end, once those packages exist.
* The GPU CUDA wheels are installed before the requirements file so that the
  `llama-cpp-python` / `torch` constraints in requirements.txt are already
  satisfied — otherwise pip would try to source-build a CPU llama-cpp-python
  (no PyPI wheels on recent versions) only for it to be replaced afterwards.
* Packages install into the interpreter that runs this script (sys.executable);
  the script does not create a venv. It checks up front whether it is inside a
  virtual environment and, if not, warns and asks before installing globally
  (and refuses outright under --yes). Activate the project's .venv first.
* The whole run is mirrored to logs/install.log.

Run:  python install.py            (interactive)
      python install.py --yes      (no prompts)
      python install.py --dry-run  (preview only)
      python install.py --cpu      (skip all GPU-specific installs)
"""

from __future__ import annotations

import argparse
import importlib.metadata as ilmeta
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
# Logs live in the project's logs/ dir alongside main.log / llm_server.log.
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "install.log"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_CACHE_DIR = PROJECT_ROOT / "model_cache"
DETECT_HW_SCRIPT = PROJECT_ROOT / "hwconfig" / "detect_hardware.py"

MIN_PYTHON = (3, 11)  # matches requires-python in pyproject.toml
# Highest Python minor we have verified has prebuilt wheels for every
# dependency. Newer interpreters are NOT blocked (no upper version gate) — they
# only get a warning (see step_check_python). Because the requirements install
# is binary-only, a missing wheel on such an interpreter fails loudly with a
# "no matching distribution" error instead of silently source-building.
WHEEL_TESTED_MAX = (3, 12)

# Dependencies that publish no wheels (pure-Python, sdist only). The
# requirements step installs binary-only to stop an unsupported interpreter
# from silently compiling a package that lacks a wheel; these few must be
# exempted or the install would fail on them on every Python version.
SOURCE_ONLY_PACKAGES = ("fastdtw", "docopt")

# The GGUF chat model. Default filename matches EXTERNAL_MODEL_PATH in
# echoloop/config.py, so the app finds it without any settings change.
GGUF_REPO_ID = "hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF"
GGUF_FILENAME = "llama-3.2-3b-instruct-q4_k_m.gguf"

# Hugging Face models EchoLoop pulls on first run; pre-fetching them here means
# the first launch is offline-ready. Repo ids match what the app requests.
HF_MODEL_REPOS = [
    ("facebook/wav2vec2-large-960h", "Wav2Vec2 (pronunciation analysis, ~1.2 GB)"),
    ("hexgrad/Kokoro-82M", "Kokoro-82M (text-to-speech)"),
    ("Systran/faster-whisper-small", "faster-whisper small (speech-to-text)"),
]

# CUDA wheel series, newest first. We pick the newest series whose CUDA version
# is not greater than the one the driver reports (CUDA 12.x is forward
# compatible at runtime, so e.g. a cu124 build runs fine on a 12.8 driver).
#
# torch publishes its own indexes (up to cu128); llama-cpp-python's prebuilt
# wheels (abetlen's index) currently top out at cu124. They are independent
# stacks, so the two lists differ on purpose.
TORCH_CU_SERIES = ["cu128", "cu126", "cu124", "cu121", "cu118"]
LLAMA_CU_SERIES = ["cu124", "cu123", "cu122", "cu121"]

TORCH_INDEX_URL = "https://download.pytorch.org/whl/{series}"
LLAMA_INDEX_URL = "https://abetlen.github.io/llama-cpp-python/whl/{series}"

PIP = [sys.executable, "-m", "pip"]

# Distribution names (PyPI names, not import names) that requirements.txt and
# its subproject files install. Used to detect whether the dependency step has
# already run. Names with dashes (faster-whisper, scikit-learn, phonemizer-fork,
# python-Levenshtein) are the distribution names importlib.metadata expects.
REQUIRED_DISTS = [
    "numpy", "soundfile", "sounddevice", "faster-whisper", "kokoro", "openai",
    "torch", "transformers", "fastapi", "uvicorn", "llama-cpp-python",
    "torchaudio", "librosa", "scipy", "scikit-learn", "fastdtw",
    "phonemizer-fork", "python-Levenshtein",
]


# ---------------------------------------------------------------------------
# "Already installed?" detection (no heavy imports — metadata only)
# ---------------------------------------------------------------------------

def dist_version(name: str) -> str | None:
    """Installed version of a distribution, or None if it is not installed."""
    try:
        return ilmeta.version(name)
    except ilmeta.PackageNotFoundError:
        return None


def is_installed(name: str) -> bool:
    return dist_version(name) is not None


def torch_is_cuda_build() -> bool:
    """True only if torch is installed as a CUDA wheel.

    CUDA wheels carry a local version tag like '2.5.1+cu124'; CPU builds are
    plain '2.5.1' or '2.5.1+cpu'. This avoids importing torch (slow/heavy).
    """
    version = dist_version("torch")
    return bool(version and "+cu" in version)


def all_requirements_installed() -> bool:
    return all(is_installed(name) for name in REQUIRED_DISTS)


def hf_repo_fully_cached(repo_id: str) -> bool:
    """True only if a COMPLETE snapshot of the repo is in the local HF cache.

    A folder existing under model_cache/hub/ is not enough: an interrupted run
    can leave a partial snapshot (missing files). snapshot_download in offline
    mode returns the path only when every file of the recorded revision is
    present, and raises otherwise — so partial downloads are correctly reported
    as not-installed and will be re-offered. Requires HF_HOME to be set first.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        # huggingface_hub not installed yet → nothing can be cached.
        return False
    try:
        snapshot_download(repo_id=repo_id, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001 — any miss/partial means "not fully cached"
        return False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class Logger:
    """Writes to both stdout and logs/install.log (append-mode, line-buffered)."""

    def __init__(self, path: Path):
        # Ensure logs/ exists, then append (keeps a history across runs).
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8", buffering=1)

    def log(self, message: str = "") -> None:
        print(message)
        self._fh.write(message + "\n")

    def banner(self, title: str) -> None:
        line = "=" * 70
        self.log("")
        self.log(line)
        self.log(title)
        self.log(line)


# ---------------------------------------------------------------------------
# Step result tracking (for the final summary)
# ---------------------------------------------------------------------------

# Status strings kept human-readable because they go straight into the summary.
DONE = "done"
SKIPPED = "skipped"
FAILED = "failed"
MANUAL = "needs manual action"


class InstallError(RuntimeError):
    """A step failed. Raised to abort the installer immediately (fail-fast).

    Continuing past a failed step leaves a half-built environment and lets a
    misleading "all done" message print at the end, so any hard failure stops
    the run instead. The argument is the step name, used in the abort message.
    """


class StepReport:
    """Collects (name, status, note) tuples for an end-of-run summary."""

    def __init__(self):
        self._rows: list[tuple[str, str, str]] = []

    def add(self, name: str, status: str, note: str = "") -> None:
        self._rows.append((name, status, note))

    def render(self) -> str:
        width = max((len(name) for name, _, _ in self._rows), default=0)
        lines = []
        for name, status, note in self._rows:
            suffix = f" — {note}" if note else ""
            lines.append(f"  {name.ljust(width)}  {status}{suffix}")
        return "\n".join(lines)

    def statuses(self) -> list[str]:
        """All recorded status strings (used to decide the closing message)."""
        return [status for _, status, _ in self._rows]


# ---------------------------------------------------------------------------
# User interaction
# ---------------------------------------------------------------------------

class Confirmer:
    """Per-step confirmation honoring --yes and --dry-run."""

    def __init__(self, log: Logger, assume_yes: bool, dry_run: bool,
                 force_reinstall: bool):
        self._log = log
        self._assume_yes = assume_yes
        self._dry_run = dry_run
        self._force_reinstall = force_reinstall

    def confirm(self, description: str, command: str | None = None, *,
                installed: bool = False) -> bool:
        """Announce a step and ask whether to run it.

        Returns True to run, False to skip. Aborts the whole installer (raises
        SystemExit) if the user answers 'n'.

        When ``installed`` is True the step's target is already present, so the
        prompt offers reinstall vs. skip (defaulting to skip) instead of the
        usual run vs. skip.
        """
        self._log.log("")
        self._log.log(f">>> {description}")
        if command:
            self._log.log(f"    command: {command}")
        if installed:
            self._log.log("    NOTE: already installed / present.")

        if self._dry_run:
            self._log.log("    [dry-run] not executed")
            return False

        if installed:
            return self._prompt_installed()
        return self._prompt_fresh()

    def _prompt_fresh(self) -> bool:
        """Not yet installed: default Yes, 's' skips, 'n' aborts."""
        if self._assume_yes:
            self._log.log("    [--yes] proceeding")
            return True
        while True:
            answer = input("    Proceed? [Y]es / [n]o-abort / [s]kip: ").strip().lower()
            if answer in ("", "y", "yes"):
                return True
            if answer in ("s", "skip"):
                self._log.log("    skipped by user")
                return False
            if answer in ("n", "no"):
                self._log.log("    aborted by user")
                raise SystemExit(1)
            print("    Please answer Y, n, or s.")

    def _prompt_installed(self) -> bool:
        """Already installed: default Skip, 'r' reinstalls, 'n' aborts."""
        if self._assume_yes:
            if self._force_reinstall:
                self._log.log("    [--yes --reinstall] reinstalling")
                return True
            self._log.log("    [--yes] already installed -> skipping")
            return False
        while True:
            answer = input("    Already installed. [r]einstall / [S]kip / "
                           "[n]o-abort: ").strip().lower()
            if answer in ("", "s", "skip"):
                self._log.log("    kept existing (skipped)")
                return False
            if answer in ("r", "reinstall"):
                return True
            if answer in ("n", "no"):
                self._log.log("    aborted by user")
                raise SystemExit(1)
            print("    Please answer r, s, or n.")

    def warn_continue(self, lines: list[str]) -> bool:
        """Show a warning and ask whether to continue or abort the installer.

        Unlike confirm(), there is no 'skip': the choice is proceed or stop.
        Honors --yes and --dry-run by proceeding (the warning is still logged).
        Returns True to continue; raises SystemExit(1) if the user aborts.
        """
        self._log.log("")
        for line in lines:
            self._log.log(f"    WARNING: {line}")

        if self._assume_yes or self._dry_run:
            note = "[--yes]" if self._assume_yes else "[dry-run]"
            self._log.log(f"    {note} continuing despite warning")
            return True

        # No default: an empty answer re-prompts. The choice is consequential
        # (the install may fail), so require an explicit continue or abort.
        while True:
            answer = input("    Continue anyway? [c]ontinue / [a]bort: "
                           ).strip().lower()
            if answer in ("c", "continue"):
                return True
            if answer in ("a", "abort"):
                self._log.log("    aborted by user")
                raise SystemExit(1)
            print("    Please answer c or a.")


# ---------------------------------------------------------------------------
# Command execution
# ---------------------------------------------------------------------------

def run_command(cmd: list[str], log: Logger) -> bool:
    """Run a subprocess, streaming combined output live to console and log.

    Output is read line-by-line as it is produced (so a long pip install shows
    progress in real time) and mirrored into logs/install.log. Returns True on exit
    code 0, False otherwise. Never raises on a non-zero exit — the caller
    decides how a failure affects the rest of the run.
    """
    log.log(f"    $ {' '.join(cmd)}")
    try:
        # Merge stderr into stdout and read incrementally. line-buffered text
        # mode keeps memory flat regardless of how much the command prints.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT),
        )
    except FileNotFoundError as exc:
        log.log(f"    ERROR: command not found: {exc}")
        return False

    # proc.stdout is guaranteed non-None given stdout=PIPE above.
    assert proc.stdout is not None
    for line in proc.stdout:
        log.log(f"    | {line.rstrip()}")
    returncode = proc.wait()

    if returncode != 0:
        log.log(f"    -> exit code {returncode}")
        return False
    return True


def run_or_fail(cmd: list[str], log: Logger, report: StepReport,
                step_name: str, note: str = "") -> None:
    """Run a command, record DONE/FAILED, and abort the installer on failure.

    This is the fail-fast wrapper around run_command: a non-zero exit records a
    FAILED row and raises InstallError so no later step runs against a broken
    environment. Callers that must keep going on failure should use run_command
    directly instead.
    """
    ok = run_command(cmd, log)
    report.add(step_name, DONE if ok else FAILED, note)
    if not ok:
        raise InstallError(step_name)


# ---------------------------------------------------------------------------
# GPU / CUDA detection (no third-party packages required)
# ---------------------------------------------------------------------------

def detect_gpu(log: Logger) -> tuple[str | None, tuple[int, int] | None]:
    """Return (gpu_name, cuda_version) using nvidia-smi only.

    Both are None when no NVIDIA GPU / nvidia-smi is found. cuda_version is the
    maximum CUDA the installed driver supports, parsed from the smi header.
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        log.log("    nvidia-smi not found — treating this machine as CPU-only.")
        return None, None

    try:
        out = subprocess.run(
            [smi], capture_output=True, text=True, timeout=15
        ).stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.log(f"    nvidia-smi failed ({exc}) — treating as CPU-only.")
        return None, None

    # Driver's max supported CUDA appears in the header: "CUDA Version: 12.8".
    cuda_version: tuple[int, int] | None = None
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    if match:
        cuda_version = (int(match.group(1)), int(match.group(2)))

    # GPU name comes from a dedicated query (robust across smi layouts).
    name = None
    try:
        name_out = subprocess.run(
            [smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        name = name_out.splitlines()[0].strip() if name_out else None
    except (OSError, subprocess.TimeoutExpired):
        pass

    log.log(f"    Detected GPU: {name or 'unknown NVIDIA GPU'}")
    log.log(f"    Driver CUDA : {'.'.join(map(str, cuda_version)) if cuda_version else 'unknown'}")
    return name, cuda_version


def _series_to_version(series: str) -> tuple[int, int]:
    """'cu124' -> (12, 4); 'cu118' -> (11, 8)."""
    digits = series[2:]
    return int(digits[:-1]), int(digits[-1])


def pick_cu_series(
    available: list[str], driver_cuda: tuple[int, int] | None
) -> str | None:
    """Newest series whose CUDA version is <= the driver's CUDA version.

    `available` is ordered newest-first. When the driver CUDA is unknown we
    optimistically pick the newest series (it usually works and the user can
    re-run with --cpu if not).
    """
    if driver_cuda is None:
        return available[0]
    for series in available:  # newest first
        if _series_to_version(series) <= driver_cuda:
            return series
    return None  # driver too old for any prebuilt wheel we know about


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def find_local_venv_name() -> str:
    """Name of a virtual-env folder in the project root, for the activate hint.

    A real venv always contains a 'pyvenv.cfg', so we look for any immediate
    subdirectory that has one (covers '.venv', 'venv', 'env', custom names).
    Falls back to '.venv' when none is found.
    """
    common = [".venv", "venv", "env", ".env"]
    # Check the usual names first, then scan for any dir with a pyvenv.cfg.
    for name in common:
        if (PROJECT_ROOT / name / "pyvenv.cfg").exists():
            return name
    try:
        for child in PROJECT_ROOT.iterdir():
            if child.is_dir() and (child / "pyvenv.cfg").exists():
                return child.name
    except OSError:
        pass
    return ".venv"


def check_virtualenv(log: Logger, args: argparse.Namespace) -> None:
    """Warn (and optionally abort) when not running inside a virtual env.

    Packages are installed into whatever interpreter runs this script
    (sys.executable). Running it with the system Python would pollute the
    global site-packages, so we detect a venv/virtualenv/conda env and, when
    absent, make the user confirm before continuing.
    """
    log.banner("Step 0/8 — Environment check")
    in_venv = (sys.prefix != sys.base_prefix
               or bool(os.environ.get("CONDA_PREFIX")))
    log.log(f"    Interpreter: {sys.executable}")

    if in_venv:
        log.log("    Running inside a virtual environment — packages stay local.")
        return

    log.log("    WARNING: NOT running inside a virtual environment.")
    log.log("    Packages would be installed into the GLOBAL Python above.")
    venv_name = find_local_venv_name()
    if sys.platform == "win32":
        log.log(f"    Activate the project venv first:  {venv_name}\\Scripts\\activate")
    else:
        log.log(f"    Activate the project venv first:  source {venv_name}/bin/activate")
    log.log("    Then re-run:  python install.py")

    if args.dry_run:
        log.log("    [dry-run] continuing anyway (nothing is installed).")
        return
    if args.yes:
        log.log("    [--yes] refusing to install globally; aborting. "
                "Activate a venv or run interactively to override.")
        raise SystemExit(1)

    answer = input("    Install into this GLOBAL interpreter anyway? "
                   "[y]es / [N]o-abort: ").strip().lower()
    if answer not in ("y", "yes"):
        log.log("    Aborted — activate a virtual environment and re-run.")
        raise SystemExit(1)
    log.log("    Proceeding with the global interpreter at the user's request.")


def step_check_python(
    log: Logger, confirmer: Confirmer, report: StepReport
) -> None:
    """Gate the interpreter: hard-fail below the minimum, warn above the tested
    maximum (no upper version block — newer Pythons may just lack wheels)."""
    log.banner("Step 1/8 — Python version")
    current = sys.version_info[:2]
    log.log(f"    Running Python {platform.python_version()} ({sys.executable})")
    if current < MIN_PYTHON:
        need = ".".join(map(str, MIN_PYTHON))
        log.log(f"    ERROR: EchoLoop needs Python >= {need}. Aborting.")
        report.add("Python version", FAILED, f"need >= {need}")
        raise InstallError("Python version")
    if current > WHEEL_TESTED_MAX:
        tested = ".".join(map(str, WHEEL_TESTED_MAX))
        confirmer.warn_continue([
            f"Python {platform.python_version()} is newer than the latest "
            f"version EchoLoop is tested against ({tested}).",
            "Prebuilt wheels may not exist yet for some dependencies on this "
            "interpreter. Installs are binary-only, so the dependency step may "
            "stop with a 'no matching distribution' error.",
        ])
    report.add("Python version", DONE, platform.python_version())


def step_install_requirements(
    log: Logger, confirmer: Confirmer, report: StepReport
) -> None:
    """Install the project requirements; the root file chains the subprojects."""
    log.banner("Step 4/8 — Project dependencies")
    if not REQUIREMENTS.exists():
        log.log(f"    ERROR: {REQUIREMENTS} not found. Aborting.")
        report.add("pip requirements", FAILED, "requirements.txt missing")
        raise InstallError("pip requirements")

    installed = all_requirements_installed()
    # Binary-only install: if pip can't find a prebuilt wheel for the current
    # interpreter it fails with "no matching distribution" instead of quietly
    # downloading an sdist and compiling it (which is what happened on an
    # untested Python and triggered a numpy source build). The few sdist-only
    # packages are exempted so they can still build.
    cmd = PIP + ["install", "-r", str(REQUIREMENTS),
                 "--only-binary", ":all:",
                 "--no-binary", ",".join(SOURCE_ONLY_PACKAGES)]
    desc = ("Install all Python dependencies (requirements.txt also pulls in "
            "llm_server/ and pronounce/ requirements).")
    if installed:
        log.log("    All expected dependencies are already installed.")
    if not confirmer.confirm(desc, " ".join(cmd), installed=installed):
        report.add("pip requirements", SKIPPED,
                   "already installed" if installed else "")
        return
    run_or_fail(cmd, log, report, "pip requirements")


def step_gpu_torch(
    log: Logger, confirmer: Confirmer, report: StepReport,
    driver_cuda: tuple[int, int] | None,
) -> None:
    """Reinstall torch + torchaudio as a matching CUDA build (together)."""
    series = pick_cu_series(TORCH_CU_SERIES, driver_cuda)
    if series is None:
        log.log("    No compatible torch CUDA wheel series for this driver.")
        report.add("torch (CUDA)", MANUAL, "see pytorch.org/get-started/locally")
        return

    index = TORCH_INDEX_URL.format(series=series)
    # torch and torchaudio are reinstalled together on purpose: replacing torch
    # alone leaves torchaudio built against the old torch (import crashes).
    cmd = PIP + ["install", "--force-reinstall", "torch", "torchaudio",
                 "--index-url", index]
    installed = torch_is_cuda_build()
    desc = (f"Install CUDA build of torch + torchaudio for {series} "
            f"(used by Wav2Vec2 pronunciation analysis).")
    if installed:
        log.log(f"    torch already a CUDA build ({dist_version('torch')}).")
    if not confirmer.confirm(desc, " ".join(cmd), installed=installed):
        report.add("torch (CUDA)", SKIPPED,
                   "already CUDA build" if installed else "")
        return
    run_or_fail(cmd, log, report, "torch (CUDA)", series)


def step_gpu_llama(
    log: Logger, confirmer: Confirmer, report: StepReport,
    driver_cuda: tuple[int, int] | None,
) -> None:
    """Install the prebuilt CUDA wheel of llama-cpp-python (newest available)."""
    series = pick_cu_series(LLAMA_CU_SERIES, driver_cuda)
    if series is None:
        log.log("    No prebuilt llama-cpp-python CUDA wheel for this driver.")
        log.log("    Build manually with CUDA: set CMAKE_ARGS=-DGGML_CUDA=on then")
        log.log("    pip install llama-cpp-python --force-reinstall --no-cache-dir")
        report.add("llama-cpp-python (CUDA)", MANUAL, "no prebuilt wheel")
        return

    index = LLAMA_INDEX_URL.format(series=series)
    # No version pin: pip picks the newest wheel published in this index.
    # --extra-index-url (not --index-url) keeps PyPI available for deps.
    cmd = PIP + ["install", "--upgrade", "--force-reinstall", "--no-cache-dir",
                 "llama-cpp-python", "--extra-index-url", index]
    # Presence-only check: metadata can't tell a CPU build from a CUDA one, so
    # any installed llama-cpp-python triggers the reinstall/skip prompt.
    installed = is_installed("llama-cpp-python")
    desc = (f"Install prebuilt CUDA wheel of llama-cpp-python from the {series} "
            f"index (newest version available there).")
    if installed:
        log.log(f"    llama-cpp-python already installed "
                f"({dist_version('llama-cpp-python')}).")
    if not confirmer.confirm(desc, " ".join(cmd), installed=installed):
        report.add("llama-cpp-python (CUDA)", SKIPPED,
                   "already installed" if installed else "")
    else:
        run_or_fail(cmd, log, report, "llama-cpp-python (CUDA)", series)

    # On Windows with no system CUDA toolkit, the CUDA runtime DLLs come from
    # these pip packages; detect_hardware.py / llm_server expect them on PATH.
    if sys.platform == "win32":
        runtime_cmd = PIP + ["install",
                             "nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12"]
        runtime_present = (is_installed("nvidia-cuda-runtime-cu12")
                           and is_installed("nvidia-cublas-cu12"))
        rdesc = ("Install CUDA runtime DLLs (nvidia-cuda-runtime-cu12, "
                 "nvidia-cublas-cu12) so llama-cpp can load on Windows.")
        if confirmer.confirm(rdesc, " ".join(runtime_cmd),
                             installed=runtime_present):
            run_or_fail(runtime_cmd, log, report, "CUDA runtime DLLs")


def step_espeak(log: Logger, confirmer: Confirmer, report: StepReport) -> None:
    """Ensure the native espeak-ng binary exists; offer to install it."""
    log.banner("Step 5/8 — espeak-ng (native binary for phonemizer)")
    if shutil.which("espeak-ng") or shutil.which("espeak"):
        log.log("    espeak-ng found on PATH.")
        report.add("espeak-ng", DONE, "already present")
        return

    log.log("    espeak-ng NOT found on PATH (required by the phoneme analyzer).")
    system = platform.system()

    if system == "Linux":
        pkg_cmd = _linux_espeak_command()
        if pkg_cmd:
            if confirmer.confirm("Install espeak-ng via the system package "
                                 "manager (needs sudo).", " ".join(pkg_cmd)):
                run_or_fail(pkg_cmd, log, report, "espeak-ng")
                return
            report.add("espeak-ng", SKIPPED)
            return
        log.log("    Could not detect a supported package manager.")

    elif system == "Darwin":
        if shutil.which("brew"):
            brew_cmd = ["brew", "install", "espeak-ng"]
            if confirmer.confirm("Install espeak-ng via Homebrew.",
                                 " ".join(brew_cmd)):
                run_or_fail(brew_cmd, log, report, "espeak-ng")
                return
            report.add("espeak-ng", SKIPPED)
            return
        log.log("    Homebrew not found. Install it from https://brew.sh first.")

    else:  # Windows and anything else: instructions only.
        log.log("    Windows: download and run the installer from")
        log.log("      https://github.com/espeak-ng/espeak-ng/releases")
        log.log("    Then ensure espeak-ng is on PATH (you may also need to set")
        log.log("    PHONEMIZER_ESPEAK_LIBRARY to the installed libespeak-ng DLL).")

    report.add("espeak-ng", MANUAL, "install separately, see log")


def _linux_espeak_command() -> list[str] | None:
    """Build the right install command for the detected Linux package manager."""
    if shutil.which("apt-get"):
        return ["sudo", "apt-get", "install", "-y", "espeak-ng"]
    if shutil.which("dnf"):
        return ["sudo", "dnf", "install", "-y", "espeak-ng"]
    if shutil.which("pacman"):
        return ["sudo", "pacman", "-S", "--noconfirm", "espeak-ng"]
    return None


def configure_hf_symlink_fallback(log: Logger) -> None:
    """On Windows without symlink privilege, make HF copy files instead.

    huggingface_hub's cache normally points snapshots/ at blobs/ via symlinks.
    Creating a symlink on Windows needs Developer Mode or admin rights; without
    them the native hf-xet downloader fails hard with WinError 1314 (it does not
    fall back to copying the way the pure-Python path does). Detect the missing
    privilege here and, when absent, disable hf-xet so downloads fall back to
    plain file copies (uses more disk, but always works). Must run before
    huggingface_hub is first imported.
    """
    if sys.platform != "win32":
        return
    MODEL_CACHE_DIR.mkdir(exist_ok=True)

    import tempfile
    supported = True
    try:
        with tempfile.TemporaryDirectory(dir=MODEL_CACHE_DIR) as tmp:
            src = Path(tmp) / "probe_src"
            src.touch()
            try:
                os.symlink(src, Path(tmp) / "probe_dst")
            except OSError:
                supported = False
    except OSError:
        supported = False

    if supported:
        log.log("    Symlink support: OK.")
        return

    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    log.log("    Symlinks unavailable (no Developer Mode / admin): HF downloads")
    log.log("    will COPY into the cache instead of symlinking (more disk use).")
    log.log("    Tip: enabling Windows Developer Mode lets HF use symlinks.")


def step_prefetch_models(
    log: Logger, confirmer: Confirmer, report: StepReport
) -> None:
    """Download the Hugging Face models into model_cache/ (HF_HOME)."""
    log.banner("Step 6/8 — Pre-download Hugging Face models")

    # Match echoloop/config.py: HF_HOME points at the project's model_cache/.
    MODEL_CACHE_DIR.mkdir(exist_ok=True)
    os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)
    # Avoid the Windows symlink-privilege crash (WinError 1314) before any
    # huggingface_hub import happens below.
    configure_hf_symlink_fallback(log)

    repos = ", ".join(repo for repo, _ in HF_MODEL_REPOS)
    installed = all(hf_repo_fully_cached(repo) for repo, _ in HF_MODEL_REPOS)
    desc = (f"Download HF models into {MODEL_CACHE_DIR.name}/ (HF_HOME): {repos}. "
            f"Several GB; already-cached files are reused.")
    if installed:
        log.log("    All three model repos already present in the cache.")
    if not confirmer.confirm(desc, installed=installed):
        report.add("HF model cache", SKIPPED,
                   "already cached" if installed else "")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log.log("    huggingface_hub not installed (did the deps step run?).")
        report.add("HF model cache", FAILED, "huggingface_hub missing")
        raise InstallError("HF model cache")

    all_ok = True
    for repo_id, label in HF_MODEL_REPOS:
        log.log(f"    Fetching {label} [{repo_id}] ...")
        try:
            snapshot_download(repo_id=repo_id)
            log.log(f"    -> done: {repo_id}")
        except Exception as exc:  # noqa: BLE001 — record which repo failed
            all_ok = False
            log.log(f"    -> FAILED: {repo_id}: {exc}")
    report.add("HF model cache", DONE if all_ok else FAILED)
    if not all_ok:
        raise InstallError("HF model cache")


def step_download_gguf(
    log: Logger, confirmer: Confirmer, report: StepReport
) -> None:
    """Download the GGUF chat model into models/ if not already present."""
    log.banner("Step 7/8 — GGUF chat model")
    MODELS_DIR.mkdir(exist_ok=True)
    target = MODELS_DIR / GGUF_FILENAME

    installed = target.exists()
    desc = (f"Download {GGUF_FILENAME} (~2 GB) from {GGUF_REPO_ID} into "
            f"{MODELS_DIR.name}/.")
    if installed:
        log.log(f"    Already present: {target}")
    if not confirmer.confirm(desc, installed=installed):
        report.add("GGUF model", SKIPPED,
                   "already downloaded" if installed else "")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log.log("    huggingface_hub not installed (did the deps step run?).")
        report.add("GGUF model", FAILED, "huggingface_hub missing")
        raise InstallError("GGUF model")

    try:
        # local_dir=models/ places the file exactly where config.py expects it
        # (rather than inside the HF cache structure).
        path = hf_hub_download(
            repo_id=GGUF_REPO_ID, filename=GGUF_FILENAME,
            local_dir=str(MODELS_DIR),
        )
        log.log(f"    -> downloaded: {path}")
        report.add("GGUF model", DONE)
    except Exception as exc:  # noqa: BLE001
        log.log(f"    -> FAILED: {exc}")
        report.add("GGUF model", FAILED)
        raise InstallError("GGUF model")


def step_detect_hardware(
    log: Logger, confirmer: Confirmer, report: StepReport
) -> None:
    """Run detect_hardware.py last, when torch/llama-cpp are installed."""
    log.banner("Step 8/8 — Hardware detection (writes hardware_config.json)")
    if not DETECT_HW_SCRIPT.exists():
        log.log(f"    {DETECT_HW_SCRIPT} not found; skipping.")
        report.add("hardware detection", SKIPPED, "script missing")
        return

    cmd = [sys.executable, str(DETECT_HW_SCRIPT)]
    desc = ("Probe the machine and write hwconfig/hardware_config.json (the app "
            "reads GPU-tuned parameters from it).")
    if not confirmer.confirm(desc, " ".join(cmd)):
        report.add("hardware detection", SKIPPED)
        return
    run_or_fail(cmd, log, report, "hardware detection")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def finish(log: Logger, report: StepReport, *, success: bool) -> None:
    """Print the end-of-run summary.

    The "run main.py" hint is printed ONLY on a fully successful run, so a
    failed or aborted install never reads as ready to launch. A successful run
    that still has manual-action items says so before the launch hint.
    """
    log.banner("Summary")
    log.log(report.render())
    log.log("")
    log.log(f"    finished: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    if not success:
        log.log("    Installation INCOMPLETE — fix the error above and re-run")
        log.log("    install.py. Do NOT run main.py until it finishes cleanly.")
        return
    if MANUAL in report.statuses():
        log.log("    Some steps need manual action (see 'needs manual action'")
        log.log("    above) before EchoLoop will fully work.")
    log.log("    Next: run `python main.py` to start EchoLoop.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EchoLoop installer — installs dependencies and downloads "
                    "the model cache and the LLM model.",
    )
    parser.add_argument("-y", "--yes", action="store_true",
                        help="auto-confirm every step (non-interactive)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print each step and command without running it")
    parser.add_argument("--reinstall", action="store_true",
                        help="with --yes, reinstall even already-installed "
                             "components (default is to skip them)")
    parser.add_argument("--cpu", action="store_true",
                        help="skip all GPU-specific (CUDA) installs")
    parser.add_argument("--gpu", action="store_true",
                        help="force GPU steps even if no GPU is auto-detected")
    parser.add_argument("--skip-models", action="store_true",
                        help="skip the Hugging Face model pre-download")
    parser.add_argument("--skip-gguf", action="store_true",
                        help="skip the GGUF chat-model download")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = Logger(LOG_FILE)
    confirmer = Confirmer(log, assume_yes=args.yes, dry_run=args.dry_run,
                          force_reinstall=args.reinstall)
    report = StepReport()

    log.banner("EchoLoop installer")
    log.log(f"    started: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    log.log(f"    platform: {platform.platform()}")
    log.log(f"    args: {vars(args)}")
    log.log(f"    log file: {LOG_FILE}")

    # All steps run inside this guard: the first one to fail raises
    # InstallError, so the run stops immediately instead of pressing on with a
    # half-built environment and printing a misleading "ready to launch" line.
    try:
        # Step 0: refuse to silently install into the global interpreter.
        check_virtualenv(log, args)

        # Step 1: Python version (hard min gate; warn above the tested max).
        step_check_python(log, confirmer, report)

        # Step 2: GPU detection (informs steps 4a/4b; no packages needed).
        log.banner("Step 2/8 — GPU / CUDA detection")
        gpu_name, driver_cuda = detect_gpu(log)
        use_gpu = (gpu_name is not None or args.gpu) and not args.cpu
        if args.cpu:
            log.log("    --cpu given: GPU steps will be skipped.")
        elif use_gpu:
            log.log("    GPU steps will be offered.")
        else:
            log.log("    No GPU detected: GPU steps will be skipped "
                    "(use --gpu to force).")
        report.add("GPU detection", DONE,
                   gpu_name or ("forced" if args.gpu else "none"))

        # Step 3: GPU-specific CUDA builds, installed BEFORE requirements so the
        # llama-cpp-python / torch constraints in requirements.txt are already
        # satisfied (avoids a doomed CPU source-build of llama-cpp-python).
        if use_gpu:
            log.banner("Step 3/8 — GPU (CUDA) builds")
            step_gpu_torch(log, confirmer, report, driver_cuda)
            step_gpu_llama(log, confirmer, report, driver_cuda)
        else:
            report.add("torch (CUDA)", SKIPPED, "CPU-only")
            report.add("llama-cpp-python (CUDA)", SKIPPED, "CPU-only")

        # Step 4: project dependencies.
        step_install_requirements(log, confirmer, report)

        # Step 5: espeak-ng.
        step_espeak(log, confirmer, report)

        # Step 6: HF model cache.
        if args.skip_models:
            report.add("HF model cache", SKIPPED, "--skip-models")
        else:
            step_prefetch_models(log, confirmer, report)

        # Step 7: GGUF.
        if args.skip_gguf:
            report.add("GGUF model", SKIPPED, "--skip-gguf")
        else:
            step_download_gguf(log, confirmer, report)

        # Step 8: hardware detection (after torch/llama exist).
        step_detect_hardware(log, confirmer, report)
    except InstallError as exc:
        log.log("")
        log.log(f"    ABORTED: step '{exc}' failed — stopping the installer "
                f"so the error is not masked.")
        finish(log, report, success=False)
        return 1

    finish(log, report, success=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
