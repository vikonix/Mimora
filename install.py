#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev
"""Mimora installer.

A standalone, idempotent setup helper that walks through everything needed to
run Mimora on a fresh machine:

  1. Verify the Python version.
  2. Detect an NVIDIA GPU / CUDA version (via nvidia-smi, no extra packages).
  3. (GPU only) install torch/torchaudio as CUDA builds.
  4. pip install the project requirements (root file pulls in the subprojects).
  5. Check for the native espeak-ng binary (optionally help install it).
  6. Pre-download the Hugging Face models into model_cache/.
  7. Pre-download the Supertonic 3 TTS model into model_cache/supertonic3/
     (the Spanish TTS backend; kept outside the HF hub cache because the
     supertonic package uses its own cache directory).
  8. Install the LLM stack: the pinned llama-server binary into bin/llama/ and
     the GGUF chat model into models/.
  9. Run tools/detect_hardware.py to write config/hardware_config.json.
 10. Write run_mimora.bat / run_mimora.sh: one-click launchers that activate
     the project's virtual environment and run main.py.

Design notes
------------
* Every step prints exactly what it will do (including the precise command)
  and asks for confirmation before running. Answer Y to run, n to abort the
  whole installer, or s to skip just that step. Use --yes to auto-confirm and
  --dry-run to print the steps without executing anything.
* If a step's target is already installed/present, the installer does NOT
  silently redo it: it says so and asks reinstall vs. skip (defaulting to
  skip). Under --yes such steps are skipped unless --reinstall is also given.
* Nothing that can be downloaded is downloaded by this script itself: the
  three fetchers live in the package (mimora/model_fetch.py,
  mimora/gguf_fetch.py, mimora/llama_server_fetch.py) and the steps below are
  thin wrappers around them. The app needs the same code for its own first-run
  check, and a PyPI install has no install.py at all - see tasks/llama-cpp.md,
  step 2a. Their log output is bridged into logs/install.log (see _LogBridge).
* GPU detection deliberately relies only on `nvidia-smi`, because
  detect_hardware.py imports torch (which may not be installed yet) - a classic
  bootstrap chicken-and-egg. detect_hardware.py is run at the very end, once
  those packages exist.
* The CUDA torch wheels are installed before the requirements file so that the
  `torch` constraint in requirements.txt is already satisfied - otherwise pip
  would pull the CPU build from PyPI only for it to be replaced afterwards.
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
import importlib
import importlib.metadata as ilmeta
import logging
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
DETECT_HW_SCRIPT = PROJECT_ROOT / "tools" / "detect_hardware.py"
LAUNCHER_BAT = PROJECT_ROOT / "run_mimora.bat"
LAUNCHER_SH = PROJECT_ROOT / "run_mimora.sh"

MIN_PYTHON = (3, 11)  # matches requires-python in pyproject.toml
# Highest Python minor we have verified has prebuilt wheels for every
# dependency. Newer interpreters are NOT blocked (no upper version gate) - they
# only get a warning (see step_check_python). Because the requirements install
# is binary-only, a missing wheel on such an interpreter fails loudly with a
# "no matching distribution" error instead of silently source-building.
WHEEL_TESTED_MAX = (3, 12)

# Dependencies that publish no wheels (pure-Python, sdist only). The
# requirements step installs binary-only to stop an unsupported interpreter
# from silently compiling a package that lacks a wheel; these few must be
# exempted or the install would fail on them on every Python version.
# unicodecsv: pulled by panphon (phoneme engine); every panphon release depends
# on it and it ships sdist-only, so --only-binary makes the resolve impossible
# without this exemption. Pure-Python - builds from sdist with no compiler.
SOURCE_ONLY_PACKAGES = ("fastdtw", "docopt", "unicodecsv")

# CUDA wheel series, newest first. We pick the newest series whose CUDA version
# is not greater than the one the driver reports (CUDA 12.x is forward
# compatible at runtime, so e.g. a cu124 build runs fine on a 12.8 driver).
TORCH_CU_SERIES = ["cu128", "cu126", "cu124", "cu121", "cu118"]

TORCH_INDEX_URL = "https://download.pytorch.org/whl/{series}"

PIP = [sys.executable, "-m", "pip"]

# Distribution names (PyPI names, not import names) that requirements.txt and
# its subproject files install. Used to detect whether the dependency step has
# already run. Names with dashes (scikit-learn, phonemizer-fork,
# python-Levenshtein) are the distribution names importlib.metadata expects.
REQUIRED_DISTS = [
    "numpy", "soundfile", "sounddevice", "kokoro", "supertonic", "openai",
    "torch", "transformers",
    "torchaudio", "librosa", "scipy", "scikit-learn", "fastdtw",
    "phonemizer-fork", "python-Levenshtein", "panphon", "sentencepiece",
    "ttkbootstrap", "pillow", "onnxruntime", "wordfreq",
]


# ---------------------------------------------------------------------------
# "Already installed?" detection (no heavy imports - metadata only)
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


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# CSI escape sequences (ESC [ ... final byte) - what pip, tqdm and any coloured
# child process emit. The terminal consumes them; a text file does not, and
# logs/install.log is exactly the file users are asked to send in when an
# install goes wrong, so it is stripped on the way to disk only.
_ANSI_CSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


class Logger:
    """Writes to both stdout and logs/install.log (append-mode, line-buffered).

    The console copy keeps whatever escape sequences a child process produced
    (pip's colours are worth having); the file copy is stripped of them.
    """

    def __init__(self, path: Path):
        # Ensure logs/ exists, then append (keeps a history across runs).
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8", buffering=1)

    def log(self, message: str = "") -> None:
        print(message)
        self._fh.write(_ANSI_CSI_RE.sub("", message) + "\n")

    def banner(self, title: str) -> None:
        line = "=" * 70
        self.log("")
        self.log(line)
        self.log(title)
        self.log(line)


class _LogBridge(logging.Handler):
    """Forwards the fetcher modules' logging output into the installer's log.

    The download steps call functions in mimora/* instead of subprocesses, so
    run_command's "stream the child's output into logs/install.log" no longer
    applies to them. Without this bridge their progress and, worse, their
    diagnostics (checksum mismatches, the CUDA device probe) would appear on
    the console but never in the log file the user is asked to send in.
    """

    def __init__(self, logger: Logger):
        super().__init__(level=logging.INFO)
        self._logger = logger

    def emit(self, record: logging.LogRecord) -> None:
        self._logger.log(f"    | {record.getMessage()}")


def bridge_module_logging(logger: Logger) -> None:
    """Route mimora.* log records into *logger* (idempotent).

    Attached to the "mimora" parent logger, so every fetcher module is covered
    by one handler. Propagation is switched off because the root logger has no
    handler here: without it, logging's last-resort handler would print every
    WARNING and ERROR to stderr a second time, next to the copy this bridge
    already wrote to stdout and the log file.
    """
    mimora_logger = logging.getLogger("mimora")
    if any(isinstance(h, _LogBridge) for h in mimora_logger.handlers):
        return
    mimora_logger.setLevel(logging.INFO)
    mimora_logger.propagate = False
    mimora_logger.addHandler(_LogBridge(logger))


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
            suffix = f" - {note}" if note else ""
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

    def _ask(self, prompt: str) -> str:
        """Read one answer, making sure the question is actually visible.

        A preceding download shows a tqdm progress bar that writes to stderr and
        keeps the cursor on its own line (carriage-return based). A plain
        ``input()`` prompt then lands on that same line and looks invisible -- the
        installer seems to hang while it is really waiting for input. Flushing both
        streams and leading with a newline forces the question onto its own line.
        """
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            return input("\n" + prompt).strip().lower()
        except EOFError:
            # stdin is closed or redirected (CI, piped input) and the questions
            # can never be answered: abort cleanly instead of crashing with a
            # traceback on every prompt.
            self._log.log("    stdin closed (no TTY) - cannot prompt. "
                          "Use --yes for unattended installs. Aborting.")
            raise SystemExit(1)

    def _prompt_fresh(self) -> bool:
        """Not yet installed: default Yes, 's' skips, 'n' aborts."""
        if self._assume_yes:
            self._log.log("    [--yes] proceeding")
            return True
        while True:
            answer = self._ask("    Proceed? [Y]es / [n]o-abort / [s]kip: ")
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
            answer = self._ask("    Already installed. [S]kip / [r]einstall / "
                               "[n]o-abort: ")
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
            answer = self._ask("    Continue anyway? [c]ontinue / [a]bort: ")
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
    code 0, False otherwise. Never raises on a non-zero exit - the caller
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


def _import_fetcher(module_name: str, log: Logger, report: StepReport,
                    step_name: str):
    """Import one of the mimora/* download helpers, failing the step cleanly.

    The fetchers live in the package rather than in this script (see the design
    notes at the top), so every download step starts by importing one. Only
    mimora.llama_server_fetch is stdlib-only; the other two need
    huggingface_hub, so a failure here almost always means the requirements
    step was skipped or aborted.
    """
    # install.py is normally run from the project root, which already puts it
    # on sys.path; this also covers `python /somewhere/else/install.py`.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        log.log(f"    Could not import {module_name}: {exc}")
        log.log("    Run the project dependencies step (step 4) first.")
        report.add(step_name, FAILED, f"{module_name} unimportable")
        raise InstallError(step_name) from exc


# ---------------------------------------------------------------------------
# GPU / CUDA detection (no third-party packages required)
# ---------------------------------------------------------------------------

def detect_gpu(log: Logger) -> tuple[str | None, tuple[int, int] | None]:
    """Return (gpu_name, cuda_version) using nvidia-smi only.

    Both are None when no NVIDIA GPU / nvidia-smi is found. cuda_version is the
    maximum CUDA the installed driver supports and comes from
    mimora.llama_server_fetch.detect_driver_cuda() rather than from a private
    copy of the parsing: the smi header was renamed in the 610 drivers
    ("CUDA UMD Version" in place of "CUDA Version") and only that module knew
    about it, so this function used to report "unknown" on a perfectly healthy
    machine. The module is stdlib-only and side-effect-free on import, so it is
    safe here, before the requirements step; its own log lines reach
    logs/install.log through bridge_module_logging().
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        log.log("    nvidia-smi not found - treating this machine as CPU-only.")
        return None, None

    cuda_version: tuple[int, int] | None = None
    # install.py is normally run from the project root, which already puts it
    # on sys.path; this also covers `python /somewhere/else/install.py`.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from mimora.llama_server_fetch import detect_driver_cuda
    except ImportError as exc:
        # Not fatal, and deliberately not an InstallError: an unknown version
        # makes pick_cu_series fall back to the newest wheel series, which is
        # what this function did on an unparsable header anyway.
        log.log(f"    Could not import mimora.llama_server_fetch ({exc}) - "
                f"driver CUDA version stays unknown.")
    else:
        cuda_version = detect_driver_cuda()

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
    log.banner("Step 0 - Environment check")
    in_venv = (sys.prefix != sys.base_prefix
               or bool(os.environ.get("CONDA_PREFIX")))
    log.log(f"    Interpreter: {sys.executable}")

    if in_venv:
        log.log("    Running inside a virtual environment - packages stay local.")
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

    sys.stdout.flush()
    sys.stderr.flush()
    try:
        answer = input("\n    Install into this GLOBAL interpreter anyway? "
                       "[N]o-abort / [y]es: ").strip().lower()
    except EOFError:
        # stdin is closed or redirected (CI, piped input): the question can
        # never be answered, and the safe default is the same as answering "no".
        log.log("    stdin closed (no TTY) - cannot prompt. Aborting - activate "
                "a virtual environment and re-run.")
        raise SystemExit(1)
    if answer not in ("y", "yes"):
        log.log("    Aborted - activate a virtual environment and re-run.")
        raise SystemExit(1)
    log.log("    Proceeding with the global interpreter at the user's request.")


def step_check_vcredist(
    log: Logger, confirmer: Confirmer, report: StepReport
) -> None:
    """On Windows, verify the MSVC runtime DLLs that torch / llama-server need.

    torch's torch_python.dll and the llama-server binary both link against the
    Microsoft Visual C++ runtime (vcruntime140.dll, vcruntime140_1.dll,
    msvcp140.dll). A clean Windows install often lacks it, and the failure only
    surfaces at RUNTIME (import torch / starting the server), long after pip
    reports success. Loading the DLLs here turns that into an up-front,
    actionable message. We do NOT auto-install the redistributable: it needs an
    elevated GUI installer, which is out of scope for this pip-only setup.
    """
    log.banner("Step 0 - Visual C++ runtime (Windows)")
    if sys.platform != "win32":
        log.log("    Not Windows; the MSVC runtime check does not apply.")
        report.add("VC++ runtime", SKIPPED, "not Windows")
        return

    import ctypes
    missing = []
    for dll in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"):
        try:
            ctypes.WinDLL(dll)
        except OSError:
            missing.append(dll)

    if not missing:
        log.log("    MSVC runtime present (vcruntime140 / msvcp140).")
        report.add("VC++ runtime", DONE, "present")
        return

    log.log(f"    MISSING: {', '.join(missing)}")
    report.add("VC++ runtime", MANUAL, "install vc_redist.x64")
    # warn_continue lets the user abort to install it now, or proceed (e.g. under
    # --yes); the report already records the manual action either way.
    confirmer.warn_continue([
        f"Microsoft Visual C++ runtime DLL(s) not found: {', '.join(missing)}.",
        "Without the Visual C++ Redistributable (x64), torch and the "
        "llama-server binary fail to load at runtime (the llama-server "
        "verification and hardware-detection steps below would fail too).",
        "Install it, then re-run install.py:",
        "  https://aka.ms/vs/17/release/vc_redist.x64.exe",
    ])


def _linux_tkinter_command() -> list[str] | None:
    """Build the right install command for the detected Linux package manager.

    Mirrors _linux_espeak_command(). Debian/Ubuntu call the package
    python3-tk, Fedora python3-tkinter, Arch just tk.
    """
    if shutil.which("apt-get"):
        return ["sudo", "apt-get", "install", "-y", "python3-tk"]
    if shutil.which("dnf"):
        return ["sudo", "dnf", "install", "-y", "python3-tkinter"]
    if shutil.which("pacman"):
        return ["sudo", "pacman", "-S", "--noconfirm", "tk"]
    return None


def step_check_tkinter(
    log: Logger, confirmer: Confirmer, report: StepReport
) -> None:
    """Verify tkinter (main.py's GUI toolkit) is importable; offer to install it.

    On Linux, tkinter is packaged separately from the interpreter itself
    (python3-tk / python3-tkinter / tk) and pip cannot install it - a venv
    built from a system Python that lacks the package fails at `import
    tkinter` with a plain ModuleNotFoundError no matter how many pip
    requirements succeed. Checked here, early, for the same reason as the
    Windows VC++ runtime check: catch it before `python main.py` does. The
    package installs into the base interpreter's stdlib location, which a venv
    sees directly (venvs only isolate site-packages, not the stdlib), so it
    becomes importable immediately with no venv recreation needed. Windows
    Python installers and the python.org macOS installer both bundle Tcl/Tk
    already, so in practice this only bites Linux (and Homebrew Python on
    macOS).
    """
    log.banner("Step 0 - tkinter (GUI toolkit)")
    if sys.platform == "win32":
        log.log("    Windows Python installers bundle tkinter; skipping.")
        report.add("tkinter", SKIPPED, "not Linux/Unix")
        return

    try:
        import tkinter  # noqa: F401
    except ImportError:
        pass
    else:
        log.log("    tkinter is importable.")
        report.add("tkinter", DONE, "present")
        return

    log.log("    tkinter is NOT importable (main.py needs it to open its window).")
    system = platform.system()

    if system == "Linux":
        pkg_cmd = _linux_tkinter_command()
        if pkg_cmd:
            if confirmer.confirm("Install tkinter via the system package "
                                 "manager (needs sudo).", " ".join(pkg_cmd)):
                run_or_fail(pkg_cmd, log, report, "tkinter")
            else:
                report.add("tkinter", SKIPPED)
            return
        log.log("    Could not detect a supported package manager. Install "
                "the 'tkinter' (or 'tk') package for your distribution "
                "manually, then re-run install.py.")

    elif system == "Darwin":
        if shutil.which("brew"):
            # Pin the formula to the running interpreter's version. Bare
            # "python-tk" resolves to Homebrew's current default Python
            # (e.g. python-tk@3.14), which installs _tkinter for the wrong
            # interpreter and leaves this 3.x venv still failing to import it.
            py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
            brew_cmd = ["brew", "install", f"python-tk@{py_ver}"]
            if confirmer.confirm("Install tkinter via Homebrew (only needed "
                                 "for Homebrew Python - the python.org "
                                 "installer bundles it already).",
                                 " ".join(brew_cmd)):
                if run_command(brew_cmd, log):
                    report.add("tkinter", DONE)
                    return
                # Homebrew keeps python-tk@X.Y formulas only for the Python
                # versions it currently ships, so the pinned name can simply
                # not exist for this interpreter. Fall back to the bare
                # formula: it targets Homebrew's default Python, which may or
                # may not be this one, but it is the only remaining shot.
                log.log(f"    brew install python-tk@{py_ver} failed (the "
                        "formula may not exist for this Python version); "
                        "trying the unpinned python-tk formula.")
                run_or_fail(["brew", "install", "python-tk"], log, report,
                            "tkinter", note="fallback: unpinned python-tk")
                return
            report.add("tkinter", SKIPPED)
            return
        log.log("    Homebrew not found. If this is a python.org install, "
                "tkinter should already be bundled - check your Python build.")

    report.add("tkinter", MANUAL, "install separately, see log")


def step_check_python(
    log: Logger, confirmer: Confirmer, report: StepReport
) -> None:
    """Gate the interpreter: hard-fail below the minimum, warn above the tested
    maximum (no upper version block - newer Pythons may just lack wheels)."""
    log.banner("Step 1 - Python version")
    current = sys.version_info[:2]
    log.log(f"    Running Python {platform.python_version()} ({sys.executable})")
    if current < MIN_PYTHON:
        need = ".".join(map(str, MIN_PYTHON))
        log.log(f"    ERROR: Mimora needs Python >= {need}. Aborting.")
        report.add("Python version", FAILED, f"need >= {need}")
        raise InstallError("Python version")
    if current > WHEEL_TESTED_MAX:
        tested = ".".join(map(str, WHEEL_TESTED_MAX))
        confirmer.warn_continue([
            f"Python {platform.python_version()} is newer than the latest "
            f"version Mimora is tested against ({tested}).",
            "Prebuilt wheels may not exist yet for some dependencies on this "
            "interpreter. Installs are binary-only, so the dependency step may "
            "stop with a 'no matching distribution' error.",
        ])
    report.add("Python version", DONE, platform.python_version())


def step_install_requirements(
    log: Logger, confirmer: Confirmer, report: StepReport
) -> None:
    """Install the project requirements; the root file chains the subprojects."""
    log.banner("Step 4 - Project dependencies")
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
            "the pronunciation/acoustic/ and pronunciation/phoneme/ "
            "requirements).")
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


def step_espeak(log: Logger, confirmer: Confirmer, report: StepReport) -> None:
    """Ensure the native espeak-ng binary exists; offer to install it."""
    log.banner("Step 5 - espeak-ng (native binary for phonemizer)")
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
                _install_espeak_soft(pkg_cmd, log, report)
                return
            report.add("espeak-ng", SKIPPED)
            return
        log.log("    Could not detect a supported package manager.")

    elif system == "Darwin":
        if shutil.which("brew"):
            brew_cmd = ["brew", "install", "espeak-ng"]
            if confirmer.confirm("Install espeak-ng via Homebrew.",
                                 " ".join(brew_cmd)):
                _install_espeak_soft(brew_cmd, log, report)
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


def _install_espeak_soft(cmd: list[str], log: Logger, report: StepReport) -> None:
    """Run the espeak-ng package install without aborting the installer.

    espeak-ng is only needed by the phoneme engine, and both declining the
    install and the Windows path merely record SKIPPED/MANUAL - so a package
    manager failure (no sudo rights, repo trouble) must not kill the whole
    install either.
    """
    if run_command(cmd, log):
        report.add("espeak-ng", DONE)
        return
    log.log("    Package install failed. Install espeak-ng manually later;")
    log.log("    everything except the phoneme engine works without it.")
    report.add("espeak-ng", MANUAL, "package install failed, see log")


def _linux_espeak_command() -> list[str] | None:
    """Build the right install command for the detected Linux package manager."""
    if shutil.which("apt-get"):
        return ["sudo", "apt-get", "install", "-y", "espeak-ng"]
    if shutil.which("dnf"):
        return ["sudo", "dnf", "install", "-y", "espeak-ng"]
    if shutil.which("pacman"):
        return ["sudo", "pacman", "-S", "--noconfirm", "espeak-ng"]
    return None


def step_prefetch_models(
    log: Logger, confirmer: Confirmer, report: StepReport
) -> None:
    """Download the Hugging Face models into model_cache/ (HF_HOME)."""
    log.banner("Step 6 - Pre-download Hugging Face models")
    model_fetch = _import_fetcher("mimora.model_fetch", log, report,
                                  "HF model cache")

    repos = ", ".join(repo.repo_id for repo in model_fetch.HF_MODEL_REPOS)
    cached = all(model_fetch.hf_repo_cached(repo.repo_id)
                 for repo in model_fetch.HF_MODEL_REPOS)
    total_mb = sum(repo.size_mb for repo in model_fetch.HF_MODEL_REPOS)
    desc = (f"Download HF models into {model_fetch.MODEL_CACHE_DIR.name}/ "
            f"(HF_HOME): {repos}. {total_mb} MB in total; already-cached files "
            f"are reused.")
    if cached:
        log.log(f"    All {len(model_fetch.HF_MODEL_REPOS)} model repos already "
                f"present in the cache.")
    if not confirmer.confirm(desc, installed=cached):
        report.add("HF model cache", SKIPPED, "already cached" if cached else "")
        return

    # force=cached: reaching this line with cached=True means the user answered
    # "[r]einstall" at the already-present prompt, so the fetcher must not skip
    # the repos its own predicate would also call cached. Every download step
    # below follows the same pattern.
    try:
        model_fetch.ensure_hf_models(force=cached)
    except model_fetch.ModelFetchError as exc:
        log.log(f"    -> FAILED: {exc}")
        report.add("HF model cache", FAILED)
        raise InstallError("HF model cache")
    report.add("HF model cache", DONE)


def step_prefetch_supertonic(
    log: Logger, confirmer: Confirmer, report: StepReport
) -> None:
    """Download the Supertonic 3 TTS model into model_cache/supertonic3/."""
    log.banner("Step 7 - Supertonic 3 TTS model (Spanish)")
    model_fetch = _import_fetcher("mimora.model_fetch", log, report,
                                  "Supertonic model")

    cache_dir = model_fetch.supertonic_cache_dir()
    cached = model_fetch.supertonic_cached()
    desc = (f"Download the Supertonic 3 TTS model "
            f"({model_fetch.SUPERTONIC_SIZE_MB} MB, weights licensed "
            f"OpenRAIL-M) into {cache_dir} - the Spanish text-to-speech "
            f"backend.")
    if cached:
        log.log(f"    Supertonic model already present: {cache_dir}")
    if not confirmer.confirm(desc, installed=cached):
        report.add("Supertonic model", SKIPPED,
                   "already downloaded" if cached else "")
        return

    try:
        model_fetch.ensure_supertonic(force=cached)
    except model_fetch.ModelFetchError as exc:
        log.log(f"    -> FAILED: {exc}")
        report.add("Supertonic model", FAILED)
        raise InstallError("Supertonic model")
    report.add("Supertonic model", DONE)


def step_llama_server(
    log: Logger, confirmer: Confirmer, report: StepReport
) -> None:
    """Install the pinned llama-server binary into bin/llama/.

    First half of the LLM stack (the GGUF model below is the second). The
    fetcher verifies the install by running --version and --list-devices, which
    is what catches llama.cpp's silent fallback to CPU when the cudart DLLs are
    missing or of the wrong major version - see tasks/llama-cpp.md, phase 0.

    A platform with no pinned build (currently everything except Windows x64)
    is recorded as a manual action rather than a hard failure: the rest of the
    install is perfectly usable, and such a machine can point
    "llama_server_path" at its own binary or switch to the lm-studio backend.
    """
    log.banner("Step 8 - LLM stack: llama-server binary")
    fetch = _import_fetcher("mimora.llama_server_fetch", log, report,
                            "llama-server binary")

    installed = fetch.installed_exe() is not None

    # Resolve the variant BEFORE asking. Two reasons: the prompt can then name
    # the real download size (a single hardcoded number fits only the CUDA
    # build, and the CPU one is 18 MB against 641), and a platform with no
    # pinned build is recognised here instead of after the user has agreed to a
    # download that cannot happen.
    try:
        variant = fetch.select_variant()
    except fetch.UnsupportedPlatformError as exc:
        # No pinned build for this OS/arch yet: say so and move on.
        log.log(f"    {exc}")
        report.add("llama-server binary", MANUAL, "no pinned build, see log")
        return

    desc = (f"Download the pinned llama.cpp release {fetch.RELEASE_TAG} "
            f"({variant}, {fetch.variant_size_mb(variant)} MB) into "
            f"{fetch.INSTALL_DIR.parent.name}/{fetch.INSTALL_DIR.name}/ and "
            f"verify that its GPU backend actually comes up.")
    if installed:
        log.log(f"    llama-server already installed: {fetch.installed_exe()}")
    if not confirmer.confirm(desc, installed=installed):
        report.add("llama-server binary", SKIPPED,
                   "already installed" if installed else "")
        return

    try:
        # Passing the resolved variant keeps the install identical to what the
        # prompt described and saves a second nvidia-smi probe.
        exe = fetch.ensure_llama_server(variant=variant, force=installed)
    except fetch.LlamaServerFetchError as exc:
        log.log(f"    -> FAILED: {exc}")
        report.add("llama-server binary", FAILED)
        raise InstallError("llama-server binary")
    report.add("llama-server binary", DONE, fetch.RELEASE_TAG)
    log.log(f"    Installed: {exe}")


def step_download_gguf(
    log: Logger, confirmer: Confirmer, report: StepReport
) -> None:
    """Download the GGUF chat model into models/ if not already present."""
    log.banner("Step 9 - LLM stack: GGUF chat model")
    gguf_fetch = _import_fetcher("mimora.gguf_fetch", log, report, "GGUF model")

    target = gguf_fetch.DEFAULT_GGUF_PATH
    present = gguf_fetch.gguf_present(target)
    desc = (f"Download {target.name} ({gguf_fetch.GGUF_SIZE_MB} MB) from "
            f"{gguf_fetch.GGUF_REPO_ID} into {target.parent.name}/.")
    if present:
        log.log(f"    Already present: {target}")
    if not confirmer.confirm(desc, installed=present):
        report.add("GGUF model", SKIPPED, "already downloaded" if present else "")
        return

    try:
        gguf_fetch.ensure_gguf(target, force=present)
    except gguf_fetch.GgufFetchError as exc:
        log.log(f"    -> FAILED: {exc}")
        report.add("GGUF model", FAILED)
        raise InstallError("GGUF model")
    report.add("GGUF model", DONE)


def step_detect_hardware(
    log: Logger, confirmer: Confirmer, report: StepReport
) -> None:
    """Run detect_hardware.py late, once torch is installed."""
    log.banner("Step 10 - Hardware detection (writes hardware_config.json)")
    if not DETECT_HW_SCRIPT.exists():
        log.log(f"    {DETECT_HW_SCRIPT} not found; skipping.")
        report.add("hardware detection", SKIPPED, "script missing")
        return

    cmd = [sys.executable, str(DETECT_HW_SCRIPT)]
    desc = ("Probe the machine and write config/hardware_config.json (the app "
            "reads GPU-tuned parameters from it).")
    if not confirmer.confirm(desc, " ".join(cmd)):
        report.add("hardware detection", SKIPPED)
        return
    run_or_fail(cmd, log, report, "hardware detection")


def step_create_launchers(
    log: Logger, confirmer: Confirmer, report: StepReport
) -> None:
    """Write the one launcher script this platform needs: run main.py with the
    venv's own python interpreter (no activate step).

    Run last, once the environment is fully set up, so the script is a
    one-click way to start Mimora afterwards. Only run_mimora.bat is written on
    Windows and only run_mimora.sh on Linux/macOS - the installer runs on the
    machine that will actually launch Mimora, so the other platform's script
    would never be used there. Calling the venv's python executable by path is
    equivalent to activating the venv and running `python main.py`, but skips
    activate.bat / activate's own quirks (Windows execution-policy prompts for
    the .ps1 variant, needing to `source` the Unix script, etc.) - the
    interpreter's own site-packages resolution already provides the same
    isolation. `Scripts\\python.exe` / `bin/python` are the two names
    guaranteed to exist regardless of whether the venv was made with the
    stdlib `venv` module or the `virtualenv` package.

    The venv folder name is auto-detected the same way check_virtualenv()
    hints at it (find_local_venv_name()), so the script still works if the
    venv was created under a name other than '.venv'.
    """
    log.banner("Step 11 - Launcher script")
    venv_name = find_local_venv_name()
    target = LAUNCHER_BAT if sys.platform == "win32" else LAUNCHER_SH

    installed = target.exists()
    desc = (f"Write {target.name}: runs main.py with '{venv_name}'s own "
            f"python interpreter.")
    if installed:
        log.log(f"    {target.name} already present (would be refreshed for "
                f"venv '{venv_name}').")
    if not confirmer.confirm(desc, installed=installed):
        report.add("Launcher script", SKIPPED,
                   "already present" if installed else "")
        return

    if target is LAUNCHER_BAT:
        # \r\n line endings: keeps the .bat readable if opened/edited on Windows.
        contents = "\r\n".join([
            "@echo off",
            "REM Launch Mimora with the venv's own python interpreter.",
            "setlocal",
            "cd /d \"%~dp0\"",
            f"\"{venv_name}\\Scripts\\python.exe\" main.py",
            "pause",
            "",
        ])
        target.write_text(contents, encoding="utf-8")
    else:
        contents = "\n".join([
            "#!/usr/bin/env bash",
            "# Launch Mimora with the venv's own python interpreter.",
            "set -e",
            "cd \"$(dirname \"$0\")\"",
            f"\"{venv_name}/bin/python\" main.py",
            "",
        ])
        target.write_text(contents, encoding="utf-8", newline="\n")
        try:
            # Best-effort +x; the file is still usable via `bash
            # run_mimora.sh` even if chmod fails (e.g. an unusual filesystem).
            target.chmod(target.stat().st_mode | 0o111)
        except OSError:
            pass

    log.log(f"    Wrote {target.name}")
    report.add("Launcher script", DONE, target.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def finish(log: Logger, report: StepReport, *, success: bool,
           dry_run: bool = False) -> None:
    """Print the end-of-run summary.

    The "run main.py" hint is printed ONLY on a fully successful run, so a
    failed or aborted install never reads as ready to launch. A successful run
    that still has manual-action items says so before the launch hint.

    Under --dry-run every step reports "skipped", which is the same word the
    installer uses for "already present, left alone". The note below is what
    tells the two apart, and it is also why the launch hint is withheld: a
    rehearsal installed nothing.
    """
    log.banner("Summary")
    log.log(report.render())
    log.log("")
    if dry_run:
        log.log("    [dry-run] nothing was executed: every 'skipped' above")
        log.log("    means 'not attempted', not 'already installed'.")
    log.log(f"    finished: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    if not success:
        log.log("    Installation INCOMPLETE - fix the error above and re-run")
        log.log("    install.py. Do NOT run main.py until it finishes cleanly.")
        return
    if dry_run:
        log.log("    Re-run without --dry-run to install.")
        return
    if MANUAL in report.statuses():
        log.log("    Some steps need manual action (see 'needs manual action'")
        log.log("    above) before Mimora will fully work.")
    launcher = LAUNCHER_BAT if sys.platform == "win32" else LAUNCHER_SH
    log.log(f"    Next: run `python main.py` to start Mimora, or use the "
            f"generated {launcher.name} launcher.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mimora installer - installs dependencies and downloads "
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
    parser.add_argument("--skip-llm", action="store_true",
                        help="skip the whole LLM stack (llama-server binary "
                             "and GGUF model) - for the lm-studio or off "
                             "backend, which need neither")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = Logger(LOG_FILE)
    # The download steps call into mimora/* instead of spawning subprocesses,
    # so their logging has to be routed into logs/install.log explicitly.
    bridge_module_logging(log)
    confirmer = Confirmer(log, assume_yes=args.yes, dry_run=args.dry_run,
                          force_reinstall=args.reinstall)
    report = StepReport()

    log.banner("Mimora installer")
    log.log(f"    started: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    log.log(f"    platform: {platform.platform()}")
    log.log(f"    args: {vars(args)}")
    log.log(f"    log file: {LOG_FILE}")

    # All steps run inside this guard: the first one to fail raises
    # InstallError, so the run stops immediately instead of pressing on with a
    # half-built environment and printing a misleading "ready to launch" line.
    try:
        # Step 0: refuse to silently install into the global interpreter, then
        # (on Windows) verify the MSVC runtime torch and llama-server need at
        # runtime.
        check_virtualenv(log, args)
        step_check_vcredist(log, confirmer, report)
        step_check_tkinter(log, confirmer, report)

        # Step 1: Python version (hard min gate; warn above the tested max).
        step_check_python(log, confirmer, report)

        # Step 2: GPU detection (informs step 3; no packages needed).
        log.banner("Step 2 - GPU / CUDA detection")
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

        # Step 3: the CUDA torch build, installed BEFORE requirements so the
        # torch constraint in requirements.txt is already satisfied (otherwise
        # pip pulls the CPU wheel from PyPI first and it is replaced right
        # after). torch ships CPU wheels on PyPI, so the CPU path needs no
        # step of its own here.
        if use_gpu:
            log.banner("Step 3 - GPU (CUDA) builds")
            step_gpu_torch(log, confirmer, report, driver_cuda)
        else:
            log.banner("Step 3 - CPU builds")
            log.log("    Nothing to pre-install: every remaining dependency "
                    "has a CPU wheel on PyPI.")
            report.add("torch (CUDA)", SKIPPED, "CPU-only")

        # Step 4: project dependencies.
        step_install_requirements(log, confirmer, report)

        # Step 5: espeak-ng.
        step_espeak(log, confirmer, report)

        # Steps 6-7: model caches (HF hub + the Supertonic cache directory).
        if args.skip_models:
            report.add("HF model cache", SKIPPED, "--skip-models")
            report.add("Supertonic model", SKIPPED, "--skip-models")
        else:
            step_prefetch_models(log, confirmer, report)
            step_prefetch_supertonic(log, confirmer, report)

        # Steps 8-9: the LLM stack - the llama-server binary and the GGUF model
        # it loads. Both are skipped together under --skip-llm: the lm-studio
        # and off backends need neither. The binary comes first because step 10
        # probes it, and both come after pip because a 2.7 GB download is a bad
        # place to discover that the dependency install fails.
        if args.skip_llm:
            report.add("llama-server binary", SKIPPED, "--skip-llm")
            report.add("GGUF model", SKIPPED, "--skip-llm")
        else:
            step_llama_server(log, confirmer, report)
            if args.skip_gguf:
                report.add("GGUF model", SKIPPED, "--skip-gguf")
            else:
                step_download_gguf(log, confirmer, report)

        # Step 10: hardware detection (after torch exists).
        step_detect_hardware(log, confirmer, report)

        # Step 11: launcher scripts, written last so they reflect the fully
        # set-up environment (correct venv folder name).
        step_create_launchers(log, confirmer, report)
    except InstallError as exc:
        log.log("")
        log.log(f"    ABORTED: step '{exc}' failed - stopping the installer "
                f"so the error is not masked.")
        finish(log, report, success=False, dry_run=args.dry_run)
        return 1

    finish(log, report, success=True, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
