#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev
"""Mimora installer.

A standalone, idempotent setup helper that walks through everything needed to
run Mimora on a fresh machine:

  1. Verify the Python version.
  2. Detect an NVIDIA GPU / CUDA version (via nvidia-smi, no extra packages).
  3. (GPU only) install torch as a CUDA build.
  4. pip install the project dependencies (read from pyproject.toml).
  5. Report which espeak-ng shared library phonemizer will use (the bundled
     espeakng-loader wheel by default), and offer to install a system one only
     if none is found.
  6. Pre-download the Hugging Face models into model_cache/.
  7. Pre-download the Supertonic 3 TTS model into model_cache/supertonic3/
     (the Spanish TTS backend; kept outside the HF hub cache because the
     supertonic package uses its own cache directory).
  8. Install the spaCy pipeline into model_cache/spacy/ (Kokoro's
     grapheme-to-phoneme step loads it; see mimora/spacy_model_fetch.py for why
     it is not simply a dependency).
  9. Install the LLM stack: the pinned llama-server binary into bin/llama/ and
     the GGUF chat model into models/.
 10. Run `python -m mimora.detect_hardware` to write config/hardware_config.json.
 11. Write run_mimora.bat / run_mimora.sh: one-click launchers that activate
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
  four fetchers live in the package (mimora/model_fetch.py,
  mimora/gguf_fetch.py, mimora/llama_server_fetch.py,
  mimora/spacy_model_fetch.py) and the steps below are thin wrappers around
  them. The app needs the same code for its own first-run
  check (mimora/first_run_download.py), and a packaged install has no
  install.py at all. Their log output is bridged into logs/install.log (see
  _LogBridge).
* GPU detection deliberately relies only on `nvidia-smi`, because
  mimora/detect_hardware.py imports torch (which may not be installed yet) - a
  classic bootstrap chicken-and-egg. The probe is run at the very end, once
  those packages exist.
* The dependency list is read from `[project.dependencies]` in pyproject.toml
  rather than from a requirements.txt, so the project states its dependencies
  once. The root requirements.txt is gone; the two under pronunciation/ remain,
  because those subpackages are installable on their own.
* The CUDA torch wheels are installed before the dependency step so that the
  `torch` constraint is already satisfied - otherwise pip would pull the CPU
  build from PyPI only for it to be replaced afterwards.
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
import importlib.util  # find_spec, for the hardware-probe module check
import logging
import os
import platform
import re
import shlex
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
# The dependency list, and the only copy of it (see project_dependencies).
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
# The app's own settings file. The installer reads and writes exactly one key
# in it, in exactly one situation - see _disable_llm_backend().
SETTINGS_FILE = PROJECT_ROOT / "config" / "settings.json"
# The hardware probe is a package module, not a script under tools/: it has to
# run on the user's machine, and tools/ is what the maintainer runs. Named as a
# module because that is how it is invoked - see step_detect_hardware.
DETECT_HW_MODULE = "mimora.detect_hardware"
# The espeak-ng probe, run the same way and for the same reason: it has to
# answer for the environment that will run Mimora, not for this process. It
# lives under pronunciation/ rather than mimora/ because it is the engines' own
# registration - see step_espeak.
ESPEAK_PROBE_MODULE = "pronunciation.common.espeak"
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

# Distribution names (PyPI names, not import names) the dependency step is
# expected to leave behind. Used to detect whether it has already run. Names
# with dashes (scikit-learn, phonemizer-fork, python-Levenshtein) are the
# distribution names importlib.metadata expects. Not derived from
# [project.dependencies] on purpose: this list also names what arrives
# transitively (onnxruntime, via supertonic), which is exactly what a "did the
# install really finish" check wants to see.
REQUIRED_DISTS = [
    "numpy", "soundfile", "sounddevice", "kokoro", "supertonic", "openai",
    "torch", "transformers",
    "librosa", "scipy", "scikit-learn", "fastdtw",
    "phonemizer-fork", "espeakng-loader", "python-Levenshtein", "panphon",
    "sentencepiece",
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

    # Frames every question. Narrower than the 70-column step banner and drawn
    # with a different character, so the two never read as the same kind of
    # line: banners announce, rules ask.
    _RULE = "    " + "-" * 62

    def _ask(self, prompt: str) -> str:
        """Read one answer, making sure the question is actually visible.

        Two different things bury a question, and both end with the installer
        looking hung while it waits for Enter.

        A preceding download shows a tqdm progress bar that writes to stderr and
        keeps the cursor on its own line (carriage-return based). A plain
        ``input()`` prompt then lands on that same line and looks invisible.
        Flushing both streams and leading with a blank line fixes that one.

        The other is plainer and was what actually caught a user out: after a
        long download the question arrives below a step banner and a few lines
        of progress notes, in the same shape as every other line of output, at
        the moment the screen stops moving and the reader stops reading. So the
        question is printed as a framed block instead of appended to the flow.
        The rules are the only lines of their kind in the whole run, which is
        what makes the block findable when scrolling back, and the answer is
        typed on its own line below them.
        """
        sys.stdout.flush()
        sys.stderr.flush()
        # Straight to stdout rather than through Logger: the frame is a console
        # affordance, and repeating it in logs/install.log would just box in the
        # ">>>" line that already records what was asked.
        print()
        print(self._RULE)
        print(prompt)
        print(self._RULE, flush=True)
        try:
            return input("    > ").strip().lower()
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
            # The Enter hint is spelled out: a capitalised [Y] is the shell
            # convention for "this is the default", but it is not obvious to
            # everyone, and a question nobody knows how to answer reads exactly
            # like a hang.
            answer = self._ask("    Proceed?  [Y]es / [n]o-abort / [s]kip"
                               "   (Enter = Yes)")
            if answer in ("", "y", "yes"):
                # Logged like the other two answers, so logs/install.log shows
                # where a run was waiting and what it was told.
                self._log.log("    proceeding")
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
            answer = self._ask("    Already installed.  [S]kip / [r]einstall / "
                               "[n]o-abort   (Enter = Skip)")
            if answer in ("", "s", "skip"):
                self._log.log("    kept existing (skipped)")
                return False
            if answer in ("r", "reinstall"):
                self._log.log("    reinstalling")
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
            # No Enter hint here, deliberately: there is no default, and
            # promising one would be a lie the loop below would not honor.
            answer = self._ask("    Continue anyway?  [c]ontinue / [a]bort")
            if answer in ("c", "continue"):
                self._log.log("    continuing despite warning")
                return True
            if answer in ("a", "abort"):
                self._log.log("    aborted by user")
                raise SystemExit(1)
            print("    Please answer c or a.")


# ---------------------------------------------------------------------------
# Command execution
# ---------------------------------------------------------------------------

# Characters that make a Windows shell do something other than pass the word
# along. The redirection pair is the one that matters here and the one easiest
# to overlook: every version specifier contains '>' or '<', so an unquoted
# `soundfile>=0.12.0` is not an argument at all - cmd reads it as "run soundfile,
# send its output to a file named =0.12.0", and PowerShell reserves both
# characters outright.
_WINDOWS_SHELL_SPECIAL = frozenset(' \t"<>|&^;,()%!')


def display_command(cmd: list[str]) -> str:
    """The command written so a shell on THIS platform would run it unchanged.

    Every step prints the exact command before running it, and that line is
    meant to be read and re-run by hand - which is the whole requirement here.
    A plain " ".join stopped meeting it when the dependency step began passing
    PEP 508 requirement strings: their markers hold spaces and a ';', so the
    joined line is not an ugly rendering of the command but a different one,
    cut short at the first semicolon.

    Quoting is per-platform because the shells disagree, and neither branch is a
    fallback for the other. shlex.join is right for POSIX, and already covers
    this: shlex.quote treats '>' as unsafe, so specifiers come out quoted.

    Windows gets its own pass rather than subprocess.list2cmdline. That function
    is correct for what subprocess uses it for - handing CreateProcess something
    the C runtime will split back into the same argv - and it therefore quotes
    on whitespace alone. A printed line has a shell for an audience, not the
    argv parser, so the redirection characters have to be covered too. Extra
    double quotes cost nothing if the line is pasted: the argv parser strips
    them, so the reconstructed argument is identical either way.

    Display only, and it assumes no argument contains a double quote of its own
    - true of every command this installer builds, and not worth the escaping
    rules it would otherwise drag in.
    """
    if os.name != "nt":
        return shlex.join(cmd)
    return " ".join(
        f'"{arg}"' if any(char in _WINDOWS_SHELL_SPECIAL for char in arg) else arg
        for arg in cmd
    )


def run_command(cmd: list[str], log: Logger) -> bool:
    """Run a subprocess, streaming combined output live to console and log.

    Output is read line-by-line as it is produced (so a long pip install shows
    progress in real time) and mirrored into logs/install.log. Returns True on exit
    code 0, False otherwise. Never raises on a non-zero exit - the caller
    decides how a failure affects the rest of the run.
    """
    log.log(f"    $ {display_command(cmd)}")
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
    # OSError, not FileNotFoundError alone: a command present but not
    # executable raises PermissionError, and this function's contract is that a
    # command that will not run becomes a FAILED row rather than a traceback
    # out of the installer's own reporting.
    except OSError as exc:
        log.log(f"    ERROR: could not run the command: {exc}")
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
    """Import one of the mimora/* helper modules, failing the step cleanly.

    The fetchers live in the package rather than in this script (see the design
    notes at the top), so every download step starts by importing one. Only
    mimora.llama_server_fetch is stdlib-only; the other two need
    huggingface_hub, so a failure here almost always means the requirements
    step was skipped or aborted.

    Also used for mimora.loader, which downloads nothing but is imported the
    same way for the same reason: the sys.path handling below is what makes
    `python /somewhere/else/install.py` work, and it should exist once.
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
    log.banner("Environment check")
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
    log.banner("Visual C++ runtime (Windows)")
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
    log.banner("tkinter (GUI toolkit)")
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


def _linux_portaudio_command() -> list[str] | None:
    """Build the right install command for the detected Linux package manager.

    Mirrors _linux_espeak_command(). Debian/Ubuntu ship the runtime library as
    libportaudio2 (the -dev package only adds headers, which nothing here
    compiles against); Fedora and Arch call it portaudio.
    """
    if shutil.which("apt-get"):
        return ["sudo", "apt-get", "install", "-y", "libportaudio2"]
    if shutil.which("dnf"):
        return ["sudo", "dnf", "install", "-y", "portaudio"]
    if shutil.which("pacman"):
        return ["sudo", "pacman", "-S", "--noconfirm", "portaudio"]
    return None


def _log_audio_devices(log: Logger) -> None:
    """Report how many devices PortAudio sees, and explain a count of zero.

    The zero case is the WSL trap: Ubuntu builds libportaudio2 with the ALSA
    backend only, while WSLg routes audio through PulseAudio, so the library
    loads cleanly, enumerates nothing, and no error anywhere says why. The
    installer used to leave the user with 'Audio: 0 input / 0 output' in
    hardware_config.json and no next step.

    Purely informational, and silent when it cannot answer: sounddevice only
    exists after the requirements step, so on a first install this says nothing
    and the same machine gets asked again at the hardware-detection step.
    """
    try:
        import sounddevice
    except (ImportError, OSError):
        return
    try:
        devices = sounddevice.query_devices()
    except (OSError, sounddevice.PortAudioError):
        return

    inputs = sum(1 for dev in devices if dev["max_input_channels"] > 0)
    outputs = sum(1 for dev in devices if dev["max_output_channels"] > 0)
    log.log(f"    Audio devices: {inputs} input / {outputs} output.")
    if inputs or outputs:
        return
    log.log("    WARNING: PortAudio is installed but exposes no audio device.")
    log.log("    On WSL and other PulseAudio-only setups this is usually the")
    log.log("    distribution's PortAudio being built without the PulseAudio")
    log.log("    backend (check with: ldd $(ldconfig -p | grep -m1 portaudio |")
    log.log("    awk '{print $NF}') | grep pulse). Two known ways out:")
    log.log("      - install libasound2-plugins and point ALSA's default")
    log.log("        device at pulse ('pcm.!default pulse' in ~/.asoundrc);")
    log.log("      - rebuild PortAudio with ./configure --with-pulseaudio.")


def step_check_portaudio(
    log: Logger, confirmer: Confirmer, report: StepReport
) -> None:
    """On Linux, verify the native PortAudio library; offer to install it.

    sounddevice is a cffi wrapper around PortAudio. Its Windows and macOS
    wheels bundle the library, its Linux wheels do not, and importing it
    without the system library raises OSError('PortAudio library not found').
    That import happens inside mimora/detect_hardware.py, the last step
    but one, so on a fresh Linux machine the installer used to abort after
    several gigabytes of downloads over a 300 kB package. Hence the check runs
    here, in the preflight block, next to the other things that only look.

    ctypes.util.find_library needs no third-party package, which is what lets
    this run before the requirements step installs anything.
    """
    log.banner("PortAudio (audio I/O library)")
    if sys.platform in ("win32", "darwin"):
        log.log("    The sounddevice wheel bundles PortAudio here; skipping.")
        report.add("PortAudio", SKIPPED, "bundled in the wheel")
        return

    import ctypes.util
    found = ctypes.util.find_library("portaudio")
    if found:
        log.log(f"    PortAudio found on the library path ({found}).")
        report.add("PortAudio", DONE, "already present")
        _log_audio_devices(log)
        return

    log.log("    PortAudio NOT found (sounddevice needs it to record and play;")
    log.log("    without it the hardware-detection step below cannot run).")
    pkg_cmd = _linux_portaudio_command()
    if pkg_cmd is None:
        log.log("    Could not detect a supported package manager. Install the")
        log.log("    PortAudio runtime (libportaudio2 / portaudio) for your")
        log.log("    distribution, then re-run install.py.")
        report.add("PortAudio", MANUAL, "install separately, see log")
        return

    if not confirmer.confirm("Install PortAudio via the system package manager "
                             "(needs sudo).", " ".join(pkg_cmd)):
        report.add("PortAudio", SKIPPED)
        return
    # Soft failure, like espeak-ng: a package manager that cannot install it
    # (no sudo rights, repo trouble) is reported and left to the user rather
    # than killing a run whose remaining steps are all still doable.
    if run_command(pkg_cmd, log):
        report.add("PortAudio", DONE)
        _log_audio_devices(log)
        return
    log.log("    Package install failed. Install the PortAudio runtime")
    log.log("    manually, then re-run install.py.")
    report.add("PortAudio", MANUAL, "package install failed, see log")


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


def project_dependencies() -> list[str]:
    """The `[project.dependencies]` list from pyproject.toml, verbatim.

    The requirement strings (markers included) are handed to pip unchanged, so
    this is a reader and not a parser - nothing here interprets a specifier.

    tomllib is imported here rather than at module level on purpose: it is
    stdlib only from 3.11, which is MIN_PYTHON, and this script has to stay
    *runnable* on an older interpreter for exactly as long as it takes to print
    "Mimora needs Python >= 3.11" in step 1. A module-level import would replace
    that message with a traceback.
    """
    import tomllib

    with PYPROJECT.open("rb") as handle:
        data = tomllib.load(handle)
    deps = data.get("project", {}).get("dependencies")
    if not isinstance(deps, list) or not deps:
        raise InstallError("pyproject.toml has no [project] dependencies")
    return [str(item) for item in deps]


def step_install_requirements(
    log: Logger, confirmer: Confirmer, report: StepReport
) -> None:
    """Install the project dependencies, as declared in pyproject.toml."""
    log.banner("Step 4 - Project dependencies")
    if not PYPROJECT.exists():
        log.log(f"    ERROR: {PYPROJECT} not found. Aborting.")
        report.add("pip requirements", FAILED, "pyproject.toml missing")
        raise InstallError("pip requirements")

    try:
        requirements = project_dependencies()
    except (OSError, ValueError, InstallError) as exc:
        # ValueError covers tomllib.TOMLDecodeError, which subclasses it.
        log.log(f"    ERROR: cannot read dependencies from {PYPROJECT}: {exc}")
        report.add("pip requirements", FAILED, "unreadable pyproject.toml")
        raise InstallError("pip requirements") from exc

    installed = all_requirements_installed()
    # Binary-only install: if pip can't find a prebuilt wheel for the current
    # interpreter it fails with "no matching distribution" instead of quietly
    # downloading an sdist and compiling it (which is what happened on an
    # untested Python and triggered a numpy source build). The few sdist-only
    # packages are exempted so they can still build.
    #
    # The requirements are passed as arguments rather than through `-r`: the
    # list lives in pyproject.toml now, and writing it to a temporary file would
    # only add a way for the two to disagree. Each string is one argv element,
    # so the markers survive without quoting.
    cmd = PIP + ["install", *requirements,
                 "--only-binary", ":all:",
                 "--no-binary", ",".join(SOURCE_ONLY_PACKAGES)]
    desc = (f"Install all {len(requirements)} Python dependencies declared in "
            f"pyproject.toml (both pronunciation engines included).")
    if installed:
        log.log("    All expected dependencies are already installed.")
    # display_command rather than " ".join, for the reason given there: this is
    # the one step whose arguments are PEP 508 requirement strings.
    if not confirmer.confirm(desc, display_command(cmd), installed=installed):
        report.add("pip requirements", SKIPPED,
                   "already installed" if installed else "")
        return
    run_or_fail(cmd, log, report, "pip requirements")


def step_gpu_torch(
    log: Logger, confirmer: Confirmer, report: StepReport,
    driver_cuda: tuple[int, int] | None,
) -> None:
    """Reinstall torch as a matching CUDA build.

    torchaudio used to be reinstalled in the same command, because replacing
    torch alone left a torchaudio built against the previous one. That pairing
    problem went away with the package itself: torchaudio is no longer a
    dependency (see the note in pyproject.toml), so there is nothing here to
    keep in step with torch.
    """
    series = pick_cu_series(TORCH_CU_SERIES, driver_cuda)
    if series is None:
        log.log("    No compatible torch CUDA wheel series for this driver.")
        report.add("torch (CUDA)", MANUAL, "see pytorch.org/get-started/locally")
        return

    index = TORCH_INDEX_URL.format(series=series)
    cmd = PIP + ["install", "--force-reinstall", "torch", "--index-url", index]
    installed = torch_is_cuda_build()
    desc = (f"Install CUDA build of torch for {series} "
            f"(used by Wav2Vec2 pronunciation analysis).")
    if installed:
        log.log(f"    torch already a CUDA build ({dist_version('torch')}).")
    if not confirmer.confirm(desc, " ".join(cmd), installed=installed):
        report.add("torch (CUDA)", SKIPPED,
                   "already CUDA build" if installed else "")
        return
    run_or_fail(cmd, log, report, "torch (CUDA)", series)


def step_espeak(log: Logger, confirmer: Confirmer, report: StepReport) -> None:
    """Report which espeak-ng the engines will use; offer to install one if none.

    Asks the same question the consumer asks. `shutil.which("espeak-ng")`, what
    this used to do, is a different question and was wrong in both directions:
    phonemizer loads a shared LIBRARY, so a machine with the executable on PATH
    but no library phonemizer can find was reported as done (on Windows that is
    the *normal* outcome of the official installer, which writes
    libespeak-ng.dll while phonemizer's search looks for espeak-ng.dll), and a
    machine with no executable but the bundled espeakng-loader wheel - the
    default setup since that wheel became a dependency - was reported as
    needing manual work it does not need.
    """
    log.banner("Step 5 - espeak-ng (shared library for phonemizer)")
    library = _resolve_espeak_library(log)
    if library:
        log.log(f"    espeak-ng resolves from {library}")
        report.add("espeak-ng", DONE, library)
        return

    log.log("    No espeak-ng library found (needed by both pronunciation "
            "engines).")
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
        log.log("    Simplest fix: reinstall the dependencies (step 4) - the")
        log.log("    espeakng-loader wheel carries the library and its data.")
        log.log("    Or install espeak-ng system-wide, from")
        log.log("      https://github.com/espeak-ng/espeak-ng/releases")
        log.log("    and then set BOTH of these, because phonemizer's own")
        log.log("    search looks for espeak-ng.dll while the installer writes")
        log.log("    libespeak-ng.dll, and the data is never found beside it:")
        log.log("      PHONEMIZER_ESPEAK_LIBRARY   -> ...\\libespeak-ng.dll")
        log.log("      PHONEMIZER_ESPEAK_DATA_PATH -> ...\\espeak-ng-data")

    report.add("espeak-ng", MANUAL, "no library found, see log")


def _resolve_espeak_library(log: Logger) -> str | None:
    """The espeak-ng library the app will use, asked of the TARGET environment.

    Runs pronunciation/common/espeak.py as a module in a subprocess rather than
    importing it here, for the same reason step_detect_hardware does: the
    answer belongs to the interpreter that will run Mimora, and an import would
    answer for whichever interpreter happens to be running install.py. That
    module prints the resolved path on stdout and its explanation on stderr, so
    nothing has to be parsed out of prose.

    Returns None when nothing was found, when the module is missing (the
    requirements step was skipped) or when the probe failed for any other
    reason - all of which mean the same thing to the caller.

    Not gated on --dry-run, like the other detection in this script: it changes
    nothing, and reporting the truth is exactly what a dry run is for. It only
    prints and exits, so the subprocess is cheap.
    """
    cmd = [sys.executable, "-m", ESPEAK_PROBE_MODULE]
    log.log(f"    $ {display_command(cmd)}")
    # The thing being printed is a filesystem path, and a redirected stdout on
    # Windows is cp1252: a path with a non-Latin-1 character would kill the
    # child with a UnicodeEncodeError and be indistinguishable from "no espeak
    # found". Force UTF-8 on both ends.
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              cwd=str(PROJECT_ROOT), env=env)
    except OSError as exc:
        log.log(f"    ERROR: could not run the espeak probe: {exc}")
        return None

    for line in (proc.stderr or "").splitlines():
        log.log(f"    | {line.rstrip()}")
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip() or None


def _install_espeak_soft(cmd: list[str], log: Logger, report: StepReport) -> None:
    """Run the espeak-ng package install without aborting the installer.

    Declining the install and the Windows path both merely record
    SKIPPED/MANUAL, so a package manager failure (no sudo rights, repo trouble)
    must not kill the whole install either. Nothing is lost by carrying on: the
    bundled espeakng-loader wheel is the default source of the library, and
    this step only matters on a machine that somehow lacks it.
    """
    if run_command(cmd, log):
        report.add("espeak-ng", DONE)
        return
    log.log("    Package install failed. Install espeak-ng manually later, or")
    log.log("    reinstall the dependencies so the espeakng-loader wheel is")
    log.log("    in place; both pronunciation engines need one of the two.")
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


def step_prefetch_spacy_model(
    log: Logger, confirmer: Confirmer, report: StepReport
) -> None:
    """Unpack the spaCy pipeline Kokoro's G2P step loads, into model_cache/spacy/.

    In a clone this step usually finds the model already there, because until
    now that is the only way it ever arrived: spaCy downloads it by shelling out
    to pip at the first synthesis, silently, and nothing in this project ever
    mentioned it. The step exists for the environment where that cannot happen -
    an installed tool has no pip - and its side benefit is that the model stops
    being something that simply turns up.
    """
    log.banner("Step 8 - spaCy pipeline (Kokoro grapheme-to-phoneme)")
    fetch = _import_fetcher("mimora.spacy_model_fetch", log, report,
                            "spaCy model")

    # The app reaches the sidecar through config, which this script never
    # imports, so it has to be activated here for the report below to describe
    # what the app will see rather than what this process happens to have.
    fetch.activate()
    present = fetch.model_available()
    desc = (f"Install the {fetch.MODEL.name} {fetch.MODEL.version} spaCy "
            f"pipeline ({fetch.MODEL.size_mb} MB) into {fetch.SIDECAR_DIR} - "
            f"Kokoro's grapheme-to-phoneme step loads it, and an installed "
            f"package has no way to fetch it on demand.")
    if present:
        log.log(f"    Already available: {fetch.resolved_location()}")
    if not confirmer.confirm(desc, installed=present):
        report.add("spaCy model", SKIPPED,
                   "already available" if present else "")
        return

    try:
        fetch.ensure_spacy_model(force=present)
    except fetch.SpacyModelFetchError as exc:
        log.log(f"    -> FAILED: {exc}")
        report.add("spaCy model", FAILED)
        raise InstallError("spaCy model")
    report.add("spaCy model", DONE)


def _disable_llm_backend(
    log: Logger, confirmer: Confirmer, report: StepReport
) -> None:
    """Set "llm_backend" to "off" after step 8 found no build for this machine.

    The default backend is "llama-server", so a machine the pinned release has
    nothing for otherwise installs cleanly, starts, fails to launch a server it
    has no binary for, and shows main.py's "no llama-server binary was found"
    on every single run. "off" is a supported configuration rather than a
    degraded one: practice phrases then come from the source text verbatim
    (mimora/phrase_source.py). It is also the same value the first-run window
    writes when the optional download is declined, so this adds no state the
    app did not already know how to reach - only a way to reach it on a machine
    where that download is never offered.

    Two things it deliberately does not do. It never overrides a backend the
    user already chose: only the "llama-server" default (or an absent value) is
    rewritten, so an lm-studio setup on a platform without a build keeps
    working, because it needs no local binary in the first place. And it asks
    first, like every other side effect in this script, which is also what
    makes it a no-op under --dry-run.
    """
    loader = _import_fetcher("mimora.loader", log, report, "LLM backend setting")
    current = loader.read_json(SETTINGS_FILE).get("llm_backend")
    if current not in (None, "llama-server"):
        log.log(f"    settings.json already selects the {current!r} backend, "
                f"which needs no binary from this step; leaving it alone.")
        return

    if not confirmer.confirm(
            'Set "llm_backend" to "off" in config/settings.json, so the app '
            'starts without an LLM instead of failing to launch one on every '
            'run. Practice phrases then come from the source text verbatim, '
            'and the setting is reversible in the settings window.'):
        report.add("LLM backend setting", SKIPPED)
        return

    if loader.save_setting(SETTINGS_FILE, "llm_backend", "off", {}):
        # The in-memory view save_setting also updates is a throwaway dict here:
        # nothing in this process reads the settings afterwards, and the app is
        # a separate run. Passing config's own _USER would mean importing
        # config, which sets HF env vars and loads the language modules.
        log.log('    settings.json: llm_backend = "off"')
        report.add("LLM backend setting", DONE, "off")
    else:
        # save_setting prints the reason (unparsable or unwritable file) and
        # never raises. A settings file the installer cannot write is not worth
        # failing an otherwise complete install for, so this becomes one more
        # line in the manual-actions list.
        report.add("LLM backend setting", MANUAL, 'set llm_backend to "off"')


def step_llama_server(
    log: Logger, confirmer: Confirmer, report: StepReport
) -> bool:
    """Install the pinned llama-server binary into bin/llama/.

    First half of the LLM stack (the GGUF model below is the second). The
    fetcher verifies the install by running --version and --list-devices, which
    is what catches llama.cpp's silent fallback to CPU when the cudart DLLs are
    missing or of the wrong major version.

    A machine the pinned release has nothing for is recorded as a manual action
    rather than a hard failure: the rest of the install is perfectly usable, and
    such a machine can point "llama_server_path" at its own binary or switch to
    the lm-studio backend. Two situations reach that branch, both raised by
    select_variant() before anything is downloaded: an OS/architecture with no
    pinned build at all (everything except Windows x64, Linux x64 and macOS),
    and a Mac older than the minimum macOS its build was compiled for.

    Returns False in exactly that case, so the caller can skip step 9: the GGUF
    model is a 2.7 GB download for a server that will not exist. Every other
    outcome returns True, including the user skipping the download - declining
    a binary is a decision about this run, not a fact about the platform, and
    the model may well be wanted for a build they install themselves.
    """
    log.banner("Step 9 - LLM stack: llama-server binary")
    fetch = _import_fetcher("mimora.llama_server_fetch", log, report,
                            "llama-server binary")

    installed_exe = fetch.installed_exe()

    # Resolve the variant BEFORE asking. Two reasons: the prompt can then name
    # the real download size (a single hardcoded number fits only the CUDA
    # build, and the CPU one is 18 MB against 641), and a platform with no
    # pinned build is recognised here instead of after the user has agreed to a
    # download that cannot happen.
    try:
        variant = fetch.select_variant()
    except fetch.UnsupportedPlatformError as exc:
        # No pinned build for this OS/arch yet: say so, leave the app in a
        # configuration that starts, and move on.
        log.log(f"    {exc}")
        report.add("llama-server binary", MANUAL, "no pinned build, see log")
        _disable_llm_backend(log, confirmer, report)
        return False

    # "Already installed" is the stamp's answer for THIS variant, not the mere
    # presence of a binary. The difference is a device-check substitution: on a
    # machine where linux-vulkan-x64 was tried and descended to linux-cpu-x64,
    # a presence test says "no Vulkan build here" and this step re-downloaded
    # the 32 MB asset on every run, failed the same check and descended again.
    # is_current() honours the recorded substitution, so the question stays
    # answered until --reinstall (force below) or an explicit --variant asks
    # for it to be reconsidered.
    installed = fetch.is_current(fetch.INSTALL_DIR, fetch.RELEASE_TAG, variant)

    desc = (f"Download the pinned llama.cpp release {fetch.RELEASE_TAG} "
            f"({variant}, {fetch.variant_size_mb(variant)} MB) into "
            f"{fetch.INSTALL_DIR.parent.name}/{fetch.INSTALL_DIR.name}/ and "
            f"verify that its GPU backend actually comes up.")
    # Reported whenever a binary is there, including when it does NOT count as
    # current (a stamp from an older release tag): "something is installed and
    # is about to be replaced" is exactly what the reader needs at that point.
    if installed_exe is not None:
        # Name the installed build, not just its path. The confirmer prints
        # *desc* right after this line, and desc describes what a reinstall
        # would fetch - which after a device-check fallback is a different
        # variant from the one on disk (linux-vulkan-x64 is selected,
        # linux-cpu-x64 is what ends up installed). Two consecutive lines
        # naming two variants read as a contradiction unless both say which
        # is which.
        stamped = fetch.installed_variant(installed_exe)
        named = f" ({stamped})" if stamped else ""
        log.log(f"    llama-server already installed: {installed_exe}{named}")
    if not confirmer.confirm(desc, installed=installed):
        report.add("llama-server binary", SKIPPED,
                   "already installed" if installed else "")
        return True

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
    return True


def step_download_gguf(
    log: Logger, confirmer: Confirmer, report: StepReport
) -> None:
    """Download the GGUF chat model into models/ if not already present."""
    log.banner("Step 10 - LLM stack: GGUF chat model")
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
    """Run the hardware probe late, once torch is installed."""
    log.banner("Step 11 - Hardware detection (writes hardware_config.json)")
    # A module, so the check is "can it be imported" rather than "is the file
    # there". importlib.util.find_spec answers without executing it, which
    # matters because the module imports torch and sounddevice when it runs.
    # install.py is normally run from the project root, which already puts it on
    # sys.path; this also covers `python /somewhere/else/install.py`.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        found = importlib.util.find_spec(DETECT_HW_MODULE) is not None
    except (ImportError, ValueError):
        # find_spec imports the parent package to look inside it, so an absent
        # or broken `mimora` raises here rather than answering None.
        found = False
    if not found:
        log.log(f"    {DETECT_HW_MODULE} not found; skipping.")
        report.add("hardware detection", SKIPPED, "module missing")
        return

    cmd = [sys.executable, "-m", DETECT_HW_MODULE]
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
    log.banner("Step 12 - Launcher script")
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
                        help="skip the model pre-downloads (Hugging Face, "
                             "Supertonic, spaCy)")
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
        # Preflight, deliberately unnumbered: these refuse to silently install
        # into the global interpreter, and check the native pieces that pip
        # cannot supply - the MSVC runtime on Windows, tkinter and PortAudio on
        # Linux. All three are needed by steps far below (or by main.py), and
        # all three are cheap to check, so they are checked before the first
        # download rather than after the last one.
        check_virtualenv(log, args)
        step_check_vcredist(log, confirmer, report)
        step_check_tkinter(log, confirmer, report)
        step_check_portaudio(log, confirmer, report)

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

        # Step 3: the CUDA torch build, installed BEFORE the dependency step so
        # its `torch` constraint is already satisfied (otherwise pip pulls the
        # CPU wheel from PyPI first and it is replaced right after). PyPI ships
        # CPU wheels for torch on Windows and macOS and CUDA wheels on Linux, so
        # the CPU path needs no step of its own here.
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

        # Steps 6-8: models that are not the LLM's - the HF hub cache, the
        # Supertonic cache directory and the spaCy pipeline sidecar. All three
        # go under --skip-models: they are the same kind of thing, a download
        # somebody may already have or may want to postpone.
        if args.skip_models:
            report.add("HF model cache", SKIPPED, "--skip-models")
            report.add("Supertonic model", SKIPPED, "--skip-models")
            report.add("spaCy model", SKIPPED, "--skip-models")
        else:
            step_prefetch_models(log, confirmer, report)
            step_prefetch_supertonic(log, confirmer, report)
            step_prefetch_spacy_model(log, confirmer, report)

        # Steps 9-10: the LLM stack - the llama-server binary and the GGUF model
        # it loads. Both are skipped together under --skip-llm: the lm-studio
        # and off backends need neither. The binary comes first because step 11
        # probes it, and both come after pip because a 2.7 GB download is a bad
        # place to discover that the dependency install fails.
        #
        # Step 9 also decides whether step 10 runs at all. The two used to be
        # independent, which on a platform with no pinned build offered a 2.7 GB
        # model right after saying that nothing here can load it.
        if args.skip_llm:
            report.add("llama-server binary", SKIPPED, "--skip-llm")
            report.add("GGUF model", SKIPPED, "--skip-llm")
        else:
            platform_has_build = step_llama_server(log, confirmer, report)
            if args.skip_gguf:
                report.add("GGUF model", SKIPPED, "--skip-gguf")
            elif not platform_has_build:
                report.add("GGUF model", SKIPPED, "no llama-server build")
            else:
                step_download_gguf(log, confirmer, report)

        # Step 11: hardware detection (after torch exists).
        step_detect_hardware(log, confirmer, report)

        # Step 12: launcher scripts, written last so they reflect the fully
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
