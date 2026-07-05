# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Early process setup for the Mimora entry point.

Two phases, split because they bracket the heavy imports in main.py:

  * ``early_init()`` must run BEFORE the heavy mimora.* imports (torch,
    transformers, Kokoro): it sets the UTF-8 environment hints, switches
    the console streams to UTF-8, and installs the library warning filters.
  * ``setup_logging()`` must run AFTER those imports: basicConfig with
    force=True replaces any handlers auto-installed by logging calls during
    the imports (e.g. the acoustic engine loads calibration.json at import
    and logs it), which would otherwise turn basicConfig into a silent
    no-op and leave main.log empty.

Only stdlib imports here, so ``from mimora import bootstrap`` stays free
and can precede everything heavy.
"""

import logging
import os
import sys
import warnings

LOG_FORMAT = "%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s"


def early_init():
    """UTF-8 console/env setup and warning filters (pre-import phase).

    Prefer UTF-8 everywhere so non-ASCII (IPA phones, espeak-ng / panphon
    data) never trips a cp1252 default on Windows. We deliberately do NOT
    re-exec the interpreter into UTF-8 mode: os.execv detaches stdout under
    some launchers (the orphaned process then fails any print with
    "[Errno 22] Invalid argument"). Instead we set the hint for child
    processes and switch our own console streams to UTF-8 where the stream
    supports it. The in-process file reads that mattered (panphon's tables)
    keep their own narrow UTF-8 fallback in pronunciation/phoneme/speech.py.
    """
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass  # stream may be None (pythonw), wrapped by an IDE, or already detached

    # Disable Hugging Face hub symlinks warning for a cleaner console output.
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # Ignore specific deprecation and model warnings from underlying libraries.
    warnings.filterwarnings("ignore", message="dropout option adds dropout.*")
    warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")


def setup_logging(log_file):
    """Install the root logging configuration (console + file, post-import).

    force=True replaces any handlers auto-installed by logging calls during
    the heavy imports; without it this basicConfig would be a silent no-op
    and the log file would stay empty. ``log_file`` is passed in by the
    caller (config.LOG_FILE) so this module never imports mimora.config,
    which pulls in torch.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ],
        force=True,
    )
