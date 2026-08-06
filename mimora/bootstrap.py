# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

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

# The command line switch that turns on append mode, spelled once. cli.py
# declares the argument with it and lifecycle.spawn_replacement() adds it to
# the command it starts the replacement process with; keeping the string here
# means the producer and the consumer cannot drift apart. It lives in this
# module rather than in cli.py because what it selects is a property of the
# logging setup, and because lifecycle.py may not import the entry point.
APPEND_LOG_FLAG = "--append-log"

# Whether log files are continued rather than truncated in this process. Set
# by setup_logging() and read through log_file_mode() below. Process-global,
# like the logging configuration it belongs to.
_append_logs = False


def log_file_mode():
    """The open() mode every log file of this process should use ("w" or "a").

    For log files opened directly rather than through ``logging``: today that
    is only logs/llm_server.log (see mimora/llm_server_ctl.py). Without this
    they would keep truncating across an in-session restart while main.log
    continues, which is the confusing half-state - the app's own log spans the
    restart and the server's log covers only what happened after it.
    """
    return "a" if _append_logs else "w"


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


def setup_logging(log_file, append=False):
    """Install the root logging configuration (console + file, post-import).

    force=True replaces any handlers auto-installed by logging calls during
    the heavy imports; without it this basicConfig would be a silent no-op
    and the log file would stay empty. ``log_file`` is passed in by the
    caller (config.LOG_FILE) so this module never imports mimora.config,
    which pulls in torch.

    ``append`` continues the existing file instead of truncating it. It is
    what an in-session restart passes to its own replacement process (see
    lifecycle.spawn_replacement): the app relaunches itself after the
    first-run window and after a restart-only setting changes, and the child
    used to open main.log with mode="w" and wipe everything the process that
    spawned it had written - which is precisely the interesting part, the
    whole first-run download or the setting that caused the restart. A fresh
    launch still starts from an empty file, so the log stays one session
    long, restarts included.

    A log file that cannot be opened costs the file and not the application.
    The console handler is installed either way, which is the half that keeps
    the failure visible, and the condition this covers is precisely a data root
    the machine cannot write to: an unhandled OSError here would end the
    startup with a traceback naming a FileHandler rather than a sentence naming
    MIMORA_HOME. paths.ensure_dirs() reports the same condition earlier and
    also carries on - the two together are what make a bad data root a bad run
    instead of a crash during an import.
    """
    global _append_logs
    _append_logs = append
    handlers = [logging.StreamHandler(sys.stdout)]
    file_error = None
    try:
        handlers.insert(0, logging.FileHandler(
            log_file, mode=log_file_mode(), encoding="utf-8"))
    except OSError as exc:
        file_error = exc
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=handlers,
        force=True,
    )
    if file_error is not None:
        # After basicConfig, so it travels through the console handler that was
        # just installed and reads like every other line of the run.
        logging.error("No log file this session: %s could not be opened (%s). "
                      "Logging to the console only. If MIMORA_HOME is set, "
                      "check that it names a writable directory.",
                      log_file, file_error)
    if append:
        # Mark the seam. Two processes write around it, and for a moment they
        # write at the same time: the parent lives on until its hard_exit(),
        # so its last records can land after the child's first ones. Without
        # a separator that stretch reads as one confused process.
        logging.info(
            "----- log continues here: restarted in-session, new pid %d -----",
            os.getpid())
