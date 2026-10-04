# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Early process setup for the Mimora entry point.

Two phases, split because they bracket the heavy imports in app.py:

  * ``early_init()`` must run BEFORE the heavy mimora.* imports (torch,
    transformers, Kokoro): it sets the UTF-8 environment hints, switches
    the console streams to UTF-8, and installs the library warning filters.
  * ``setup_logging()`` must run AFTER those imports: basicConfig with
    force=True replaces any handlers auto-installed by logging calls during
    the imports (e.g. the acoustic engine loads calibration.json at import
    and logs it), which would otherwise turn basicConfig into a silent
    no-op and leave main.log empty.

``install_log()`` is a third piece and belongs to neither phase: a second,
append-only file for the stretch of a run that installs something, so that
what a first run downloaded is still readable after the next launch has
truncated main.log.

Only stdlib imports here, so ``from mimora import bootstrap`` stays free
and can precede everything heavy.
"""

import contextlib
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

# The rule that opens every run's header. 80 columns because that is what a
# console is, and a rule rather than a worded line because its whole job is to
# be findable by eye: with --append-log several runs share one file, and this
# is where each of them starts.
_HEADER_RULE = "=" * 80


def _launch_command() -> str:
    """How this process was started, as one printable line.

    ``sys.executable`` and ``sys.argv`` rather than a reconstruction: this is a
    log header, so the honest answer is the one the interpreter reports.
    lifecycle.relaunch_command() looks similar and is not interchangeable - it
    answers "what would start this process again" and deliberately drops the
    interpreter for a console script, which would hide the very difference this
    line exists to record.

    Arguments containing spaces are quoted, so a Windows path does not read as
    two arguments. That is also what makes the line usable as-is when somebody
    pastes it back into a shell to reproduce a report.
    """
    parts = [sys.executable or "<unknown interpreter>", *sys.argv]
    return " ".join(f'"{part}"' if " " in part else part for part in parts)


def _header_lines(append: bool) -> list[str]:
    """The lines a log opens with, as text.

    Built rather than logged so the same header can be written to a file that
    is not the root logger's - see :func:`install_log`. Two spellings of it
    would drift, and the header is the one part of a log that is read before
    anything is known about the run.

    The three facts are the ones every bug report opens by asking: which build
    is running, which process wrote these lines, and how it was started - the
    last one tells a source checkout from an installed console script, which is
    the difference paths.py branches on and therefore the difference between
    two entirely different sets of file locations.
    """
    # Function-local so the module-level rule above ("only stdlib imports")
    # stays literally true. mimora/__init__.py is a version string and a
    # docstring, and it has necessarily been imported already for this module
    # to exist, so the cost is a dictionary lookup.
    from mimora import __version__

    lines = []
    if append:
        # Only when continuing a file: the blank line separates this run from
        # the previous one, and a fresh log is supposed to OPEN with the rule
        # rather than with an empty line.
        lines.append("")
    lines.append(_HEADER_RULE)
    lines.append(f"Mimora {__version__}  |  pid {os.getpid()}"
                 + ("  |  continues the log above (restarted in-session)"
                    if append else ""))
    lines.append(f"Launched: {_launch_command()}")
    return lines


def open_log_section(handler: logging.Handler, log_file, intro=()) -> None:
    """Open one handler's file with the run header, below what it holds.

    For the two append-only logs, `install_log`'s and the hardware probe's:
    both are read by scrolling to the run one is interested in, so each run has
    to start with something findable. A blank line separates it from the
    previous run and is written only when there IS one, so a fresh file still
    opens with the rule.

    ``intro`` is written into this file only, under the header - the caller's
    summary of what the section is about. Nothing here travels through the root
    logger, so the console and main.log see none of it.
    """
    try:
        # Asked before anything is written, and of the path rather than the
        # handler: the handler has already created the file, so its own view
        # cannot tell an empty new file from one that holds ten runs.
        continued = os.path.getsize(log_file) > 0
    except OSError:
        continued = False
    _emit_to(handler, ([""] if continued else []) + _header_lines(append=False)
             + list(intro))


def _emit_to(handler: logging.Handler, lines) -> None:
    """Write *lines* to one handler as bare text, bypassing the root logger.

    For the files that open with a header of their own (see
    :func:`open_log_section`): the records are built here and handed straight
    to the handler, so nothing reaches the console or main.log a second time -
    the caller's summary would otherwise be logged twice in the same run.

    The formatter is swapped for a bare one and put back, the same trick and
    for the same reason as :func:`_log_header` - a header line is a header
    line, and a timestamp in front of the rule would make it just another
    record.
    """
    previous = handler.formatter
    handler.setFormatter(logging.Formatter("%(message)s"))
    try:
        for line in lines:
            handler.handle(logging.LogRecord(
                name=__name__, level=logging.INFO, pathname=__file__, lineno=0,
                msg="%s", args=(line,), exc_info=None))
    finally:
        handler.setFormatter(previous)


def _log_header(append: bool) -> None:
    """Open the run's log with a bare rule and what this process is.

    Emitted through ``logging`` so it reaches the console as well, but with the
    handlers' formatter swapped for a bare one for the length of it: the first
    line is meant to be a rule, and a timestamp and a level in front of it
    would make it just another record. The original formatter objects are put
    back immediately, so nothing downstream can tell this happened.

    What the lines say, and why those facts, is in :func:`_header_lines`.

    In append mode it also marks the seam. Two processes write around it, and
    for a moment they write at the same time: the parent lives on until its
    hard_exit(), so its last records can land after the child's first ones.
    Without a separator that stretch reads as one confused process, and the pid
    on the line above is what tells the two apart.
    """
    root = logging.getLogger()
    bare = logging.Formatter("%(message)s")
    previous = [handler.formatter for handler in root.handlers]
    for handler in root.handlers:
        handler.setFormatter(bare)
    try:
        for line in _header_lines(append):
            logging.info("%s", line)
    finally:
        # Restored even if a handler raises mid-header: a log that lost its
        # timestamps for the rest of the run would be a worse outcome than a
        # missing header.
        for handler, formatter in zip(root.handlers, previous):
            handler.setFormatter(formatter)


def log_file_mode():
    """The open() mode every log file of this process should use ("w" or "a").

    For log files opened directly rather than through ``logging``: today that
    is only logs/llm_server.log (see mimora/llm_server_ctl.py). Without this
    they would keep truncating across an in-session restart while main.log
    continues, which is the confusing half-state - the app's own log spans the
    restart and the server's log covers only what happened after it.
    """
    return "a" if _append_logs else "w"


# Kept OUT of the install log, matched on the top-level logger name. httpx
# logs one INFO line per request and a first run is hundreds of them: of the
# first install log measured, 106 KB of 112 KB were request lines carrying
# signed CDN URLs. They stay in main.log, which covers one run and is replaced
# by the next launch - this file is appended to for the life of the machine, so
# there the same lines accumulate instead. Nothing diagnostic is lost: a failed
# download reaches this file as the traceback first_run_download logs.
_INSTALL_LOG_MUTED = ("httpx",)


def _wanted_in_install_log(record: logging.LogRecord) -> bool:
    """False for the third-party chatter the install log does not keep."""
    return record.name.split(".", 1)[0] not in _INSTALL_LOG_MUTED


@contextlib.contextmanager
def install_log(log_file, intro=()):
    """Mirror everything logged inside the block into an append-only file.

    What the first-run window does is an installation, and main.log cannot keep
    it: that file covers one run and the next launch truncates it, so the
    record of a four-gigabyte download is gone by the time anybody asks what
    was downloaded. This is the file that survives, and it is the wheel
    install's counterpart to the logs/install.log that install.py writes -
    which is not in the wheel at all.

    The mode is ``"a"``, hardcoded rather than taken from
    :func:`log_file_mode`: never truncating is the whole point of this file,
    and an in-session restart (which is what that function answers for) would
    only add one more section.

    The handler goes on the ROOT logger, so the block captures every module and
    every thread, the download worker included. Keeping only ``mimora.*``
    instead would be the tempting shortcut and would drop exactly the
    third-party traceback that explains a failure; what is dropped is named
    instead, one logger at a time, in :data:`_INSTALL_LOG_MUTED`, and only on
    this handler.

    ``intro`` is written into this file only, under the header (see
    :func:`open_log_section`): the caller's summary of what is about to be
    installed.

    A file that cannot be opened costs the file and not the installation, the
    same rule as :func:`setup_logging` - the download itself is unaffected and
    is still logged as usual.
    """
    try:
        handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    except OSError as exc:
        logging.error("No install log this run: %s could not be opened (%s). "
                      "The download itself is unaffected and is logged as "
                      "usual.", log_file, exc)
        yield
        return

    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.addFilter(_wanted_in_install_log)
    # Written before the handler is attached, so the header cannot be preceded
    # by a record of some other thread that logged in the meantime.
    open_log_section(handler, log_file, intro)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield
    finally:
        # Removed before it is closed, and in this order: a background thread
        # that logs between the two would otherwise reach a closed file.
        root.removeHandler(handler)
        handler.close()


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
    # A windowless launch (the mimora-gui script) has no console, and both
    # streams are None there. print() accepts that, but a library that calls
    # sys.stderr.write() directly stops the application with an
    # AttributeError, so each missing stream gets a sink.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass  # stream may be None (pythonw), wrapped by an IDE, or already detached

    # Disable Hugging Face hub symlinks warning for a cleaner console output.
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # Stop transformers from converting our .bin checkpoints to safetensors
    # behind our back. Online, with a repo that ships no safetensors,
    # from_pretrained starts a background "Thread-auto_conversion" that opens a
    # conversion PR on the Hub and downloads the converted weights from
    # refs/pr/<n> - gigabytes nobody asked for, since we never request
    # safetensors.
    #
    # It also corrupts the cache in a way that does not heal: killed mid-flight
    # by the app exiting, it leaves blobs/<sha>.incomplete behind, which
    # loader.models_cached reads as "not cached", so a complete repo is reported
    # missing at every start and no download clears it (the download does not
    # need that file). model_fetch._sweep_incomplete_blobs heals such a cache.
    #
    # DISABLE_SAFETENSORS_CONVERSION is transformers' own switch and is read per
    # call rather than frozen at import, so setting it here covers every load.
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"

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
    first-run window and after a restart-only setting changes. Without it the
    child opens main.log with mode="w" and wipes everything the process that
    spawned it wrote - which is precisely the interesting part, the whole
    first-run download or the setting that caused the restart. A fresh
    launch still starts from an empty file, so the log stays one session
    long, restarts included.

    Either way the file opens with the run header (see _log_header): a rule,
    the build and pid, and the command line. In append mode that header is
    also the seam between the two processes.

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
    # Before anything else this process logs, including the failure below: the
    # header is what makes the first line of a log identifiable, and a run that
    # could not open its file still writes it to the console.
    _log_header(append)
    if file_error is not None:
        # After basicConfig, so it travels through the console handler that was
        # just installed and reads like every other line of the run.
        logging.error("No log file this session: %s could not be opened (%s). "
                      "Logging to the console only. If MIMORA_HOME is set, "
                      "check that it names a writable directory.",
                      log_file, file_error)
