# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Lifecycle control for the local LLM server subprocess.

Used with the "llama-server" backend, the only one that runs a server of its
own: the controller launches the official llama.cpp binary, waits until it
answers, and terminates it on app shutdown. Phrase generation itself goes
through LLMManager (llm.py).

The server speaks the same OpenAI-compatible API as LM Studio and answers 503
while the model is still loading, so the readiness poll is just
LLMManager.check_connection against config.LLM_SERVER_URL.
"""

import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from mimora import config, llama_server_fetch
from mimora.llm import LLMManager

# How long to wait for a graceful exit before killing the subprocess.
SERVER_TERMINATE_TIMEOUT_SEC = 5

# llama-server tuning that must never be left to the binary's own defaults.
# All three were measured against the previous backend before this one became
# the default; the numbers below are what brought the two to parity:
#   --parallel 1     : the default (-1) opens several slots, splits the context
#                      between them and routes a request to whichever slot is
#                      most similar, which fragments the prefix cache.
#   --cache-reuse 256: brings prefix reuse up to parity with llama-cpp-python,
#                      which the sliding-window prompt in llm.py relies on.
# --ctx-size is passed explicitly for the same reason: the default (0) takes
# the model's own training context (131072 for Llama 3.2) and inflates the KV
# cache to fill free VRAM.
LLAMA_SERVER_PARALLEL_SLOTS = 1
LLAMA_SERVER_CACHE_REUSE = 256


def llama_server_command(exe_path: str, model_path: str, host: str, port: int,
                         n_gpu_layers: int, n_ctx: int, api_key: str) -> list:
    """Command line for the llama.cpp binary (the "llama-server" backend).

    --no-ui drops the bundled browser UI: the app talks HTTP only, and not
    serving the assets keeps the surface small. (--no-webui is the same switch
    under its former name; the binary now reports that spelling as deprecated.)

    --api-key answers the warning llama-server prints when it starts without
    one. The server allows every CORS origin, so without a key any page open in
    a browser could call this port and read the answer. The value is the one
    LLMManager already sends, so requiring it costs nothing.
    """
    return [
        exe_path,
        "-m", model_path,
        "--host", host,
        "--port", str(port),
        "--n-gpu-layers", str(n_gpu_layers),
        "--ctx-size", str(n_ctx),
        "--parallel", str(LLAMA_SERVER_PARALLEL_SLOTS),
        "--cache-reuse", str(LLAMA_SERVER_CACHE_REUSE),
        "--api-key", api_key,
        "--no-ui",
    ]


def log_compute_devices(exe_path: str) -> None:
    """Record which devices llama-server sees; warn on a silent CPU fallback.

    llama.cpp drops to the CPU *without an error* when a CUDA build cannot load
    its runtime DLLs: it still logs "offloaded N/N layers to GPU", still
    answers every request, and is merely about three times slower. At the
    default verbosity the server's own log shows neither the buffer names nor
    the chosen devices, so nothing in a normal run would reveal this - only a
    speed comparison would, and only if someone thought to make one.

    Purely diagnostic: the probe loads no model, and any failure here is
    reported and shrugged off rather than blocking a server that would have
    started fine.
    """
    exe = Path(exe_path)
    try:
        devices = llama_server_fetch.list_devices(exe).strip()
    except llama_server_fetch.LlamaServerFetchError as exc:
        logging.warning("Could not list llama-server compute devices: %s", exc)
        return
    logging.info("llama-server compute devices:\n%s", devices)

    variant_name = llama_server_fetch.installed_variant(exe)
    if variant_name is None:
        # A binary the user manages themselves: we have no idea which backend
        # it was built with, so the listing above is the whole report.
        return
    pattern = llama_server_fetch.VARIANTS[variant_name].device_pattern
    if pattern is None or re.search(pattern, devices):
        return
    logging.warning(
        "llama-server is installed as %s but reports no matching compute "
        "device - it will run on the CPU, roughly three times slower, without "
        "reporting an error. Re-run `python -m mimora.llama_server_fetch "
        "--force` to repair the installation.", variant_name)


class LLMServerController:
    """Starts and stops the llama-server subprocess."""

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._log_file = None
        # Serializes subprocess creation (start) against teardown (shutdown)
        # and makes shutdown() itself safe to reach from two threads at once
        # (loader thread on a start() failure path + Tk thread in quit_app).
        self._shutdown_lock = threading.Lock()
        # Set by shutdown(). A start() that loses the race to a quit_app
        # shutdown must not spawn a server afterwards - nothing would ever
        # terminate it. One-way by design: start() runs once per process
        # (load_components) and is never retried after shutdown.
        self._shutdown_requested = False

    def _build_command(self) -> Optional[list]:
        """Command line for llama-server, or None on a bad setup.

        Every "cannot start" reason is logged here and reported to the caller
        as None, so start() has a single failure path and main.py keeps its
        one error message for the user.

        The binary is located here rather than read from a constant frozen at
        config's import: the first-run window may have downloaded it after that
        import, and a stale empty string would then refuse to start a server
        this machine now has. Everything else this method reads is already
        taken at call time for the same reason.
        """
        model_path = config.EXTERNAL_MODEL_PATH
        if not model_path:
            logging.error("EXTERNAL_MODEL_PATH is empty - cannot start the LLM server.")
            return None

        exe_path = config.resolve_llama_server_path()
        if not exe_path:
            logging.error(
                "llama-server binary not found: settings.json "
                "'llama_server_path' is empty, %s holds no installation and "
                "no llama-server is on PATH. Run "
                "`python -m mimora.llama_server_fetch` or set the path.",
                llama_server_fetch.INSTALL_DIR)
            return None
        if not os.path.isfile(exe_path):
            logging.error("llama-server binary not found at %s (settings.json "
                          "'llama_server_path').", exe_path)
            return None
        return llama_server_command(
            exe_path, model_path, config.LLM_SERVER_HOST,
            config.LLM_SERVER_PORT, config.EXTERNAL_N_GPU_LAYERS,
            config.EXTERNAL_N_CTX, config.LLM_SERVER_API_KEY)

    def start(self, llm_mgr: LLMManager) -> bool:
        """Launch the server subprocess and block until it responds.

        Readiness is probed through ``llm_mgr``, whose client is (re)pointed
        at the local server here - the same client the app then uses for
        generation. Returns False on an unusable configuration (see
        _build_command), an early subprocess exit, a startup timeout, or when
        shutdown() has already been requested.
        """
        cmd = self._build_command()
        if cmd is None:
            return False

        # Before the model load makes the wait long: the probe is sub-second
        # and its answer is the only record of which backend came up.
        log_compute_devices(cmd[0])

        log_path = config.LLM_SERVER_LOG_FILE
        logging.info(f"Starting LLM server: {' '.join(cmd)}")
        # Creation runs under the same lock as shutdown(), so the two cannot
        # interleave: either shutdown() runs first and the flag stops the
        # launch, or the subprocess is fully published before shutdown() gets
        # the lock and terminates it. Without this, a quit during startup
        # could leave a freshly spawned server orphaned.
        with self._shutdown_lock:
            if self._shutdown_requested:
                logging.info("LLM server start aborted: shutdown requested.")
                return False
            self._log_file = open(log_path, "w", encoding="utf-8", buffering=1)
            try:
                self._process = subprocess.Popen(
                    cmd, stdout=self._log_file, stderr=self._log_file)
            except Exception:
                # Don't leak the just-opened log file when the launch itself
                # fails (e.g. a missing interpreter); the exception still
                # propagates to the caller's error handling.
                self._log_file.close()
                self._log_file = None
                raise

        deadline = time.time() + config.LLM_SERVER_STARTUP_TIMEOUT
        llm_mgr.init_client(base_url=config.LLM_SERVER_URL,
                            api_key=config.LLM_SERVER_API_KEY)
        while time.time() < deadline:
            # Snapshot the process reference: shutdown() (called from quit_app
            # on the Tk main thread while this loop runs on the loader thread)
            # sets self._process to None, and reading it twice would race that
            # and crash on None.poll(). A cleared reference means the app is
            # quitting - stop waiting quietly.
            process = self._process
            if process is None:
                logging.info("LLM server startup aborted: shutdown requested.")
                return False
            if process.poll() is not None:
                logging.error(f"LLM server exited unexpectedly (code {process.returncode}).")
                self.shutdown()  # nothing to terminate; closes the log file
                return False
            if llm_mgr.check_connection(silent=True):
                logging.info("LLM server is ready.")
                return True
            time.sleep(1.0)

        logging.error("LLM server did not become ready within the timeout.")
        # The subprocess may still be loading the model - terminate it now
        # instead of leaving it holding VRAM until the app exits.
        self.shutdown()
        return False

    def shutdown(self):
        """Terminate the subprocess (kill on timeout) and close its log file.

        Safe to call repeatedly and when the server was never started - every
        step is a no-op then. Also called by start() on its failure paths, so
        it can run concurrently on the loader thread and the Tk main thread
        (quit_app); the lock makes the check-then-use on the process and log
        file atomic - the loser of the race sees None and does nothing. Also
        flags the controller so a start() still ahead of its Popen call aborts
        instead of spawning a server nothing would terminate.
        """
        with self._shutdown_lock:
            self._shutdown_requested = True
            process, self._process = self._process, None
            log_file, self._log_file = self._log_file, None

        if process is not None:
            if process.poll() is None:
                logging.info("Terminating LLM server subprocess...")
                process.terminate()
                try:
                    process.wait(timeout=SERVER_TERMINATE_TIMEOUT_SEC)
                except subprocess.TimeoutExpired:
                    logging.warning("LLM server did not exit cleanly - killing it.")
                    process.kill()
                    process.wait()  # reap the killed process (avoids a zombie on POSIX)

        if log_file is not None:
            log_file.close()
