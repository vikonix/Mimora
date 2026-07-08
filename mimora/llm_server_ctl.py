# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Lifecycle control for the local LLM server subprocess.

Used only with the "local_server" LLM backend: launches llm_server/server.py
with the configured model, waits until it answers, and terminates it on app
shutdown. Phrase generation itself goes through LLMManager (llm.py).
"""

import logging
import subprocess
import sys
import threading
import time
from typing import Optional

from mimora import config
from mimora.llm import LLMManager

SERVER_SCRIPT = str(config.BASE_DIR / "llm_server" / "server.py")

# How long to wait for a graceful exit before killing the subprocess.
SERVER_TERMINATE_TIMEOUT_SEC = 5


class LLMServerController:
    """Starts and stops the llm_server/server.py subprocess."""

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

    def start(self, llm_mgr: LLMManager) -> bool:
        """Launch the server subprocess and block until it responds.

        Readiness is probed through ``llm_mgr``, whose client is (re)pointed
        at the local server here - the same client the app then uses for
        generation. Returns False on a missing model path, an early subprocess
        exit, a startup timeout, or when shutdown() has already been requested.
        """
        model_path = config.EXTERNAL_MODEL_PATH
        if not model_path:
            logging.error("EXTERNAL_MODEL_PATH is empty - cannot start local server.")
            return False

        cmd = [
            sys.executable,
            SERVER_SCRIPT,
            "--model", model_path,
            "--host", config.LOCAL_SERVER_HOST,
            "--port", str(config.LOCAL_SERVER_PORT),
            "--n-gpu-layers", str(config.EXTERNAL_N_GPU_LAYERS),
            "--n-ctx", str(config.EXTERNAL_N_CTX),
        ]
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

        deadline = time.time() + config.LOCAL_SERVER_STARTUP_TIMEOUT
        llm_mgr.init_client(base_url=config.LOCAL_SERVER_URL,
                            api_key=config.LOCAL_SERVER_API_KEY)
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
