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

    def start(self, llm_mgr: LLMManager) -> bool:
        """Launch the server subprocess and block until it responds.

        Readiness is probed through ``llm_mgr``, whose client is (re)pointed
        at the local server here — the same client the app then uses for
        generation. Returns False on a missing model path, an early subprocess
        exit, or a startup timeout.
        """
        model_path = config.EXTERNAL_MODEL_PATH
        if not model_path:
            logging.error("EXTERNAL_MODEL_PATH is empty — cannot start local server.")
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
        self._log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        self._process = subprocess.Popen(cmd, stdout=self._log_file, stderr=self._log_file)

        deadline = time.time() + config.LOCAL_SERVER_STARTUP_TIMEOUT
        llm_mgr.init_client(base_url=config.LOCAL_SERVER_URL,
                            api_key=config.LOCAL_SERVER_API_KEY)
        while time.time() < deadline:
            if self._process.poll() is not None:
                logging.error(f"LLM server exited unexpectedly (code {self._process.returncode}).")
                self.shutdown()  # nothing to terminate; closes the log file
                return False
            if llm_mgr.check_connection(silent=True):
                logging.info("LLM server is ready.")
                return True
            time.sleep(1.0)

        logging.error("LLM server did not become ready within the timeout.")
        # The subprocess may still be loading the model — terminate it now
        # instead of leaving it holding VRAM until the app exits.
        self.shutdown()
        return False

    def shutdown(self):
        """Terminate the subprocess (kill on timeout) and close its log file.

        Safe to call repeatedly and when the server was never started — every
        step is a no-op then. Also called by start() on its failure paths.
        """
        if self._process is not None:
            if self._process.poll() is None:
                logging.info("Terminating LLM server subprocess...")
                self._process.terminate()
                try:
                    self._process.wait(timeout=SERVER_TERMINATE_TIMEOUT_SEC)
                except subprocess.TimeoutExpired:
                    logging.warning("LLM server did not exit cleanly — killing it.")
                    self._process.kill()
                    self._process.wait()  # reap the killed process (avoids a zombie on POSIX)
            self._process = None

        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
