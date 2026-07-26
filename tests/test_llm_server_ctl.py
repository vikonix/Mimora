# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Unit tests for the LLM-server command building (mimora/llm_server_ctl.py).

The command line carries every tuning decision that keeps llama-server at
parity with the app's expectations, and it is built by a pure function these
tests can check without ever spawning a process. Run from the project root with:

    python -m unittest tests.test_llm_server_ctl
"""

import sys
import unittest
from unittest.mock import patch

from mimora import config
from mimora.llm_server_ctl import LLMServerController, llama_server_command

MODEL = "/models/llama-3.2-3b-instruct-q4_k_m.gguf"
HOST = "127.0.0.1"
PORT = 8765
NGL = 20
NCTX = 2048


def flag_value(cmd, flag):
    """Value following *flag* in a command list, or None when absent."""
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


class LlamaServerCommandTests(unittest.TestCase):
    def setUp(self):
        self.cmd = llama_server_command(
            "/opt/llama/llama-server", MODEL, HOST, PORT, NGL, NCTX)

    def test_runs_the_binary_directly(self):
        self.assertEqual(self.cmd[0], "/opt/llama/llama-server")
        self.assertNotIn(sys.executable, self.cmd)

    def test_passes_the_configured_values(self):
        self.assertEqual(flag_value(self.cmd, "-m"), MODEL)
        self.assertEqual(flag_value(self.cmd, "--host"), HOST)
        self.assertEqual(flag_value(self.cmd, "--port"), str(PORT))
        self.assertEqual(flag_value(self.cmd, "--n-gpu-layers"), str(NGL))

    def test_context_size_is_explicit(self):
        # Never leave it to the default (-c 0): that takes the model's own
        # training context (131072 for Llama 3.2) and inflates the KV cache.
        self.assertEqual(flag_value(self.cmd, "--ctx-size"), str(NCTX))
        # --n-ctx is llama-cpp-python's spelling and llama-server rejects it.
        self.assertNotIn("--n-ctx", self.cmd)

    def test_single_slot_and_prefix_reuse_are_explicit(self):
        # Both defaults are actively harmful here: several slots fragment the
        # prefix cache the sliding-window prompt depends on, and cache-reuse 0
        # gives up prefix reuse the previous backend had.
        self.assertEqual(flag_value(self.cmd, "--parallel"), "1")
        self.assertEqual(flag_value(self.cmd, "--cache-reuse"), "256")

    def test_web_ui_is_disabled(self):
        self.assertIn("--no-webui", self.cmd)

    def test_every_argument_is_a_string(self):
        for arg in self.cmd:
            self.assertIsInstance(arg, str)


class BuildCommandTests(unittest.TestCase):
    """The wiring and the "cannot start" paths of _build_command."""

    def _build(self, **overrides):
        values = {
            "EXTERNAL_MODEL_PATH": MODEL,
            "LLM_SERVER_HOST": HOST,
            "LLM_SERVER_PORT": PORT,
            "EXTERNAL_N_GPU_LAYERS": NGL,
            "EXTERNAL_N_CTX": NCTX,
            # __file__ stands in for the binary: _build_command only checks
            # that the path exists, and this file certainly does.
            "LLAMA_SERVER_PATH": __file__,
        }
        values.update(overrides)
        with patch.multiple(config, **values):
            return LLMServerController()._build_command()

    def test_builds_the_binary_command_from_config(self):
        cmd = self._build()
        self.assertEqual(cmd[0], __file__)
        self.assertNotIn(sys.executable, cmd)
        self.assertEqual(flag_value(cmd, "-m"), MODEL)
        self.assertEqual(flag_value(cmd, "--port"), str(PORT))
        self.assertEqual(flag_value(cmd, "--ctx-size"), str(NCTX))
        self.assertIn("--no-webui", cmd)

    def _assert_refused(self, **overrides):
        """The build returns None AND says why.

        assertLogs doubles as noise control: it captures the record instead of
        letting the expected error reach the console during a test run.
        """
        with self.assertLogs(level="ERROR") as captured:
            self.assertIsNone(self._build(**overrides))
        return captured.output[0]

    def test_missing_model_path_is_refused(self):
        self.assertIn("EXTERNAL_MODEL_PATH",
                      self._assert_refused(EXTERNAL_MODEL_PATH=""))

    def test_unresolvable_binary_is_refused(self):
        # Nothing configured, nothing installed, nothing on PATH: the message
        # has to point at the fetch command, since there is no path to blame.
        message = self._assert_refused(LLAMA_SERVER_PATH="")
        self.assertIn("llama_server_fetch", message)

    def test_binary_that_does_not_exist_is_refused(self):
        message = self._assert_refused(LLAMA_SERVER_PATH="/no/such/llama-server")
        self.assertIn("/no/such/llama-server", message)


class BackendListTests(unittest.TestCase):
    def test_llama_server_is_the_default_and_an_offered_choice(self):
        self.assertIn("llama-server", config.LLM_BACKEND_CHOICES)
        self.assertEqual(config.USER_SETTING_DEFAULTS["llm_backend"],
                         "llama-server")

    def test_retired_backend_is_gone(self):
        # local_server (the llama-cpp-python wrapper) must not come back as a
        # selectable value; settings.json files naming it are migrated instead
        # (loader.migrate_llm_backend).
        self.assertNotIn("local_server", config.LLM_BACKEND_CHOICES)


if __name__ == "__main__":
    unittest.main()
