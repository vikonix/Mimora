# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for the LLM-server control module (mimora/llm_server_ctl.py).

The command line carries every tuning decision that keeps llama-server at
parity with the app's expectations, and it is built by a pure function these
tests can check without ever spawning a process. The other half is the fork
that decides whether to launch at all: a server already listening on the port
(left by a crashed session, or started by another local app) is used as it is,
and a port held by anything else is refused with a sentence the window can
show. Run from the project root with:

    python -m unittest tests.test_llm_server_ctl
"""

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from mimora import config, llama_server_fetch, llm_server_ctl
from mimora.llm_server_ctl import LLMServerController, llama_server_command

MODEL = "/models/llama-3.2-3b-instruct-q4_k_m.gguf"
MODEL_FILE = "llama-3.2-3b-instruct-q4_k_m.gguf"
HOST = "127.0.0.1"
PORT = 8765
URL = f"http://{HOST}:{PORT}/v1"
NGL = 20
NCTX = 2048
API_KEY = "local"


def flag_value(cmd, flag):
    """Value following *flag* in a command list, or None when absent."""
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


def fake_manager(connected=True, served=(MODEL,), list_error=None):
    """An LLMManager stand-in with only the members the module uses.

    The model listing is shaped like the OpenAI client's answer (a page with a
    .data list of objects with an .id), because that is the only part of the
    library the controller reads.
    """
    manager = Mock()
    manager.check_connection.return_value = connected
    if list_error is not None:
        manager.client.models.list.side_effect = list_error
    else:
        manager.client.models.list.return_value = SimpleNamespace(
            data=[SimpleNamespace(id=name) for name in served])
    return manager


class LlamaServerCommandTests(unittest.TestCase):
    def setUp(self):
        self.cmd = llama_server_command(
            "/opt/llama/llama-server", MODEL, HOST, PORT, NGL, NCTX, API_KEY)

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
        # --no-webui is the same switch under its former name; the pinned build
        # accepts it but reports it as deprecated.
        self.assertIn("--no-ui", self.cmd)
        self.assertNotIn("--no-webui", self.cmd)

    def test_api_key_is_required_by_the_server(self):
        # Without it llama-server accepts unauthenticated requests from any
        # CORS origin - i.e. from any page open in a browser on this machine.
        self.assertEqual(flag_value(self.cmd, "--api-key"), API_KEY)

    def test_every_argument_is_a_string(self):
        for arg in self.cmd:
            self.assertIsInstance(arg, str)


class BuildCommandTests(unittest.TestCase):
    """The wiring and the "cannot start" paths of _build_command."""

    def _build(self, exe=__file__, **overrides):
        """Build a command with *exe* as the binary the resolver reports.

        exe is the resolver's answer rather than a constant because
        _build_command locates the binary when it runs (so a download that
        finished after config was imported is still seen). __file__ stands in
        for the binary - only its existence is checked, and this file exists.
        exe=None leaves the resolver alone for a test that patches it itself.
        """
        values = {
            "EXTERNAL_MODEL_PATH": MODEL,
            "LLM_SERVER_HOST": HOST,
            "LLM_SERVER_PORT": PORT,
            "EXTERNAL_N_GPU_LAYERS": NGL,
            "EXTERNAL_N_CTX": NCTX,
            "LLM_SERVER_API_KEY": API_KEY,
        }
        if exe is not None:
            values["resolve_llama_server_path"] = lambda: exe
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
        self.assertEqual(flag_value(cmd, "--api-key"), API_KEY)
        self.assertIn("--no-ui", cmd)

    def test_the_binary_is_located_when_the_command_is_built(self):
        # Not read from a value frozen at config's import: the first-run window
        # may download the binary after that import, and a stale empty string
        # would refuse to start a server this machine now has.
        resolve = Mock(return_value=__file__)
        with patch.object(config, "resolve_llama_server_path", resolve):
            self.assertIsNotNone(self._build(exe=None))
        resolve.assert_called_once_with()

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
        message = self._assert_refused(exe="")
        self.assertIn("llama_server_fetch", message)

    def test_binary_that_does_not_exist_is_refused(self):
        message = self._assert_refused(exe="/no/such/llama-server")
        self.assertIn("/no/such/llama-server", message)


class PortProbeTests(unittest.TestCase):
    """port_is_busy: one TCP connection, and every failure means "free"."""

    def test_a_listening_port_is_busy(self):
        with patch.object(llm_server_ctl.socket,
                          "create_connection") as connect:
            self.assertTrue(llm_server_ctl.port_is_busy(HOST, PORT))
        connect.assert_called_once_with((HOST, PORT),
                                        llm_server_ctl.PORT_PROBE_TIMEOUT_SEC)

    def test_a_refused_port_is_free(self):
        with patch.object(llm_server_ctl.socket, "create_connection",
                          side_effect=ConnectionRefusedError):
            self.assertFalse(llm_server_ctl.port_is_busy(HOST, PORT))

    def test_a_probe_failure_is_free_too(self):
        # Every OSError reads as "nothing is listening": the probe decides
        # whether to launch and must never raise out of start().
        with patch.object(llm_server_ctl.socket, "create_connection",
                          side_effect=TimeoutError):
            self.assertFalse(llm_server_ctl.port_is_busy(HOST, PORT))


class ServerPropertiesTests(unittest.TestCase):
    """server_properties: a plain GET of llama.cpp's own /props endpoint."""

    PROPS = {"default_generation_settings": {"n_ctx": NCTX}}

    def _read(self, body=None, error=None):
        """Read the properties with urlopen stubbed; return them and the stub.

        MagicMock and not Mock for the answer: urlopen's result is used as a
        context manager, which needs __enter__.
        """
        if error is not None:
            opener = Mock(side_effect=error)
        else:
            answer = MagicMock()
            answer.__enter__.return_value = io.BytesIO(body)
            opener = Mock(return_value=answer)
        with patch.object(llm_server_ctl.urllib.request, "urlopen", opener):
            return llm_server_ctl.server_properties(HOST, PORT, API_KEY), opener

    def test_the_answer_is_returned_as_a_dictionary(self):
        properties, _ = self._read(body=json.dumps(self.PROPS).encode())
        self.assertEqual(properties, self.PROPS)

    def test_the_request_goes_to_props_outside_the_v1_prefix(self):
        # /props is llama.cpp's own endpoint and is NOT under /v1, where the
        # OpenAI base URL points. A request to /v1/props answers 404.
        _, opener = self._read(body=b"{}")
        request = opener.call_args.args[0]
        self.assertEqual(request.full_url, f"http://{HOST}:{PORT}/props")
        self.assertEqual(opener.call_args.kwargs["timeout"],
                         llm_server_ctl.PROPS_TIMEOUT_SEC)

    def test_the_api_key_is_sent(self):
        # The server is started with --api-key and answers 401 without one.
        _, opener = self._read(body=b"{}")
        request = opener.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"),
                         f"Bearer {API_KEY}")

    def test_a_failed_request_is_not_an_error(self):
        # Every caller is diagnostic, so nothing here may reach start().
        with self.assertLogs(level="INFO"):
            properties, _ = self._read(error=OSError("connection reset"))
        self.assertIsNone(properties)

    def test_a_body_that_is_not_json_is_not_an_error(self):
        # What another program listening on the port would answer.
        with self.assertLogs(level="INFO"):
            properties, _ = self._read(body=b"<html>not this server</html>")
        self.assertIsNone(properties)

    def test_a_json_answer_that_is_not_an_object_is_refused(self):
        properties, _ = self._read(body=b"[1, 2]")
        self.assertIsNone(properties)


class RunningServerTests(unittest.TestCase):
    """The busy-port fork: adopt the server, or refuse the port."""

    def _start(self, manager, busy=True, props=None):
        """start() with the port probe, /props and the subprocess stubbed out.

        Popen is patched to assert it is NOT reached: every case here is
        decided before the launch, and a test that spawned llama-server would
        be an integration test with a GGUF model behind it. server_properties
        is patched for a plainer reason - unpatched it would open a socket to
        this machine's port 8765, which a unit test must not do.
        """
        with patch.object(llm_server_ctl, "port_is_busy", return_value=busy), \
                patch.object(llm_server_ctl, "server_properties",
                             return_value=props), \
                patch.object(llm_server_ctl.subprocess, "Popen") as popen, \
                patch.multiple(config, LLM_SERVER_HOST=HOST,
                               LLM_SERVER_PORT=PORT, LLM_SERVER_URL=URL,
                               LLM_SERVER_API_KEY=API_KEY,
                               EXTERNAL_MODEL_PATH=MODEL,
                               EXTERNAL_N_CTX=NCTX):
            controller = LLMServerController()
            with self.assertLogs(level="INFO") as captured:
                started = controller.start(manager)
        return controller, started, popen, "\n".join(captured.output)

    def test_a_running_server_is_used_instead_of_a_second_one(self):
        # The point of the whole fork: a launch could not bind this port.
        controller, started, popen, _ = self._start(fake_manager())
        self.assertTrue(started)
        self.assertTrue(controller.adopted)
        self.assertIsNone(controller.last_error)
        popen.assert_not_called()

    def test_the_client_points_at_the_server_before_the_port_is_asked(self):
        # The check of a busy port goes through this client, and the app then
        # generates phrases through the same one.
        manager = fake_manager()
        self._start(manager)
        manager.init_client.assert_called_once_with(base_url=URL,
                                                    api_key=API_KEY)
        calls = [name for name, _, _ in manager.mock_calls]
        self.assertLess(calls.index("init_client"),
                        calls.index("check_connection"))

    def test_using_a_foreign_server_is_a_warning(self):
        # Its model, context size and GPU layers are not the configured ones,
        # which is the first thing to know when phrases look odd.
        self.assertIn("WARNING", self._start(fake_manager())[3])

    def test_the_model_of_the_adopted_server_is_recorded(self):
        self.assertIn(MODEL, self._start(fake_manager())[3])

    def test_a_different_model_on_the_adopted_server_is_a_warning(self):
        # The intended case when another app serves a larger model: still
        # used, and the log names the model this run expected.
        controller, started, _, log = self._start(
            fake_manager(served=("some-other-model.gguf",)))
        self.assertTrue(started)
        self.assertTrue(controller.adopted)
        self.assertIn(MODEL_FILE, log)

    def test_an_unreadable_model_list_does_not_refuse_the_server(self):
        # Diagnostics may not turn a server that answers into a failed start.
        controller, started, _, _ = self._start(
            fake_manager(list_error=RuntimeError("no /v1/models here")))
        self.assertTrue(started)
        self.assertTrue(controller.adopted)

    def test_a_busy_port_without_the_api_names_the_port(self):
        # Something else holds it, so a launch would fail to bind. The message
        # has to name the port, because closing that program is the only cure.
        controller, started, popen, _ = self._start(
            fake_manager(connected=False))
        self.assertFalse(started)
        self.assertFalse(controller.adopted)
        self.assertIn(str(PORT), controller.last_error)
        popen.assert_not_called()

    def test_the_context_of_the_adopted_server_is_recorded(self):
        log = self._start(
            fake_manager(),
            props={"default_generation_settings": {"n_ctx": NCTX}})[3]
        self.assertIn(str(NCTX), log)

    def test_a_smaller_context_on_the_adopted_server_is_a_warning(self):
        # The harmful direction: that server cuts a prompt this run is sized
        # for, on its side, and nothing else would report it.
        controller, started, _, log = self._start(
            fake_manager(),
            props={"default_generation_settings": {"n_ctx": NCTX // 2}})
        self.assertTrue(started)
        self.assertTrue(controller.adopted)
        self.assertIn("smaller context", log)
        self.assertIn(str(NCTX // 2), log)

    def test_the_context_of_the_adopted_server_is_kept_for_the_window(self):
        controller, _, _, _ = self._start(
            fake_manager(),
            props={"default_generation_settings": {"n_ctx": NCTX // 2}})
        self.assertEqual(controller.served_n_ctx, NCTX // 2)

    def test_an_unreported_context_is_none(self):
        controller, _, _, _ = self._start(fake_manager(), props=None)
        self.assertIsNone(controller.served_n_ctx)

    def test_a_larger_context_is_not_a_warning(self):
        # It only costs VRAM on the other server, which is not our business.
        log = self._start(
            fake_manager(),
            props={"default_generation_settings": {"n_ctx": NCTX * 2}})[3]
        self.assertNotIn("smaller context", log)

    def test_an_unreadable_props_answer_does_not_refuse_the_server(self):
        # props=None is what server_properties returns for every failure.
        controller, started, _, _ = self._start(fake_manager(), props=None)
        self.assertTrue(started)
        self.assertTrue(controller.adopted)

    def test_a_props_answer_without_a_context_size_is_only_reported(self):
        controller, started, _, log = self._start(
            fake_manager(), props={"default_generation_settings": "unexpected"})
        self.assertTrue(started)
        self.assertTrue(controller.adopted)
        self.assertIn("did not report a context size", log)

    def test_an_adopted_server_is_not_terminated_on_shutdown(self):
        # It is not this process's subprocess, and another app may use it.
        controller, _, popen, _ = self._start(fake_manager())
        controller.shutdown()
        popen.return_value.terminate.assert_not_called()

    def test_a_free_port_is_not_adopted(self):
        # No API call at all in the normal case: the socket probe answers it.
        manager = fake_manager()
        with patch.object(llm_server_ctl, "port_is_busy", return_value=False), \
                patch.multiple(config, LLM_SERVER_HOST=HOST,
                               LLM_SERVER_PORT=PORT):
            controller = LLMServerController()
            self.assertFalse(controller._use_running_server(manager))
        self.assertFalse(controller.adopted)
        self.assertIsNone(controller.last_error)
        manager.check_connection.assert_not_called()


class LaunchedServerContextTests(unittest.TestCase):
    """A server this run started is asked for its context too.

    It is the only record of the context the server actually got, so the
    launched server gets the same /props check as an adopted one.
    """

    def _start(self, props):
        """start() through a launch, with every outside effect stubbed.

        The subprocess is a stub that never exits, the port is free, the
        device probe and /props are replaced, and the server log goes to a
        temporary directory.
        """
        process = Mock()
        process.poll.return_value = None
        manager = fake_manager()
        with tempfile.TemporaryDirectory() as log_dir, \
                patch.object(llm_server_ctl, "port_is_busy",
                             return_value=False), \
                patch.object(llm_server_ctl, "log_compute_devices"), \
                patch.object(llm_server_ctl, "server_properties",
                             return_value=props) as read_props, \
                patch.object(llm_server_ctl.subprocess, "Popen",
                             return_value=process), \
                patch.object(llm_server_ctl.bootstrap, "log_file_mode",
                             return_value="w"), \
                patch.multiple(config, LLM_SERVER_HOST=HOST,
                               LLM_SERVER_PORT=PORT, LLM_SERVER_URL=URL,
                               LLM_SERVER_API_KEY=API_KEY,
                               EXTERNAL_MODEL_PATH=MODEL,
                               EXTERNAL_N_GPU_LAYERS=NGL,
                               EXTERNAL_N_CTX=NCTX,
                               LLM_SERVER_LOG_FILE=str(
                                   Path(log_dir) / "llm_server.log"),
                               resolve_llama_server_path=lambda: __file__):
            controller = LLMServerController()
            with self.assertLogs(level="INFO") as captured:
                started = controller.start(manager)
            # Closes the log file before the directory is removed.
            controller._log_file.close()
        return controller, started, read_props, "\n".join(captured.output)

    def test_the_launched_server_is_asked_for_its_context(self):
        controller, started, read_props, _ = self._start(
            {"default_generation_settings": {"n_ctx": NCTX}})
        self.assertTrue(started)
        self.assertFalse(controller.adopted)
        read_props.assert_called_once_with(HOST, PORT, API_KEY)
        self.assertEqual(controller.served_n_ctx, NCTX)

    def test_a_shrunk_context_is_a_warning_and_not_a_failure(self):
        controller, started, _, log = self._start(
            {"default_generation_settings": {"n_ctx": NCTX // 4}})
        self.assertTrue(started)
        self.assertEqual(controller.served_n_ctx, NCTX // 4)
        self.assertIn("smaller context", log)

    def test_an_unreadable_answer_does_not_fail_the_start(self):
        controller, started, _, _ = self._start(None)
        self.assertTrue(started)
        self.assertIsNone(controller.served_n_ctx)


class LogComputeDevicesTests(unittest.TestCase):
    """The startup device probe - diagnostic only, must never raise."""

    CUDA_LISTING = "Available devices:\n  CUDA0: NVIDIA GeForce RTX 3090 (24576 MiB)"
    CPU_LISTING = "Available devices:\n  (none)"

    def _run(self, listing=CUDA_LISTING, variant="win-cuda-12.4-x64",
             error=None):
        """Run the probe with the fetch module's two lookups stubbed out.

        Both stubs stand in for a subprocess and a file on disk, neither of
        which a unit test should need.
        """
        list_devices = (Mock(side_effect=error) if error
                        else Mock(return_value=listing))
        with patch.multiple(llama_server_fetch,
                            list_devices=list_devices,
                            installed_variant=Mock(return_value=variant)):
            with self.assertLogs(level="INFO") as captured:
                llm_server_ctl.log_compute_devices("/opt/llama/llama-server")
        return captured.output

    def test_device_listing_is_recorded(self):
        self.assertIn("CUDA0", "\n".join(self._run()))

    def test_missing_gpu_device_is_a_warning(self):
        # The silent CPU fallback: the promised backend is simply not there.
        output = "\n".join(self._run(listing=self.CPU_LISTING))
        self.assertIn("WARNING", output)
        self.assertIn("llama_server_fetch", output)

    def test_cpu_variant_is_not_warned_about(self):
        # Nothing was promised, so nothing is missing.
        output = self._run(listing=self.CPU_LISTING, variant="win-cpu-x64")
        self.assertNotIn("WARNING", "\n".join(output))

    def test_foreign_binary_is_only_reported(self):
        # No stamp means no documented expectation to compare the listing to.
        output = self._run(listing=self.CPU_LISTING, variant=None)
        self.assertNotIn("WARNING", "\n".join(output))

    def test_probe_failure_does_not_propagate(self):
        # A server that would have started must not be blocked by diagnostics.
        error = llama_server_fetch.LlamaServerFetchError("binary would not run")
        output = "\n".join(self._run(error=error))
        self.assertIn("WARNING", output)
        self.assertIn("binary would not run", output)


class BackendListTests(unittest.TestCase):
    def test_llama_server_is_the_default_and_an_offered_choice(self):
        self.assertIn("llama-server", config.LLM_BACKEND_CHOICES)
        self.assertEqual(config.USER_SETTING_DEFAULTS["llm_backend"],
                         "llama-server")


if __name__ == "__main__":
    unittest.main()
