# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Unit tests for mimora.loader - the pure configuration machinery.

These exercise the validation rules and fallbacks in isolation: loader imports
only the standard library, so nothing here touches the HuggingFace stack or
config.py's import-time side effects. The one function that reaches for torch
(detect_device, lazily) gets a stub in sys.modules, so these stay fast and pass
with or without torch and a GPU. Run from the project root with:

    python -m unittest tests.test_loader
"""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from mimora import loader


class ReadJsonTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _write(self, name: str, text: str) -> Path:
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_missing_file_is_silent_empty(self):
        err = io.StringIO()
        with redirect_stderr(err):
            result = loader.read_json(self.dir / "absent.json")
        self.assertEqual(result, {})
        self.assertEqual(err.getvalue(), "")  # absence must not warn

    def test_valid_object(self):
        path = self._write("ok.json", '{"a": 1, "b": "x"}')
        self.assertEqual(loader.read_json(path), {"a": 1, "b": "x"})

    def test_invalid_json_warns_and_empties(self):
        path = self._write("broken.json", "{not valid")
        err = io.StringIO()
        with redirect_stderr(err):
            result = loader.read_json(path)
        self.assertEqual(result, {})
        self.assertIn("[config]", err.getvalue())

    def test_non_object_json_warns_and_empties(self):
        path = self._write("list.json", "[1, 2, 3]")
        err = io.StringIO()
        with redirect_stderr(err):
            result = loader.read_json(path)
        self.assertEqual(result, {})
        self.assertIn("must contain a JSON object", err.getvalue())


class UserNumberTests(unittest.TestCase):
    def _silent(self, *args, **kwargs):
        with redirect_stderr(io.StringIO()):
            return loader.user_number(*args, **kwargs)

    def test_missing_key_returns_default(self):
        self.assertEqual(loader.user_number({}, "k", 20), 20)

    def test_valid_value_passes_through(self):
        self.assertEqual(loader.user_number({"k": 5}, "k", 20), 5)

    def test_float_value_passes_through(self):
        self.assertEqual(loader.user_number({"k": 1.5}, "k", 1.0), 1.5)

    def test_non_numeric_returns_default(self):
        self.assertEqual(self._silent({"k": "x"}, "k", 20), 20)

    def test_bool_is_rejected(self):
        # bool is a subclass of int - must not be silently accepted as a number.
        self.assertEqual(self._silent({"k": True}, "k", 20), 20)

    def test_below_minimum_returns_default(self):
        self.assertEqual(self._silent({"k": 0}, "k", 20, minimum=1), 20)

    def test_above_maximum_returns_default(self):
        self.assertEqual(self._silent({"k": 150}, "k", 70, maximum=100), 70)

    def test_within_bounds_passes(self):
        self.assertEqual(loader.user_number({"k": 50}, "k", 70, minimum=0,
                                            maximum=100), 50)


class UserPathTests(unittest.TestCase):
    def setUp(self):
        self.base = Path("/base/dir")

    def test_missing_key_returns_default_str(self):
        default = self.base / "d.txt"
        self.assertEqual(loader.user_path({}, self.base, "k", default),
                         str(default))

    def test_relative_value_resolved_against_base(self):
        result = loader.user_path({"k": "sub/f.txt"}, self.base, "k",
                                  self.base / "d.txt")
        self.assertEqual(result, str(self.base / "sub/f.txt"))

    def test_non_string_returns_default(self):
        default = self.base / "d.txt"
        with redirect_stderr(io.StringIO()):
            result = loader.user_path({"k": 123}, self.base, "k", default)
        self.assertEqual(result, str(default))

    def test_blank_string_returns_default(self):
        default = self.base / "d.txt"
        with redirect_stderr(io.StringIO()):
            result = loader.user_path({"k": "   "}, self.base, "k", default)
        self.assertEqual(result, str(default))


class UserBoolTests(unittest.TestCase):
    def test_missing_key_returns_default(self):
        self.assertTrue(loader.user_bool({}, "k", True))

    def test_valid_bool_passes(self):
        self.assertFalse(loader.user_bool({"k": False}, "k", True))

    def test_non_bool_returns_default(self):
        with redirect_stderr(io.StringIO()):
            self.assertTrue(loader.user_bool({"k": "yes"}, "k", True))


class ServerUrlTests(unittest.TestCase):
    """server_url - every accepted spelling normalizes to the same base URL."""

    def test_bare_host_gets_default_port(self):
        self.assertEqual(loader.server_url("localhost", 1234),
                         "http://localhost:1234/v1")

    def test_host_with_port_kept(self):
        self.assertEqual(loader.server_url("100.96.0.54:5000", 1234),
                         "http://100.96.0.54:5000/v1")

    def test_full_url_passes_through(self):
        self.assertEqual(loader.server_url("http://100.96.0.54:1234", 1234),
                         "http://100.96.0.54:1234/v1")

    def test_full_url_with_v1_not_doubled(self):
        self.assertEqual(loader.server_url("http://host:1234/v1", 1234),
                         "http://host:1234/v1")

    def test_trailing_slash_stripped(self):
        self.assertEqual(loader.server_url("http://host:1234/v1/", 1234),
                         "http://host:1234/v1")

    def test_https_scheme_preserved(self):
        self.assertEqual(loader.server_url("https://host", 1234),
                         "https://host:1234/v1")

    def test_whitespace_stripped(self):
        self.assertEqual(loader.server_url("  host:1234 ", 1234),
                         "http://host:1234/v1")


class SaveSettingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "settings.json"
        self.addCleanup(self._tmp.cleanup)

    def test_writes_and_updates_memory(self):
        memory = {}
        ok = loader.save_setting(self.path, "voice", "af_heart", memory)
        self.assertTrue(ok)
        self.assertEqual(memory["voice"], "af_heart")
        on_disk = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, {"voice": "af_heart"})

    def test_preserves_existing_keys(self):
        self.path.write_text('{"_comment": "keep me", "voice": "old"}',
                             encoding="utf-8")
        loader.save_setting(self.path, "voice", "new", {})
        on_disk = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, {"_comment": "keep me", "voice": "new"})


class ModelsCachedTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.hub = Path(self._tmp.name) / "hub"
        self.addCleanup(self._tmp.cleanup)
        self.repos = ("org/model-a",)

    def _repo_dir(self, repo: str) -> Path:
        return self.hub / ("models--" + repo.replace("/", "--"))

    def _make_cached(self, repo: str):
        snap = self._repo_dir(repo) / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}", encoding="utf-8")
        (self._repo_dir(repo) / "blobs").mkdir(parents=True)

    def test_missing_hub_is_not_cached(self):
        self.assertFalse(loader.models_cached(self.hub, self.repos))

    def test_empty_snapshots_is_not_cached(self):
        (self._repo_dir("org/model-a") / "snapshots").mkdir(parents=True)
        self.assertFalse(loader.models_cached(self.hub, self.repos))

    def test_complete_repo_is_cached(self):
        self._make_cached("org/model-a")
        self.assertTrue(loader.models_cached(self.hub, self.repos))

    def test_incomplete_blob_blocks_cached(self):
        self._make_cached("org/model-a")
        (self._repo_dir("org/model-a") / "blobs" / "x.incomplete").write_text(
            "", encoding="utf-8")
        self.assertFalse(loader.models_cached(self.hub, self.repos))


class _FakeTorch:
    """Stand-in for the torch module, so these tests need neither it nor a GPU.

    Installed into sys.modules, which is what `import torch` inside
    detect_device consults first. A sys.modules entry of None is the documented
    way to make that same import raise ImportError, which covers the third case.
    """

    def __init__(self, cuda_available: bool):
        self.cuda = type("cuda", (), {
            "is_available": staticmethod(lambda: cuda_available)})


class DetectDeviceTests(unittest.TestCase):
    def _detect(self, hw_value, torch_module) -> tuple[str, str]:
        """Return (device, stderr) with *torch_module* standing in for torch."""
        err = io.StringIO()
        with mock.patch.dict(sys.modules, {"torch": torch_module}), \
                redirect_stderr(err):
            return loader.detect_device(hw_value), err.getvalue()

    def test_cpu_short_circuits_without_asking_torch(self):
        # The fake reports CUDA: anything other than "cpu" here would mean the
        # value was probed rather than trusted, i.e. the ~1s import was paid.
        device, err = self._detect("cpu", _FakeTorch(True))
        self.assertEqual(device, "cpu")
        self.assertEqual(err, "")

    def test_cuda_confirmed_by_torch(self):
        device, err = self._detect("cuda", _FakeTorch(True))
        self.assertEqual(device, "cuda")
        self.assertEqual(err, "")

    def test_stale_cuda_falls_back_to_cpu(self):
        device, err = self._detect("cuda", _FakeTorch(False))
        self.assertEqual(device, "cpu")
        self.assertIn("hardware_config.json", err)

    def test_cuda_without_torch_falls_back_to_cpu(self):
        device, err = self._detect("cuda", None)  # None => import raises
        self.assertEqual(device, "cpu")
        self.assertIn("hardware_config.json", err)

    def test_no_hw_value_probes_torch(self):
        self.assertEqual(self._detect(None, _FakeTorch(True))[0], "cuda")
        self.assertEqual(self._detect(None, _FakeTorch(False))[0], "cpu")

    def test_unknown_hw_value_probes_torch_silently(self):
        # An unrecognised value is not a claim about CUDA, so it is not worth a
        # message about a stale file.
        device, err = self._detect("gpu", _FakeTorch(False))
        self.assertEqual(device, "cpu")
        self.assertEqual(err, "")


if __name__ == "__main__":
    unittest.main()
