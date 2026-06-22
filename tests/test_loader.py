# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Unit tests for mimora.loader — the pure configuration machinery.

These exercise the validation rules and fallbacks in isolation: loader imports
only the standard library, so nothing here touches torch, the HuggingFace stack,
or config.py's import-time side effects. Run from the project root with:

    python -m unittest tests.test_loader
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

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
        # bool is a subclass of int — must not be silently accepted as a number.
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


class DetectDeviceTests(unittest.TestCase):
    # Only the short-circuit branches are tested: a valid hw_value must be
    # returned verbatim without importing torch, so these stay fast and do not
    # depend on torch being installed or a GPU being present.
    def test_cuda_passes_through(self):
        self.assertEqual(loader.detect_device("cuda"), "cuda")

    def test_cpu_passes_through(self):
        self.assertEqual(loader.detect_device("cpu"), "cpu")


if __name__ == "__main__":
    unittest.main()
