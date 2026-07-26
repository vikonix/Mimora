# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Unit tests for the GGUF downloader (mimora/gguf_fetch.py).

No network: what matters here is the presence check install.py and the app
branch on, and that ensure_gguf() writes where config.EXTERNAL_MODEL_PATH
expects. Run from the project root with:

    python -m unittest tests.test_gguf_fetch
"""

import contextlib
import io
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mimora import config, gguf_fetch

_saved_levels: dict = {}


def setUpModule():
    """Silence the fetchers' INFO chatter for the duration of the suite.

    model_fetch is included because gguf_fetch delegates prepare_hf_env() to
    it, and that is where the Windows symlink notice comes from.
    """
    for name in (gguf_fetch.__name__, "mimora.model_fetch"):
        logger = logging.getLogger(name)
        _saved_levels[name] = logger.level
        logger.setLevel(logging.CRITICAL)


def tearDownModule():
    for name, level in _saved_levels.items():
        logging.getLogger(name).setLevel(level)


class DefaultTargetTests(unittest.TestCase):
    def test_default_path_matches_the_app_default(self):
        # The point of naming the file in two places is that the app finds the
        # download without a settings change. Compared against the DEFAULT, not
        # the live EXTERNAL_MODEL_PATH, so a developer whose settings.json
        # points at another GGUF does not fail the suite.
        default = Path(config.USER_SETTING_DEFAULTS["external_model_path"])
        self.assertEqual(gguf_fetch.DEFAULT_GGUF_PATH.name, default.name)
        self.assertEqual(gguf_fetch.DEFAULT_GGUF_PATH.parent.name,
                         default.parent.name)


class GgufPresentTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.target = Path(self._tmp.name) / "model.gguf"

    def test_absent_file(self):
        self.assertFalse(gguf_fetch.gguf_present(self.target))

    def test_present_file(self):
        self.target.write_bytes(b"gguf")
        self.assertTrue(gguf_fetch.gguf_present(self.target))

    def test_a_directory_is_not_the_model(self):
        self.target.mkdir()
        self.assertFalse(gguf_fetch.gguf_present(self.target))


class EnsureGgufTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.target = Path(self._tmp.name) / "models" / "model.gguf"

    def test_present_file_is_not_downloaded_again(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b"gguf")
        with patch("huggingface_hub.hf_hub_download", create=True) as download:
            result = gguf_fetch.ensure_gguf(self.target)
        download.assert_not_called()
        self.assertEqual(result, self.target)

    def test_force_downloads_over_a_present_file(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b"gguf")
        with patch("huggingface_hub.hf_hub_download",
                   return_value=str(self.target), create=True) as download:
            gguf_fetch.ensure_gguf(self.target, force=True)
        download.assert_called_once()

    def test_downloads_into_the_targets_directory(self):
        with patch("huggingface_hub.hf_hub_download",
                   return_value=str(self.target), create=True) as download:
            gguf_fetch.ensure_gguf(self.target)
        _, kwargs = download.call_args
        self.assertEqual(kwargs["repo_id"], gguf_fetch.GGUF_REPO_ID)
        self.assertEqual(kwargs["filename"], self.target.name)
        self.assertEqual(kwargs["local_dir"], str(self.target.parent))
        # The parent must exist before hf_hub_download is asked to write there.
        self.assertTrue(self.target.parent.is_dir())

    def test_download_failure_becomes_a_gguf_fetch_error(self):
        with patch("huggingface_hub.hf_hub_download",
                   side_effect=RuntimeError("no network"), create=True):
            with self.assertRaises(gguf_fetch.GgufFetchError):
                gguf_fetch.ensure_gguf(self.target)


class CliTests(unittest.TestCase):
    def setUp(self):
        # main() configures logging for its CLI use; letting it install a root
        # handler here would leak into every test module that runs afterwards.
        patcher = patch("logging.basicConfig")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_defaults(self):
        args = gguf_fetch.parse_args([])
        self.assertEqual(args.target, gguf_fetch.DEFAULT_GGUF_PATH)
        self.assertFalse(args.force)
        self.assertFalse(args.list)

    def test_list_runs_no_download(self):
        with patch.object(gguf_fetch, "_print_status") as printer, \
                patch.object(gguf_fetch, "ensure_gguf") as ensure:
            self.assertEqual(gguf_fetch.main(["--list"]), 0)
        printer.assert_called_once()
        ensure.assert_not_called()

    def test_failure_becomes_a_nonzero_exit(self):
        # The error goes to stderr for the user; captured so the test report
        # does not look like something went wrong.
        stderr = io.StringIO()
        with patch.object(gguf_fetch, "ensure_gguf",
                          side_effect=gguf_fetch.GgufFetchError("boom")), \
                contextlib.redirect_stderr(stderr):
            self.assertEqual(gguf_fetch.main([]), 1)
        self.assertIn("boom", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
