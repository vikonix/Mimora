# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Unit tests for the model downloader (mimora/model_fetch.py).

Nothing here touches the network or the real cache: the "is it downloaded?"
predicates and the environment preparation are the parts install.py and the app
branch on, and both are pure enough to check against a temporary directory.
Run from the project root with:

    python -m unittest tests.test_model_fetch
"""

import contextlib
import io
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mimora import model_fetch, models_info

_saved_level = logging.NOTSET


def setUpModule():
    """Silence the module's own INFO chatter for the duration of the suite.

    "Fetching X", "Already cached: Y" and the Windows symlink notice are the
    point of the module when a user runs it, and 30 lines of noise around the
    assertions when unittest does.
    """
    global _saved_level
    logger = logging.getLogger(model_fetch.__name__)
    _saved_level = logger.level
    logger.setLevel(logging.CRITICAL)


def tearDownModule():
    logging.getLogger(model_fetch.__name__).setLevel(_saved_level)


class EnvHelperTests(unittest.TestCase):
    """hf_home() / supertonic_cache_dir() must follow the environment."""

    def test_hf_home_falls_back_to_model_cache(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_HOME", None)
            self.assertEqual(model_fetch.hf_home(),
                             model_fetch.MODEL_CACHE_DIR)

    def test_hf_home_honours_external_setting(self):
        with patch.dict(os.environ, {"HF_HOME": "/elsewhere/hf"}):
            self.assertEqual(model_fetch.hf_home(), Path("/elsewhere/hf"))

    def test_supertonic_dir_falls_back_to_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SUPERTONIC_CACHE_DIR", None)
            self.assertEqual(model_fetch.supertonic_cache_dir(),
                             model_fetch.DEFAULT_SUPERTONIC_CACHE_DIR)

    def test_supertonic_dir_honours_external_setting(self):
        with patch.dict(os.environ, {"SUPERTONIC_CACHE_DIR": "/elsewhere/st"}):
            self.assertEqual(model_fetch.supertonic_cache_dir(),
                             Path("/elsewhere/st"))


class PrepareHfEnvTests(unittest.TestCase):
    """prepare_hf_env() sets defaults without overriding what is already set."""

    def test_sets_both_cache_variables(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_HOME", None)
            os.environ.pop("SUPERTONIC_CACHE_DIR", None)
            model_fetch.prepare_hf_env()
            self.assertEqual(os.environ["HF_HOME"],
                             str(model_fetch.MODEL_CACHE_DIR))
            self.assertEqual(os.environ["SUPERTONIC_CACHE_DIR"],
                             str(model_fetch.DEFAULT_SUPERTONIC_CACHE_DIR))

    def test_keeps_an_externally_configured_cache(self):
        # setdefault, not assignment: a user who points HF_HOME at a shared
        # cache must keep it, or the installer would silently download twice.
        with patch.dict(os.environ, {"HF_HOME": "/shared/hf",
                                     "SUPERTONIC_CACHE_DIR": "/shared/st"}):
            model_fetch.prepare_hf_env()
            self.assertEqual(os.environ["HF_HOME"], "/shared/hf")
            self.assertEqual(os.environ["SUPERTONIC_CACHE_DIR"], "/shared/st")


class SupertonicCachedTests(unittest.TestCase):
    """A present, non-empty directory means a complete download."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache_dir = Path(self._tmp.name) / "supertonic3"
        patcher = patch.dict(os.environ,
                             {"SUPERTONIC_CACHE_DIR": str(self.cache_dir)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_missing_directory(self):
        self.assertFalse(model_fetch.supertonic_cached())

    def test_empty_directory(self):
        self.cache_dir.mkdir()
        self.assertFalse(model_fetch.supertonic_cached())

    def test_directory_with_a_file(self):
        self.cache_dir.mkdir()
        (self.cache_dir / "model.onnx").write_bytes(b"x")
        self.assertTrue(model_fetch.supertonic_cached())


class HfRepoCachedTests(unittest.TestCase):
    """The predicate ensure_hf_models() skips a repo on.

    An over-generous answer is the expensive direction: the repo is skipped and
    no later run completes it, so a half-fetched cache stays half-fetched until
    somebody thinks to pass --force.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = patch.dict(os.environ, {"HF_HOME": self._tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.repo_dir = (Path(self._tmp.name) / "hub" / "models--repo--one")

    def _populate(self, *, incomplete: bool):
        (self.repo_dir / "snapshots" / "abc123").mkdir(parents=True)
        (self.repo_dir / "snapshots" / "abc123" / "config.json").write_bytes(b"{}")
        blobs = self.repo_dir / "blobs"
        blobs.mkdir()
        if incomplete:
            (blobs / "deadbeef.incomplete").write_bytes(b"half a file")

    def test_absent_repo(self):
        self.assertFalse(model_fetch.hf_repo_cached("repo/one"))

    def test_complete_snapshot(self):
        self._populate(incomplete=False)
        self.assertTrue(model_fetch.hf_repo_cached("repo/one"))

    def test_snapshot_with_an_interrupted_blob(self):
        # The case the previous implementation missed: it asked
        # snapshot_download(local_files_only=True), whose completeness check is
        # skipped when trees/<commit>.json is absent - which it is for a cache
        # transformers filled file by file, i.e. after a first run.
        self._populate(incomplete=True)
        self.assertFalse(model_fetch.hf_repo_cached("repo/one"))

    def test_needs_no_huggingface_hub(self):
        # install.py calls this before the requirements step, and the app calls
        # it on the way to a download; neither should depend on the library
        # being importable just to ask a filesystem question.
        self._populate(incomplete=False)
        with patch.dict(sys.modules, {"huggingface_hub": None}):
            self.assertTrue(model_fetch.hf_repo_cached("repo/one"))


class EnsureHfModelsTests(unittest.TestCase):
    """Every repo is attempted even when one fails, and the failures are
    reported together rather than aborting on the first one."""

    def setUp(self):
        self.repos = (models_info.HfRepo("repo/one", "first", 1),
                      models_info.HfRepo("repo/two", "second", 2))

    def test_skips_cached_repos(self):
        downloaded = []
        with patch.object(model_fetch, "hf_repo_cached", return_value=True), \
                patch("huggingface_hub.snapshot_download",
                      side_effect=lambda repo_id: downloaded.append(repo_id),
                      create=True):
            model_fetch.ensure_hf_models(self.repos)
        self.assertEqual(downloaded, [])

    def test_force_downloads_cached_repos(self):
        downloaded = []
        with patch.object(model_fetch, "hf_repo_cached", return_value=True), \
                patch("huggingface_hub.snapshot_download",
                      side_effect=lambda repo_id: downloaded.append(repo_id),
                      create=True):
            model_fetch.ensure_hf_models(self.repos, force=True)
        self.assertEqual(downloaded, ["repo/one", "repo/two"])

    def test_reports_every_failure_at_once(self):
        def fail(repo_id):
            raise RuntimeError("no network")

        with patch.object(model_fetch, "hf_repo_cached", return_value=False), \
                patch("huggingface_hub.snapshot_download", side_effect=fail,
                      create=True):
            with self.assertRaises(model_fetch.ModelFetchError) as ctx:
                model_fetch.ensure_hf_models(self.repos)
        message = str(ctx.exception)
        self.assertIn("repo/one", message)
        self.assertIn("repo/two", message)


class CliTests(unittest.TestCase):
    """The flags install.py and a user typing the command both rely on."""

    def setUp(self):
        # main() configures logging for its CLI use; letting it install a root
        # handler here would leak into every test module that runs afterwards.
        patcher = patch("logging.basicConfig")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_defaults_select_nothing_explicitly(self):
        args = model_fetch.parse_args([])
        self.assertFalse(args.hf)
        self.assertFalse(args.supertonic)
        self.assertFalse(args.force)
        self.assertFalse(args.list)

    def test_single_target_flags(self):
        self.assertTrue(model_fetch.parse_args(["--hf"]).hf)
        self.assertTrue(model_fetch.parse_args(["--supertonic"]).supertonic)

    def test_list_runs_no_download(self):
        with patch.object(model_fetch, "_print_status") as printer, \
                patch.object(model_fetch, "ensure_hf_models") as hf, \
                patch.object(model_fetch, "ensure_supertonic") as st:
            self.assertEqual(model_fetch.main(["--list"]), 0)
        printer.assert_called_once()
        hf.assert_not_called()
        st.assert_not_called()

    def test_no_flag_means_everything(self):
        with patch.object(model_fetch, "ensure_hf_models") as hf, \
                patch.object(model_fetch, "ensure_supertonic") as st:
            self.assertEqual(model_fetch.main([]), 0)
        hf.assert_called_once()
        st.assert_called_once()

    def test_hf_flag_skips_supertonic(self):
        with patch.object(model_fetch, "ensure_hf_models") as hf, \
                patch.object(model_fetch, "ensure_supertonic") as st:
            self.assertEqual(model_fetch.main(["--hf"]), 0)
        hf.assert_called_once()
        st.assert_not_called()

    def test_failure_becomes_a_nonzero_exit(self):
        # The error goes to stderr for the user; captured so the test report
        # does not look like something went wrong.
        stderr = io.StringIO()
        with patch.object(model_fetch, "ensure_hf_models",
                          side_effect=model_fetch.ModelFetchError("boom")), \
                contextlib.redirect_stderr(stderr):
            self.assertEqual(model_fetch.main(["--hf"]), 1)
        self.assertIn("boom", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
