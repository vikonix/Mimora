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
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mimora import model_fetch

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


class MissingModelsTests(unittest.TestCase):
    """missing_models() is the app's first-run question, so it must name
    everything that is absent - and stay empty when nothing is."""

    def test_lists_repos_and_supertonic_when_nothing_is_present(self):
        with patch.object(model_fetch, "hf_repo_cached", return_value=False), \
                patch.object(model_fetch, "supertonic_cached",
                             return_value=False):
            missing = model_fetch.missing_models()
        expected = [repo for repo, _ in model_fetch.HF_MODEL_REPOS]
        expected.append(model_fetch.SUPERTONIC_MODEL_NAME)
        self.assertEqual(missing, expected)

    def test_empty_when_everything_is_present(self):
        with patch.object(model_fetch, "hf_repo_cached", return_value=True), \
                patch.object(model_fetch, "supertonic_cached",
                             return_value=True):
            self.assertEqual(model_fetch.missing_models(), [])

    def test_supertonic_alone_is_reported(self):
        with patch.object(model_fetch, "hf_repo_cached", return_value=True), \
                patch.object(model_fetch, "supertonic_cached",
                             return_value=False):
            self.assertEqual(model_fetch.missing_models(),
                             [model_fetch.SUPERTONIC_MODEL_NAME])


class EnsureHfModelsTests(unittest.TestCase):
    """Every repo is attempted even when one fails, and the failures are
    reported together rather than aborting on the first one."""

    def setUp(self):
        self.repos = (("repo/one", "first"), ("repo/two", "second"))

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
