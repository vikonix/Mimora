# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for the spaCy model sidecar (mimora/spacy_model_fetch.py).

No network: every test that installs anything replaces ``_download`` with a
function that writes a wheel built here. What is under test is the decision and
the bookkeeping around the download, not the download.

Two of these are worth more than the rest.

* **A half-unpacked directory must never survive a failure.** A tree holding a
  .dist-info and no package satisfies ``spacy.util.is_package`` and then fails
  inside ``spacy.load`` - which is a worse state than having nothing at all,
  because the app's own "is it there" answer becomes a lie.
* **activate() appends.** An environment that has its own copy of the model
  keeps using it; this directory is a fallback, and a test says so, because
  nothing else in the code would notice the day somebody changes it to insert.

Run from the project root with:

    python -m unittest tests.test_spacy_model_fetch
"""

import ast
import sys
import unittest
import zipfile
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from mimora import models_info, spacy_model_fetch

MODEL = models_info.SPACY_EN
# A hash that is merely not the placeholder: every test that installs replaces
# _download, so no real digest is ever compared against it.
PINNED = "ab" * 32


def _write_wheel(path: Path, *, package: bool = True,
                 dist_info: bool = True) -> None:
    """Build a minimal wheel with the two entries the unpacking looks for."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        if package:
            archive.writestr(f"{MODEL.name}/__init__.py", "")
        if dist_info:
            archive.writestr(f"{MODEL.name}-{MODEL.version}.dist-info/METADATA",
                             f"Name: {MODEL.name}\n")


def _fake_download(**wheel_kwargs):
    """A _download stand-in that produces a wheel instead of fetching one."""
    def download(_url, target, _expected_sha256, _progress):
        _write_wheel(Path(target), **wheel_kwargs)
        return PINNED
    return download


def _imports(module) -> set:
    """Every name *module* imports, read from its source rather than run."""
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
    return names


class _TempDirTest(unittest.TestCase):
    """Base for the tests that need a throwaway directory tree."""

    def setUp(self):
        tmp = TemporaryDirectory(prefix="spacy-fetch-test-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.dest = self.root / "spacy"


class PinTests(unittest.TestCase):
    """The catalogue entry the fetcher is built around."""

    def test_the_pin_is_a_full_sha256_digest(self):
        # 64 hex characters whether or not it has been measured yet: the
        # placeholder has the same shape on purpose, so a value pasted in the
        # wrong column is a failure here rather than at download time.
        self.assertEqual(len(MODEL.sha256), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in MODEL.sha256))

    def test_the_wheel_name_and_url_agree_with_the_pin(self):
        expected = f"{MODEL.name}-{MODEL.version}-py3-none-any.whl"
        self.assertEqual(spacy_model_fetch.wheel_filename(), expected)
        self.assertTrue(spacy_model_fetch.download_url().endswith(expected))


class UnmeasuredPinTests(_TempDirTest):
    """An unmeasured hash stops the download instead of skipping the check."""

    def test_downloading_refuses_and_says_how_to_fix_it(self):
        unpinned = MODEL._replace(sha256=models_info.UNMEASURED_SHA256)
        with mock.patch.object(spacy_model_fetch, "MODEL", unpinned), \
                mock.patch.object(spacy_model_fetch, "_download") as download:
            with self.assertRaises(spacy_model_fetch.SpacyModelFetchError) as caught:
                spacy_model_fetch.ensure_spacy_model(self.dest)
        download.assert_not_called()
        self.assertIn("--measure", str(caught.exception))


class SidecarStateTests(_TempDirTest):
    """What counts as an installed model in our own directory."""

    def _make(self, *, package: bool, dist_info: bool) -> None:
        if package:
            (self.dest / MODEL.name).mkdir(parents=True)
        if dist_info:
            (self.dest / f"{MODEL.name}-{MODEL.version}.dist-info").mkdir(
                parents=True)

    def test_a_missing_directory_is_absent_rather_than_an_error(self):
        self.assertFalse(spacy_model_fetch.model_in_sidecar(self.dest))

    def test_both_directories_are_required(self):
        # Each half satisfies a different consumer - spacy.load imports the
        # package, is_package looks up the .dist-info - so one without the
        # other passes one test and fails the other, somewhere else entirely.
        for package, dist_info in ((True, False), (False, True)):
            with self.subTest(package=package, dist_info=dist_info):
                with TemporaryDirectory() as tmp:
                    self.dest = Path(tmp) / "spacy"
                    self._make(package=package, dist_info=dist_info)
                    self.assertFalse(
                        spacy_model_fetch.model_in_sidecar(self.dest))

    def test_both_present_is_installed(self):
        self._make(package=True, dist_info=True)
        self.assertTrue(spacy_model_fetch.model_in_sidecar(self.dest))


class ActivateTests(_TempDirTest):
    """Putting the sidecar where importlib.metadata will look."""

    def setUp(self):
        super().setUp()
        self.addCleanup(self._restore_sys_path, list(sys.path))

    @staticmethod
    def _restore_sys_path(original):
        sys.path[:] = original

    def _install(self) -> None:
        (self.dest / MODEL.name).mkdir(parents=True)
        (self.dest / f"{MODEL.name}-{MODEL.version}.dist-info").mkdir(parents=True)

    def test_an_empty_sidecar_is_a_no_op(self):
        # The normal state until the first run downloads something. Adding a
        # directory that holds nothing would be harmless but misleading.
        before = list(sys.path)
        self.assertFalse(spacy_model_fetch.activate(self.dest))
        self.assertEqual(sys.path, before)

    def test_the_directory_is_appended_not_prepended(self):
        # Load-bearing: a model somebody installed into the environment on
        # purpose must keep winning over ours.
        self._install()
        self.assertTrue(spacy_model_fetch.activate(self.dest))
        self.assertEqual(sys.path[-1], str(self.dest))

    def test_activating_twice_adds_one_entry(self):
        self._install()
        spacy_model_fetch.activate(self.dest)
        spacy_model_fetch.activate(self.dest)
        self.assertEqual(sys.path.count(str(self.dest)), 1)


class AvailabilityTests(unittest.TestCase):
    """"Is it there" is asked of the interpreter, not of the filesystem."""

    def test_an_unknown_distribution_resolves_nowhere(self):
        # model_in_sidecar is stubbed rather than trusted: the machine running
        # the tests may well have the real sidecar installed, and then this
        # would be asserting nothing.
        with mock.patch.object(metadata, "distribution",
                               side_effect=metadata.PackageNotFoundError), \
                mock.patch.object(spacy_model_fetch, "model_in_sidecar",
                                  return_value=False):
            self.assertIsNone(spacy_model_fetch.resolved_location())
            self.assertFalse(spacy_model_fetch.model_available())

    def test_an_unpacked_sidecar_is_enough_on_its_own(self):
        # Availability must not depend on whether the ASKING process has
        # activated the sidecar. install.py and this module's CLI never import
        # config, which is where the application activates it, and with only
        # the metadata lookup they reported a working install as missing.
        with mock.patch.object(metadata, "distribution",
                               side_effect=metadata.PackageNotFoundError), \
                mock.patch.object(spacy_model_fetch, "model_in_sidecar",
                                  return_value=True):
            self.assertTrue(spacy_model_fetch.model_available())

    def test_a_known_distribution_reports_where_it_lives(self):
        found = mock.Mock()
        found.locate_file.return_value = Path("/somewhere/site-packages")
        with mock.patch.object(metadata, "distribution", return_value=found), \
                mock.patch.object(spacy_model_fetch, "model_in_sidecar",
                                  return_value=False):
            self.assertEqual(spacy_model_fetch.resolved_location(),
                             Path("/somewhere/site-packages"))
            self.assertTrue(spacy_model_fetch.model_available())

    def test_a_broken_dist_info_is_absent_rather_than_fatal(self):
        # This runs while the first-run plan is being built, before any window
        # exists; an exception here would be a traceback instead of a dialog.
        with mock.patch.object(metadata, "distribution",
                               side_effect=ValueError("bad metadata")), \
                mock.patch.object(spacy_model_fetch, "model_in_sidecar",
                                  return_value=False):
            self.assertFalse(spacy_model_fetch.model_available())


class ExtractTests(_TempDirTest):
    """Unpacking, and what it refuses to unpack."""

    def test_a_member_pointing_outside_the_target_is_refused(self):
        wheel = self.root / "evil.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("../escaped.py", "")
        target = self.root / "into"
        target.mkdir()
        with self.assertRaises(spacy_model_fetch.SpacyModelFetchError):
            spacy_model_fetch._extract_wheel(wheel, target)
        self.assertFalse((self.root / "escaped.py").exists())


class InstallTests(_TempDirTest):
    """ensure_spacy_model, with the download replaced."""

    def setUp(self):
        super().setUp()
        pinned = MODEL._replace(sha256=PINNED)
        patcher = mock.patch.object(spacy_model_fetch, "MODEL", pinned)
        self.addCleanup(patcher.stop)
        patcher.start()

    def _staging(self) -> Path:
        return self.dest.with_name(self.dest.name + ".new")

    def test_a_successful_install_leaves_the_model_and_no_staging(self):
        with mock.patch.object(spacy_model_fetch, "_download",
                               _fake_download()):
            result = spacy_model_fetch.ensure_spacy_model(self.dest)
        self.assertEqual(result, self.dest)
        self.assertTrue(spacy_model_fetch.model_in_sidecar(self.dest))
        self.assertFalse(self._staging().exists())

    def test_a_failed_download_leaves_nothing_behind(self):
        # The staging directory is the whole point: a half-filled one would be
        # picked up by the next run as if it were an install.
        def boom(*_args, **_kwargs):
            raise spacy_model_fetch.SpacyModelFetchError("checksum mismatch")

        with mock.patch.object(spacy_model_fetch, "_download", boom):
            with self.assertRaises(spacy_model_fetch.SpacyModelFetchError):
                spacy_model_fetch.ensure_spacy_model(self.dest)
        self.assertFalse(self._staging().exists())
        self.assertFalse(self.dest.exists())

    def test_a_wheel_missing_the_package_is_rejected_before_the_swap(self):
        with mock.patch.object(spacy_model_fetch, "_download",
                               _fake_download(package=False)):
            with self.assertRaises(spacy_model_fetch.SpacyModelFetchError):
                spacy_model_fetch.ensure_spacy_model(self.dest)
        self.assertFalse(self._staging().exists())
        self.assertFalse(self.dest.exists())

    def test_an_existing_install_is_not_downloaded_again(self):
        with mock.patch.object(spacy_model_fetch, "_download",
                               _fake_download()):
            spacy_model_fetch.ensure_spacy_model(self.dest)
        with mock.patch.object(spacy_model_fetch, "_download") as download:
            spacy_model_fetch.ensure_spacy_model(self.dest)
        download.assert_not_called()

    def test_force_downloads_again_over_an_existing_install(self):
        with mock.patch.object(spacy_model_fetch, "_download",
                               _fake_download()):
            spacy_model_fetch.ensure_spacy_model(self.dest)
        download = mock.Mock(side_effect=_fake_download())
        with mock.patch.object(spacy_model_fetch, "_download", download):
            spacy_model_fetch.ensure_spacy_model(self.dest, force=True)
        download.assert_called_once()
        self.assertTrue(spacy_model_fetch.model_in_sidecar(self.dest))

    def test_a_leftover_staging_directory_does_not_block_a_retry(self):
        # Killed mid-install, the staging directory can outlive the process.
        staging = self._staging()
        staging.mkdir(parents=True)
        (staging / "leftover.txt").write_text("junk", encoding="utf-8")
        with mock.patch.object(spacy_model_fetch, "_download",
                               _fake_download()):
            spacy_model_fetch.ensure_spacy_model(self.dest)
        self.assertTrue(spacy_model_fetch.model_in_sidecar(self.dest))
        self.assertFalse((self.dest / "leftover.txt").exists())


class ImportDisciplineTests(unittest.TestCase):
    """The rule every fetcher lives by."""

    def test_the_fetcher_does_not_import_config(self):
        # config flips HF_HUB_OFFLINE=1 once the models are cached, which would
        # switch the network off exactly when a download is wanted. Asserted
        # rather than described, like the same rule in tests/test_first_run.py.
        names = _imports(spacy_model_fetch)
        self.assertNotIn("config", names)
        self.assertNotIn("mimora.config", names)

    def test_spacy_itself_is_not_imported(self):
        # Everything this module decides is a filesystem or metadata question.
        # Importing spaCy to answer it would drag thinc into install.py and
        # into the first-run plan, both of which run before the requirements
        # are guaranteed to be installed.
        self.assertNotIn("spacy", _imports(spacy_model_fetch))


if __name__ == "__main__":
    unittest.main()
