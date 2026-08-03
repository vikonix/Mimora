# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Unit tests for the path resolution (mimora/paths.py).

This module decides where every file the app touches lives, and it answers
differently depending on how Mimora was installed. Only one of those answers -
the source tree - is observable on the machine these tests run on, so the
package branch is exercised by stubbing the repository marker and the platform,
which is also the only way to check all three operating systems from one.

What is worth pinning here, in order of what would hurt most if it broke:

* the roots stay separate (a package's downloads must not be looked for among
  its code, and its shipped resources must not be looked for among the
  downloads);
* the shipped resources sit INSIDE the package, because that is the only place
  a wheel carries a non-Python file from - a directory at the top of the source
  tree is skipped by both the build and the install without a word;
* MIMORA_HOME wins over both modes, since it is the documented way out of
  every wrong automatic answer;
* the marker is the file next to the package, not a "site-packages" test on
  __file__, because that one misreports editable installs;
* the layout under the root is the same in both modes, which is what lets one
  set of instructions describe both.

Nothing here touches the filesystem except through tmp_path-style stubs. Run
from the project root with:

    python -m unittest tests.test_paths
"""

import os
import unittest
from pathlib import Path
from unittest import mock

from mimora import paths


def _no_marker():
    """Force package mode: pretend pyproject.toml is not next to the package."""
    return mock.patch.object(paths, "repo_mode", return_value=False)


def _with_marker():
    """Force repository mode."""
    return mock.patch.object(paths, "repo_mode", return_value=True)


def _clean_env(**overrides):
    """Environment with MIMORA_HOME removed unless *overrides* puts it back."""
    env = dict(os.environ)
    env.pop(paths.HOME_ENV_VAR, None)
    env.update(overrides)
    return mock.patch.dict(os.environ, env, clear=True)


class RepoModeTests(unittest.TestCase):
    """The marker, and what running from a clone implies."""

    def test_marker_is_found_in_this_checkout(self):
        # These tests run from the source tree, so the real answer must be True.
        # If this ever fails, every path in the app has silently moved.
        self.assertTrue(paths.repo_mode())

    def test_repo_mode_puts_data_beside_the_code(self):
        with _clean_env():
            self.assertEqual(paths.data_root(), paths.resource_root())

    def test_marker_is_the_file_next_to_the_package(self):
        # Pinned as a path, not as a behaviour: an editable install points at a
        # source tree from inside site-packages, and testing __file__ for
        # "site-packages" instead would call that a package install.
        self.assertTrue((paths.resource_root() / "pyproject.toml").is_file())


class EnvOverrideTests(unittest.TestCase):
    """MIMORA_HOME beats both automatic answers."""

    def test_override_wins_in_repo_mode(self):
        with _with_marker(), _clean_env(**{paths.HOME_ENV_VAR: "/tmp/elsewhere"}):
            self.assertEqual(paths.data_root(), Path("/tmp/elsewhere").absolute())

    def test_override_wins_in_package_mode(self):
        with _no_marker(), _clean_env(**{paths.HOME_ENV_VAR: "/tmp/elsewhere"}):
            self.assertEqual(paths.data_root(), Path("/tmp/elsewhere").absolute())

    def test_blank_override_is_ignored(self):
        # An exported-but-empty variable is the shell's normal way of saying
        # "unset", and treating it as a path would put the data root at "".
        with _with_marker(), _clean_env(**{paths.HOME_ENV_VAR: "   "}):
            self.assertEqual(paths.data_root(), paths.resource_root())

    def test_override_is_made_absolute(self):
        with _with_marker(), _clean_env(**{paths.HOME_ENV_VAR: "relative/dir"}):
            self.assertTrue(paths.data_root().is_absolute())

    def test_surrounding_quotes_are_dropped(self):
        # `set MIMORA_HOME="D:\dir"` in cmd keeps the quotes IN the value,
        # unlike a POSIX shell, and a quote cannot appear in a Windows
        # filename - so keeping them turns ensure_dirs() into an OSError raised
        # while config is being imported.
        for quoted in ('"/tmp/elsewhere"', "'/tmp/elsewhere'"):
            with self.subTest(value=quoted):
                with _with_marker(), _clean_env(**{paths.HOME_ENV_VAR: quoted}):
                    self.assertEqual(paths.data_root(),
                                     Path("/tmp/elsewhere").absolute())

    def test_quotes_alone_are_treated_as_unset(self):
        with _with_marker(), _clean_env(**{paths.HOME_ENV_VAR: '"  "'}):
            self.assertEqual(paths.data_root(), paths.resource_root())

    def test_unpaired_quote_is_left_alone(self):
        # Only a matching pair is shell quoting; a lone quote is either a typo
        # or, on a POSIX filesystem, a legal part of the name. Stripping one
        # side would silently point the data root somewhere else.
        with _with_marker(), _clean_env(**{paths.HOME_ENV_VAR: '"/tmp/odd'}):
            self.assertEqual(paths.data_root(), Path('"/tmp/odd').absolute())

    def test_override_does_not_move_the_resources(self):
        # It redirects what this machine writes; it cannot move files that
        # arrive with the code.
        with _no_marker(), _clean_env(**{paths.HOME_ENV_VAR: "/tmp/elsewhere"}):
            self.assertNotEqual(paths.resource_root(), Path("/tmp/elsewhere"))


class PackageModeLocationTests(unittest.TestCase):
    """One OS directory per platform, and never the package's own directory."""

    def test_windows_uses_appdata(self):
        with _no_marker(), \
                mock.patch.object(paths.sys, "platform", "win32"), \
                _clean_env(APPDATA=r"C:\Users\someone\AppData\Roaming"):
            self.assertEqual(paths.data_root(),
                             Path(r"C:\Users\someone\AppData\Roaming") / "Mimora")

    def test_windows_without_appdata_falls_back_to_the_home_directory(self):
        with _no_marker(), \
                mock.patch.object(paths.sys, "platform", "win32"), \
                mock.patch.object(paths.Path, "home",
                                  return_value=Path("/home/someone")), \
                _clean_env():
            os.environ.pop("APPDATA", None)
            self.assertEqual(
                paths.data_root(),
                Path("/home/someone") / "AppData" / "Roaming" / "Mimora")

    def test_macos_uses_application_support(self):
        with _no_marker(), \
                mock.patch.object(paths.sys, "platform", "darwin"), \
                mock.patch.object(paths.Path, "home",
                                  return_value=Path("/Users/someone")), \
                _clean_env():
            self.assertEqual(
                paths.data_root(),
                Path("/Users/someone") / "Library" / "Application Support" / "Mimora")

    def test_linux_honours_xdg_data_home(self):
        with _no_marker(), \
                mock.patch.object(paths.sys, "platform", "linux"), \
                _clean_env(XDG_DATA_HOME="/home/someone/.local/share"):
            self.assertEqual(paths.data_root(),
                             Path("/home/someone/.local/share") / "mimora")

    def test_linux_without_xdg_uses_the_spec_default(self):
        with _no_marker(), \
                mock.patch.object(paths.sys, "platform", "linux"), \
                mock.patch.object(paths.Path, "home",
                                  return_value=Path("/home/someone")), \
                _clean_env():
            os.environ.pop("XDG_DATA_HOME", None)
            self.assertEqual(paths.data_root(),
                             Path("/home/someone") / ".local" / "share" / "mimora")

    def test_package_mode_never_writes_next_to_the_code(self):
        # The whole point of the split: site-packages belongs to the installer,
        # and `uv tool upgrade` rebuilds it.
        for platform_name, env in (("win32", {"APPDATA": "/appdata"}),
                                   ("darwin", {}),
                                   ("linux", {"XDG_DATA_HOME": "/xdg"})):
            with self.subTest(platform=platform_name):
                with _no_marker(), \
                        mock.patch.object(paths.sys, "platform", platform_name), \
                        _clean_env(**env):
                    self.assertNotEqual(paths.data_root(), paths.resource_root())


class LayoutTests(unittest.TestCase):
    """Every named location is a child of the root, identically in both modes."""

    _ACCESSORS = ("config_dir", "themes_dir", "models_dir", "model_cache_dir",
                  "llama_dir", "log_dir")

    def _relative_layout(self):
        root = paths.data_root()
        return {name: getattr(paths, name)().relative_to(root)
                for name in self._ACCESSORS}

    def test_layout_is_identical_in_both_modes(self):
        # Not aesthetics: it is what makes one set of instructions, one set of
        # log paths and one set of error messages true everywhere, and what
        # lets a user carry the directory between machines.
        with _with_marker(), _clean_env():
            repo_layout = self._relative_layout()
        with _no_marker(), \
                mock.patch.object(paths.sys, "platform", "linux"), \
                _clean_env(XDG_DATA_HOME="/xdg"):
            package_layout = self._relative_layout()
        self.assertEqual(repo_layout, package_layout)

    def test_themes_live_under_the_config_directory(self):
        with _clean_env():
            self.assertEqual(paths.themes_dir().parent, paths.config_dir())

    def test_shipped_themes_are_not_part_of_the_writable_layout(self):
        # The counterpart of the test above, and the reason the theme lookup
        # has two places rather than one: this directory is inside the package,
        # so in package mode it is not under the data root at all and a user
        # cannot add a file to it.
        with _clean_env():
            self.assertEqual(paths.shipped_themes_dir().parent,
                             paths.shipped_root())
            self.assertNotEqual(paths.shipped_themes_dir(), paths.themes_dir())

    def test_logs_sit_beside_the_settings_rather_than_in_an_os_log_directory(self):
        # A separate log convention exists on only two of the three platforms,
        # and logs are wanted exactly when somebody is already looking at the
        # config directory.
        with _clean_env():
            self.assertEqual(paths.log_dir().parent, paths.data_root())


class ShippedRootTests(unittest.TestCase):
    """What ships inside the package, and that it really is inside it.

    These are the tests that would have caught the packaging bug this root was
    introduced for: the practice texts and theme schemas used to live at the
    top of the source tree, which works perfectly in a clone and produces a
    wheel without them.
    """

    def test_shipped_root_is_the_package_directory(self):
        self.assertEqual(paths.shipped_root(), Path(paths.__file__).resolve().parent)

    def test_shipped_root_sits_inside_the_resource_root(self):
        # The two answer different questions and must not be confused: the
        # resource root is the parent, where the sibling pronunciation package
        # keeps its committed calibrations.
        self.assertEqual(paths.shipped_root().parent, paths.resource_root())

    def test_the_environment_override_does_not_move_it(self):
        # MIMORA_HOME redirects what the machine writes. These files arrive
        # with the code and cannot be anywhere else.
        with _no_marker(), _clean_env(**{paths.HOME_ENV_VAR: "/tmp/elsewhere"}):
            self.assertEqual(paths.shipped_root(),
                             Path(paths.__file__).resolve().parent)

    def test_package_mode_keeps_the_shipped_files_away_from_the_downloads(self):
        with _no_marker(), \
                mock.patch.object(paths.sys, "platform", "linux"), \
                _clean_env(XDG_DATA_HOME="/xdg"):
            self.assertNotEqual(paths.shipped_root(), paths.data_root())

    def test_the_shipped_files_exist_in_this_checkout(self):
        # Pinned by name rather than by count: these are the files config.py
        # falls back to when settings.json names nothing, and a rename that
        # missed one would show up as a missing theme or an empty practice
        # panel, both of which the app survives quietly.
        for relative in ("texts/practice_text.txt",
                         "texts/practice_text_es.txt",
                         "themes/dark_schema.json",
                         "themes/light_schema.json"):
            with self.subTest(file=relative):
                self.assertTrue((paths.shipped_root() / relative).is_file(),
                                f"{relative} is missing from the package")


class EnsureDirsTests(unittest.TestCase):
    """Creation must work where nothing exists yet."""

    def test_creates_parents(self):
        # In package mode the data root itself is missing on a first run, so
        # mkdir without parents=True would raise instead of creating anything.
        created = []

        def fake_mkdir(self, parents=False, exist_ok=False):
            created.append((self, parents, exist_ok))

        with _with_marker(), _clean_env(), \
                mock.patch.object(paths.Path, "mkdir", fake_mkdir):
            paths.ensure_dirs()

        self.assertTrue(created, "ensure_dirs created nothing")
        for path, parents, exist_ok in created:
            with self.subTest(path=path):
                self.assertTrue(parents)
                self.assertTrue(exist_ok)

    def _created_by_ensure_dirs(self):
        created = []
        with _with_marker(), _clean_env(), \
                mock.patch.object(paths.Path, "mkdir",
                                  lambda self, **kw: created.append(self)):
            paths.ensure_dirs()
        return created

    def test_creates_the_config_directory(self):
        # Without it loader.save_setting fails on every write and only says so
        # on stderr, so no preference would ever persist.
        self.assertIn(paths.config_dir(), self._created_by_ensure_dirs())

    def test_creates_the_user_themes_directory(self):
        # The one entry nothing writes to. It is created because it is the
        # documented place to drop a theme, and in package mode it does not
        # otherwise exist - an instruction beginning "first create this
        # directory" is one nobody follows.
        self.assertIn(paths.themes_dir(), self._created_by_ensure_dirs())

    def test_does_not_create_anything_inside_the_package(self):
        # The shipped resources are read-only and belong to the installation.
        created = self._created_by_ensure_dirs()
        self.assertNotIn(paths.shipped_themes_dir(), created)
        self.assertNotIn(paths.shipped_root(), created)


if __name__ == "__main__":
    unittest.main()
