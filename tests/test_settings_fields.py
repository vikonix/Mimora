# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for the settings-window field model (mimora/settings_window.py).

These validate the declarative Field/Section model without creating any Tk
widgets: build_sections() and all_fields() are plain data, so a broken key,
kind, or choices callable is caught here instead of at window-open time. Run
from the project root with:

    python -m unittest tests.test_settings_fields
"""

import importlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# These tests validate the field model against the BUILT-IN defaults, not
# against this machine's config/settings.json or hardware_config.json: a valid
# but unusual local value would otherwise fail the suite although the code is
# correct (a flaky "unit" test under discover). read_json is stubbed to "no
# overrides" while mimora.config builds its module constants, then restored;
# the reload covers the case where another test module (or an earlier import
# in this process) already built config from the real files. settings_window
# reads config lazily per field, so it sees the default-built module.
from mimora import loader as _loader

_real_read_json = _loader.read_json
_loader.read_json = lambda path: {}
try:
    from mimora import config
    importlib.reload(config)
finally:
    _loader.read_json = _real_read_json

from mimora.settings_window import all_fields, build_sections

VALID_KINDS = {"bool", "choice", "number", "scale", "text", "path"}

# Settings bound at startup (engine wiring, Kokoro pipeline, theme, LLM
# subprocess): the window must mark exactly these as restart-only. A field
# moving in or out of this set is a deliberate behavior change - update the
# test together with the field.
EXPECTED_RESTART_KEYS = {
    "practice_language",
    "accent",
    "engine",
    "phoneme_good_mode",
    "color_theme",
    "llm_backend",
    "lm_studio_host",
    "external_model_path",
    "external_n_ctx",
    "warm_up",
}


class FieldModelTests(unittest.TestCase):
    def test_every_key_is_a_known_user_key(self):
        # A field writing an unknown key would be silently ignored at the next
        # startup (config warns and skips it), so the two lists must agree.
        for field in all_fields():
            self.assertIn(field.key, config._KNOWN_USER_KEYS,
                          f"{field.key!r} is not in config._KNOWN_USER_KEYS")

    def test_keys_are_unique(self):
        keys = [field.key for field in all_fields()]
        self.assertEqual(len(keys), len(set(keys)),
                         "duplicate settings key in build_sections()")

    def test_kinds_are_valid(self):
        for field in all_fields():
            self.assertIn(field.kind, VALID_KINDS,
                          f"{field.key!r} has unknown kind {field.kind!r}")

    def test_restart_flags_match_expectation(self):
        actual = {field.key for field in all_fields() if field.restart}
        self.assertEqual(actual, EXPECTED_RESTART_KEYS)

    def test_restart_fields_expose_a_runtime_value(self):
        # The pending-restart hint compares the saved value against the value
        # the process runs with; a restart field without that getter would
        # silently drop out of the hint.
        for field in all_fields():
            if field.restart:
                self.assertIsNotNone(
                    field.runtime_value,
                    f"restart field {field.key!r} lacks runtime_value")
                self.assertIsNotNone(field.runtime_value(), field.key)

    def test_labels_are_non_empty(self):
        for field in all_fields():
            self.assertTrue(field.label.strip(),
                            f"{field.key!r} has an empty label")

    def test_sections_are_non_empty(self):
        for section in build_sections():
            self.assertTrue(section.title.strip())
            self.assertTrue(section.fields,
                            f"section {section.title!r} has no fields")


class FieldValueTests(unittest.TestCase):
    """get_value / choices consistency against the real config module."""

    def test_get_value_returns_something(self):
        # Text fields may legitimately be empty ("" is a valid user_name) but
        # must still be strings; every other kind must produce a real value.
        for field in all_fields():
            value = field.get_value()
            if field.kind == "text":
                self.assertIsInstance(value, str,
                                      f"{field.key!r} returned a non-string")
            else:
                self.assertIsNotNone(value, f"{field.key!r} returned None")

    def test_choice_fields_have_choices_containing_current_value(self):
        for field in all_fields():
            if field.kind != "choice":
                continue
            self.assertIsNotNone(field.choices,
                                 f"{field.key!r} is a choice without choices")
            choices = field.choices()
            self.assertTrue(choices, f"{field.key!r} has empty choices")
            self.assertIn(field.get_value(), choices,
                          f"{field.key!r}: current value not among choices")

    def test_number_fields_are_numeric_and_in_range(self):
        for field in all_fields():
            if field.kind != "number":
                continue
            value = field.get_value()
            self.assertIsInstance(value, (int, float),
                                  f"{field.key!r} value is not a number")
            if field.minimum is not None:
                self.assertGreaterEqual(value, field.minimum, field.key)
            if field.maximum is not None:
                self.assertLessEqual(value, field.maximum, field.key)

    def test_scale_fields_are_numeric_and_bounded(self):
        # A slider needs both bounds (they define its extent) and a current
        # value inside them, or the thumb has nowhere valid to sit.
        for field in all_fields():
            if field.kind != "scale":
                continue
            value = field.get_value()
            self.assertIsInstance(value, (int, float),
                                  f"{field.key!r} value is not a number")
            self.assertIsNotNone(field.minimum, f"{field.key!r} lacks minimum")
            self.assertIsNotNone(field.maximum, f"{field.key!r} lacks maximum")
            self.assertGreaterEqual(value, field.minimum, field.key)
            self.assertLessEqual(value, field.maximum, field.key)

    def test_path_fields_declare_file_types(self):
        for field in all_fields():
            if field.kind == "path":
                self.assertTrue(field.file_types,
                                f"{field.key!r} has no file_types filter")


class ConfigHelperTests(unittest.TestCase):
    """The public config accessors the settings window relies on."""

    def test_accent_choices_cover_profiles(self):
        choices = config.accent_choices()
        self.assertIn("american", choices)
        self.assertIn("british", choices)

    def test_accent_voices_and_default_agree(self):
        for accent in config.accent_choices():
            voices = config.accent_voices(accent)
            self.assertTrue(voices, f"no voices for accent {accent!r}")
            self.assertIn(config.accent_default_voice(accent), voices)

    def test_unknown_accent_degrades_quietly(self):
        self.assertEqual(config.accent_voices("klingon"), ())
        self.assertEqual(config.accent_default_voice("klingon"), "")

    def test_available_themes_include_builtin_dark(self):
        themes = config.available_themes()
        self.assertIn("dark", themes)
        self.assertEqual(tuple(sorted(themes)), themes)  # stable order

    def test_available_themes_include_the_shipped_light_schema(self):
        # Against the real directories, so this fails if light_schema.json ever
        # stops being found where the package keeps it. Without the shipped
        # half of the lookup an installed copy offers "dark" alone - and does so
        # silently, because the built-in palette IS dark and nothing errors.
        self.assertIn("light", config.available_themes())

    def test_user_setting_falls_back(self):
        self.assertEqual(
            config.user_setting("no_such_key_ever", "fallback"), "fallback")


class ThemeLookupTests(unittest.TestCase):
    """Two places hold theme schemas, and which one wins is a decision.

    The user's config/themes/ is searched before the schemas inside the
    package, so a theme can be added without touching the installation and a
    shipped one can be replaced by reusing its name. Both directories are
    stubbed here: the real ones hold the project's own themes, and a test that
    depended on them would break the day one is added.
    """

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        root = Path(self._temp.name)
        self.user_dir = root / "user"
        self.shipped_dir = root / "shipped"
        self.user_dir.mkdir()
        self.shipped_dir.mkdir()
        patches = (
            mock.patch.object(config.paths, "themes_dir",
                              return_value=self.user_dir),
            mock.patch.object(config.paths, "shipped_themes_dir",
                              return_value=self.shipped_dir),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self._temp.cleanup)

    @staticmethod
    def _write(directory, name):
        (directory / f"{name}_schema.json").write_text("{}", encoding="utf-8")

    def test_a_shipped_theme_is_found_when_the_user_has_none(self):
        # The package-mode default: config/themes/ exists (ensure_dirs creates
        # it) and is empty.
        self._write(self.shipped_dir, "light")
        self.assertEqual(config._theme_file("light"),
                         self.shipped_dir / "light_schema.json")

    def test_a_user_file_overrides_the_shipped_one_of_the_same_name(self):
        self._write(self.shipped_dir, "light")
        self._write(self.user_dir, "light")
        self.assertEqual(config._theme_file("light"),
                         self.user_dir / "light_schema.json")

    def test_a_missing_theme_resolves_to_the_shipped_path(self):
        # Nothing reads it - loader.read_json answers {} either way - but it is
        # the path named in the "missing or invalid" message, and pointing at
        # the user's empty directory would send the reader to the wrong place.
        self.assertEqual(config._theme_file("nosuch"),
                         self.shipped_dir / "nosuch_schema.json")

    def test_available_themes_merge_both_places_without_duplicates(self):
        self._write(self.shipped_dir, "light")
        self._write(self.shipped_dir, "dark")
        self._write(self.user_dir, "light")   # overrides, not a second entry
        self._write(self.user_dir, "solar")
        self.assertEqual(config.available_themes(), ("dark", "light", "solar"))

    def test_a_missing_user_directory_is_not_an_error(self):
        # It only exists because ensure_dirs makes it, and a user is free to
        # delete it; the shipped themes must keep working.
        self._write(self.shipped_dir, "light")
        self.user_dir.rmdir()
        self.assertIn("light", config.available_themes())


class DefaultsTests(unittest.TestCase):
    """USER_SETTING_DEFAULTS - consumed by the settings window's Default reset."""

    def test_defaults_cover_exactly_the_known_keys(self):
        self.assertEqual(set(config.USER_SETTING_DEFAULTS),
                         set(config._KNOWN_USER_KEYS))

    def test_resolved_defaults_omit_machine_derived_keys(self):
        resolved = config.default_user_settings()
        self.assertNotIn("external_n_ctx", resolved)  # hardware detection decides
        self.assertIn("voice", resolved)              # None resolved to a voice

    def test_default_voice_belongs_to_default_accent(self):
        resolved = config.default_user_settings()
        accent = resolved["english_accent"]
        self.assertIn(resolved["voice"], config.accent_voices(accent))

    def test_choice_defaults_are_valid_choices(self):
        # "voice" is skipped: its choices follow the machine's *running*
        # accent, while the default voice follows the *default* accent - that
        # pairing is covered by test_default_voice_belongs_to_default_accent.
        resolved = config.default_user_settings()
        for field in all_fields():
            if field.kind != "choice" or field.key not in resolved \
                    or field.key == "voice":
                continue
            self.assertIn(resolved[field.key], field.choices(),
                          f"default for {field.key!r} not among its choices")

    def test_number_defaults_are_in_range(self):
        resolved = config.default_user_settings()
        for field in all_fields():
            if field.kind != "number" or field.key not in resolved:
                continue
            value = resolved[field.key]
            if field.minimum is not None:
                self.assertGreaterEqual(value, field.minimum, field.key)
            if field.maximum is not None:
                self.assertLessEqual(value, field.maximum, field.key)


if __name__ == "__main__":
    unittest.main()
