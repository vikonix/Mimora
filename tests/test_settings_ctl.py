# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Unit tests for mimora/settings_ctl.py (no Tk widgets).

Focus on the "Default" reset semantics - the diffing that decides which
defaults are re-dispatched live - and the persist suppression during the
reset. config attributes are patched per-test and restored afterwards. Run
from the project root with:

    python -m unittest tests.test_settings_ctl
"""

import unittest
from unittest import mock

from mimora import config
from mimora.settings_ctl import (
    LIVE_CONFIG_ATTRS,
    SETTING_LIVE_ATTRS,
    SettingsGlue,
)


def make_glue(dispatch=None):
    """A SettingsGlue with recording fakes for the controller touchpoints."""
    errors = []
    glue = SettingsGlue(
        report_error=errors.append,
        get_window=lambda: None,
        dispatch=dispatch if dispatch is not None else (lambda key, value: None),
    )
    return glue, errors


class TestTables(unittest.TestCase):
    def test_tables_do_not_overlap(self):
        # A key in both tables would be diffed against two different config
        # attributes depending on lookup order - keep them disjoint.
        self.assertEqual(set(LIVE_CONFIG_ATTRS) & set(SETTING_LIVE_ATTRS),
                         set())

    def test_every_table_attr_exists_on_config(self):
        # A typo'd attribute name would make the diffing raise at reset time.
        for attr, _cast in LIVE_CONFIG_ATTRS.values():
            self.assertTrue(hasattr(config, attr), attr)
        for attr in SETTING_LIVE_ATTRS.values():
            self.assertTrue(hasattr(config, attr), attr)


class TestDefaultDiffing(unittest.TestCase):
    def test_restart_only_key_never_differs(self):
        # Keys absent from both tables (restart-only) must answer False:
        # their dispatch would be persist-only and the overrides are already
        # removed from settings.json by the reset.
        glue, _ = make_glue()
        self.assertFalse(
            glue._default_differs_from_live("engine", "acoustic"))

    def test_int_float_mismatch_is_not_a_change(self):
        glue, _ = make_glue()
        with mock.patch.object(config, "MAX_RECORD_SECONDS", 30.0):
            self.assertFalse(
                glue._default_differs_from_live("max_record_seconds", 30))
            self.assertTrue(
                glue._default_differs_from_live("max_record_seconds", 25))

    def test_bool_is_compared_as_bool_not_number(self):
        # isinstance(True, int) holds, so the numeric branch must exclude
        # bools: show_face True vs default True is "no change".
        glue, _ = make_glue()
        with mock.patch.object(config, "SHOW_FACE", True):
            self.assertFalse(
                glue._default_differs_from_live("show_face", True))
            self.assertTrue(
                glue._default_differs_from_live("show_face", False))

    def test_practice_text_path_compares_normalized(self):
        glue, _ = make_glue()
        with mock.patch.object(config, "PRACTICE_TEXT_FILE",
                               "texts\\practice_text.txt"):
            self.assertFalse(glue._default_differs_from_live(
                "practice_text_file", "texts/practice_text.txt"))
            self.assertTrue(glue._default_differs_from_live(
                "practice_text_file", "texts/other.txt"))

    def test_plain_string_setting_diffs_by_equality(self):
        glue, _ = make_glue()
        with mock.patch.object(config, "TTS_VOICE", "af_heart"):
            self.assertFalse(glue._default_differs_from_live(
                "voice", "af_heart"))
            self.assertTrue(glue._default_differs_from_live(
                "voice", "af_bella"))


class TestReset(unittest.TestCase):
    def test_reset_dispatches_only_actual_changes(self):
        # Defaults equal to the live values are skipped (re-dispatching an
        # unchanged voice would needlessly regenerate the current phrase).
        dispatched = []
        glue, _ = make_glue(dispatch=lambda k, v: dispatched.append(k))
        defaults = {"voice": "af_heart", "show_face": False}
        with mock.patch.object(config, "reset_user_settings",
                               return_value=True), \
             mock.patch.object(config, "default_user_settings",
                               return_value=defaults), \
             mock.patch.object(config, "TTS_VOICE", "af_heart"), \
             mock.patch.object(config, "SHOW_FACE", True):
            self.assertTrue(glue.reset_to_defaults())
        self.assertEqual(dispatched, ["show_face"])

    def test_persist_is_suppressed_during_reset_dispatch(self):
        # The re-applied defaults must not be written back as overrides.
        glue, _ = make_glue(dispatch=lambda k, v: self.assertTrue(
            glue.persist(k, v)))
        with mock.patch.object(config, "reset_user_settings",
                               return_value=True), \
             mock.patch.object(config, "default_user_settings",
                               return_value={"voice": "af_bella"}), \
             mock.patch.object(config, "TTS_VOICE", "af_heart"), \
             mock.patch.object(config, "save_user_setting") as save:
            self.assertTrue(glue.reset_to_defaults())
            save.assert_not_called()

    def test_persist_resumes_after_reset(self):
        glue, _ = make_glue()
        with mock.patch.object(config, "reset_user_settings",
                               return_value=True), \
             mock.patch.object(config, "default_user_settings",
                               return_value={}), \
             mock.patch.object(config, "save_user_setting",
                               return_value=True) as save:
            glue.reset_to_defaults()
            glue.persist("voice", "af_bella")
            save.assert_called_once_with("voice", "af_bella")

    def test_failed_reset_reports_and_returns_false(self):
        glue, errors = make_glue()
        with mock.patch.object(config, "reset_user_settings",
                               return_value=False):
            self.assertFalse(glue.reset_to_defaults())
        self.assertEqual(len(errors), 1)

    def test_failed_persist_reports_and_returns_false(self):
        glue, errors = make_glue()
        with mock.patch.object(config, "save_user_setting",
                               return_value=False):
            self.assertFalse(glue.persist("voice", "af_bella"))
        self.assertEqual(len(errors), 1)
