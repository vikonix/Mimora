# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Unit tests for the language/variant configuration model (mimora/config.py).

Stage 1 of the multilingual refactor turned the practice language into data
(LANGUAGE_PROFILES) and introduced the settings.json keys ``practice_language``
and ``accent`` (with the legacy ``english_accent`` honored as a read fallback).
These tests build ``mimora.config`` against a controlled settings dict - never
this machine's real config/settings.json - so the layering, migration and
derived constants are checked in isolation, without a filesystem or the ML
stack. Run from the project root with:

    python -m unittest tests.test_language_profiles
"""

import contextlib
import importlib
import io
import unittest

from mimora import loader as _loader


def _build_config(user_settings: dict):
    """Reload mimora.config as if settings.json held exactly *user_settings*.

    read_json is stubbed so the settings file returns *user_settings* and the
    hardware file returns "no overrides"; the module is reloaded under the stub
    and returned. Callers restore a defaults-built config in tearDown so the
    reloaded global state never leaks into another test module.

    config reports invalid settings to stderr (never raises). Some tests feed
    deliberately invalid values to exercise the fallbacks, so the reload runs
    under redirect_stderr to keep the test output clean - the fallback itself is
    verified by the returned constants, not by the message.
    """
    def fake_read_json(path):
        if path.name == "settings.json":
            return dict(user_settings)
        return {}  # hardware_config.json and any other file: no overrides

    real_read_json = _loader.read_json
    _loader.read_json = fake_read_json
    try:
        from mimora import config
        with contextlib.redirect_stderr(io.StringIO()):
            importlib.reload(config)
        return config
    finally:
        _loader.read_json = real_read_json


class _ConfigTestBase(unittest.TestCase):
    def tearDown(self):
        # Leave mimora.config rebuilt on the built-in defaults so a later test
        # module that imported it does not observe this test's overrides.
        _build_config({})


class ProfileStructureTests(_ConfigTestBase):
    """Every LANGUAGE_PROFILES entry is internally consistent."""

    def setUp(self):
        self.config = _build_config({})

    def test_english_profile_present(self):
        self.assertIn("english", self.config.LANGUAGE_PROFILES)
        self.assertEqual(self.config.language_choices()[:1], ("english",))

    def test_profiles_have_required_fields(self):
        for name, profile in self.config.LANGUAGE_PROFILES.items():
            for key in ("display_name", "flores_code", "default_variant",
                        "engines", "practice_text_file", "variants"):
                self.assertIn(key, profile, f"{name!r} missing {key!r}")
            self.assertTrue(profile["display_name"].strip(), name)
            # FLORES-200 codes are "<lang>_<Script>", e.g. "eng_Latn".
            self.assertRegex(profile["flores_code"], r"^[a-z]{3}_[A-Z][a-z]{3}$",
                             f"{name!r} flores_code")
            self.assertTrue(profile["practice_text_file"].endswith(".txt"), name)

    def test_engines_are_a_nonempty_subset_of_choices(self):
        for name, profile in self.config.LANGUAGE_PROFILES.items():
            engines = profile["engines"]
            self.assertTrue(engines, f"{name!r} has no engines")
            for engine in engines:
                self.assertIn(engine, self.config.ENGINE_CHOICES,
                              f"{name!r} lists unknown engine {engine!r}")

    def test_variants_are_valid_and_default_belongs(self):
        for name, profile in self.config.LANGUAGE_PROFILES.items():
            variants = profile["variants"]
            self.assertTrue(variants, f"{name!r} has no variants")
            self.assertIn(profile["default_variant"], variants,
                          f"{name!r} default_variant not among its variants")
            for vname, variant in variants.items():
                for key in ("kokoro_lang_code", "espeak_language",
                            "default_voice", "voices"):
                    self.assertIn(key, variant, f"{name}/{vname} missing {key!r}")
                self.assertTrue(variant["voices"], f"{name}/{vname} has no voices")
                self.assertIn(variant["default_voice"], variant["voices"],
                              f"{name}/{vname} default_voice not in voices")
                # espeak/kokoro codes must be non-empty strings.
                self.assertTrue(variant["espeak_language"].strip())
                self.assertTrue(str(variant["kokoro_lang_code"]).strip())

    def test_voices_are_unique_within_a_variant(self):
        for name, profile in self.config.LANGUAGE_PROFILES.items():
            for vname, variant in profile["variants"].items():
                voices = variant["voices"]
                self.assertEqual(len(voices), len(set(voices)),
                                 f"{name}/{vname} has duplicate voices")


class HelperTests(_ConfigTestBase):
    """The public language/variant accessors."""

    def setUp(self):
        self.config = _build_config({})

    def test_accent_choices_cover_english_variants(self):
        choices = self.config.accent_choices()
        self.assertIn("american", choices)
        self.assertIn("british", choices)

    def test_accent_voices_and_default_agree(self):
        for accent in self.config.accent_choices():
            voices = self.config.accent_voices(accent)
            self.assertTrue(voices, f"no voices for variant {accent!r}")
            self.assertIn(self.config.accent_default_voice(accent), voices)

    def test_unknown_accent_degrades_quietly(self):
        self.assertEqual(self.config.accent_voices("klingon"), ())
        self.assertEqual(self.config.accent_default_voice("klingon"), "")

    def test_available_engines_default_is_active_language(self):
        self.assertEqual(self.config.available_engines(),
                         self.config.available_engines("english"))
        self.assertEqual(self.config.available_engines("english"),
                         ("phoneme", "acoustic", "none"))

    def test_available_engines_unknown_language_is_empty(self):
        self.assertEqual(self.config.available_engines("klingon"), ())


class ActiveSelectionTests(_ConfigTestBase):
    """practice_language / accent resolution and the derived constants."""

    def test_defaults_select_american_english(self):
        config = _build_config({})
        self.assertEqual(config.PRACTICE_LANGUAGE, "english")
        self.assertEqual(config.TARGET_LANGUAGE, "English")
        self.assertEqual(config.ACCENT, "american")
        self.assertEqual(config.KOKORO_LANG_CODE, "a")
        self.assertEqual(config.ESPEAK_LANGUAGE, "en-us")
        self.assertEqual(config.KOKORO_VOICE, "af_heart")

    def test_accent_key_selects_variant(self):
        config = _build_config({"accent": "british"})
        self.assertEqual(config.ACCENT, "british")
        self.assertEqual(config.KOKORO_LANG_CODE, "b")
        self.assertEqual(config.ESPEAK_LANGUAGE, "en-gb")
        self.assertEqual(config.KOKORO_VOICE, "bf_emma")

    def test_unknown_accent_falls_back_to_default_variant(self):
        config = _build_config({"accent": "klingon"})
        self.assertEqual(config.ACCENT, "american")


class LegacyMigrationTests(_ConfigTestBase):
    """The legacy english_accent key is read as a fallback for accent."""

    def test_legacy_english_accent_is_honored(self):
        config = _build_config({"english_accent": "british"})
        self.assertEqual(config.ACCENT, "british")
        self.assertEqual(config.KOKORO_LANG_CODE, "b")

    def test_new_accent_key_wins_over_legacy(self):
        # A settings.json carrying both keys prefers the new one.
        config = _build_config({"accent": "british",
                                "english_accent": "american"})
        self.assertEqual(config.ACCENT, "british")

    def test_legacy_voice_still_resolves_against_variant(self):
        config = _build_config({"english_accent": "british", "voice": "bm_george"})
        self.assertEqual(config.KOKORO_VOICE, "bm_george")


class PracticeLanguageTests(_ConfigTestBase):
    def test_unknown_practice_language_falls_back_to_english(self):
        config = _build_config({"practice_language": "klingon"})
        self.assertEqual(config.PRACTICE_LANGUAGE, "english")
        self.assertEqual(config.TARGET_LANGUAGE, "English")

    def test_practice_text_file_default_comes_from_profile(self):
        config = _build_config({})
        # The default resolves the profile's relative text path to an absolute
        # one under the project root.
        self.assertTrue(config.PRACTICE_TEXT_FILE.endswith("practice_text.txt"))


class KeyRegistryTests(_ConfigTestBase):
    """The new keys are registered so they are neither warned about nor lost."""

    def setUp(self):
        self.config = _build_config({})

    def test_new_keys_are_known(self):
        self.assertIn("practice_language", self.config._KNOWN_USER_KEYS)
        self.assertIn("accent", self.config._KNOWN_USER_KEYS)

    def test_defaults_cover_exactly_the_known_keys(self):
        self.assertEqual(set(self.config.USER_SETTING_DEFAULTS),
                         set(self.config._KNOWN_USER_KEYS))

    def test_default_voice_matches_default_variant(self):
        resolved = self.config.default_user_settings()
        self.assertIn(resolved["voice"],
                      self.config.accent_voices(resolved["accent"]))


class ProfileTextTests(_ConfigTestBase):
    """Stage 2: language-specific text lives in the profile and is derived."""

    def setUp(self):
        self.config = _build_config({})

    def test_phrase_gen_block_is_complete(self):
        pg = self.config.LANGUAGE_PROFILES["english"]["phrase_gen"]
        for key in ("system", "fragment_system", "full_ask", "fragment_ask"):
            self.assertIn(key, pg)
            self.assertTrue(pg[key].strip(), key)

    def test_derived_prompt_constants_match_profile(self):
        pg = self.config.LANGUAGE_PROFILES["english"]["phrase_gen"]
        self.assertEqual(self.config.PHRASE_GEN_SYSTEM_PROMPT, pg["system"])
        self.assertEqual(self.config.PHRASE_GEN_FRAGMENT_SYSTEM_PROMPT,
                         pg["fragment_system"])
        self.assertEqual(self.config.PHRASE_GEN_FULL_ASK, pg["full_ask"])
        self.assertEqual(self.config.PHRASE_GEN_FRAGMENT_ASK, pg["fragment_ask"])

    def test_preview_warmup_and_flores_are_derived(self):
        profile = self.config.LANGUAGE_PROFILES["english"]
        self.assertEqual(self.config.PREVIEW_PHRASE, profile["preview_phrase"])
        self.assertEqual(self.config.TRANSLATOR_WARMUP,
                         profile["translator_warmup"])
        self.assertEqual(self.config.SOURCE_FLORES_CODE, profile["flores_code"])

    def test_every_profile_carries_language_text(self):
        # A new language must ship all its language text, so enabling it needs
        # no code branch - only a profile entry.
        for name, profile in self.config.LANGUAGE_PROFILES.items():
            self.assertIn("phrase_gen", profile, name)
            self.assertIn("preview_phrase", profile, name)
            self.assertIn("translator_warmup", profile, name)


class TranslationTargetTests(_ConfigTestBase):
    def setUp(self):
        self.config = _build_config({})

    def test_off_choice_and_other_languages_present(self):
        targets = self.config.translation_targets()
        self.assertIn("", targets)          # "translation off"
        self.assertIn("Spanish", targets)   # a valid target for English practice

    def test_active_language_is_excluded(self):
        # Translating into the practiced language is pointless, so its display
        # name never appears among the targets.
        self.assertNotIn(self.config.TARGET_LANGUAGE,
                         self.config.translation_targets())

    def test_english_practice_keeps_the_base_list(self):
        # English is not in the base list today, so nothing is removed.
        self.assertEqual(self.config.translation_targets(),
                         self.config.TRANSLATION_LANGUAGES)


class PhonemeExampleTests(_ConfigTestBase):
    def setUp(self):
        self.config = _build_config({})

    def test_example_for_uses_active_language(self):
        from mimora import phoneme_examples
        # A known English phone resolves to its example word; stress/length
        # marks are tolerated (mirrors the engine's symbols).
        self.assertEqual(phoneme_examples.example_for("i"), "see")
        self.assertEqual(phoneme_examples.example_for("ˈiː"), "see")

    def test_example_for_unknown_language_returns_none(self):
        from mimora import phoneme_examples
        self.assertIsNone(phoneme_examples.example_for("i", language="klingon"))

    def test_registry_has_english_table(self):
        from mimora import phoneme_examples
        self.assertIn("english",
                      phoneme_examples.PHONEME_EXAMPLES_BY_LANGUAGE)


if __name__ == "__main__":
    unittest.main()
