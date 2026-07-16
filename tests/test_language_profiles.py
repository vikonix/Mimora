# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Unit tests for the language/variant configuration model (mimora/config.py).

Stages 1-3 of the multilingual refactor turned the practice language into data
(LANGUAGE_PROFILES), introduced the settings.json keys ``practice_language``
and ``accent`` (with the legacy ``english_accent`` honored as a read fallback),
and enabled Spanish (Castilian) as the second language.
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
from unittest import mock

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
                for key in ("espeak_language", "default_voice", "voices"):
                    self.assertIn(key, variant, f"{name}/{vname} missing {key!r}")
                self.assertTrue(variant["voices"], f"{name}/{vname} has no voices")
                self.assertIn(variant["default_voice"], variant["voices"],
                              f"{name}/{vname} default_voice not in voices")
                self.assertTrue(variant["espeak_language"].strip())
                # The TTS backend is per-variant data; an absent key means
                # Kokoro (see config.TTS_BACKEND). The backend determines
                # which language-code key the variant must carry.
                backend = variant.get("tts_backend", "kokoro")
                self.assertIn(backend, self.config.TTS_BACKEND_CHOICES,
                              f"{name}/{vname} unknown tts_backend {backend!r}")
                lang_key = ("kokoro_lang_code" if backend == "kokoro"
                            else "tts_lang_code")
                self.assertIn(lang_key, variant,
                              f"{name}/{vname} missing {lang_key!r}")
                self.assertTrue(str(variant[lang_key]).strip(),
                                f"{name}/{vname} empty {lang_key!r}")

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
        self.assertEqual(config.TTS_BACKEND, "kokoro")
        self.assertEqual(config.TTS_LANG_CODE, "a")
        self.assertEqual(config.ESPEAK_LANGUAGE, "en-us")
        self.assertEqual(config.TTS_VOICE, "af_heart")

    def test_accent_key_selects_variant(self):
        config = _build_config({"accent": "british"})
        self.assertEqual(config.ACCENT, "british")
        self.assertEqual(config.TTS_BACKEND, "kokoro")
        self.assertEqual(config.TTS_LANG_CODE, "b")
        self.assertEqual(config.ESPEAK_LANGUAGE, "en-gb")
        self.assertEqual(config.TTS_VOICE, "bf_emma")

    def test_unknown_accent_falls_back_to_default_variant(self):
        config = _build_config({"accent": "klingon"})
        self.assertEqual(config.ACCENT, "american")


class LegacyMigrationTests(_ConfigTestBase):
    """The legacy english_accent key is read as a fallback for accent."""

    def test_legacy_english_accent_is_honored(self):
        config = _build_config({"english_accent": "british"})
        self.assertEqual(config.ACCENT, "british")
        self.assertEqual(config.TTS_LANG_CODE, "b")

    def test_new_accent_key_wins_over_legacy(self):
        # A settings.json carrying both keys prefers the new one.
        config = _build_config({"accent": "british",
                                "english_accent": "american"})
        self.assertEqual(config.ACCENT, "british")

    def test_legacy_voice_still_resolves_against_variant(self):
        config = _build_config({"english_accent": "british", "voice": "bm_george"})
        self.assertEqual(config.TTS_VOICE, "bm_george")


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
        self.assertEqual(self.config.TTS_WARMUP, profile["tts_warmup"])
        self.assertEqual(self.config.SOURCE_FLORES_CODE, profile["flores_code"])

    def test_every_profile_carries_language_text(self):
        # A new language must ship all its language text, so enabling it needs
        # no code branch - only a profile entry.
        for name, profile in self.config.LANGUAGE_PROFILES.items():
            for key in ("phrase_gen", "preview_phrase", "translator_warmup",
                        "tts_warmup", "greeting_named", "greeting_anonymous",
                        "practice_text_fallback"):
                self.assertIn(key, profile, name)

    def test_greeting_templates_are_well_formed(self):
        for name, profile in self.config.LANGUAGE_PROFILES.items():
            # The named form must carry the placeholder and format cleanly...
            self.assertIn("{name}", profile["greeting_named"], name)
            self.assertIn("Ana", profile["greeting_named"].format(name="Ana"))
            # ...while the anonymous form is a ready sentence without one.
            self.assertNotIn("{", profile["greeting_anonymous"], name)

    def test_derived_greeting_and_fallback_match_profile(self):
        profile = self.config.LANGUAGE_PROFILES["english"]
        self.assertEqual(self.config.GREETING_NAMED, profile["greeting_named"])
        self.assertEqual(self.config.GREETING_ANONYMOUS,
                         profile["greeting_anonymous"])
        self.assertEqual(self.config.PRACTICE_TEXT_FALLBACK,
                         profile["practice_text_fallback"])


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

    def test_english_practice_drops_english_target(self):
        # English is now a base target (for practicing other languages), so
        # English practice removes it while keeping every other choice.
        targets = self.config.translation_targets()
        self.assertNotIn("English", targets)
        self.assertEqual(
            targets,
            tuple(label for label in self.config.TRANSLATION_LANGUAGES
                  if label != "English"))


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


class SpanishSelectionTests(_ConfigTestBase):
    """Stage 3: selecting Spanish wires every derived constant from its profile."""

    def setUp(self):
        self.config = _build_config({"practice_language": "spanish"})

    def test_derived_constants_follow_spanish_profile(self):
        self.assertEqual(self.config.PRACTICE_LANGUAGE, "spanish")
        self.assertEqual(self.config.TARGET_LANGUAGE, "Spanish")
        self.assertEqual(self.config.ACCENT, "castilian")
        # Spanish runs the Supertonic 3 backend (10 voices, ISO lang code) -
        # see tasks/supertonic_tts_backend_task.md.
        self.assertEqual(self.config.TTS_BACKEND, "supertonic")
        self.assertEqual(self.config.TTS_LANG_CODE, "es")
        self.assertEqual(self.config.ESPEAK_LANGUAGE, "es")
        self.assertEqual(self.config.TTS_VOICE, "F1")
        self.assertEqual(len(self.config.TTS_VOICES), 10)
        self.assertEqual(self.config.TTS_TOTAL_STEPS, 8)
        self.assertEqual(self.config.SOURCE_FLORES_CODE, "spa_Latn")
        self.assertTrue(
            self.config.PRACTICE_TEXT_FILE.endswith("practice_text_es.txt"))

    def test_spanish_prompts_come_from_spanish_profile(self):
        pg = self.config.LANGUAGE_PROFILES["spanish"]["phrase_gen"]
        self.assertEqual(self.config.PHRASE_GEN_SYSTEM_PROMPT, pg["system"])
        self.assertEqual(self.config.PHRASE_GEN_FRAGMENT_SYSTEM_PROMPT,
                         pg["fragment_system"])
        self.assertEqual(self.config.PHRASE_GEN_FULL_ASK, pg["full_ask"])
        self.assertEqual(self.config.PHRASE_GEN_FRAGMENT_ASK, pg["fragment_ask"])

    def test_preview_and_warmup_are_spanish(self):
        profile = self.config.LANGUAGE_PROFILES["spanish"]
        self.assertEqual(self.config.PREVIEW_PHRASE, profile["preview_phrase"])
        self.assertEqual(self.config.TRANSLATOR_WARMUP,
                         profile["translator_warmup"])
        self.assertEqual(self.config.TTS_WARMUP, profile["tts_warmup"])
        self.assertEqual(self.config.TTS_WARMUP, "Hola.")

    def test_greeting_and_fallback_are_spanish(self):
        profile = self.config.LANGUAGE_PROFILES["spanish"]
        self.assertEqual(self.config.GREETING_NAMED, profile["greeting_named"])
        self.assertEqual(self.config.GREETING_ANONYMOUS,
                         profile["greeting_anonymous"])
        self.assertEqual(self.config.PRACTICE_TEXT_FALLBACK,
                         profile["practice_text_fallback"])


class TTSBackendDefaultTests(_ConfigTestBase):
    """The tts_backend variant field defaults to Kokoro (english.py unchanged)."""

    def test_english_variants_carry_no_backend_field(self):
        # The default keeps existing Kokoro profiles untouched: a variant
        # without the field runs Kokoro.
        config = _build_config({})
        for variant in config.LANGUAGE_PROFILES["english"]["variants"].values():
            self.assertNotIn("tts_backend", variant)
        self.assertEqual(config.TTS_BACKEND, "kokoro")

    def test_total_steps_defaults_when_absent(self):
        # english variants name no total_steps (Kokoro ignores it); the
        # constant still resolves to the documented default.
        config = _build_config({})
        self.assertEqual(config.TTS_TOTAL_STEPS, 8)


class EngineAvailabilityTests(_ConfigTestBase):
    """An engine not offered for the language falls back to the first available."""

    def test_unavailable_engine_falls_back(self):
        # The acoustic engine is English-only ASR, so Spanish rejects it.
        config = _build_config({"practice_language": "spanish",
                                "engine": "acoustic"})
        self.assertEqual(config.ENGINE, "phoneme")

    def test_available_engine_is_kept(self):
        config = _build_config({"practice_language": "spanish", "engine": "none"})
        self.assertEqual(config.ENGINE, "none")

    def test_english_keeps_every_engine(self):
        config = _build_config({"engine": "acoustic"})
        self.assertEqual(config.ENGINE, "acoustic")


class EngineExperimentalTests(_ConfigTestBase):
    """The experimental flag is a data rule: a missing model calibration file."""

    # A language that can never have a committed calibration file, so the
    # "missing calibration" branch is tested independently of when the real
    # Spanish calibration (tasks/spanish_language_task.md) lands.
    _FAKE_PROFILE = {
        "display_name": "Klingon",
        "flores_code": "tlh_Latn",
        "default_variant": "standard",
        "engines": ("phoneme", "none"),
        "practice_text_file": "texts/practice_text.txt",
        "phrase_gen": {"system": "s", "fragment_system": "f",
                       "full_ask": "a", "fragment_ask": "b"},
        "preview_phrase": "Qapla'.",
        "translator_warmup": "Qapla'.",
        "variants": {
            "standard": {
                "kokoro_lang_code": "k",
                "espeak_language": "xx",
                "default_voice": "kf_one",
                "voices": ["kf_one"],
            },
        },
    }

    def setUp(self):
        self.config = _build_config({})

    def test_english_phoneme_is_never_experimental(self):
        # en_model_calibration.json is committed with the repo.
        self.assertFalse(self.config.engine_experimental(
            "phoneme", "english", "american"))
        self.assertFalse(self.config.PHONEME_EXPERIMENTAL)

    def test_non_phoneme_engines_are_never_experimental(self):
        self.assertFalse(self.config.engine_experimental(
            "none", "spanish", "castilian"))
        self.assertFalse(self.config.engine_experimental(
            "acoustic", "english", "american"))

    def test_missing_calibration_flags_experimental(self):
        # tearDown reloads config, so the artificial entry cannot leak.
        self.config.LANGUAGE_PROFILES["klingon"] = self._FAKE_PROFILE
        self.assertTrue(self.config.engine_experimental(
            "phoneme", "klingon", "standard"))

    def test_spanish_flag_tracks_the_calibration_file(self):
        # Experimental exactly while es_model_calibration.json is absent; this
        # stays green when the Spanish calibration lands later.
        expected = not (self.config._PHONEME_CALIBRATION_DIR
                        / "es_model_calibration.json").is_file()
        self.assertEqual(self.config.engine_experimental(
            "phoneme", "spanish", "castilian"), expected)

    def test_module_flag_matches_active_selection(self):
        config = _build_config({"practice_language": "spanish"})
        self.assertEqual(config.PHONEME_EXPERIMENTAL,
                         config.engine_experimental())


class SpanishTranslationTargetTests(_ConfigTestBase):
    def setUp(self):
        self.config = _build_config({"practice_language": "spanish"})

    def test_spanish_practice_keeps_english_and_drops_spanish(self):
        targets = self.config.translation_targets()
        self.assertIn("English", targets)
        self.assertNotIn("Spanish", targets)
        self.assertIn("", targets)  # "translation off" always stays


class TranslationValidationTests(_ConfigTestBase):
    """translation_language is validated against translation_targets()."""

    def test_practiced_language_as_target_falls_back_to_off(self):
        # A stale target left over from practicing another language: "Spanish"
        # is a member of TRANSLATION_LANGUAGES but not a sensible target while
        # practicing Spanish, so it falls back to "" like any invalid value.
        config = _build_config({"practice_language": "spanish",
                                "translation_language": "Spanish"})
        self.assertEqual(config.TRANSLATION_LANGUAGE, "")

    def test_valid_target_is_kept(self):
        config = _build_config({"practice_language": "spanish",
                                "translation_language": "English"})
        self.assertEqual(config.TRANSLATION_LANGUAGE, "English")

    def test_english_practice_rejects_english_target(self):
        config = _build_config({"translation_language": "English"})
        self.assertEqual(config.TRANSLATION_LANGUAGE, "")


class SpanishPhonemeExampleTests(_ConfigTestBase):
    def setUp(self):
        self.config = _build_config({"practice_language": "spanish"})

    def test_spanish_symbols_resolve(self):
        from mimora import phoneme_examples
        self.assertEqual(phoneme_examples.example_for("β", "spanish"), "lobo")
        self.assertEqual(phoneme_examples.example_for("ɲ", "spanish"), "niño")

    def test_tap_and_trill_have_distinct_examples(self):
        from mimora import phoneme_examples
        self.assertNotEqual(phoneme_examples.example_for("ɾ", "spanish"),
                            phoneme_examples.example_for("r", "spanish"))

    def test_active_language_default_is_spanish(self):
        # With Spanish active, example_for without a language argument answers
        # from the Spanish table (the badge tooltips follow the practice).
        from mimora import phoneme_examples
        self.assertEqual(phoneme_examples.example_for("x"), "jamón")

    def test_stress_marks_are_tolerated(self):
        from mimora import phoneme_examples
        self.assertEqual(phoneme_examples.example_for("ˈa", "spanish"), "casa")


class SingleVoiceProfileTests(_ConfigTestBase):
    """The Random-voice rule is data-driven: fewer than two voices disables it.

    Checked on an artificial single-voice profile (task §4.2), not through the
    UI: settings_window.py enables "Random voice per phrase" only when the
    running variant offers at least two voices (its ``enabled`` lambda tests
    ``len(config.TTS_VOICES) >= 2``), which is the same voices-list length
    rule exercised here.
    """

    def setUp(self):
        self.config = _build_config({})
        # Mirrors the future French entry (§1: one voice, ff_siwis); tearDown
        # reloads config, so the artificial entry cannot leak.
        self.config.LANGUAGE_PROFILES["french"] = {
            "display_name": "French",
            "flores_code": "fra_Latn",
            "default_variant": "standard",
            "engines": ("phoneme", "none"),
            "practice_text_file": "texts/practice_text.txt",
            "phrase_gen": {"system": "s", "fragment_system": "f",
                           "full_ask": "a", "fragment_ask": "b"},
            "preview_phrase": "Salut.",
            "translator_warmup": "Salut.",
            "variants": {
                "standard": {
                    "kokoro_lang_code": "f",
                    "espeak_language": "fr-fr",
                    "default_voice": "ff_siwis",
                    "voices": ["ff_siwis"],
                },
            },
        }

    def test_single_voice_variant_fails_the_random_voice_rule(self):
        voices = self.config.accent_voices("standard", "french")
        self.assertEqual(len(voices), 1)
        self.assertLess(len(voices), 2)

    def test_multi_voice_variant_passes_the_random_voice_rule(self):
        for language in ("english", "spanish"):
            accent = self.config.default_accent(language)
            with self.subTest(language=language):
                self.assertGreaterEqual(
                    len(self.config.accent_voices(accent, language)), 2)


class MigrationWriteTests(_ConfigTestBase):
    """Persisting a new language key drops the legacy english_accent key."""

    def test_saving_new_keys_drops_legacy_key(self):
        for key, value in (("accent", "british"),
                           ("practice_language", "english")):
            with self.subTest(key=key):
                config = _build_config({"english_accent": "british"})
                with mock.patch.object(_loader, "save_setting",
                                       return_value=True), \
                     mock.patch.object(_loader, "reset_settings",
                                       return_value=True) as reset:
                    self.assertTrue(config.save_user_setting(key, value))
                reset.assert_called_once()
                self.assertEqual(set(reset.call_args[0][1]), {"english_accent"})

    def test_saving_unrelated_key_keeps_legacy_key(self):
        config = _build_config({"english_accent": "british"})
        with mock.patch.object(_loader, "save_setting", return_value=True), \
             mock.patch.object(_loader, "reset_settings",
                               return_value=True) as reset:
            self.assertTrue(config.save_user_setting("voice", "bm_george"))
        reset.assert_not_called()

    def test_no_legacy_key_means_no_cleanup(self):
        config = _build_config({"accent": "british"})
        with mock.patch.object(_loader, "save_setting", return_value=True), \
             mock.patch.object(_loader, "reset_settings",
                               return_value=True) as reset:
            self.assertTrue(config.save_user_setting("accent", "american"))
        reset.assert_not_called()


class PhraseLevelTests(_ConfigTestBase):
    """Proficiency levels (tasks/phrase_level_task.md): schema and constants."""

    def setUp(self):
        self.config = _build_config({})

    def test_every_profile_carries_six_levels(self):
        for name, profile in self.config.LANGUAGE_PROFILES.items():
            levels = profile["phrase_gen"]["levels"]
            self.assertEqual(len(levels), 6, name)
            for index, level in enumerate(levels):
                label = f"{name} level {index}"
                self.assertTrue(level["vocab_hint"].strip(), label)
                self.assertTrue(level["grammar_hint"].strip(), label)
                low, high = level["words"]
                self.assertTrue(1 <= low <= high, label)
                if level["min_zipf"] is not None:
                    self.assertGreater(level["min_zipf"], 0, label)

    def test_every_profile_carries_focus_stopwords(self):
        # Focus-word stopwords are per-language data (llm.py
        # _pick_focus_word): the frequency filter prefers frequent words,
        # so a language without its function-word list would steer the
        # pick toward them.
        for name, profile in self.config.LANGUAGE_PROFILES.items():
            stopwords = profile["phrase_gen"]["stopwords"].split()
            self.assertTrue(stopwords, name)
            self.assertEqual(stopwords,
                             [word.lower() for word in stopwords], name)

    def test_stopword_constant_is_derived_from_the_active_profile(self):
        self.assertEqual(
            self.config.PHRASE_GEN_STOPWORDS,
            frozenset(self.config.LANGUAGE_PROFILES["english"]
                      ["phrase_gen"]["stopwords"].split()))

    def test_zipf_floors_relax_as_the_level_rises(self):
        # A higher level must never demand MORE frequent vocabulary; None
        # (no floor) is treated as the loosest possible value.
        for name, profile in self.config.LANGUAGE_PROFILES.items():
            floors = [level["min_zipf"] if level["min_zipf"] is not None
                      else float("-inf")
                      for level in profile["phrase_gen"]["levels"]]
            self.assertEqual(floors, sorted(floors, reverse=True), name)

    def test_system_prompt_is_a_word_count_template(self):
        # {min_words}/{max_words} are filled per level by mimora/llm.py; the
        # template must format cleanly and contain no other placeholders.
        for name, profile in self.config.LANGUAGE_PROFILES.items():
            system = profile["phrase_gen"]["system"]
            self.assertIn("{min_words}", system, name)
            self.assertIn("{max_words}", system, name)
            formatted = system.format(min_words=3, max_words=8)
            self.assertNotIn("{", formatted, name)

    def test_level_constants_are_derived_from_the_active_profile(self):
        self.assertEqual(self.config.PHRASE_GEN_LEVEL, 3)
        self.assertIs(
            self.config.PHRASE_GEN_LEVELS,
            self.config.LANGUAGE_PROFILES["english"]["phrase_gen"]["levels"])
        self.assertEqual(self.config.PHRASE_GEN_WORDFREQ_LANG, "en")

    def test_spanish_selection_switches_levels_and_wordfreq_lang(self):
        config = _build_config({"practice_language": "spanish"})
        self.assertIs(
            config.PHRASE_GEN_LEVELS,
            config.LANGUAGE_PROFILES["spanish"]["phrase_gen"]["levels"])
        self.assertEqual(config.PHRASE_GEN_WORDFREQ_LANG, "es")

    def test_level_setting_is_read_and_validated(self):
        self.assertEqual(_build_config({"phrase_gen_level": 5}).PHRASE_GEN_LEVEL, 5)
        # Out-of-range and non-numeric values fall back to the default.
        self.assertEqual(_build_config({"phrase_gen_level": 99}).PHRASE_GEN_LEVEL, 3)
        self.assertEqual(_build_config({"phrase_gen_level": "high"}).PHRASE_GEN_LEVEL, 3)

    def test_level_key_is_registered(self):
        self.assertIn("phrase_gen_level", self.config._KNOWN_USER_KEYS)
        self.assertIn("phrase_gen_level", self.config.USER_SETTING_DEFAULTS)


if __name__ == "__main__":
    unittest.main()
