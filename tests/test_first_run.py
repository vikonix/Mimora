# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for the startup download plan (mimora/first_run.py).

No network and no downloads: every predicate is patched out, so what is under
test is the decision, not the machine the tests run on.

The decision worth testing is which components each level contains, and which
level they land in. Getting the first wrong makes the first-run dialog ask for
the wrong number of gigabytes, which is the one thing the dialog exists to
state honestly; getting the second wrong makes one Skip switch off a feature
the user never refused, because the level is what ensure_ready maps to a
settings.json write.

The specific trap is NLLB. It must not be in the required level (translation is
off by default, and asking for 2483 MB would nearly double what a default run
needs), and it must not be folded into the optional one either (that level
already means "the chat model", and its refusal writes llm_backend "off").

Run from the project root with:

    python -m unittest tests.test_first_run
"""

import ast
import unittest
from pathlib import Path
from unittest import mock

from mimora import (config, first_run, gguf_fetch, llama_server_fetch, loader,
                    model_fetch, models_info, spacy_model_fetch)

ENGINES = ("phoneme", "acoustic", "none")
TTS_BACKENDS = ("kokoro", "supertonic")


def _component(key="x", label="X", size_mb=1, present=False):
    return first_run.Component(key, label, size_mb, present)


class RequiredLevelTests(unittest.TestCase):
    """Which models a run cannot start without."""

    def test_default_configuration_is_kokoro_plus_the_phoneme_recognizer(self):
        # The spaCy pipeline is in there because Kokoro is: its G2P step loads
        # one, and on an installed tool nothing else can put it in place.
        self.assertEqual(first_run.required_models("phoneme", "kokoro"),
                         (models_info.KOKORO, models_info.SPACY_EN,
                          models_info.WAV2VEC2_PHONEME))

    def test_only_the_active_engine_recognizer_is_required(self):
        # The dispatcher never loads the inactive engine's weights, so
        # requiring them would ask for 1262 MB the run does not touch.
        self.assertNotIn(models_info.WAV2VEC2_ACOUSTIC,
                         first_run.required_models("phoneme", "kokoro"))
        self.assertNotIn(models_info.WAV2VEC2_PHONEME,
                         first_run.required_models("acoustic", "kokoro"))

    def test_only_the_active_tts_backend_model_is_required(self):
        self.assertNotIn(models_info.SUPERTONIC,
                         first_run.required_models("phoneme", "kokoro"))
        self.assertNotIn(models_info.KOKORO,
                         first_run.required_models("phoneme", "supertonic"))

    def test_the_none_engine_requires_no_recognizer(self):
        # "none" disables scoring and loads no recognizer at all; the TTS model
        # and whatever that backend needs alongside it are all that is left.
        for tts_backend in TTS_BACKENDS:
            with self.subTest(tts_backend=tts_backend):
                expected = ((first_run._TTS_MODEL[tts_backend],)
                            + first_run._TTS_SUPPORT_MODELS.get(tts_backend, ()))
                self.assertEqual(first_run.required_models("none", tts_backend),
                                 expected)

    def test_the_spacy_pipeline_follows_the_tts_backend_not_the_engine(self):
        # Spanish synthesises through Supertonic, which never touches spaCy.
        # Asking that machine to agree to a download it will never read is the
        # same mistake as putting NLLB in the required level.
        for engine in ENGINES:
            with self.subTest(engine=engine):
                self.assertIn(models_info.SPACY_EN,
                              first_run.required_models(engine, "kokoro"))
                self.assertNotIn(models_info.SPACY_EN,
                                 first_run.required_models(engine, "supertonic"))

    def test_nllb_is_never_required(self):
        # It has a level of its own (see TranslatorLevelTests). Even with
        # translation on, the app starts perfectly well while the model is
        # missing - the phrase is simply shown without a translation - so it
        # cannot be part of the level whose refusal means "quit".
        for engine in ENGINES:
            for tts_backend in TTS_BACKENDS:
                with self.subTest(engine=engine, tts_backend=tts_backend):
                    self.assertNotIn(
                        models_info.NLLB,
                        first_run.required_models(engine, tts_backend))

    def test_every_combination_yields_known_catalogue_records(self):
        # A typo in either table would show up as a component with no size.
        catalogue = {*models_info.HF_REPOS, models_info.SUPERTONIC,
                     models_info.SPACY_EN}
        for engine in ENGINES:
            for tts_backend in TTS_BACKENDS:
                with self.subTest(engine=engine, tts_backend=tts_backend):
                    for model in first_run.required_models(engine, tts_backend):
                        self.assertIn(model, catalogue)

    def test_totals_match_the_measured_sizes(self):
        # These three numbers are what the dialog reads out to the user, so a
        # re-snap of models_info has to fail here rather than quietly change
        # what the app asks for.
        cases = {("phoneme", "kokoro"): 1640,     # default configuration
                 ("acoustic", "kokoro"): 1638,    # the other engine
                 ("phoneme", "supertonic"): 1668}  # Spanish, no spaCy pipeline
        for (engine, tts_backend), expected in cases.items():
            with self.subTest(engine=engine, tts_backend=tts_backend):
                models = first_run.required_models(engine, tts_backend)
                self.assertEqual(sum(m.size_mb for m in models), expected)

    def test_the_engine_table_agrees_with_config(self):
        # config._ENGINE_MODEL_REPO makes the same choice for the offline gate.
        # The two are built independently on purpose (this one drops NLLB), but
        # they must not disagree about which recognizer an engine uses.
        for engine, model in first_run._ENGINE_RECOGNIZER.items():
            with self.subTest(engine=engine):
                self.assertEqual(config._ENGINE_MODEL_REPO[engine],
                                 model.repo_id)


class TranslatorLevelTests(unittest.TestCase):
    """The level that exists only when translation is on."""

    def test_no_translation_language_means_no_download(self):
        # The default. A machine that never enables translation is never asked
        # for the 2483 MB, and - the other half of the same fact - its offline
        # gate does not wait for a model it will not load.
        self.assertEqual(first_run.translator_models(""), ())

    def test_a_selected_language_requires_nllb(self):
        for language in ("Russian", "Japanese"):
            with self.subTest(language=language):
                self.assertEqual(first_run.translator_models(language),
                                 (models_info.NLLB,))

    def test_the_model_is_the_one_config_loads(self):
        # config binds NLLB_TRANSLATOR_MODEL_NAME from the same record. If the
        # two ever named different repos, the window would download one model
        # and the app would then look for another.
        self.assertEqual(first_run._TRANSLATOR_MODEL.repo_id,
                         config.NLLB_TRANSLATOR_MODEL_NAME)

    def test_the_offline_gate_asks_for_it_on_the_same_condition(self):
        # The two decisions are made in different modules and must agree, in
        # both directions. Requiring NLLB with translation off would leave the
        # Hub online at every start for a model that is never loaded (which is
        # what it used to do); not requiring it with translation on would flip
        # the process offline while the model is still missing, and then the
        # download could not happen at all.
        self.assertEqual(
            models_info.NLLB.repo_id in config._CACHED_REPOS,
            bool(config.TRANSLATION_LANGUAGE))

    def test_the_level_follows_the_setting(self):
        with mock.patch.object(config, "TRANSLATION_LANGUAGE", ""), \
                mock.patch.object(first_run, "model_present",
                                  return_value=False):
            self.assertEqual(first_run._translator_components(), ())
        with mock.patch.object(config, "TRANSLATION_LANGUAGE", "Russian"), \
                mock.patch.object(first_run, "model_present",
                                  return_value=False):
            components = first_run._translator_components()
        self.assertEqual([c.key for c in components],
                         [models_info.NLLB.repo_id])
        self.assertEqual(components[0].size_mb, models_info.NLLB.size_mb)


class PresenceTests(unittest.TestCase):
    """Presence is asked of the app, not of our own downloaders."""

    def test_a_hub_repo_goes_through_the_predicate_config_also_uses(self):
        # loader.models_cached directly, not model_fetch.hf_repo_cached, which
        # gives the same answer but also runs prepare_hf_env() and would change
        # how the process downloads just by being asked what is on disk.
        with mock.patch.object(loader, "models_cached",
                               return_value=True) as cached:
            self.assertTrue(first_run.model_present(models_info.KOKORO))
        (hub_dir, repos), _ = cached.call_args
        self.assertEqual(Path(hub_dir).name, "hub")
        self.assertEqual(tuple(repos), (models_info.KOKORO.repo_id,))

    def test_a_packaged_model_goes_through_its_own_cache_check(self):
        with mock.patch.object(model_fetch, "supertonic_cached",
                               return_value=False) as cached:
            self.assertFalse(first_run.model_present(models_info.SUPERTONIC))
        cached.assert_called_once_with()

    def test_a_wheel_model_asks_whether_the_interpreter_can_find_it(self):
        # model_available, not model_in_sidecar: a checkout whose environment
        # already has the pipeline, and an installed tool that was given one
        # with --with, both need no download. Asking our own directory would
        # offer one to a machine that is already fine.
        with mock.patch.object(spacy_model_fetch, "model_available",
                               return_value=True) as available:
            self.assertTrue(first_run.model_present(models_info.SPACY_EN))
        available.assert_called_once_with()

    def test_a_wheel_model_is_keyed_by_its_distribution_name(self):
        # The key is what first_run_download dispatches on, and a wheel has no
        # repo id to be identified by.
        with mock.patch.object(spacy_model_fetch, "model_available",
                               return_value=True):
            component = first_run._model_component(models_info.SPACY_EN)
        self.assertEqual(component.key, models_info.SPACY_EN.name)

    def _status(self, path):
        with mock.patch.object(config, "resolve_llama_server_path",
                               return_value=path):
            return first_run.llama_server_status()

    def test_the_binary_is_taken_from_config_not_from_bin_llama(self):
        # config.resolve_llama_server_path() also covers a binary named in
        # settings.json or found on PATH; installed_exe() would miss both and
        # offer a 641 MB download to a machine that already works.
        self.assertEqual(self._status(__file__), first_run.SERVER_PRESENT)

    def test_no_path_at_all_means_absent(self):
        self.assertEqual(self._status(""), first_run.SERVER_ABSENT)

    def test_a_path_naming_nothing_is_misconfigured_not_absent(self):
        # The difference decides whether a download is offered. Only the
        # settings branch of the resolver can return a path that is not there,
        # and a download would go to bin/llama/ while that setting keeps
        # winning - so this case must not read as "absent, go fetch one".
        missing = str(Path(__file__).with_name("no-such-llama-server"))
        self.assertEqual(self._status(missing),
                         first_run.SERVER_MISCONFIGURED)


class OptionalLevelTests(unittest.TestCase):
    """The llama-server level, and when it is not offered at all."""

    def setUp(self):
        # Default for every case below: the backend that needs the level, and
        # nothing installed. Individual tests override what they are about.
        patches = (
            mock.patch.object(config, "LLM_BACKEND", "llama-server"),
            mock.patch.object(first_run, "llama_server_status",
                              return_value=first_run.SERVER_ABSENT),
            mock.patch.object(gguf_fetch, "gguf_present", return_value=False),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_the_other_backends_are_not_asked_for_anything(self):
        # lm-studio talks to a server somebody else runs, off loads no LLM at
        # all; asking either to agree to 2.6 GB would be a plain mistake.
        for backend in ("lm-studio", "off"):
            with self.subTest(backend=backend):
                with mock.patch.object(config, "LLM_BACKEND", backend), \
                        mock.patch.object(llama_server_fetch,
                                          "select_variant") as select:
                    components, blocked = first_run._optional_components()
                self.assertEqual(components, ())
                # Not blocked: nothing is missing, the level does not apply.
                self.assertIsNone(blocked)
                select.assert_not_called()

    def test_a_missing_binary_is_sized_from_the_selected_variant(self):
        with mock.patch.object(llama_server_fetch, "select_variant",
                               return_value="win-cpu-x64"):
            components, blocked = first_run._optional_components()
        self.assertIsNone(blocked)
        binary = components[0]
        self.assertEqual(binary.key, first_run.KEY_LLAMA_SERVER)
        self.assertFalse(binary.present)
        self.assertEqual(binary.size_mb,
                         llama_server_fetch.variant_size_mb("win-cpu-x64"))
        # Carried, not re-derived at download time: the size above is the size
        # OF THIS BUILD, and a later select_variant() could answer otherwise.
        self.assertEqual(binary.variant, "win-cpu-x64")

    def test_only_the_binary_names_a_variant(self):
        # Every other component has exactly one form, so a variant on one would
        # be a value nothing could act on.
        with mock.patch.object(llama_server_fetch, "select_variant",
                               return_value="win-cpu-x64"):
            components, _ = first_run._optional_components()
        gguf = components[1]
        self.assertEqual(gguf.key, first_run.KEY_GGUF)
        self.assertIsNone(gguf.variant)

    def test_an_installed_binary_costs_no_variant_resolution(self):
        # select_variant() shells out to nvidia-smi. Nothing needs the size of
        # a binary that is already there, so a normal start must not pay for it.
        with mock.patch.object(first_run, "llama_server_status",
                               return_value=first_run.SERVER_PRESENT), \
                mock.patch.object(llama_server_fetch,
                                  "select_variant") as select:
            components, blocked = first_run._optional_components()
        select.assert_not_called()
        self.assertIsNone(blocked)
        binary = components[0]
        self.assertTrue(binary.present)
        self.assertIsNone(binary.size_mb)

    def test_a_platform_without_a_build_is_offered_nothing(self):
        # Not "the GGUF only": one model without a server does not start the
        # backend. The dialog points at PATH or lm-studio instead.
        unsupported = llama_server_fetch.UnsupportedPlatformError("no build")
        with mock.patch.object(llama_server_fetch, "select_variant",
                               side_effect=unsupported):
            components, blocked = first_run._optional_components()
        self.assertEqual(components, ())
        self.assertEqual(blocked, first_run.BLOCKED_NO_BUILD)

    def test_a_binary_named_by_a_broken_setting_is_offered_nothing(self):
        # The download would land in bin/llama/ while the settings.json path
        # goes on winning in the resolver, so it would cost 641 MB and change
        # nothing. select_variant() is not even consulted.
        with mock.patch.object(first_run, "llama_server_status",
                               return_value=first_run.SERVER_MISCONFIGURED), \
                mock.patch.object(llama_server_fetch,
                                  "select_variant") as select:
            components, blocked = first_run._optional_components()
        self.assertEqual(components, ())
        self.assertEqual(blocked, first_run.BLOCKED_BAD_SETTING)
        select.assert_not_called()

    def test_the_gguf_is_looked_for_where_settings_json_points(self):
        # gguf_fetch may not import config, so its own default is models/<name>
        # and it cannot see an overridden external_model_path.
        with mock.patch.object(llama_server_fetch, "select_variant",
                               return_value="win-cpu-x64"), \
                mock.patch.object(gguf_fetch, "gguf_present",
                                  return_value=False) as present:
            first_run._optional_components()
        present.assert_called_once_with(config.EXTERNAL_MODEL_PATH)


class PlanTests(unittest.TestCase):
    """The sums the dialog quotes and the progress bar divides by."""

    def test_missing_and_totals_ignore_what_is_already_there(self):
        plan = first_run.Plan(
            required=(_component("a", size_mb=100, present=True),
                      _component("b", size_mb=200, present=False)),
            optional=(_component("c", size_mb=300, present=False),),
            llama_server_blocked=None)
        self.assertEqual([c.key for c in plan.missing_required], ["b"])
        self.assertEqual(plan.missing_required_mb, 200)
        self.assertEqual(plan.missing_optional_mb, 300)

    def test_an_empty_level_costs_nothing(self):
        plan = first_run.Plan(required=(), optional=(),
                              llama_server_blocked=None)
        self.assertEqual(plan.missing_required_mb, 0)
        self.assertEqual(plan.missing_optional_mb, 0)
        # Also the default value of the field, which every construction that
        # predates the translator level relies on.
        self.assertEqual(plan.translator, ())
        self.assertEqual(plan.missing_translator_mb, 0)

    def test_the_translator_level_is_counted_on_its_own(self):
        # Separately from the optional one, because the window offers them as
        # separate choices and each total is quoted next to its own checkbox.
        plan = first_run.Plan(
            required=(),
            optional=(_component("gguf", size_mb=2019, present=False),),
            llama_server_blocked=None,
            translator=(_component("nllb", size_mb=2483, present=False),))
        self.assertEqual(plan.missing_optional_mb, 2019)
        self.assertEqual(plan.missing_translator_mb, 2483)
        self.assertEqual([c.key for c in plan.missing_translator], ["nllb"])

    def test_a_present_binary_never_reaches_a_total(self):
        # size_mb is None there, and a None slipping into the denominator would
        # silently shrink it rather than fail.
        plan = first_run.Plan(
            required=(),
            optional=(_component(first_run.KEY_LLAMA_SERVER, size_mb=None,
                                 present=True),
                      _component("gguf", size_mb=2019, present=False)),
            llama_server_blocked=None)
        self.assertEqual(plan.missing_optional_mb, 2019)

    def test_totalling_a_sizeless_component_fails_loudly(self):
        with self.assertRaises(RuntimeError):
            first_run.total_mb((_component("odd", size_mb=None, present=False),))


class BuildPlanTests(unittest.TestCase):
    """The levels assembled, with the machine patched out."""

    def test_a_bare_machine_needs_both_levels(self):
        with mock.patch.object(config, "ENGINE", "phoneme"), \
                mock.patch.object(config, "TTS_BACKEND", "kokoro"), \
                mock.patch.object(config, "LLM_BACKEND", "llama-server"), \
                mock.patch.object(config, "TRANSLATION_LANGUAGE", ""), \
                mock.patch.object(first_run, "model_present",
                                  return_value=False), \
                mock.patch.object(first_run, "llama_server_status",
                                  return_value=first_run.SERVER_ABSENT), \
                mock.patch.object(gguf_fetch, "gguf_present",
                                  return_value=False), \
                mock.patch.object(llama_server_fetch, "select_variant",
                                  return_value="win-cpu-x64"):
            plan = first_run.build_plan()

        self.assertEqual([c.key for c in plan.required],
                         [models_info.KOKORO.repo_id,
                          models_info.SPACY_EN.name,
                          models_info.WAV2VEC2_PHONEME.repo_id])
        self.assertEqual(plan.missing_required_mb, 1640)
        self.assertEqual([c.key for c in plan.optional],
                         [first_run.KEY_LLAMA_SERVER, first_run.KEY_GGUF])
        self.assertIsNone(plan.llama_server_blocked)
        # A first run is a machine with translation off, so the third level is
        # empty even here. It is what keeps the default first run at 1640 MB.
        self.assertEqual(plan.translator, ())

    def test_translation_on_adds_the_third_level_and_nothing_else(self):
        # The state the app restarts into after the user turns translation on:
        # everything else is already installed, so this level appears alone.
        with mock.patch.object(config, "LLM_BACKEND", "llama-server"), \
                mock.patch.object(config, "TRANSLATION_LANGUAGE", "Russian"), \
                mock.patch.object(first_run, "model_present",
                                  side_effect=lambda m: m is not models_info.NLLB), \
                mock.patch.object(first_run, "llama_server_status",
                                  return_value=first_run.SERVER_PRESENT), \
                mock.patch.object(gguf_fetch, "gguf_present",
                                  return_value=True):
            plan = first_run.build_plan()

        self.assertEqual(plan.missing_required, ())
        self.assertEqual(plan.missing_optional, ())
        self.assertEqual([c.key for c in plan.missing_translator],
                         [models_info.NLLB.repo_id])
        self.assertEqual(plan.missing_translator_mb, models_info.NLLB.size_mb)

    def test_a_fully_installed_machine_needs_nothing(self):
        with mock.patch.object(config, "LLM_BACKEND", "llama-server"), \
                mock.patch.object(config, "TRANSLATION_LANGUAGE", "Russian"), \
                mock.patch.object(first_run, "model_present",
                                  return_value=True), \
                mock.patch.object(first_run, "llama_server_status",
                                  return_value=first_run.SERVER_PRESENT), \
                mock.patch.object(gguf_fetch, "gguf_present",
                                  return_value=True):
            plan = first_run.build_plan()

        self.assertEqual(plan.missing_required, ())
        self.assertEqual(plan.missing_optional, ())
        self.assertEqual(plan.missing_translator, ())
        self.assertEqual(plan.missing_required_mb, 0)
        self.assertEqual(plan.missing_optional_mb, 0)
        self.assertEqual(plan.missing_translator_mb, 0)


class ImportDisciplineTests(unittest.TestCase):
    """This module may read config; the fetchers may not read this module.

    config flips HF_HUB_OFFLINE=1 once the models are cached, so a fetcher that
    reached config through here would switch the network off exactly when a
    download is wanted.
    """

    def _imports(self, module) -> set:
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add(node.module or "")
                names.update(alias.name for alias in node.names)
        return names

    def test_no_fetcher_imports_this_module(self):
        for module in (model_fetch, gguf_fetch, llama_server_fetch,
                       spacy_model_fetch):
            with self.subTest(module=module.__name__):
                self.assertNotIn("first_run", self._imports(module))

    def test_this_module_is_the_one_that_knows_about_config(self):
        # Stated as a test so the dependency direction is asserted rather than
        # merely described in the docstring.
        self.assertIn("config", self._imports(first_run))


class BlockedReasonCoverageTests(unittest.TestCase):
    """Every reason the plan can give has a note the window can show.

    The window indexes its texts by reason, so a reason added without one would
    raise KeyError while building the first-run dialog - i.e. on a machine that
    has nothing yet, at the only moment this code ever runs.
    """

    def test_the_window_has_a_text_for_every_blocked_reason(self):
        # Imported here rather than at the top: this file is about first_run,
        # and only this one assertion needs the tkinter-importing window.
        from mimora import first_run_window

        reasons = {value for name, value in vars(first_run).items()
                   if name.startswith("BLOCKED_")}
        self.assertEqual(set(first_run_window._BLOCKED_TEXT), reasons)


class StillMissingTests(unittest.TestCase):
    """A Plan is a snapshot, so the window has to subtract what it fetched.

    The bug this covers: the window read plan.missing_required for the whole of
    its life, so after a download that finished the required level and then
    failed on the optional one, it still believed nothing had arrived. Its
    second button stayed "Quit" instead of becoming "Skip", and pressing it
    exited the app - throwing away a download that had just succeeded, on a
    machine that was by then perfectly startable.
    """

    def setUp(self):
        from mimora import first_run_window
        self.still_missing = first_run_window.still_missing
        self.components = (_component("a"), _component("b"))

    def test_nothing_fetched_leaves_the_list_alone(self):
        self.assertEqual(self.still_missing(self.components, set()),
                         self.components)

    def test_a_fetched_component_drops_out(self):
        self.assertEqual(self.still_missing(self.components, {"a"}),
                         (_component("b"),))

    def test_everything_fetched_leaves_nothing(self):
        # This is the state that decides "Skip" over "Quit".
        self.assertEqual(self.still_missing(self.components, {"a", "b"}), ())

    def test_keys_from_another_level_are_ignored(self):
        # The window keeps one set of fetched keys for both levels, so the
        # required list must not shrink because an optional component landed.
        self.assertEqual(self.still_missing(self.components, {"gguf-chat"}),
                         self.components)

    def test_order_is_preserved(self):
        # It is also the retry's fetch order, and the required level has to
        # keep coming before the optional one.
        three = (_component("a"), _component("b"), _component("c"))
        self.assertEqual([c.key for c in self.still_missing(three, {"b"})],
                         ["a", "c"])


class EnsureReadyTests(unittest.TestCase):
    """What app.run is told once the window has closed.

    The rule: if the machine changed, config is stale and the app restarts.
    config read the machine while it was being imported - what was cached, what
    the hardware probe had written - and the window then changes exactly that.
    The offline gate is the sharp one: decided before the download, it would
    leave a session that just fetched the last missing model revalidating over
    the network everything it loads.

    "Changed" is asked of the plan rather than of the window having run, and
    that difference is a loop guard. A download can report success while the
    plan still calls the component missing, and restarting there would reopen
    the same window forever.
    """

    def _window_outcome(self, **kwargs):
        """An Outcome with everything false except what the caller names.

        Not spelled `_outcome`: unittest.TestCase.run() assigns an _Outcome
        object to self._outcome, and an instance attribute shadows a method of
        the same name, so every call died with "'_Outcome' object is not
        callable".
        """
        from mimora import first_run_window

        fields = dict(quit_requested=False, optional_declined=False,
                      translator_declined=False)
        fields.update(kwargs)
        return first_run_window.Outcome(**fields)

    def _ensure_ready(self, outcome, plan, after=None, probed=False):
        """Run ensure_ready with the window and the machine stubbed out.

        *plan* is what build_plan answers before the window runs and *after*
        what it answers once it has; the default is "everything it wanted is
        now on disk". The two are separate because the decision under test is
        precisely the difference between them.
        """
        from mimora import first_run_window

        if after is None:
            after = first_run.Plan(required=(), optional=(),
                                   llama_server_blocked=None)
        with mock.patch.object(first_run, "build_plan",
                               side_effect=[plan, after]), \
                mock.patch.object(first_run_window, "FirstRunWindow") as window, \
                mock.patch.object(first_run_window, "_detect_hardware_once",
                                  return_value=probed) as probe, \
                mock.patch.object(config, "save_user_setting") as saved, \
                mock.patch.object(config, "LLM_BACKEND", config.LLM_BACKEND), \
                mock.patch.object(config, "TRANSLATION_LANGUAGE",
                                  config.TRANSLATION_LANGUAGE):
            # The last two patches only save and restore: the decline branches
            # assign to them, and a unit test must not leave the live config
            # (or, through save_user_setting, settings.json) changed.
            window.return_value.run.return_value = outcome
            result = first_run_window.ensure_ready()
            # Read while the patches are still in place: they restore the real
            # values on the way out, so a test that looked afterwards would be
            # asserting about this machine's settings.json instead.
            self.live_backend = config.LLM_BACKEND
            self.live_translation = config.TRANSLATION_LANGUAGE
        return result, window, probe, saved

    def _translator_plan(self):
        return first_run.Plan(
            required=(), optional=(), llama_server_blocked=None,
            translator=(_component("nllb", size_mb=2483),))

    def test_nothing_missing_never_builds_the_window(self):
        from mimora import first_run_window

        plan = first_run.Plan(required=(), optional=(),
                              llama_server_blocked=None)
        result, window, _probe, saved = self._ensure_ready(
            self._window_outcome(), plan)
        self.assertEqual(result, first_run_window.READY)
        window.assert_not_called()
        saved.assert_not_called()

    def test_a_completed_download_restarts(self):
        from mimora import first_run_window

        result, _window, probe, saved = self._ensure_ready(
            self._window_outcome(), self._translator_plan())
        self.assertEqual(result, first_run_window.RESTART)
        # The probe has to run here whatever it answers: this is the moment
        # the llama-server binary exists.
        probe.assert_called_once_with()
        saved.assert_not_called()

    def test_nothing_actually_fetched_starts_instead_of_looping(self):
        # The state that makes "the window ran, so restart" wrong: a stray
        # *.incomplete blob keeps loader.models_cached answering "not cached",
        # the download fetches nothing because it needs nothing, and the plan
        # comes back identical. Restarting there would reopen the same window
        # forever, so the app starts and says so instead.
        from mimora import first_run_window

        plan = self._translator_plan()
        # assertLogs rather than letting it through: the warning is half of
        # what this test is about (a machine in this state has to say which
        # repo, or nobody will ever find the stray file), and capturing it also
        # keeps logging's last-resort handler from printing it into the middle
        # of an otherwise quiet test run.
        with self.assertLogs("mimora.first_run_window", "WARNING") as logs:
            result, _window, _probe, _saved = self._ensure_ready(
                self._window_outcome(), plan, after=plan)
        self.assertEqual(result, first_run_window.READY)
        self.assertIn("nllb", logs.output[0])
        self.assertIn("incomplete", logs.output[0])

    def test_the_hardware_probe_still_forces_a_restart_on_its_own(self):
        # A true first run whose plan happens to come back unchanged (the case
        # above) still has to restart, because the probe has just rewritten the
        # file config read at import.
        from mimora import first_run_window

        plan = self._translator_plan()
        result, _window, _probe, _saved = self._ensure_ready(
            self._window_outcome(), plan, after=plan, probed=True)
        self.assertEqual(result, first_run_window.RESTART)

    def test_quitting_is_the_one_path_that_does_not_restart(self):
        from mimora import first_run_window

        result, _window, _probe, saved = self._ensure_ready(
            self._window_outcome(quit_requested=True), self._translator_plan())
        self.assertEqual(result, first_run_window.CANCELLED)
        # Nothing is recorded on the way out: the question returns next time.
        saved.assert_not_called()

    def test_a_declined_translator_is_written_back_and_restarts(self):
        from mimora import first_run_window

        result, _window, _probe, saved = self._ensure_ready(
            self._window_outcome(translator_declined=True), self._translator_plan())
        self.assertEqual(result, first_run_window.RESTART)
        saved.assert_called_once_with("translation_language", "")
        # Applied live as well, because this process goes on to build the plan
        # and the settings window from the same constant.
        self.assertEqual(self.live_translation, "")

    def test_a_declined_optional_level_is_written_back_and_restarts(self):
        from mimora import first_run_window

        plan = first_run.Plan(
            required=(), optional=(_component("gguf", size_mb=2019),),
            llama_server_blocked=None)
        result, _window, _probe, saved = self._ensure_ready(
            self._window_outcome(optional_declined=True), plan)
        self.assertEqual(result, first_run_window.RESTART)
        saved.assert_called_once_with("llm_backend", "off")
        self.assertEqual(self.live_backend, "off")


class RefusalTests(unittest.TestCase):
    """"Skip" refuses exactly the levels that are still missing, and no others.

    Each flag is mapped to a settings.json write by ensure_ready, so a flag set
    for a level that was never on screen turns a feature off that nobody
    declined. That was reachable: the window used to record the refusal as a
    single unconditional boolean, which was harmless only as long as there was
    one refusable level and it could not be empty whenever this ran.

    The window is built without __init__ on purpose. What is under test is the
    bookkeeping, the constructor is entirely Tk, and a test that needed a
    display would not run in CI.
    """

    def _window(self, plan, fetched=()):
        from mimora import first_run_window
        window = object.__new__(first_run_window.FirstRunWindow)
        window.plan = plan
        window._fetched = set(fetched)
        window._thread = None      # nothing running, so _downloading() is False
        window._state = None
        window._closed = False
        window.root = mock.Mock()  # _close() only calls destroy()
        window._outcome = first_run_window.Outcome(True, False, False)
        return window

    def test_an_empty_level_is_not_refused(self):
        plan = first_run.Plan(
            required=(), optional=(), llama_server_blocked=None,
            translator=(_component("nllb", size_mb=2483),))
        window = self._window(plan)
        window._on_secondary()
        self.assertFalse(window._outcome.quit_requested)
        self.assertTrue(window._outcome.translator_declined)
        # The one that matters: no chat model was ever offered here, so
        # ensure_ready must not switch the LLM backend off.
        self.assertFalse(window._outcome.optional_declined)

    def test_a_level_that_arrived_is_not_refused(self):
        # Skip after a partial download: what was fetched is not a refusal.
        plan = first_run.Plan(
            required=(), optional=(_component("gguf"),),
            llama_server_blocked=None,
            translator=(_component("nllb"),))
        window = self._window(plan, fetched={"nllb"})
        window._on_secondary()
        self.assertTrue(window._outcome.optional_declined)
        self.assertFalse(window._outcome.translator_declined)

    def test_a_missing_required_level_is_a_quit_not_a_refusal(self):
        plan = first_run.Plan(
            required=(_component("kokoro"),), optional=(),
            llama_server_blocked=None,
            translator=(_component("nllb"),))
        window = self._window(plan)
        window._on_secondary()
        self.assertTrue(window._outcome.quit_requested)
        # Nothing is recorded on the way out: the question returns next time.
        self.assertFalse(window._outcome.optional_declined)
        self.assertFalse(window._outcome.translator_declined)


if __name__ == "__main__":
    unittest.main()
