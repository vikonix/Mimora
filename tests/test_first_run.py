# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Unit tests for the startup download plan (mimora/first_run.py).

No network and no downloads: every predicate is patched out, so what is under
test is the decision, not the machine the tests run on.

The decision worth testing is which components each level contains. Getting it
wrong does not crash anything - it makes the first-run dialog ask for the wrong
number of gigabytes, which is the one thing the dialog exists to state
honestly. The specific trap is NLLB: config._CACHED_REPOS requires it
unconditionally, and reusing that constant would have the dialog ask for 4110 MB
where a default run needs 1627.

Run from the project root with:

    python -m unittest tests.test_first_run
"""

import ast
import unittest
from pathlib import Path
from unittest import mock

from mimora import (config, first_run, gguf_fetch, llama_server_fetch, loader,
                    model_fetch, models_info)

ENGINES = ("phoneme", "acoustic", "none")
TTS_BACKENDS = ("kokoro", "supertonic")


def _component(key="x", label="X", size_mb=1, present=False):
    return first_run.Component(key, label, size_mb, present)


class RequiredLevelTests(unittest.TestCase):
    """Which models a run cannot start without."""

    def test_default_configuration_is_kokoro_plus_the_phoneme_recognizer(self):
        self.assertEqual(first_run.required_models("phoneme", "kokoro"),
                         (models_info.KOKORO, models_info.WAV2VEC2_PHONEME))

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
        # "none" disables scoring and loads no recognizer at all; only the TTS
        # model is left.
        for tts_backend in TTS_BACKENDS:
            with self.subTest(tts_backend=tts_backend):
                self.assertEqual(first_run.required_models("none", tts_backend),
                                 (first_run._TTS_MODEL[tts_backend],))

    def test_nllb_is_never_required(self):
        # The whole reason this set is built here instead of being taken from
        # config._CACHED_REPOS. Translation is off by default and the app runs
        # without its 2483 MB.
        for engine in ENGINES:
            for tts_backend in TTS_BACKENDS:
                with self.subTest(engine=engine, tts_backend=tts_backend):
                    self.assertNotIn(
                        models_info.NLLB,
                        first_run.required_models(engine, tts_backend))

    def test_every_combination_yields_known_catalogue_records(self):
        # A typo in either table would show up as a component with no size.
        catalogue = {*models_info.HF_REPOS, models_info.SUPERTONIC}
        for engine in ENGINES:
            for tts_backend in TTS_BACKENDS:
                with self.subTest(engine=engine, tts_backend=tts_backend):
                    for model in first_run.required_models(engine, tts_backend):
                        self.assertIn(model, catalogue)

    def test_totals_match_the_measured_sizes(self):
        # These three numbers are what the dialog reads out to the user, so a
        # re-snap of models_info has to fail here rather than quietly change
        # what the app asks for.
        cases = {("phoneme", "kokoro"): 1627,     # default configuration
                 ("acoustic", "kokoro"): 1625,    # the other engine
                 ("phoneme", "supertonic"): 1668}  # Spanish
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
    """The two levels assembled, with the machine patched out."""

    def test_a_bare_machine_needs_both_levels(self):
        with mock.patch.object(config, "ENGINE", "phoneme"), \
                mock.patch.object(config, "TTS_BACKEND", "kokoro"), \
                mock.patch.object(config, "LLM_BACKEND", "llama-server"), \
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
                          models_info.WAV2VEC2_PHONEME.repo_id])
        self.assertEqual(plan.missing_required_mb, 1627)
        self.assertEqual([c.key for c in plan.optional],
                         [first_run.KEY_LLAMA_SERVER, first_run.KEY_GGUF])
        self.assertIsNone(plan.llama_server_blocked)

    def test_a_fully_installed_machine_needs_nothing(self):
        with mock.patch.object(config, "LLM_BACKEND", "llama-server"), \
                mock.patch.object(first_run, "model_present",
                                  return_value=True), \
                mock.patch.object(first_run, "llama_server_status",
                                  return_value=first_run.SERVER_PRESENT), \
                mock.patch.object(gguf_fetch, "gguf_present",
                                  return_value=True):
            plan = first_run.build_plan()

        self.assertEqual(plan.missing_required, ())
        self.assertEqual(plan.missing_optional, ())
        self.assertEqual(plan.missing_required_mb, 0)
        self.assertEqual(plan.missing_optional_mb, 0)


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
        for module in (model_fetch, gguf_fetch, llama_server_fetch):
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


if __name__ == "__main__":
    unittest.main()
