# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for the first-run downloader (mimora/first_run_download.py).

Nothing here downloads anything. Two things are worth testing without a
network, and they are the two that fail silently rather than loudly:

* the arithmetic behind the progress bar - a denominator that drifts or a
  double-counted byte stream makes the bar lie, and no exception is ever
  raised;
* the shape of the tqdm stand-in against the exact call forms huggingface_hub
  uses. The library is not pinned anywhere (it arrives transitively), and if a
  future version stops calling our object the way we expect, the bar simply
  stops moving. A red test here is the only warning we get.

Run from the project root with:

    python -m unittest tests.test_first_run_download
"""

import inspect
import logging
import os
import unittest
from unittest import mock

from mimora import (config, first_run, first_run_download, gguf_fetch,
                    llama_server_fetch, model_fetch, models_info,
                    spacy_model_fetch)

MB = first_run_download.BYTES_PER_MB

_saved_level = logging.NOTSET


def setUpModule():
    """Silence the module's own logging for the duration of the suite.

    DownloadLoopTests deliberately fails a fetch, and the worker answers with
    log.exception - which is exactly right in production and a full traceback
    in the middle of a passing test run. Same treatment as tests/test_model_fetch.py.
    """
    global _saved_level
    logger = logging.getLogger(first_run_download.__name__)
    _saved_level = logger.level
    logger.setLevel(logging.CRITICAL)


def tearDownModule():
    logging.getLogger(first_run_download.__name__).setLevel(_saved_level)


def _component(key="k", label="L", size_mb=10, present=False, variant=None):
    return first_run.Component(key, label, size_mb, present, variant)


class ProgressStateTests(unittest.TestCase):
    """The numbers the bar is drawn from."""

    def setUp(self):
        self.state = first_run_download.ProgressState(100 * MB)

    def test_starts_empty(self):
        snapshot = self.state.snapshot()
        self.assertEqual(snapshot.done_bytes, 0)
        self.assertEqual(snapshot.fraction, 0.0)
        self.assertFalse(snapshot.finished)

    def test_absolute_reports_do_not_accumulate(self):
        # advance() takes the byte count reached, not a delta; two reports of
        # the same number must not add up.
        self.state.begin(_component("first"))
        self.state.advance(5 * MB)
        self.state.advance(5 * MB)
        self.assertEqual(self.state.snapshot().done_bytes, 5 * MB)

    def test_a_second_reporter_does_not_double_count(self):
        # On the xet path huggingface_hub keeps a transfer bar and a
        # reconstruction bar, both of which become instances of our stand-in.
        self.state.begin(_component("first"))
        self.state.advance(8 * MB)   # reconstruction
        self.state.advance(6 * MB)   # transfer, behind
        self.assertEqual(self.state.snapshot().done_bytes, 8 * MB)

    def test_progress_never_moves_backwards(self):
        # huggingface_hub subtracts the resumed size when it retries a file.
        self.state.begin(_component("first"))
        self.state.advance(9 * MB)
        self.state.advance(2 * MB)
        self.assertEqual(self.state.snapshot().done_bytes, 9 * MB)

    def test_a_finished_component_is_banked_at_its_planned_size(self):
        self.state.begin(_component("first", size_mb=40))
        self.state.advance(3 * MB)
        self.state.complete()
        self.assertEqual(self.state.snapshot().done_bytes, 40 * MB)
        # The next component starts its own count from zero on top of that.
        self.state.begin(_component("second"))
        self.state.advance(5 * MB)
        self.assertEqual(self.state.snapshot().done_bytes, 45 * MB)

    def test_finished_components_are_named_in_order(self):
        # What the dialog reads after a failure to tell what is now on disk
        # from what its (never-updated) plan still calls missing.
        self.assertEqual(self.state.snapshot().completed_keys, ())
        self.state.begin(_component("first"))
        self.state.complete()
        self.state.begin(_component("second"))
        self.assertEqual(self.state.snapshot().completed_keys, ("first",))
        self.state.complete()
        self.assertEqual(self.state.snapshot().completed_keys,
                         ("first", "second"))

    def test_an_unfinished_component_is_not_named(self):
        # complete() is what banks a component; a begin() that never got there
        # must leave no trace, or a failed download would report itself done.
        self.state.begin(_component("first"))
        self.state.advance(9 * MB)
        self.state.fail("no network")
        self.assertEqual(self.state.snapshot().completed_keys, ())

    def test_the_denominator_never_changes(self):
        # The plan fixes it before the first byte; recomputing mid-flight is
        # what would make the percentage jump.
        self.state.begin(_component("first"))
        self.state.advance(50 * MB)
        self.assertEqual(self.state.snapshot().total_bytes, 100 * MB)

    def test_overshooting_the_plan_is_clamped(self):
        # Sizes are a pin, not a live lookup, so reality can exceed them.
        self.state.begin(_component("first"))
        self.state.advance(500 * MB)
        snapshot = self.state.snapshot()
        self.assertEqual(snapshot.done_bytes, 100 * MB)
        self.assertEqual(snapshot.fraction, 1.0)

    def test_failure_and_success_are_distinguishable(self):
        self.state.fail("no network")
        snapshot = self.state.snapshot()
        self.assertTrue(snapshot.finished)
        self.assertEqual(snapshot.error, "no network")
        self.assertIsNone(first_run_download.ProgressState(0).snapshot().error)


class TqdmStandInTests(unittest.TestCase):
    """The stand-in must survive every shape huggingface_hub calls it in."""

    def setUp(self):
        self.state = first_run_download.ProgressState(100 * MB)
        self.state.begin(_component("component"))
        self.cls = first_run_download.make_tqdm_class(self.state)

    def test_byte_bar_updates_reach_the_state(self):
        # _create_progress_bar passes keywords only for a non-hf class.
        bar = self.cls(total=10 * MB, initial=0, unit="B", unit_scale=True,
                       desc="Reconstructing")
        bar.update(3 * MB)
        bar.update(2 * MB)
        self.assertEqual(self.state.snapshot().done_bytes, 5 * MB)

    def test_a_resumed_download_counts_its_head_start(self):
        self.cls(total=10 * MB, initial=4 * MB, unit="B")
        self.assertEqual(self.state.snapshot().done_bytes, 4 * MB)

    def test_the_file_counter_contributes_no_bytes(self):
        # thread_map wraps the file iterator in tqdm_class; that bar counts
        # files, and its "updates" must not be mistaken for bytes.
        bar = self.cls(["a", "b"], desc="Fetching 2 files")
        self.assertEqual(list(bar), ["a", "b"])
        self.assertEqual(self.state.snapshot().done_bytes, 0)

    def test_total_is_readable_and_writable_as_a_number(self):
        # _AggregatedTqdm does `bar.total = (bar.total or 0) + total`, and
        # _update_transfer_bar compares bar.n against it - a stand-in whose
        # attributes were callables would raise inside the download thread.
        bar = self.cls(total=None, unit="B")
        bar.total = (bar.total or 0) + 7
        self.assertEqual(bar.total, 7)
        bar.update(1)
        self.assertEqual(bar.n, 1)

    def test_display_calls_are_accepted_and_ignored(self):
        bar = self.cls(total=1, unit="B")
        with bar:
            bar.refresh()
            bar.set_description("x")
            bar.set_postfix_str("y", refresh=False)
            self.assertEqual(bar.format_dict.get("rate"), None)
            bar.some_method_a_later_version_invents()
        bar.close()

    def test_the_upstream_hook_still_exists(self):
        # The whole approach rests on this argument being public. If a version
        # bump removes it, this fails here instead of silently freezing the bar.
        from huggingface_hub import hf_hub_download, snapshot_download
        for func in (hf_hub_download, snapshot_download):
            with self.subTest(func=func.__name__):
                self.assertIn("tqdm_class",
                              inspect.signature(func).parameters)

    def test_the_fetchers_accept_and_forward_it(self):
        for func in (model_fetch.ensure_hf_models, gguf_fetch.ensure_gguf):
            with self.subTest(func=func.__name__):
                self.assertIn("tqdm_class",
                              inspect.signature(func).parameters)


class BinaryProgressTests(unittest.TestCase):
    """ensure_llama_server reports per asset; the CUDA build has two."""

    def test_the_second_archive_continues_where_the_first_stopped(self):
        state = first_run_download.ProgressState(1000 * MB)
        state.begin(_component(first_run.KEY_LLAMA_SERVER, size_mb=641))
        report = first_run_download._binary_progress(state)

        report("llama.zip", 100 * MB, 250 * MB)
        report("llama.zip", 250 * MB, 250 * MB)
        self.assertEqual(state.snapshot().done_bytes, 250 * MB)

        # New name: its count restarts at zero but the component's does not.
        report("cudart.zip", 50 * MB, 391 * MB)
        self.assertEqual(state.snapshot().done_bytes, 300 * MB)
        report("cudart.zip", 391 * MB, 391 * MB)
        self.assertEqual(state.snapshot().done_bytes, 641 * MB)


class DispatchTests(unittest.TestCase):
    """Every component key reaches the downloader that owns it."""

    def setUp(self):
        self.state = first_run_download.ProgressState(0)

    def test_the_binary_goes_to_llama_server_fetch(self):
        with mock.patch.object(llama_server_fetch,
                               "ensure_llama_server") as ensure:
            first_run_download._fetch(
                _component(key=first_run.KEY_LLAMA_SERVER), self.state)
        self.assertIn("progress", ensure.call_args.kwargs)

    def test_the_binary_is_fetched_as_the_variant_the_plan_priced(self):
        # Passing it on is what stops ensure_llama_server from running
        # select_variant (and nvidia-smi) again, and what guarantees the build
        # downloaded is the one whose size the user agreed to.
        with mock.patch.object(llama_server_fetch,
                               "ensure_llama_server") as ensure:
            first_run_download._fetch(
                _component(key=first_run.KEY_LLAMA_SERVER,
                           variant="win-cpu-x64"), self.state)
        self.assertEqual(ensure.call_args.kwargs["variant"], "win-cpu-x64")

    def test_the_gguf_goes_to_the_configured_path(self):
        with mock.patch.object(gguf_fetch, "ensure_gguf") as ensure:
            first_run_download._fetch(_component(key=first_run.KEY_GGUF),
                                      self.state)
        self.assertEqual(ensure.call_args.args[0], config.EXTERNAL_MODEL_PATH)
        self.assertIn("tqdm_class", ensure.call_args.kwargs)

    def test_a_hub_repo_is_fetched_alone(self):
        # One repo per component, not the whole HF_MODEL_REPOS list: the plan
        # already decided which ones this run needs.
        with mock.patch.object(model_fetch, "ensure_hf_models") as ensure:
            first_run_download._fetch(
                _component(key=models_info.KOKORO.repo_id), self.state)
        self.assertEqual(list(ensure.call_args.args[0]), [models_info.KOKORO])

    def test_supertonic_goes_to_its_own_downloader(self):
        with mock.patch.object(model_fetch, "ensure_supertonic") as ensure:
            first_run_download._fetch(
                _component(key=models_info.SUPERTONIC.name), self.state)
        ensure.assert_called_once_with()

    def test_the_spacy_pipeline_goes_to_its_own_fetcher_and_is_activated(self):
        # activate() right after the download is what makes the model usable in
        # THIS process: config put the directory on sys.path at import, when it
        # did not exist yet, so without this the download would only take effect
        # at the next launch - and the models load moments later.
        with mock.patch.object(spacy_model_fetch, "ensure_spacy_model") as ensure, \
                mock.patch.object(spacy_model_fetch, "activate") as activate:
            first_run_download._fetch(
                _component(key=models_info.SPACY_EN.name), self.state)
        self.assertIn("progress", ensure.call_args.kwargs)
        activate.assert_called_once_with()

    def test_an_unknown_component_fails_loudly(self):
        # A component added to the plan without a branch here would otherwise
        # be skipped, and the app would start missing exactly what it asked
        # permission to download.
        with self.assertRaises(RuntimeError):
            first_run_download._fetch(_component(key="something-new"),
                                      self.state)


class DownloadLoopTests(unittest.TestCase):
    """The worker's own contract: it never raises, and it records why."""

    def test_components_are_banked_in_order(self):
        components = (_component("a", size_mb=10), _component("b", size_mb=20))
        state = first_run_download.ProgressState(30 * MB)
        with mock.patch.object(first_run_download, "_fetch"):
            first_run_download.download(components, state)
        snapshot = state.snapshot()
        self.assertEqual(snapshot.done_bytes, 30 * MB)
        self.assertTrue(snapshot.finished)
        self.assertIsNone(snapshot.error)

    def test_offline_mode_is_lifted_for_the_download_and_restored(self):
        """The bug that made the very first real download fail.

        config.py sets HF_HUB_OFFLINE=1 while it is imported, as soon as every
        hub repo the run needs is cached. The GGUF lives outside that cache, so
        a machine with the repos and without the GGUF reached the downloader
        with the network already off. Clearing the environment is not enough:
        huggingface_hub freezes the flag into constants at import time, and
        app.py imports it long before the window opens.
        """
        from huggingface_hub import constants as hub_constants

        seen = {}

        def record(_component, _state):
            seen["env"] = os.environ.get("HF_HUB_OFFLINE")
            seen["constant"] = hub_constants.HF_HUB_OFFLINE

        state = first_run_download.ProgressState(10 * MB)
        with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}), \
                mock.patch.object(hub_constants, "HF_HUB_OFFLINE", True), \
                mock.patch.object(first_run_download, "_fetch",
                                  side_effect=record):
            first_run_download.download((_component("a", size_mb=10),), state)

            self.assertIsNone(seen["env"])
            self.assertFalse(seen["constant"])
            # Restored, so the app's own model loading stays offline-fast.
            self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
            self.assertTrue(hub_constants.HF_HUB_OFFLINE)

    def test_the_library_is_read_before_the_environment_is_cleared(self):
        """The regression that broke offline mode for the whole process.

        huggingface_hub computes HF_HUB_OFFLINE from the environment when its
        constants module is imported, and at this point in a run it may not be
        imported yet - transformers and kokoro pull it in lazily, later. When
        this function cleared the environment first, its own import became the
        first one and froze the constant as False, so restoring it afterwards
        pinned the process online. Nothing raised; the only symptom was the app
        quietly making seventy network round-trips while loading models.
        """
        from huggingface_hub import constants as hub_constants

        order = []
        real_pop = os.environ.pop

        def read():
            order.append("read")
            return hub_constants

        def clear(*args):
            order.append("clear")
            return real_pop(*args)

        with mock.patch.object(first_run_download, "_hub_constants", read), \
                mock.patch.object(os.environ, "pop", clear):
            with first_run_download._hub_online():
                pass

        self.assertEqual(order[0], "read")

    def test_the_copy_fallback_is_armed_before_the_library_is_read(self):
        """The other half of the same ordering, and just as silent when broken.

        prepare_hf_env() sets HF_HUB_DISABLE_SYMLINKS and HF_HUB_DISABLE_XET,
        and huggingface_hub freezes those into constants at import time exactly
        as it freezes HF_HUB_OFFLINE. Every other caller gets the ordering for
        free because the ensure_* functions call prepare_hf_env themselves - but
        they run inside this context manager, by which point the library has
        already been imported, so _hub_online has to call it first and on its
        own. Reaching the library first left the Windows copy fallback off for
        the rest of the process, and a download then died with WinError 1314 on
        whichever small file lost the race inside are_symlinks_supported (see
        model_fetch._configure_symlink_fallback).
        """
        from huggingface_hub import constants as hub_constants

        order = []

        def prepare():
            order.append("prepare")

        def read():
            order.append("read")
            return hub_constants

        with mock.patch.object(model_fetch, "prepare_hf_env", prepare), \
                mock.patch.object(first_run_download, "_hub_constants", read):
            with first_run_download._hub_online():
                pass

        self.assertEqual(order, ["prepare", "read"])

    def test_offline_mode_is_restored_after_a_failure_too(self):
        from huggingface_hub import constants as hub_constants

        state = first_run_download.ProgressState(10 * MB)
        with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}), \
                mock.patch.object(hub_constants, "HF_HUB_OFFLINE", True), \
                mock.patch.object(first_run_download, "_fetch",
                                  side_effect=OSError("no route to host")):
            first_run_download.download((_component("a", size_mb=10),), state)
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
        self.assertTrue(hub_constants.HF_HUB_OFFLINE)

    def test_a_failure_stops_the_run_and_is_reported(self):
        components = (_component("a", size_mb=10), _component("b", size_mb=20))
        state = first_run_download.ProgressState(30 * MB)
        with mock.patch.object(first_run_download, "_fetch",
                               side_effect=OSError("disk full")) as fetch:
            first_run_download.download(components, state)
        snapshot = state.snapshot()
        self.assertEqual(fetch.call_count, 1)   # did not carry on
        self.assertTrue(snapshot.finished)
        self.assertIn("disk full", snapshot.error)

    def test_a_failure_still_reports_what_had_already_finished(self):
        """The half-success the dialog has to know about.

        Components run in order, required ones first, so a network drop on the
        optional level leaves the app perfectly startable. Without this the
        dialog would go on believing its opening plan, offer "Quit" instead of
        "Skip", and throw the finished download away.
        """
        components = (_component("a", size_mb=10), _component("b", size_mb=20))
        state = first_run_download.ProgressState(30 * MB)
        calls = []

        def fetch(component, _state):
            calls.append(component.key)
            if component.key == "b":
                raise OSError("no route to host")

        with mock.patch.object(first_run_download, "_fetch",
                               side_effect=fetch):
            first_run_download.download(components, state)

        snapshot = state.snapshot()
        self.assertEqual(calls, ["a", "b"])
        self.assertEqual(snapshot.completed_keys, ("a",))
        self.assertIn("no route to host", snapshot.error)


if __name__ == "__main__":
    unittest.main()
