# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Checks for the shared espeak-ng registration (pronunciation/common/espeak.py).

Everything here runs on stubs and needs no espeak-ng of any kind: what is being
asserted is the wiring, and the wiring is exactly what broke in the ways this
module's docstring records.

Four questions, one per class:
  * Does the registration set BOTH the library and the data path? Setting only
    the library is the bug that showed up as an access violation inside a C
    library rather than as an exception, so it deserves a test of its own.
  * Does a failure stay soft and stay visible - False, a warning, no raise?
  * Does the ACOUSTIC engine register before its first phonemize call? It had
    no registration at all and worked only by accident, on machines where a
    system espeak-ng existed or where importing Kokoro had registered one.
  * Does install.py step 5 ask the consumer's question (which library resolves)
    rather than `shutil.which("espeak-ng")` (does the executable exist)? The
    two disagree, and the old check was wrong in both directions.
"""

import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import install
from pronunciation.common import espeak

BUNDLED_LIBRARY = "/fake/site-packages/espeakng_loader/libespeak-ng.so"
BUNDLED_DATA = "/fake/site-packages/espeakng_loader/espeak-ng-data"


class FakeEspeakWrapper:
    """Records what was registered, the way phonemizer's wrapper stores it.

    The real class keeps these as class attributes with exactly this meaning,
    which is why a test can assert against them instead of against call order.
    """

    _ESPEAK_LIBRARY = None
    _ESPEAK_DATA_PATH = None

    @classmethod
    def set_library(cls, library):
        cls._ESPEAK_LIBRARY = library

    @classmethod
    def set_data_path(cls, data_path):
        cls._ESPEAK_DATA_PATH = data_path

    @classmethod
    def reset(cls):
        cls._ESPEAK_LIBRARY = None
        cls._ESPEAK_DATA_PATH = None


def _stub_modules(loader):
    """Patch the two modules ensure_espeak imports, for the duration of a test.

    Both are imported inside the function, so replacing them in sys.modules is
    enough - and necessary, because phonemizer and espeakng_loader are really
    installed here and would otherwise answer for this machine.
    """
    wrapper_module = SimpleNamespace(EspeakWrapper=FakeEspeakWrapper)
    return mock.patch.dict(sys.modules, {
        "espeakng_loader": loader,
        "phonemizer.backend.espeak.wrapper": wrapper_module,
    })


def _working_loader():
    return SimpleNamespace(
        get_library_path=lambda: BUNDLED_LIBRARY,
        get_data_path=lambda: BUNDLED_DATA,
    )


class RegistrationTests(unittest.TestCase):
    """What ensure_espeak() hands to phonemizer."""

    def setUp(self):
        # The result is cached for the life of the process, so every test has
        # to start from "not attempted yet".
        espeak._bundled_registered = None
        FakeEspeakWrapper.reset()
        self.addCleanup(setattr, espeak, "_bundled_registered", None)
        self.addCleanup(FakeEspeakWrapper.reset)

    def test_registers_both_the_library_and_the_data_path(self):
        # The point of the test: not one of the two, both. phonemizer copies
        # the library to a temporary directory before loading it and passes the
        # data path separately, so the data is never found beside the library
        # and a library-only registration dies inside espeak with an access
        # violation instead of raising something readable.
        with _stub_modules(_working_loader()):
            self.assertTrue(espeak.ensure_espeak())
        self.assertEqual(FakeEspeakWrapper._ESPEAK_LIBRARY, BUNDLED_LIBRARY)
        self.assertEqual(FakeEspeakWrapper._ESPEAK_DATA_PATH, BUNDLED_DATA)

    def test_second_call_does_not_register_again(self):
        with _stub_modules(_working_loader()):
            espeak.ensure_espeak()
            FakeEspeakWrapper.reset()
            self.assertTrue(espeak.ensure_espeak())
        # Nothing was written the second time: the analysis path calls this per
        # word, so it has to be free after the first call.
        self.assertIsNone(FakeEspeakWrapper._ESPEAK_LIBRARY)

    def test_missing_data_directory_registers_nothing_at_all(self):
        # espeakng_loader.get_data_path() raises when the directory is absent.
        # Asking for it before touching the wrapper is what keeps the process
        # out of the half-registered state described above - so assert that the
        # library was NOT set either.
        def explode():
            raise RuntimeError("data path not exists at /nowhere")

        loader = SimpleNamespace(get_library_path=lambda: BUNDLED_LIBRARY,
                                 get_data_path=explode)
        with _stub_modules(loader):
            with self.assertLogs(espeak.__name__, level="WARNING"):
                self.assertFalse(espeak.ensure_espeak())
        self.assertIsNone(FakeEspeakWrapper._ESPEAK_LIBRARY)
        self.assertIsNone(FakeEspeakWrapper._ESPEAK_DATA_PATH)

    def test_absent_loader_package_is_a_soft_failure(self):
        # A standalone install of one pronunciation subpackage may not have the
        # wheel; a system espeak-ng is the answer there, so this must not raise.
        with mock.patch.dict(sys.modules, {"espeakng_loader": None}):
            with self.assertLogs(espeak.__name__, level="WARNING") as logs:
                self.assertFalse(espeak.ensure_espeak())
        # Silence was the old behaviour (`except Exception: pass`) and made a
        # failure indistinguishable from success, so the message matters.
        self.assertIn("system espeak-ng", "\n".join(logs.output))

    def test_success_says_where_the_library_came_from(self):
        with _stub_modules(_working_loader()):
            with self.assertLogs(espeak.__name__, level="INFO") as logs:
                espeak.ensure_espeak()
        self.assertIn(BUNDLED_LIBRARY, "\n".join(logs.output))


class ResolvedLibraryTests(unittest.TestCase):
    """resolved_library() answers through phonemizer, not through PATH."""

    def test_returns_the_path_phonemizer_would_load(self):
        wrapper = SimpleNamespace(
            EspeakWrapper=SimpleNamespace(library=lambda: BUNDLED_LIBRARY))
        with mock.patch.dict(
                sys.modules, {"phonemizer.backend.espeak.wrapper": wrapper}):
            self.assertEqual(espeak.resolved_library(), BUNDLED_LIBRARY)

    def test_returns_none_when_phonemizer_finds_nothing(self):
        # phonemizer raises RuntimeError rather than returning None; the caller
        # only needs "there is one" or "there is not".
        def explode():
            raise RuntimeError("failed to find espeak library")

        wrapper = SimpleNamespace(EspeakWrapper=SimpleNamespace(library=explode))
        with mock.patch.dict(
                sys.modules, {"phonemizer.backend.espeak.wrapper": wrapper}):
            self.assertIsNone(espeak.resolved_library())


class AcousticEngineRegistrationTests(unittest.TestCase):
    """The acoustic engine registers espeak before it phonemizes anything.

    This is the engine that had no registration of its own. Importing it here
    is deliberate: the assertion is about the call in the real module, not
    about a copy of the intent.
    """

    def test_phonemize_word_registers_first(self):
        from pronunciation.acoustic import speech

        calls = []
        # The cache is keyed by word alone and survives between tests, so a
        # warm entry would skip the body being tested.
        speech._phonemize_word.cache_clear()
        self.addCleanup(speech._phonemize_word.cache_clear)

        with mock.patch.object(speech, "ensure_espeak",
                               side_effect=lambda: calls.append("register")), \
             mock.patch.object(speech, "phonemize",
                               side_effect=lambda *a, **k: (
                                   calls.append("phonemize") or "h ə l oʊ")):
            speech._phonemize_word("hello")

        self.assertEqual(calls, ["register", "phonemize"])

    def test_load_models_is_not_the_only_place_it_happens(self):
        # Stated as a test because it is the whole point of putting the call in
        # _phonemize_word: get_word_phonemes() reaches espeak on the reference
        # text without the recognizer ever being loaded.
        from pronunciation.acoustic import speech

        speech._phonemize_word.cache_clear()
        self.addCleanup(speech._phonemize_word.cache_clear)

        with mock.patch.object(speech, "ensure_espeak") as register, \
             mock.patch.object(speech, "phonemize", return_value="h aɪ"), \
             mock.patch.object(speech, "load_models",
                               side_effect=AssertionError(
                                   "must not load the recognizer")):
            speech.get_word_phonemes("hi")

        register.assert_called()


class _StubLogger:
    """Enough of install.Logger for a step to run: it only prints."""

    def __init__(self):
        self.lines = []

    def log(self, message=""):
        self.lines.append(message)

    def banner(self, title):
        self.lines.append(title)

    def text(self):
        return "\n".join(self.lines)


class InstallerStepTests(unittest.TestCase):
    """Step 5 asks which library resolves, not whether an executable exists."""

    def test_the_probe_is_the_engines_own_module(self):
        # Deliberately the module under pronunciation/, not one under mimora/:
        # the installer has to run the same registration the engines run.
        self.assertEqual(install.ESPEAK_PROBE_MODULE,
                         "pronunciation.common.espeak")

    def test_probe_runs_the_module_in_the_target_interpreter(self):
        # `-m` in a subprocess rather than an import here, for the same reason
        # step_detect_hardware does it: the answer belongs to the environment
        # that will run Mimora.
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=BUNDLED_LIBRARY + "\n", stderr="")
        with mock.patch.object(install.subprocess, "run",
                               return_value=completed) as run:
            library = install._resolve_espeak_library(_StubLogger())

        self.assertEqual(library, BUNDLED_LIBRARY)
        self.assertEqual(run.call_args.args[0],
                         [sys.executable, "-m", "pronunciation.common.espeak"])

    def test_probe_reports_nothing_when_the_module_exits_non_zero(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="no library")
        with mock.patch.object(install.subprocess, "run",
                               return_value=completed):
            self.assertIsNone(install._resolve_espeak_library(_StubLogger()))

    def test_step_reports_the_resolved_path_and_never_consults_which(self):
        report = install.StepReport()
        log = _StubLogger()
        with mock.patch.object(install, "_resolve_espeak_library",
                               return_value=BUNDLED_LIBRARY), \
             mock.patch.object(install.shutil, "which",
                               side_effect=AssertionError(
                                   "step 5 must not look for an executable")):
            install.step_espeak(log, confirmer=None, report=report)

        self.assertEqual(report.statuses(), [install.DONE])
        # The note is the path itself: the summary has to say WHICH espeak-ng
        # the app will use, not merely that one exists.
        self.assertIn(BUNDLED_LIBRARY, report.render())

    def test_step_records_manual_when_no_library_resolves(self):
        report = install.StepReport()
        log = _StubLogger()
        # Windows: the branch that only prints instructions, so the confirmer
        # is never reached and the step stays non-interactive.
        with mock.patch.object(install, "_resolve_espeak_library",
                               return_value=None), \
             mock.patch.object(install.platform, "system",
                               return_value="Windows"):
            install.step_espeak(log, confirmer=None, report=report)

        self.assertEqual(report.statuses(), [install.MANUAL])
        # Both variables, because one is not enough - the trap this whole
        # change was written around.
        self.assertIn("PHONEMIZER_ESPEAK_LIBRARY", log.text())
        self.assertIn("PHONEMIZER_ESPEAK_DATA_PATH", log.text())


if __name__ == "__main__":
    unittest.main()
