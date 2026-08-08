# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Checks for how KokoroBackend builds its model and pipeline (mimora/tts.py).

Everything here runs on stubs: no kokoro, no torch, no weights, no network.
What is asserted is the wiring of three keyword arguments and one method call,
and that wiring is exactly what was wrong until 2026-08-08.

The defect: `KPipeline(lang_code=...)` alone left both `repo_id` and `model` at
their library defaults, and nothing called `.eval()` on the model. Three
separate consequences, hence the tests below:

  * without `.eval()` the synthesis model stays in TRAINING mode. KModel is
    constructed here rather than loaded through `from_pretrained`, and
    torch.nn.Module starts with `training=True`; kokoro then applies
    `nn.Dropout` unconditionally in `forward()`, both in TextEncoder and in the
    F0/N blocks of ProsodyPredictor. So a fifth of the linguistic
    representation and of the prosody prediction was being zeroed at random -
    in the reference audio the user imitates and the acoustic engine scores
    against;
  * without `model=`, the parameter defaults to True and KPipeline builds a
    SECOND KModel, onto a device it picks itself (`cuda` when available), past
    config.DEVICE. Nothing ever read that copy: every call site passes
    `model=self.model`. It was 82M parameters held for the session on the
    machine that also has to fit Wav2Vec2 and llama.cpp;
  * without `repo_id=`, kokoro prints `WARNING: Defaulting repo_id ...` on
    stdout at every start, and decides for itself which repo `load_voice()`
    fetches voices from.

Importing mimora.tts pulls in sounddevice (through mimora.audio_io), so this
module needs PortAudio present the way the app itself does. It needs nothing
else: kokoro is replaced in sys.modules before load_model() imports it.

Run from the project root with:

    python -m unittest tests.test_tts
"""

import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from mimora import config, models_info, tts

# Deliberately not "cuda" or "cpu": the assertion is that whatever config.DEVICE
# says is what the model is moved to, not that the value happens to be right.
DEVICE = "test-device"


class FakeKModel:
    """kokoro.KModel reduced to the traits these tests are about.

    ``training`` starts True because torch.nn.Module does - that default is the
    whole reason the fix exists. ``to()`` and ``eval()`` return ``self`` because
    the real ones do, so the chained call in tts.py reads here the way it reads
    there. Every instance is recorded, so a test can ask how many were built.
    """

    instances: list = []

    # The signature mirrors the real one (kokoro/model.py) so that a call with
    # an unexpected keyword fails here rather than passing silently.
    def __init__(self, repo_id=None, config=None, model=None,
                 disable_complex=False):
        self.repo_id = repo_id
        self.training = True
        self.device = None
        FakeKModel.instances.append(self)

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.training = False
        return self


class FakeKPipeline:
    """kokoro.KPipeline reduced to what it was handed.

    It deliberately does NOT reproduce the library's
    ``isinstance(model, KModel) ... elif model:`` branch (kokoro/pipeline.py):
    a stub that re-implements the library only proves the copy agrees with
    itself. What is recorded is the argument - and the argument is what decides
    which branch the real class takes.
    """

    def __init__(self, lang_code, repo_id=None, model=True, trf=False,
                 en_callable=None, device=None):
        self.lang_code = lang_code
        self.repo_id = repo_id
        self.model = model
        self.device = device
        self.loaded_voices = []

    def load_voice(self, voice):
        # _prefetch_voices walks config.TTS_VOICES through this. Recorded rather
        # than omitted so a failure in that loop reads as a failure, not as a
        # stub that forgot a method.
        self.loaded_voices.append(voice)
        return voice


class KokoroBackendSetupTests(unittest.TestCase):
    """What load_model() hands to KModel and KPipeline."""

    def setUp(self):
        FakeKModel.instances = []
        self.addCleanup(setattr, FakeKModel, "instances", [])

    def _loaded_backend(self):
        """Runs KokoroBackend.load_model() against the stubs.

        kokoro is imported inside load_model(), so replacing it in sys.modules
        is enough - and necessary, because the real package is installed here
        and would otherwise download config.json and 82M weights.
        """
        backend = tts.KokoroBackend()
        stub = SimpleNamespace(KModel=FakeKModel, KPipeline=FakeKPipeline)
        with mock.patch.dict(sys.modules, {"kokoro": stub}), \
                mock.patch.object(tts.config, "DEVICE", DEVICE):
            backend.load_model()
        return backend

    def test_the_model_is_switched_out_of_training_mode(self):
        # The defect itself. Measured before the fix: two syntheses of one
        # phrase in one voice differed by 0.67 peak on a waveform in [-1, 1].
        self.assertFalse(self._loaded_backend().model.training)

    def test_the_model_lands_on_the_configured_device(self):
        # config.DEVICE is the project's single answer to "which device", and
        # it already accounts for a torch build that cannot use CUDA. The
        # library's own device choice (`cuda if available`) does not.
        self.assertEqual(self._loaded_backend().model.device, DEVICE)

    def test_the_pipeline_is_handed_the_model_rather_than_building_one(self):
        backend = self._loaded_backend()
        # Passing an instance is what makes the library take its
        # `isinstance(model, KModel)` branch instead of the one that builds a
        # second network.
        self.assertIs(backend.pipeline.model, backend.model)
        # Our side of the same statement: exactly one model is constructed
        # here. The stub pipeline builds none, so this pins tts.py alone.
        self.assertEqual(len(FakeKModel.instances), 1)

    def test_both_constructors_get_the_repo_id_from_the_catalogue(self):
        # One identifier, from the record the first-run check also reads. A
        # second copy could send the download after one repo while synthesis
        # loaded another - and an absent one makes kokoro pick its own and say
        # so on stdout at every start.
        backend = self._loaded_backend()
        self.assertEqual(backend.model.repo_id, models_info.KOKORO.repo_id)
        self.assertEqual(backend.pipeline.repo_id, models_info.KOKORO.repo_id)

    def test_the_pipeline_gets_its_language_code_from_the_profile(self):
        # The language variant decides, never an `if language` branch in code.
        self.assertEqual(self._loaded_backend().pipeline.lang_code,
                         config.TTS_LANG_CODE)


if __name__ == "__main__":
    unittest.main()
