# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for the model catalogue (mimora/models_info.py).

The sizes themselves cannot be checked here - that needs the network, and
tools/measure_model_sizes.py is where it happens. What CAN be checked without a
network is that nobody added a model or a release asset and left its size out,
which is the failure this module exists to prevent: a missing number does not
crash anything, it just makes the first-run dialog and the progress bar quietly
wrong.

The same applies to the other fields of llama_server_fetch.VARIANTS, whose
rows are checked here for internal consistency (backend against device check,
fallback against the table) for the same reason: an inconsistent row does not
fail loudly, it makes the module describe a build as something it is not.

Run from the project root with:

    python -m unittest tests.test_models_info
"""

import ast
import re
import unittest
from pathlib import Path

from mimora import gguf_fetch, llama_server_fetch, model_fetch, models_info


def _all_models():
    """Every record in the catalogue, whatever its type."""
    return (*models_info.HF_REPOS, models_info.SUPERTONIC,
            models_info.GGUF_CHAT, models_info.SPACY_EN)


class SizeCompletenessTests(unittest.TestCase):
    """Every downloadable thing carries a usable size."""

    def test_every_model_has_a_positive_size(self):
        for model in _all_models():
            with self.subTest(model=model.label):
                self.assertIsInstance(model.size_mb, int)
                self.assertGreater(model.size_mb, 0)

    def test_every_release_asset_has_a_positive_size(self):
        # Bumping RELEASE_TAG means re-snapping these next to the checksums;
        # this is what catches a new variant added without its numbers.
        for name, variant in llama_server_fetch.VARIANTS.items():
            for asset in variant.assets:
                with self.subTest(variant=name, asset=asset.name):
                    self.assertIsInstance(asset.size_mb, int)
                    self.assertGreater(asset.size_mb, 0)

    def test_variant_size_is_the_sum_of_its_assets(self):
        for name, variant in llama_server_fetch.VARIANTS.items():
            with self.subTest(variant=name):
                self.assertEqual(
                    llama_server_fetch.variant_size_mb(name),
                    sum(asset.size_mb for asset in variant.assets))


class VariantTableShapeTests(unittest.TestCase):
    """Internal consistency of llama_server_fetch.VARIANTS.

    Same reason as the sizes above: none of these mistakes crashes anything,
    they just make the module describe a build as something it is not.
    """

    def test_cpu_backend_matches_the_absence_of_a_device_check(self):
        # The two fields answer the same question from different sides, and
        # several messages read one to speak about the other: --list prints
        # the backend, verify_devices branches on it while the device check
        # itself is driven by the pattern. A row where they disagree would
        # promise a GPU build and then verify nothing, or the reverse.
        for name, variant in llama_server_fetch.VARIANTS.items():
            with self.subTest(variant=name):
                self.assertEqual(variant.backend == "CPU",
                                 variant.device_pattern is None)

    def test_every_variant_names_a_backend(self):
        for name, variant in llama_server_fetch.VARIANTS.items():
            with self.subTest(variant=name):
                self.assertIsInstance(variant.backend, str)
                self.assertTrue(variant.backend.strip())

    def test_fallbacks_point_at_a_known_variant_and_terminate(self):
        # ensure_llama_server walks this chain after a failed device check.
        # A typo would surface there as "Unknown variant", mid-install and
        # after a download; a cycle would only be caught by its own guard.
        for name, variant in llama_server_fetch.VARIANTS.items():
            with self.subTest(variant=name):
                seen = [name]
                current = variant.fallback
                while current is not None:
                    self.assertIn(current, llama_server_fetch.VARIANTS)
                    self.assertNotIn(current, seen)
                    seen.append(current)
                    current = llama_server_fetch.VARIANTS[current].fallback


class CatalogueShapeTests(unittest.TestCase):
    """The catalogue is the single source of these facts, so it has to be
    internally consistent before anyone reads it."""

    def test_repo_ids_are_unique(self):
        repo_ids = [repo.repo_id for repo in models_info.HF_REPOS]
        self.assertCountEqual(repo_ids, set(repo_ids))

    def test_labels_carry_no_size(self):
        # Sizes used to live inside the display labels, which is how they came
        # to disagree with reality. The number belongs to size_mb and is
        # formatted where it is shown. Matching a digit before the unit keeps
        # this from tripping over a model whose name merely contains "GB".
        for model in _all_models():
            with self.subTest(model=model.label):
                self.assertIsNone(re.search(r"\d\s*[MG]B", model.label))

    def test_supertonic_is_not_a_hub_repo(self):
        # It has its own cache directory and its own ensure_*; listing it among
        # the hub repos would cache a snapshot the app never reads.
        self.assertNotIn(models_info.SUPERTONIC.name,
                         [repo.repo_id for repo in models_info.HF_REPOS])


class BindingTests(unittest.TestCase):
    """The consumers bind to the catalogue rather than restating it. These
    assertions are what turns a re-typed literal back into a red test."""

    def test_model_fetch_exposes_the_catalogue_itself(self):
        self.assertIs(model_fetch.HF_MODEL_REPOS, models_info.HF_REPOS)
        self.assertEqual(model_fetch.SUPERTONIC_MODEL_NAME,
                         models_info.SUPERTONIC.name)
        self.assertEqual(model_fetch.SUPERTONIC_SIZE_MB,
                         models_info.SUPERTONIC.size_mb)

    def test_gguf_fetch_binds_to_the_catalogue(self):
        self.assertEqual(gguf_fetch.GGUF_REPO_ID, models_info.GGUF_CHAT.repo_id)
        self.assertEqual(gguf_fetch.GGUF_FILENAME,
                         models_info.GGUF_CHAT.filename)
        self.assertEqual(gguf_fetch.GGUF_SIZE_MB, models_info.GGUF_CHAT.size_mb)

    def test_gguf_default_path_uses_the_catalogue_filename(self):
        # The default target has to match config.EXTERNAL_MODEL_PATH's default,
        # or the app looks for a file the installer put somewhere else.
        self.assertEqual(gguf_fetch.DEFAULT_GGUF_PATH.name,
                         models_info.GGUF_CHAT.filename)


class ImportDisciplineTests(unittest.TestCase):
    """models_info is readable by config AND by the fetchers only because it
    depends on neither. A stray import here would break that quietly."""

    def test_module_imports_nothing_of_the_project(self):
        # Parsed rather than grepped: prose in the docstring must not be able to
        # pass for an import, in either direction.
        tree = ast.parse(Path(models_info.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        self.assertEqual(imported, {"__future__", "typing"})


if __name__ == "__main__":
    unittest.main()
