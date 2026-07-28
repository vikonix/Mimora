# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Identity and download size of every model Mimora fetches.

Pure data: this module imports nothing but the standard library and has no side
effects, which is exactly what lets every layer read it. The fetchers must never
import mimora/config.py (config flips HF_HUB_OFFLINE=1 once the models are
cached, switching the network off precisely when a download is wanted), so facts
they share with config have to live in a module that depends on neither side.
install.py reads it too, before the requirements step has run, hence stdlib only.

Two reasons it exists:

* a model's identity used to be written down two to four times each -
  mimora/config.py, mimora/model_fetch.py and mimora/tts.py all spelled the same
  repo ids out, three of them as bare literals - so the copies could drift;
* the first-run download dialog has to name volumes as numbers before the user
  agrees to them, and the sizes were prose inside a display label (or, for
  Kokoro, absent). See tasks/first-run-fetch.md, work 5.

Deliberately NOT here
---------------------
* **The llama-server binary.** The size of a release asset belongs next to its
  name and sha256 in mimora/llama_server_fetch.py: bumping the pinned release
  rewrites all three together, and a size left behind in another file would not
  fail loudly, it would quietly mis-scale the progress bar. The binary is also
  not a model.
* **Cache paths.** They stay in mimora/model_fetch.py, next to the code that
  writes into them.
* **Which level a model belongs to** (required / optional / lazy). That depends
  on the active engine, the active TTS backend and llm_backend, so it is the
  result of a computation at startup, not a static property of a model.
* **The defaults in pronunciation/acoustic/config.py and
  pronunciation/phoneme/config.py.** Those subpackages are reusable,
  GUI-agnostic libraries that must work without mimora.config, so their model
  names are standalone fallbacks rather than a second copy of the app's choice.
"""

from __future__ import annotations

from typing import NamedTuple

# ---------------------------------------------------------------------------
# What size_mb means
# ---------------------------------------------------------------------------
#
# BYTES OVER THE NETWORK, in decimal MB (bytes / 1_000_000).
#
# Three different numbers could be meant here and they diverge by up to a
# factor of two, so the choice is load-bearing:
#
#   1. what the download costs  <- this one
#   2. what it occupies on disk after unpacking (the llama.cpp archives) or
#      after conversion
#   3. what the HF cache occupies on Windows without symlink privileges, where
#      model_fetch._configure_symlink_fallback puts downloads on the HTTP path,
#      which COPIES files into snapshots/ instead of linking them
#
# The first-run dialog promises traffic ("you are about to download X GB"), and
# somebody on a metered connection has to be able to trust it, so traffic is
# what is stored. A dialog that also wants to warn about free disk space needs
# (2) or (3) and must not reuse these numbers.
#
# Decimal MB, not MiB, because that is how download sizes are advertised and what
# the pre-existing GGUF_SIZE_MB already meant. llama_server_fetch._human(), which
# renders live progress under the same "MB" label, divides by 1_000_000 to match;
# it used to divide by 1024**2, which made a finished download report a number
# visibly smaller than the size the user had just agreed to.

# ---------------------------------------------------------------------------
# Record types
# ---------------------------------------------------------------------------


class HfRepo(NamedTuple):
    """A Hugging Face repository fetched whole into the hub cache.

    label is display text (logs, the installer's prompts, the first-run dialog)
    and carries no size of its own: the number lives in size_mb and is
    formatted where it is shown, so the two can never contradict each other.
    """

    repo_id: str
    label: str
    size_mb: int


class HfFile(NamedTuple):
    """A single file pulled out of a Hugging Face repository.

    Separate from HfRepo because the size of one file is not the size of its
    repo, and because the download goes through hf_hub_download into a plain
    directory rather than through snapshot_download into the cache layout.
    """

    repo_id: str
    filename: str
    label: str
    size_mb: int


class PackagedModel(NamedTuple):
    """A model some other package downloads into its own cache directory.

    name is what that package calls the model, not a Hugging Face repo id: it
    is passed to the package's own loader, which resolves it however it likes.
    """

    name: str
    label: str
    size_mb: int


# ---------------------------------------------------------------------------
# Hugging Face hub repositories
# ---------------------------------------------------------------------------
#
# Re-snap the sizes with `python tools/measure_model_sizes.py` and commit the
# result whenever a repo id changes; they are a one-off per pin rather than a
# runtime lookup. Measuring at startup was rejected on purpose: it would put a
# network round-trip in front of the very dialog that exists for people with no
# network or an expensive one.
#
# Every number below is measured. For the record, the prose they replaced was
# optimistic across the board (~1.2 GB for repos that are 1.26 GB, ~2.4 GB for
# one that is 2.48), which is the concrete argument against eyeballing them.
#
# Measure the WHOLE snapshot, not the weights the app loads: snapshot_download
# without allow_patterns fetches every file in the repo, and these repos ship
# several weight formats side by side (pytorch_model.bin, model.safetensors,
# flax_model.msgpack).

WAV2VEC2_ACOUSTIC = HfRepo(
    "facebook/wav2vec2-large-960h",
    "Wav2Vec2 (acoustic pronunciation engine)",
    size_mb=1262,  # measured 2026-07-28
)

WAV2VEC2_PHONEME = HfRepo(
    "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
    "Wav2Vec2 phoneme engine (espeak IPA ASR)",
    size_mb=1264,  # measured 2026-07-28
)

KOKORO = HfRepo(
    "hexgrad/Kokoro-82M",
    "Kokoro-82M (text-to-speech, English)",
    size_mb=363,  # measured 2026-07-28
)

NLLB = HfRepo(
    "facebook/nllb-200-distilled-600M",
    "NLLB-200 distilled 600M (offline translator)",
    size_mb=2483,  # measured 2026-07-28
)

# Every hub repo, in the order model_fetch downloads them. Supertonic is NOT in
# this tuple: it does not use the hub cache (see below), so caching its repo
# under HF_HOME/hub would be dead weight the app never reads.
HF_REPOS: tuple[HfRepo, ...] = (
    WAV2VEC2_ACOUSTIC,
    WAV2VEC2_PHONEME,
    KOKORO,
    NLLB,
)

# ---------------------------------------------------------------------------
# Models that live outside the hub cache
# ---------------------------------------------------------------------------

# Supertonic keeps its weights in its own directory: the package downloads them
# with snapshot_download(local_dir=...) into the directory named by the
# SUPERTONIC_CACHE_DIR env var. The weights are OpenRAIL-M licensed (the code is
# MIT), which is why they are downloaded rather than shipped with Mimora.
SUPERTONIC = PackagedModel(
    "supertonic-3",
    "Supertonic 3 TTS (Spanish; weights OpenRAIL-M)",
    # Measured on disk rather than over the wire: which files the supertonic
    # package pulls out of Supertone/supertonic-3 is its own business, so the
    # repo total would be an upper bound. The cache directory IS the download -
    # the package writes with local_dir=, skipping the hub's blob/symlink
    # duplication.
    size_mb=404,  # measured 2026-07-28
)

# The GGUF chat model the llama-server backend loads. The filename matches the
# default of config.EXTERNAL_MODEL_PATH, so the app finds it without a settings
# change.
GGUF_CHAT = HfFile(
    "hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF",
    "llama-3.2-3b-instruct-q4_k_m.gguf",
    "Llama 3.2 3B Instruct Q4_K_M (chat model for llama-server)",
    size_mb=2019,  # measured 2026-07-28
)
