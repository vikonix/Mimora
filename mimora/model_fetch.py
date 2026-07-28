# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Download the models Mimora always needs into model_cache/.

Covers the four downloads that are required whatever the LLM backend is: the
two Wav2Vec2 recognizers (acoustic and phoneme engines), Kokoro TTS, the
NLLB-200 translator - all through the Hugging Face hub cache - plus the
Supertonic 3 TTS weights, which live in their own cache directory.

The LLM stack is deliberately NOT here: `mimora/llama_server_fetch.py` fetches
the llama-server binary and `mimora/gguf_fetch.py` the GGUF chat model. Both
are unnecessary when `llm_backend` is "lm-studio" or "off", while everything in
this module is chosen by the active engine and TTS backend rather than by a
setting, so the split follows what a run can actually skip.

Run it directly:

    python -m mimora.model_fetch              # everything that is missing
    python -m mimora.model_fetch --list       # what is present, what is not
    python -m mimora.model_fetch --hf --force # re-fetch the hub repos only

Design notes
------------
* No side effects at import time and no heavy imports at module level:
  huggingface_hub and supertonic are imported inside the functions that need
  them, so install.py can import this module before the requirements step has
  run and still get the "is it downloaded?" predicates. The two project modules
  imported at the top, models_info and loader, are pure and stdlib-only, so
  they cost nothing and break no rule.
* This module must NEVER import mimora.config. config sets HF_HUB_OFFLINE=1 as
  soon as the models are cached, which would switch the network off exactly
  when a download is wanted. The dependency runs the other way: config takes
  the cache paths and the Supertonic predicate from here.
* What each model IS (repo id, label, download size) comes from
  mimora/models_info.py, which config reads as well - that is how the same
  facts reach both sides without either importing the other. It is pure data
  with no imports of its own, so it costs this module nothing. Paths stay here,
  next to the code that writes into them.
* prepare_hf_env() must run before huggingface_hub is first imported anywhere
  in the process - HF_HOME and HF_HUB_DISABLE_XET are read at import time. Each
  ensure_* function calls it, so callers do not have to remember the ordering.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

if __package__ in (None, ""):
    # Executed as a plain script (python mimora/model_fetch.py) rather than
    # with -m: that form gives the module no package context and the relative
    # import below fails. Same shim, and the same reason, as in gguf_fetch.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "mimora"

from . import loader, models_info

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
# Mirrors config.MODEL_CACHE_DIR; config imports this constant instead of
# spelling the path a second time.
MODEL_CACHE_DIR = BASE_DIR / "model_cache"

# Supertonic keeps its weights OUTSIDE the HF hub cache: the package downloads
# them with snapshot_download(local_dir=...) into the directory named by the
# SUPERTONIC_CACHE_DIR env var (whose own default would be ~/.cache/supertonic3).
# Pinning it under model_cache/ keeps the weights next to the code.
DEFAULT_SUPERTONIC_CACHE_DIR = MODEL_CACHE_DIR / "supertonic3"

# Identity and size of every model live in mimora/models_info.py. The names
# below BIND to those records rather than restating them: a binding cannot
# drift from what it points at, which a second copy of the string could and did
# (the same repo ids used to be spelled out here, in config.py and in tts.py).
SUPERTONIC_MODEL_NAME = models_info.SUPERTONIC.name
SUPERTONIC_SIZE_MB = models_info.SUPERTONIC.size_mb

# Repos Mimora pulls on first run; pre-fetching them makes the first launch
# offline-ready. Repo ids match what the app requests.
# Supertonic is NOT in this list on purpose: it does not use the hub cache
# (see above), so caching its repo under HF_HOME/hub would be dead weight the
# app never reads. It has its own ensure_supertonic() instead.
HF_MODEL_REPOS: tuple[models_info.HfRepo, ...] = models_info.HF_REPOS


class ModelFetchError(RuntimeError):
    """A download did not finish, so the model is not usable."""


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def hf_home() -> Path:
    """Effective HF cache root: the env var when set, model_cache/ otherwise.

    Read through the environment rather than from MODEL_CACHE_DIR directly so
    that an externally set HF_HOME is honored by the predicates too.
    """
    return Path(os.environ.get("HF_HOME") or MODEL_CACHE_DIR)


def supertonic_cache_dir() -> Path:
    """Effective Supertonic cache directory (env var wins, as for HF_HOME)."""
    return Path(os.environ.get("SUPERTONIC_CACHE_DIR")
                or DEFAULT_SUPERTONIC_CACHE_DIR)


def prepare_hf_env() -> None:
    """Point the HF caches at model_cache/ and arm the Windows fallbacks.

    Shared by every download here and by gguf_fetch, because each of them can
    also run standalone from its own CLI and so cannot rely on another step
    having done this first. Must run before huggingface_hub is imported:
    HF_HOME and HF_HUB_DISABLE_XET are read at import time. Idempotent, and
    setdefault throughout, so an externally configured cache stays untouched.
    """
    MODEL_CACHE_DIR.mkdir(exist_ok=True)
    os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
    os.environ.setdefault("SUPERTONIC_CACHE_DIR",
                          str(DEFAULT_SUPERTONIC_CACHE_DIR))
    _configure_symlink_fallback()


# Result of the one-off symlink probe below; None until it has run. The probe
# creates a temporary directory and its outcome cannot change while the process
# lives, so repeating it on every prepare_hf_env() call would be pure cost -
# and every predicate in this module calls prepare_hf_env().
_symlink_supported: Optional[bool] = None


def _probe_symlink_support() -> bool:
    """Can this process create a symlink inside the model cache?"""
    try:
        with tempfile.TemporaryDirectory(dir=MODEL_CACHE_DIR) as tmp:
            src = Path(tmp) / "probe_src"
            src.touch()
            try:
                os.symlink(src, Path(tmp) / "probe_dst")
            except OSError:
                return False
    except OSError:
        return False
    return True


def _configure_symlink_fallback() -> None:
    """On Windows, keep HF off the hf-xet path that crashes without symlinks.

    huggingface_hub's cache points snapshots/ at blobs/ via symlinks. Creating a
    symlink on Windows needs Developer Mode or admin rights.

    The native hf-xet downloader links files into the cache itself and fails hard
    with WinError 1314 when that privilege is missing - and, unlike the pure-
    Python HTTP path, it does NOT fall back to copying. Crucially it can hit this
    even when a plain symlink probe passes (the privilege can be present at probe
    time yet unavailable to xet's linker), so a probe is not a reliable gate.

    We therefore disable hf-xet on every Windows run. Downloads then take the
    HTTP path, which checks symlink support itself and copies into the cache when
    symlinks are unavailable (uses more disk, but always works).

    The env vars are re-applied on every call (they are cheap, and a caller that
    restored os.environ must not silently lose them), while the probe and its
    log line happen exactly once per process.
    """
    global _symlink_supported
    if sys.platform != "win32":
        return

    # Unconditional: xet is the only path that raises 1314 without a copy
    # fallback, and it can do so regardless of the symlink probe below.
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    first_call = _symlink_supported is None
    if first_call:
        _symlink_supported = _probe_symlink_support()

    if _symlink_supported:
        if first_call:
            log.info("Symlink support: OK (hf-xet disabled on Windows for "
                     "safety).")
        return

    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    if first_call:
        log.info("Symlinks unavailable (no Developer Mode / admin): HF "
                 "downloads will COPY into the cache instead of symlinking "
                 "(more disk use). Tip: enabling Windows Developer Mode lets "
                 "HF use symlinks.")


# ---------------------------------------------------------------------------
# "Already downloaded?" predicates
# ---------------------------------------------------------------------------

def hf_repo_cached(repo_id: str) -> bool:
    """Is the repo present in the local HF cache and free of partial files?

    Delegates to loader.models_cached so that the installer, config's
    offline gate and the app's startup check all answer this question the same
    way. A snapshot directory has to exist and no *.incomplete blob may be left
    behind by an interrupted download - the latter matters because
    ensure_hf_models() SKIPS a repo this returns True for, so an over-generous
    answer here means a half-fetched repo that no later run ever completes.

    This used to ask snapshot_download(local_files_only=True) instead, on the
    understanding that it verifies every file of the recorded revision. It only
    does so when trees/<commit>.json is cached, and that listing is written as a
    side effect of snapshot_download itself (huggingface_hub 1.24.0,
    _snapshot_download._raise_if_incomplete_snapshot returns early without it) -
    so for a cache filled file by file by transformers, which is how a first run
    fills it, the check silently did nothing. The filesystem check is both
    stricter and free of the huggingface_hub import.

    Neither check can see a download interrupted exactly BETWEEN two files: no
    *.incomplete is left then. With files of this size an interrupt lands
    mid-file in nearly every case, and --force remains the way out.
    """
    prepare_hf_env()
    return loader.models_cached(hf_home() / "hub", (repo_id,))


def supertonic_cached() -> bool:
    """True when the Supertonic 3 model is fully present in its cache dir.

    Unlike the hub cache above, no manifest check is needed: the supertonic
    package downloads atomically (into a temp directory that is renamed onto
    the cache dir only on success), so a present, non-empty directory is a
    complete download. Pure filesystem work, so config.py can call it during
    its own import without pulling huggingface_hub in.
    """
    cache_dir = supertonic_cache_dir()
    try:
        return cache_dir.is_dir() and any(cache_dir.iterdir())
    except OSError:
        return False


def missing_models() -> list[str]:
    """Human-readable names of everything this module would still download.

    Empty means a run needs no network for these models. Intended for the app's
    first-run check as much as for the installer.
    """
    missing = [repo.repo_id for repo in HF_MODEL_REPOS
               if not hf_repo_cached(repo.repo_id)]
    if not supertonic_cached():
        missing.append(SUPERTONIC_MODEL_NAME)
    return missing


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

def ensure_hf_models(repos: Optional[Sequence[models_info.HfRepo]] = None, *,
                     force: bool = False,
                     tqdm_class: Optional[type] = None) -> None:
    """Download every Hugging Face repo Mimora needs into the hub cache.

    Already-cached repos are skipped unless *force* is set; already-downloaded
    files inside a partially fetched repo are reused either way (that is
    snapshot_download's own behaviour). Every repo is attempted even when one
    fails, so a single flaky download does not hide the state of the rest;
    the failures are collected and reported together.

    *tqdm_class* is huggingface_hub's own hook for replacing the progress bar,
    forwarded untouched. The app's first-run window passes a stand-in that
    records bytes instead of drawing (mimora/first_run_download.py); the CLI
    and install.py pass nothing and keep the normal bars. It is omitted from
    the call rather than passed as None so that a huggingface_hub without the
    argument still works - it is not pinned anywhere and arrives transitively.
    """
    prepare_hf_env()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ModelFetchError(
            "huggingface_hub is not installed - install the project "
            "requirements first (python install.py).") from exc

    progress_arg = {} if tqdm_class is None else {"tqdm_class": tqdm_class}

    failures: list[str] = []
    for repo in repos if repos is not None else HF_MODEL_REPOS:
        repo_id = repo.repo_id
        if not force and hf_repo_cached(repo_id):
            log.info("Already cached: %s", repo_id)
            continue
        log.info("Fetching %s, %d MB [%s] ...", repo.label, repo.size_mb, repo_id)
        try:
            snapshot_download(repo_id=repo_id, **progress_arg)
            log.info("-> done: %s", repo_id)
        except Exception as exc:  # noqa: BLE001 - record which repo failed
            log.error("-> FAILED: %s: %s", repo_id, exc)
            failures.append(repo_id)

    if failures:
        raise ModelFetchError(
            f"Could not download: {', '.join(failures)}. Check the network / "
            f"proxy and re-run; finished repos are not fetched again.")


def ensure_supertonic(*, force: bool = False) -> None:
    """Download the Supertonic 3 TTS model into its own cache directory.

    The Spanish TTS backend (mimora/tts.py SupertonicBackend). Separate from
    the hub repos because the supertonic package does not read the HF hub
    cache. Pre-fetching matters for offline mode: the app flips
    HF_HUB_OFFLINE=1 once its models are cached, and this download goes through
    huggingface_hub, so it must happen while the Hub is still online. The
    weights are OpenRAIL-M licensed (the code is MIT), which is why they are
    downloaded rather than shipped with Mimora.
    """
    prepare_hf_env()
    if not force and supertonic_cached():
        log.info("Already downloaded: %s (%s)",
                 SUPERTONIC_MODEL_NAME, supertonic_cache_dir())
        return

    try:
        # The loader-level functions download without loading the ONNX sessions
        # (no synthesis warm-up is wanted at install time). get_cache_dir honors
        # SUPERTONIC_CACHE_DIR, so the download lands where the app looks.
        from supertonic.loader import download_model, get_cache_dir
    except ImportError as exc:
        raise ModelFetchError(
            "the supertonic package is not installed - install the project "
            "requirements first (python install.py).") from exc

    try:
        target = get_cache_dir(SUPERTONIC_MODEL_NAME)
        log.info("Fetching Supertonic 3 [%s] into %s ...",
                 SUPERTONIC_MODEL_NAME, target)
        download_model(target, SUPERTONIC_MODEL_NAME)
        log.info("-> done: Supertonic 3")
    except Exception as exc:  # noqa: BLE001 - network, disk, licence prompts
        raise ModelFetchError(
            f"Could not download the Supertonic 3 model: {exc}") from exc


def ensure_all(*, force: bool = False) -> None:
    """Download everything this module owns, hub repos first."""
    ensure_hf_models(force=force)
    ensure_supertonic(force=force)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _print_status() -> None:
    prepare_hf_env()
    print(f"HF cache : {hf_home()}")
    print(f"Supertonic cache: {supertonic_cache_dir()}")
    print("Models   :")
    for repo in HF_MODEL_REPOS:
        mark = "present" if hf_repo_cached(repo.repo_id) else "MISSING"
        print(f"    [{mark:>7}] {repo.repo_id}  - {repo.label}, "
              f"{repo.size_mb} MB")
    supertonic = models_info.SUPERTONIC
    mark = "present" if supertonic_cached() else "MISSING"
    print(f"    [{mark:>7}] {supertonic.name}  - {supertonic.label}, "
          f"{supertonic.size_mb} MB")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Hugging Face and Supertonic models Mimora "
                    "needs into model_cache/.")
    parser.add_argument("--hf", action="store_true",
                        help="only the Hugging Face hub repos")
    parser.add_argument("--supertonic", action="store_true",
                        help="only the Supertonic 3 TTS model")
    parser.add_argument("--force", action="store_true",
                        help="download even when the model is already present")
    parser.add_argument("--list", action="store_true",
                        help="show what is present and what is missing, then exit")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    # stream=stdout, not the default stderr: _print_status() print()s to stdout
    # and two streams with different buffering interleave in whatever order the
    # OS feels like - the symlink notice ended up in the middle of the model
    # list.
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stdout)

    if args.list:
        _print_status()
        return 0

    # Neither flag given means "everything"; both given means the same.
    want_hf = args.hf or not args.supertonic
    want_supertonic = args.supertonic or not args.hf
    try:
        if want_hf:
            ensure_hf_models(force=args.force)
        if want_supertonic:
            ensure_supertonic(force=args.force)
    except ModelFetchError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
