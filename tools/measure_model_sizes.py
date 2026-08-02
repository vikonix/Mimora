# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Measure the download size of everything Mimora fetches.

The sizes in mimora/models_info.py and mimora/llama_server_fetch.py are
hardcoded on purpose: the first-run dialog has to name volumes BEFORE the user
agrees to download them, so it cannot afford a network round-trip first - the
dialog exists for people with no network or an expensive one. This script is how
those hardcoded numbers get their values: run it once per pin change, paste the
block it prints, commit.

    python tools/measure_model_sizes.py

Nothing is downloaded. Hub repositories are measured through the Hugging Face
metadata API and llama.cpp release assets through HTTP HEAD requests, so the
whole run costs a few kilobytes.

Two properties of the measurement matter, both of them reasons not to eyeball
these numbers:

* a repo is measured WHOLE. snapshot_download without allow_patterns fetches
  every file, and the Wav2Vec2 repos ship several weight formats side by side
  (pytorch_model.bin, model.safetensors, flax_model.msgpack), so the download is
  considerably larger than the weights the app ends up loading;
* sizes are decimal MB (bytes / 1_000_000), matching how downloads are
  advertised and what models_info.size_mb is documented to mean.

Supertonic is the one entry measured on disk rather than over the wire. Its
upstream repo is Supertone/supertonic-3 (named in the README), but what the
supertonic package actually pulls out of it is the package's business, not ours,
so the repo total would be an upper bound rather than the download. The cache
directory is the download: the package writes with local_dir=, which puts files
down directly instead of through the hub's blob/symlink layout. The cost is that
the model has to be present already.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The tool runs as a script, so sys.path starts at tools/ and the package next
# door is not importable without this (same as tools/preview_first_run.py).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mimora import llama_server_fetch, model_fetch, models_info  # noqa: E402

_HTTP_TIMEOUT_SEC = 30
# Mirrors llama_server_fetch._download: some corporate proxies reject clients
# they cannot name, and an explicit agent is kinder to whoever reads their log.
_USER_AGENT = "mimora-measure-model-sizes/1.0"


def _mb(size_bytes: int) -> int:
    """Bytes to decimal MB, which is what size_mb means everywhere."""
    return round(size_bytes / 1_000_000)


# ---------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------

def _sibling_size(sibling) -> Optional[int]:
    """Size in bytes of one file in a repo listing, or None if unreported.

    huggingface_hub reports plain files in .size and LFS files in .lfs, and the
    latter has been both an object and a dict across versions - hence the two
    lookups rather than one attribute access.
    """
    size = getattr(sibling, "size", None)
    if size is not None:
        return size
    lfs = getattr(sibling, "lfs", None)
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        return lfs.get("size")
    return getattr(lfs, "size", None)


def _repo_file_sizes(repo_id: str) -> dict[str, int]:
    """Every file of *repo_id* and its size in bytes, from the metadata API.

    Raises RuntimeError with the repo id in the message: a failure here is
    per-repo and the caller reports the rest anyway.
    """
    model_fetch.prepare_hf_env()
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed - install the project "
            "requirements first (python install.py).") from exc

    try:
        info = HfApi().model_info(repo_id, files_metadata=True)
    except Exception as exc:  # noqa: BLE001 - network, auth, gated repo, 404
        raise RuntimeError(f"{repo_id}: {exc}") from exc

    sizes: dict[str, int] = {}
    unreported: list[str] = []
    for sibling in info.siblings or ():
        size = _sibling_size(sibling)
        if size is None:
            unreported.append(sibling.rfilename)
        else:
            sizes[sibling.rfilename] = size
    if unreported:
        # Silence here would turn into a total that is quietly too small, which
        # is the one failure mode the progress bar cannot survive.
        raise RuntimeError(
            f"{repo_id}: the API reported no size for "
            f"{len(unreported)} file(s), e.g. {unreported[0]} - the total would "
            f"be too low, so nothing is printed for this repo.")
    if not sizes:
        raise RuntimeError(f"{repo_id}: the API listed no files at all.")
    return sizes


def measure_hf_repo(repo: models_info.HfRepo) -> int:
    """Download size in MB of a whole repository snapshot."""
    return _mb(sum(_repo_file_sizes(repo.repo_id).values()))


def measure_hf_file(entry: models_info.HfFile) -> int:
    """Download size in MB of one named file inside a repository."""
    sizes = _repo_file_sizes(entry.repo_id)
    if entry.filename not in sizes:
        raise RuntimeError(
            f"{entry.repo_id}: no file named {entry.filename} - the pinned "
            f"filename is wrong or the repo was re-published. Files: "
            f"{', '.join(sorted(sizes)[:5])} ...")
    return _mb(sizes[entry.filename])


# ---------------------------------------------------------------------------
# Supertonic (on disk, see the module docstring)
# ---------------------------------------------------------------------------

def measure_supertonic_on_disk() -> int:
    """Size in MB of the Supertonic cache directory.

    A proxy for the download rather than the download itself; the caller labels
    it as such.
    """
    cache_dir = model_fetch.supertonic_cache_dir()
    if not model_fetch.supertonic_cached():
        raise RuntimeError(
            f"nothing in {cache_dir} - download it first with "
            f"'python -m mimora.model_fetch --supertonic', then re-run.")
    total = sum(path.stat().st_size
                for path in cache_dir.rglob("*") if path.is_file())
    return _mb(total)


# ---------------------------------------------------------------------------
# llama.cpp release assets
# ---------------------------------------------------------------------------

def measure_release_asset(asset_name: str, tag: str) -> int:
    """Size in MB of one llama.cpp release asset, via a HEAD request.

    GitHub answers the download URL with a redirect to its object store; urllib
    follows it and keeps the method, so Content-Length comes back from the store
    without a single byte of payload.
    """
    url = llama_server_fetch.DOWNLOAD_URL.format(tag=tag, asset=asset_name)
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SEC) as response:
            length = response.headers.get("Content-Length")
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"{asset_name}: {exc}") from exc
    if not length or not length.isdigit():
        raise RuntimeError(
            f"{asset_name}: the server reported no usable Content-Length "
            f"({length!r}).")
    return _mb(int(length))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _line(target: str, current: int, measured: Optional[int],
          error: Optional[str], stamp: str) -> str:
    """One report row: where the number goes, what it was, what it is now."""
    if measured is None:
        return f"  {target:<40} {current:>6} -> FAILED  ({error})"
    delta = measured - current
    drift = "unchanged" if delta == 0 else f"{delta:+d} MB"
    return (f"  {target:<40} {current:>6} -> {measured:>6}   {drift}\n"
            f"      size_mb={measured},  # measured {stamp}")


def main() -> int:
    stamp = date.today().isoformat()
    print(f"Measuring download sizes ({stamp}). Nothing is downloaded.\n")

    # (where the number goes, current value, how to measure it)
    jobs: list[tuple[str, int, Callable[[], int]]] = []
    for name, repo in (("WAV2VEC2_ACOUSTIC", models_info.WAV2VEC2_ACOUSTIC),
                       ("WAV2VEC2_PHONEME", models_info.WAV2VEC2_PHONEME),
                       ("KOKORO", models_info.KOKORO),
                       ("NLLB", models_info.NLLB)):
        jobs.append((f"models_info.{name}", repo.size_mb,
                     lambda r=repo: measure_hf_repo(r)))
    jobs.append(("models_info.GGUF_CHAT", models_info.GGUF_CHAT.size_mb,
                 lambda: measure_hf_file(models_info.GGUF_CHAT)))
    jobs.append(("models_info.SUPERTONIC*", models_info.SUPERTONIC.size_mb,
                 measure_supertonic_on_disk))

    tag = llama_server_fetch.RELEASE_TAG
    for _, variant in sorted(llama_server_fetch.VARIANTS.items()):
        for asset in variant.assets:
            # Named by the archive rather than by its variant: the CUDA variant
            # has two assets, and "win-cuda-12.4-x64 asset" twice in a row does
            # not say which line goes where.
            asset_name = asset.name.format(tag=tag)
            jobs.append((asset_name, asset.size_mb,
                         lambda n=asset_name: measure_release_asset(n, tag)))

    failures = 0
    for target, current, measure in jobs:
        try:
            measured: Optional[int] = measure()
            error: Optional[str] = None
        except RuntimeError as exc:
            measured, error = None, str(exc)
            failures += 1
        print(_line(target, current, measured, error, stamp))

    print("\n  * Supertonic is measured on disk, not over the wire - see the "
          "module docstring.")
    if failures:
        print(f"\n{failures} entr(ies) could not be measured; the rest are "
              f"usable. Fix the cause and re-run rather than guessing.",
              file=sys.stderr)
        return 1
    print("\nPaste each size_mb line into the record it names, then commit.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
