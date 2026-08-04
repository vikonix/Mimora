# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Install the spaCy pipeline Kokoro's grapheme-to-phoneme step needs.

The fourth downloader, next to mimora/model_fetch.py, mimora/gguf_fetch.py and
mimora/llama_server_fetch.py, and the only one whose model no Mimora code ever
asks for: it is reached from inside Kokoro.

Why it exists
-------------
kokoro builds misaki, and misaki's English G2P constructor does this::

    if not spacy.util.is_package(name):
        spacy.cli.download(name)
    self.nlp = spacy.load(name, enable=components)

``spacy.cli.download`` shells out to ``sys.executable -m pip install <url>``. An
environment created by ``uv tool install`` contains no pip, so the subprocess
returns non-zero and ``spacy.util.run_command`` ends the process with
``sys.exit`` - inside the loader thread, as a BaseException, which is why the
application used to sit on "Loading models..." with nothing in the log.

Declaring the model as a dependency is not open to us: spaCy models are not
published on PyPI, they live on the ``explosion/spacy-models`` release pages,
and a published package may not name a dependency by URL. Nor is there a hook
in misaki - the model name is built inside the constructor and the finished
pipeline cannot be handed in from outside. So the model has to be in place
BEFORE misaki looks, and it has to look like an installed distribution, because
``is_package`` asks importlib.metadata rather than trying an import. Unpacking
the wheel and putting its directory on sys.path satisfies exactly that.

Where it goes, and why not site-packages
----------------------------------------
Into ``paths.spacy_model_dir()`` under the data root; the reasoning is written
out there. The short version: the environment of an installed tool belongs to
the tool that installed it, and ``uv tool upgrade`` rebuilds it.

sys.path order
--------------
:func:`activate` APPENDS. A properly installed model - one somebody pip-installed
into the environment, or the copy spaCy downloaded for itself in a development
checkout - keeps winning, and this directory is consulted only when there is no
other. That also means the sidecar is not normally exercised in a clone whose
virtual environment already has the model; :func:`model_available` reports where
the answer came from, and the CLI prints it.

Run it directly::

    python -m mimora.spacy_model_fetch            # install if missing
    python -m mimora.spacy_model_fetch --list     # target, state, resolved path
    python -m mimora.spacy_model_fetch --force    # reinstall over what is there
    python -m mimora.spacy_model_fetch --measure  # print sha256 and size, install nothing

Design notes
------------
* No side effects at import, like the other three fetchers, so install.py can
  import this module before the requirements step has run.
* **It must never import mimora.config**: config flips ``HF_HUB_OFFLINE=1`` once
  the models are cached, which would switch the network off exactly when a
  download is wanted. ``paths`` is stdlib-only and allowed - that ban is about
  config's import-time side effects, not about layering.
* spaCy itself is not imported here either. Everything this module needs to
  decide is a question about the filesystem and about importlib.metadata, and
  answering it must not drag torch and thinc into install.py or into the
  first-run plan.
* No install stamp, unlike llama_server_fetch: a wheel unpacks a
  ``<name>-<version>.dist-info`` directory that already records exactly which
  version is on disk, and inventing a second record of the same fact would let
  the two disagree.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import logging
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from importlib import metadata
from pathlib import Path
from typing import Callable, Optional

if __package__ in (None, ""):
    # Executed as a plain script rather than with -m: that form puts THIS
    # directory on sys.path instead of the project root, so "import mimora"
    # below would not resolve. Same shim, for the same reason, as the other
    # three fetchers.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mimora import models_info, paths

log = logging.getLogger(__name__)

# Identity, version, size and hash all come from the catalogue; this module owns
# the mechanics only. See mimora/models_info.py WheelModel for why those four
# facts share one record.
MODEL = models_info.SPACY_EN

# paths.py is stdlib-only, so importing it does not breach the ban on importing
# config (that ban is about config's import-time side effects).
SIDECAR_DIR = paths.spacy_model_dir()

DOWNLOAD_URL = ("https://github.com/explosion/spacy-models/releases/download/"
                "{name}-{version}/{name}-{version}-py3-none-any.whl")
RELEASE_PAGE_URL = ("https://github.com/explosion/spacy-models/releases/tag/"
                    "{name}-{version}")

_CHUNK_SIZE = 1024 * 1024
_HTTP_TIMEOUT_SEC = 60

# The same shape llama_server_fetch.ProgressFn has - (asset name, bytes so far,
# total or None) - and deliberately so: first_run_download already owns an
# adapter from it to the progress bar, and a second convention would need a
# second adapter saying the same thing.
ProgressFn = Callable[[str, int, Optional[int]], None]


class SpacyModelFetchError(RuntimeError):
    """The spaCy model is not in place and could not be installed."""


def wheel_filename() -> str:
    """Name of the release asset, which is also the wheel's own filename."""
    return f"{MODEL.name}-{MODEL.version}-py3-none-any.whl"


def download_url() -> str:
    return DOWNLOAD_URL.format(name=MODEL.name, version=MODEL.version)


# ---------------------------------------------------------------------------
# What is on this machine
# ---------------------------------------------------------------------------

def _dist_info_dir(dest: Path) -> Path:
    """The .dist-info directory the wheel unpacks, at the pinned version."""
    return dest / f"{MODEL.name}-{MODEL.version}.dist-info"


def model_in_sidecar(dest: Path = SIDECAR_DIR) -> bool:
    """True when *dest* holds the pinned version, unpacked and complete.

    Both halves are checked, because they answer different questions: the
    package directory is what ``spacy.load`` imports, and the .dist-info
    directory is what ``spacy.util.is_package`` looks up. A tree with only one
    of them would pass one test and fail the other, and the failure would come
    out of spaCy rather than out of here.
    """
    try:
        return (dest / MODEL.name).is_dir() and _dist_info_dir(dest).is_dir()
    except OSError:
        return False


def resolved_location() -> Optional[Path]:
    """Where the current interpreter would find the model, or None.

    Asked of importlib.metadata rather than of the filesystem, because that is
    what spaCy asks: ``is_package`` is a ``metadata.distribution`` lookup, so
    the honest answer to "does this machine have it" is whatever that lookup
    returns on the sys.path the application is actually running with.
    """
    try:
        distribution = metadata.distribution(MODEL.name)
    except metadata.PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - a broken .dist-info must not be fatal
        log.debug("Looking up %s in the environment failed.", MODEL.name,
                  exc_info=True)
        return None
    located = distribution.locate_file("")
    return None if located is None else Path(str(located))


def model_available() -> bool:
    """True when THE APPLICATION would find the model, from wherever.

    Two halves, because there are two ways it can be there: unpacked in our own
    directory, which ``config`` puts on sys.path while it is imported, or
    already installed in the environment. The second half is the important one
    in spirit - a development checkout whose virtual environment already has the
    model, and an install that was given one with ``--with``, both need no
    download, which is the same principle the rest of the first-run plan is
    built on: presence is asked of the app, not of our downloaders.

    The first half is what makes the answer independent of the process asking.
    ``install.py`` and this module's own CLI never import ``config``, so with
    only the metadata lookup they would call a perfectly good sidecar missing -
    which is exactly what the first live ``--list`` did.
    """
    return model_in_sidecar() or resolved_location() is not None


# ---------------------------------------------------------------------------
# Making it visible
# ---------------------------------------------------------------------------

def activate(dest: Path = SIDECAR_DIR) -> bool:
    """Put *dest* on sys.path so the unpacked model can be found; report whether.

    Appended, not inserted: see the module docstring. Idempotent, and a no-op
    when nothing has been unpacked yet - a directory that does not exist on
    sys.path is harmless but pointless, and the absence is the normal state
    right up until the first run downloads something.

    ``invalidate_caches`` is insurance, not a known fix. Both the import system
    and importlib.metadata cache per sys.path entry, keyed by the entry itself,
    so a brand new one is scanned fresh either way. It costs nothing and this
    happens at most twice per process, while a stale listing here would show up
    as spaCy deciding the model is absent right after it was downloaded.
    """
    if not model_in_sidecar(dest):
        return False
    entry = str(dest)
    if entry in sys.path:
        return True
    sys.path.append(entry)
    importlib.invalidate_caches()
    log.info("spaCy model directory added to sys.path: %s", entry)
    return True


# ---------------------------------------------------------------------------
# Download and unpack
# ---------------------------------------------------------------------------

def _download(url: str, target: Path, expected_sha256: Optional[str],
              progress: Optional[ProgressFn]) -> str:
    """Fetch the wheel; return its sha256, checking it when one is expected.

    The hash is computed while streaming, so the file is never held in memory
    whole and a corrupted download is caught before anything is unpacked.
    *expected_sha256* is None only in --measure mode, which exists to produce
    the value in the first place.
    """
    name = target.name
    request = urllib.request.Request(
        url, headers={"User-Agent": "mimora-spacy-model-fetch/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SEC) as response:
            length = response.headers.get("Content-Length")
            total = int(length) if length and length.isdigit() else None
            digest = hashlib.sha256()
            downloaded = 0
            with open(target, "wb") as handle:
                while True:
                    chunk = response.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(name, downloaded, total)
    except (urllib.error.URLError, OSError) as exc:
        raise SpacyModelFetchError(
            f"Download of {name} failed: {exc}. The release is at "
            f"{RELEASE_PAGE_URL.format(name=MODEL.name, version=MODEL.version)} "
            f"if you need to fetch it manually (proxy, firewall, or offline "
            f"machine).") from exc

    actual = digest.hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        # Keep nothing questionable on disk: the next run must start clean.
        target.unlink(missing_ok=True)
        raise SpacyModelFetchError(
            f"Checksum mismatch for {name}: expected {expected_sha256}, got "
            f"{actual}. The download was corrupted or the pinned release asset "
            f"was changed upstream.")
    return actual


def _extract_wheel(wheel: Path, into: Path) -> None:
    """Unpack the wheel into *into*.

    A wheel is a zip archive and nothing here needs the executable bit, so this
    is simpler than llama_server_fetch._extract - but it keeps the same guard
    against members that point outside the destination. ZipFile.extract already
    strips absolute paths and '..'; the explicit check keeps that guarantee
    visible and local.
    """
    root = into.resolve()
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            if not (root / member.filename).resolve().is_relative_to(root):
                raise SpacyModelFetchError(
                    f"Refusing to unpack {wheel.name}: member "
                    f"{member.filename!r} points outside the target directory.")
        archive.extractall(into)


def _swap(staging: Path, dest: Path) -> None:
    """Replace *dest* with *staging* (staging is left alone on failure)."""
    if dest.exists():
        try:
            shutil.rmtree(dest)
        except OSError as exc:
            raise SpacyModelFetchError(
                f"Could not remove the existing model at {dest}: {exc}. The "
                f"new copy is ready in {staging}.") from exc
    try:
        staging.replace(dest)
    except OSError as exc:
        raise SpacyModelFetchError(
            f"Could not move {staging} onto {dest}: {exc}") from exc


def ensure_spacy_model(dest: Path = SIDECAR_DIR, *,
                       force: bool = False,
                       progress: Optional[ProgressFn] = None) -> Path:
    """Make sure the pinned model is unpacked under *dest*; return that path.

    Does nothing when the model is already there. It deliberately does NOT
    consult :func:`model_available`: the caller decides whether an existing copy
    elsewhere is good enough (the first-run plan does exactly that), while this
    function's job is to fill *dest*.
    """
    if not force and model_in_sidecar(dest):
        log.info("%s %s is already installed in %s.",
                 MODEL.name, MODEL.version, dest)
        return dest

    if MODEL.sha256 == models_info.UNMEASURED_SHA256:
        raise SpacyModelFetchError(
            f"The sha256 of {wheel_filename()} has not been measured yet, so "
            f"the download cannot be verified. Run "
            f"`python -m mimora.spacy_model_fetch --measure` and put the value "
            f"it prints into models_info.SPACY_EN.")

    # Built in a staging directory and swapped in at the end, so an interrupted
    # or failed run never leaves a half-unpacked model behind - which here would
    # be worse than nothing, because a directory with a .dist-info and no
    # package satisfies is_package and then fails inside spacy.load.
    staging = dest.with_name(dest.name + ".new")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    try:
        with tempfile.TemporaryDirectory(prefix="spacy-model-") as tmp:
            wheel = Path(tmp) / wheel_filename()
            log.info("Downloading %s (%d MB) ...", wheel.name, MODEL.size_mb)
            _download(download_url(), wheel, MODEL.sha256, progress)
            log.info("Unpacking %s ...", wheel.name)
            _extract_wheel(wheel, staging)

        if not model_in_sidecar(staging):
            raise SpacyModelFetchError(
                f"{wheel_filename()} did not unpack into the expected layout: "
                f"{MODEL.name}/ and {_dist_info_dir(staging).name}/ were "
                f"supposed to be there.")
    except BaseException:
        # Any failure (network, checksum, bad layout, Ctrl-C) must not leave a
        # half-filled staging directory for the next run to trip over.
        shutil.rmtree(staging, ignore_errors=True)
        raise

    _swap(staging, dest)
    log.info("%s %s installed in %s.", MODEL.name, MODEL.version, dest)
    return dest


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _cli_progress() -> ProgressFn:
    """Print a single rewritten line of progress, in the same MB as the plan."""
    def report(name: str, downloaded: int, total: Optional[int]) -> None:
        if total:
            percent = f"{downloaded / total * 100:5.1f}%"
        else:
            percent = "  ?  "
        print(f"\r  {name}: {percent} "
              f"({downloaded / 1_000_000:.1f} MB)", end="", flush=True)
        if total and downloaded >= total:
            print()
    return report


def _print_status(dest: Path) -> None:
    """Report the pin, the sidecar and where the model actually resolves from."""
    resolved = resolved_location()
    print(f"Model    : {MODEL.name} {MODEL.version}  ({MODEL.size_mb} MB)")
    print(f"URL      : {download_url()}")
    print(f"Sidecar  : {dest}")
    print(f"State    : {'present' if model_in_sidecar(dest) else 'MISSING'}")
    # The two can differ, and which one wins is the point of the line: an
    # environment that has its own copy keeps it, because activate() appends.
    print(f"Resolves : {resolved if resolved is not None else 'nowhere'}")


def _measure() -> int:
    """Download the wheel to a temporary file and print its hash and size.

    The one-off step behind the pin in models_info: there is no published
    checksum for these release assets, so the value has to be produced once and
    committed. Installs nothing, exactly so it can be run on a machine that
    already has the model.
    """
    with tempfile.TemporaryDirectory(prefix="spacy-model-") as tmp:
        wheel = Path(tmp) / wheel_filename()
        digest = _download(download_url(), wheel, None, _cli_progress())
        size = wheel.stat().st_size
    print(f"\nsha256   : {digest}")
    print(f"bytes    : {size}")
    print(f"size_mb  : {round(size / 1_000_000)}  "
          f"(decimal MB, the unit models_info states)")
    return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install the spaCy pipeline Kokoro's G2P step needs.")
    parser.add_argument("--dest", type=Path, default=SIDECAR_DIR,
                        help=f"destination directory (default: {SIDECAR_DIR})")
    parser.add_argument("--force", action="store_true",
                        help="reinstall even when the model is already there")
    parser.add_argument("--list", action="store_true",
                        help="show the target and its state, then exit")
    parser.add_argument("--measure", action="store_true",
                        help="print the wheel's sha256 and size without "
                             "installing it")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    # stream=stdout, not the default stderr: this CLI also print()s a status
    # report, and two streams with different buffering interleave in whatever
    # order the OS feels like.
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stdout)

    if args.list:
        # Activated before reporting, because the "Resolves" line is meant to
        # answer for the APPLICATION and the application activates the sidecar
        # through config, which this CLI never imports. Without this the line
        # said "nowhere" while the app would have found the model perfectly
        # well - a report answering a question nobody asked. Mutating sys.path
        # in a process that is about to exit costs nothing.
        activate(args.dest)
        _print_status(args.dest)
        return 0

    try:
        if args.measure:
            return _measure()
        ensure_spacy_model(args.dest, force=args.force,
                           progress=_cli_progress())
    except SpacyModelFetchError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
