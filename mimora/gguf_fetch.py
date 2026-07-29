# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Download the GGUF chat model the llama-server backend runs.

Second half of the LLM stack: mimora/llama_server_fetch.py fetches the binary,
this module fetches the weights it loads. Both are unnecessary when
`llm_backend` is "lm-studio" (LM Studio manages its own model) or "off" (no LLM
at all), which is why they live apart from mimora/model_fetch.py - the models
there are picked by the active engine and TTS backend and are always needed.

Run it directly:

    python -m mimora.gguf_fetch              # download if missing
    python -m mimora.gguf_fetch --list       # show target path and state
    python -m mimora.gguf_fetch --force      # download even if present

Design notes
------------
* Same import discipline as model_fetch: no side effects at import time and
  huggingface_hub imported inside the functions, so install.py can import this
  module before the requirements step has run.
* This module must NEVER import mimora.config, for the same reason model_fetch
  must not: config flips HF_HUB_OFFLINE=1 once the models are cached.
* It deliberately does NOT resolve settings.json's "external_model_path". Doing
  so would mean reading the user config, which is exactly the dependency the
  point above forbids. The default target matches config.EXTERNAL_MODEL_PATH's
  own default, and a caller that wants to honour an overridden path passes it
  as *target* - which is what mimora/first_run_download.py does.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    # Executed as a plain script (python mimora/gguf_fetch.py) rather than with
    # -m: that form puts THIS directory on sys.path instead of the project
    # root, so the "import mimora" below would not resolve. Putting the root
    # there makes both invocations work, which matters because running a file
    # directly is what an IDE's "run" button does.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mimora import model_fetch, models_info

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# The GGUF chat model. Identity and size come from mimora/models_info.py, where
# every model's facts live; these names bind to that record rather than
# restating it, so they cannot drift from what config and the installer see.
# The filename matches the default of config.EXTERNAL_MODEL_PATH, so the app
# finds the file without a settings change.
GGUF_REPO_ID = models_info.GGUF_CHAT.repo_id
GGUF_FILENAME = models_info.GGUF_CHAT.filename
DEFAULT_GGUF_PATH = MODELS_DIR / GGUF_FILENAME
# Download size, for the "are you sure?" prompts a GUI will want.
GGUF_SIZE_MB = models_info.GGUF_CHAT.size_mb


class GgufFetchError(RuntimeError):
    """The GGUF model is not on disk and could not be downloaded."""


def gguf_present(target: Optional[Path] = None) -> bool:
    """True when the GGUF file exists at *target* (default: models/<name>).

    Existence only: hf_hub_download writes the final file atomically, so a
    present file is a complete one, and hashing 2 GB on every start would cost
    more than the check is worth.
    """
    path = Path(target) if target is not None else DEFAULT_GGUF_PATH
    try:
        return path.is_file()
    except OSError:
        return False


def ensure_gguf(target: Optional[Path] = None, *,
                force: bool = False,
                tqdm_class: Optional[type] = None) -> Path:
    """Make sure the GGUF chat model sits at *target*; return its path.

    local_dir is used rather than the plain hub cache so the file lands exactly
    where config.EXTERNAL_MODEL_PATH points, instead of inside the cache's
    blobs/snapshots layout.

    *tqdm_class* is huggingface_hub's own progress-bar hook, forwarded
    untouched; see model_fetch.ensure_hf_models for why it is omitted from the
    call instead of being passed as None.
    """
    path = Path(target) if target is not None else DEFAULT_GGUF_PATH
    if not force and gguf_present(path):
        log.info("Already downloaded: %s", path)
        return path

    model_fetch.prepare_hf_env()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise GgufFetchError(
            "huggingface_hub is not installed - install the project "
            "requirements first (python install.py).") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s (%d MB) from %s into %s ...",
             path.name, GGUF_SIZE_MB, GGUF_REPO_ID, path.parent)
    try:
        downloaded = hf_hub_download(
            repo_id=GGUF_REPO_ID, filename=path.name,
            local_dir=str(path.parent),
            **({} if tqdm_class is None else {"tqdm_class": tqdm_class}),
        )
    except Exception as exc:  # noqa: BLE001 - network, disk, gated repo
        raise GgufFetchError(
            f"Could not download {path.name} from {GGUF_REPO_ID}: {exc}") from exc
    log.info("-> downloaded: %s", downloaded)
    return Path(downloaded)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _print_status(target: Path) -> None:
    """Report where the model would go and whether it is already there.

    A named helper rather than four print()s inside main(), mirroring
    model_fetch._print_status - it keeps the CLI's two paths (report vs.
    download) separable, in tests as well as by eye.
    """
    print(f"Repo   : {GGUF_REPO_ID}")
    print(f"File   : {GGUF_FILENAME}  ({GGUF_SIZE_MB} MB)")
    print(f"Target : {target}")
    print(f"State  : {'present' if gguf_present(target) else 'MISSING'}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the GGUF chat model into models/.")
    parser.add_argument("--target", type=Path, default=DEFAULT_GGUF_PATH,
                        help=f"destination file (default: {DEFAULT_GGUF_PATH})")
    parser.add_argument("--force", action="store_true",
                        help="download even when the file is already present")
    parser.add_argument("--list", action="store_true",
                        help="show the target path and its state, then exit")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    # stream=stdout, not the default stderr: this CLI also print()s its status
    # report, and two streams with different buffering interleave in whatever
    # order the OS feels like - the log line ended up in the middle of the
    # model list.
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stdout)

    if args.list:
        _print_status(args.target)
        return 0

    try:
        ensure_gguf(args.target, force=args.force)
    except GgufFetchError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
