# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Download the model components that the user selected in the installer.

install.cmd runs this file with the interpreter of the installed tool
environment, so every fetcher writes into the user data directory that the
application reads. A component that is not downloaded here is not lost: the
first-run window of the application offers it again.

Usage: python fetch_models.py core chat translator
"""

import argparse
import logging
import sys
from typing import Callable

from mimora import (detect_hardware, gguf_fetch, llama_server_fetch,
                    model_fetch, models_info, spacy_model_fetch)

Step = Callable[[], int]


def _hf_repos(*repos: models_info.HfRepo) -> Step:
    """A step that downloads the given Hugging Face repositories."""
    def run() -> int:
        try:
            model_fetch.ensure_hf_models(list(repos))
        except model_fetch.ModelFetchError as exc:
            print(f"\nERROR: {exc}", file=sys.stderr)
            return 1
        return 0
    return run


# One entry for each check box of the installer (the [Tasks] names in
# mimora.iss). The grouping follows what a refusal means in the first-run
# window: "core" is what the default configuration cannot start without.
COMPONENTS: dict[str, tuple[Step, ...]] = {
    "core": (
        _hf_repos(models_info.WAV2VEC2_PHONEME, models_info.KOKORO),
        lambda: spacy_model_fetch.main([]),
    ),
    "chat": (
        lambda: llama_server_fetch.main([]),
        lambda: gguf_fetch.main([]),
    ),
    "translator": (_hf_repos(models_info.NLLB),),
    "spanish": (lambda: model_fetch.main(["--supertonic"]),),
    "acoustic": (_hf_repos(models_info.WAV2VEC2_ACOUSTIC),),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("components", nargs="+", choices=sorted(COMPONENTS))
    args = parser.parse_args()

    # stdout, as in the fetchers themselves: two streams with different
    # buffering mix their lines in the console.
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stdout)

    failed = []
    for name in args.components:
        print(f"\n=== {name} ===", flush=True)
        # Every step runs, also after a failure: one unavailable download must
        # not prevent the others.
        if any([step() != 0 for step in COMPONENTS[name]]):
            failed.append(name)

    # Last, because the probe reports whether the llama-server binary that
    # "chat" installed can use the GPU.
    print("\n=== hardware ===", flush=True)
    if detect_hardware.main() != 0:
        failed.append("hardware")

    if failed:
        print(f"\nFailed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
