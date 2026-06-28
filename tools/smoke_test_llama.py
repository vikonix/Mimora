#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Smoke-test the installed llama-cpp-python build by actually loading a model.

Why this exists
---------------
On CPUs that lack an instruction set the prebuilt wheel was compiled for (most
often AVX512 on older Intel/AMD parts), llama-cpp-python imports fine but then
dies the moment it touches its SIMD compute kernels, i.e. during
``llama_model_load_from_file`` / the first decode. On Windows that shows up as
exit code 0xC000001D (STATUS_ILLEGAL_INSTRUCTION). The crash is a hard process
abort: it CANNOT be caught with try/except, so the only reliable way to detect it
is to run the risky code in a separate process and inspect that process's exit
code. This script is that separate process.

A plain ``import llama_cpp`` is NOT enough to trigger the crash (the user's own
failure happened at model load, after a successful import), so this test really
loads a GGUF and runs a single decode, which exercises the compute kernels where
the illegal instruction lives.

Model selection (option a with fallback b)
------------------------------------------
1. ``--model PATH`` if given.
2. else the project's own GGUF in ``models/`` (downloaded by install.py).
3. else a tiny model downloaded on demand (``FALLBACK_REPO`` / ``FALLBACK_FILE``).

Exit codes (kept distinct so the sweeper can tell them apart)
------------------------------------------------------------
* 0  -> OK: the build loaded a model and decoded a token on this machine.
* 3  -> INCONCLUSIVE: the test could not run (llama-cpp-python not installed,
        no model available and the fallback download failed, or any other caught
        Python error). This says nothing about CPU compatibility.
* anything else (e.g. 0xC000001D / 3221225501 on Windows, or a negative signal
        on POSIX) -> the build is INCOMPATIBLE with this CPU: it crashed.

Run it directly to check the currently installed build:

    python tools/smoke_test_llama.py
    python tools/smoke_test_llama.py --model path/to/model.gguf
"""

from __future__ import annotations

import argparse
import logging
import os
import site
import sys
from pathlib import Path

# Our own clean exit codes. The hard-crash "incompatible" verdict is never one we
# set ourselves: it is whatever the OS reports when the process is killed by an
# illegal instruction, which is always nonzero and never 3 or 4.
EXIT_OK = 0
# 3 = genuinely cannot test (llama-cpp-python not installed, or no model
#     available). Says nothing about this version; the sweeper should stop.
EXIT_INCONCLUSIVE = 3
# 4 = llama_cpp imported, but loading/decoding the model raised a catchable
#     error. For the version sweep this counts as "this version does not work
#     here", so the sweeper should reject it and try the next one.
EXIT_LOAD_FAILED = 4

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Matches GGUF_FILENAME / MODELS_DIR in install.py, so the test reuses the model
# the installer already downloaded instead of fetching anything.
PROJECT_MODEL = PROJECT_ROOT / "models" / "llama-3.2-3b-instruct-q4_k_m.gguf"

# Tiny fallback model (~1 MB) used only when no real model is present, e.g. after
# install.py was run with --skip-gguf. It is a genuine GGUF, so it still drives
# the SIMD code path. If this repo ever moves, update these two constants.
FALLBACK_REPO = "ggml-org/models-moved"
FALLBACK_FILE = "tinyllamas/stories260K.gguf"

LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "smoke_test_llama.log"

logger = logging.getLogger("smoke_test_llama")


def _setup_logging() -> None:
    """Mirror console output to logs/smoke_test_llama.log (overwritten per run)."""
    LOG_DIR.mkdir(exist_ok=True)
    handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False


def _say(message: str) -> None:
    """Print to console and log in one call (the two should never diverge)."""
    print(message)
    logger.info(message)


def _register_native_dll_dirs() -> None:
    """Make CUDA runtime DLLs from pip-installed nvidia-* packages findable.

    A CUDA build of llama-cpp-python needs cudart64_12.dll / cublas64_12.dll,
    which ship in the nvidia-cuda-runtime-cu12 / nvidia-cublas-cu12 packages
    under site-packages/nvidia/*/bin, NOT on the default DLL search path.
    Without this, importing llama_cpp on a GPU machine fails with "Could not
    find module llama.dll (or one of its dependencies)". llama_cpp loads
    llama.dll with the legacy Windows search (PATH is consulted, the
    os.add_dll_directory() dirs are not), so the dirs are added to PATH as well.
    Mirrors the helper in detect_hardware.py / llm_server/server.py. A no-op off
    Windows or when the nvidia packages are absent (the CPU build needs nothing).
    """
    if sys.platform != "win32":
        return
    for site_dir in site.getsitepackages():
        for bin_dir in Path(site_dir).glob("nvidia/*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


def resolve_model(explicit: str | None) -> Path | None:
    """Pick the model to load, downloading the tiny fallback only if needed.

    Returns the model path, or None when no model can be obtained (the caller
    then exits INCONCLUSIVE). Any download failure is reported but not raised:
    a missing model is a "cannot test" condition, not a CPU verdict.
    """
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
        _say(f"--model not found: {path}")
        return None

    if PROJECT_MODEL.is_file():
        _say(f"Using project model: {PROJECT_MODEL}")
        return PROJECT_MODEL

    _say(f"No project model at {PROJECT_MODEL}; downloading tiny fallback "
         f"({FALLBACK_REPO}/{FALLBACK_FILE}).")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        _say("huggingface_hub is not installed; cannot fetch the fallback model.")
        return None
    try:
        # Cache under model_cache/ so repeated sweeper iterations reuse the file.
        downloaded = hf_hub_download(
            repo_id=FALLBACK_REPO, filename=FALLBACK_FILE,
            cache_dir=str(PROJECT_ROOT / "model_cache"),
        )
        return Path(downloaded)
    except Exception as exc:  # noqa: BLE001 - any download error means "cannot test"
        _say(f"Fallback model download failed: {exc}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load a GGUF in this process and decode one token to detect "
                    "an illegal-instruction crash in the installed "
                    "llama-cpp-python build.",
    )
    parser.add_argument("--model", default=None,
                        help="GGUF model path to load (defaults to the project "
                             "model, then a tiny downloaded fallback)")
    args = parser.parse_args()

    _setup_logging()
    _say(f"Python: {sys.version.split()[0]} ({sys.executable})")

    # Put the nvidia DLL dirs on PATH first, or a CUDA build fails to import.
    _register_native_dll_dirs()
    try:
        import llama_cpp
    except ImportError:
        _say("llama-cpp-python is not installed -> INCONCLUSIVE.")
        return EXIT_INCONCLUSIVE
    except Exception as exc:  # noqa: BLE001 - a catchable native load error
        _say(f"Importing llama_cpp failed: {exc} -> INCONCLUSIVE.")
        return EXIT_INCONCLUSIVE
    _say(f"llama-cpp-python {getattr(llama_cpp, '__version__', 'unknown')} imported.")

    model_path = resolve_model(args.model)
    if model_path is None:
        _say("No model to load -> INCONCLUSIVE (pass --model PATH to override).")
        return EXIT_INCONCLUSIVE

    # The point of no return: loading and decoding run the SIMD kernels. If the
    # build is incompatible with this CPU, the process is killed RIGHT HERE with
    # an illegal-instruction code and never reaches the success print below.
    _say(f"Loading model and decoding one token: {model_path}")
    try:
        llm = llama_cpp.Llama(
            model_path=str(model_path),
            n_ctx=256,       # tiny: we only need the kernels to run, not real output
            n_gpu_layers=0,  # force the CPU backend, which is what can crash
            verbose=False,
        )
        llm.create_completion("Hi", max_tokens=1)
    except Exception as exc:  # noqa: BLE001 - any load error means this version failed here
        # In llama-cpp-python 0.3.x an illegal instruction during the CPU-backend
        # probe surfaces as a catchable WinError 0xC000001D instead of killing
        # the process. Detect that exact code and label it plainly; either way
        # the build does not work on this machine, so report LOAD_FAILED and let
        # the sweeper move on to an older version. 0xC000001D is shown by Python
        # as the signed value -1073741795.
        winerror = getattr(exc, "winerror", None)
        if winerror in (-1073741795, 0xC000001D):
            _say("Illegal instruction (0xC000001D) on model load: this build "
                 "needs CPU features this machine lacks -> LOAD FAILED.")
        else:
            _say(f"Model load/decode raised (not a crash): {exc} -> LOAD FAILED.")
        return EXIT_LOAD_FAILED

    _say("OK: model loaded and decoded a token. This build runs on this CPU.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
