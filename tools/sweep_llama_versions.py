#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Find and install the newest llama-cpp-python CPU wheel that runs on this CPU.

Why this exists
---------------
abetlen's prebuilt CPU index ships exactly ONE wheel per version (no AVX2 vs
AVX512 variants to choose from), and recent wheels are built for an instruction
set some CPUs lack, so they crash with 0xC000001D on first model load. install.py
installs the latest wheel (right for modern CPUs); this tool is the repair path
when that latest wheel crashes on an older CPU. It installs an older version whose
wheel uses a lower CPU baseline, with no source build required.

The search is CPU-aware so it does not waste time on wheels that cannot work:

    read this CPU's AVX512 support (from numpy)
    if no AVX512: skip every wheel newer than AVX2_SAFE_CEILING (they need AVX512)
    for each remaining version, newest first:
        pip install --force-reinstall llama-cpp-python==<version>  (CPU index)
        run tools/smoke_test_llama.py in a separate process
        if it exits 0 -> this version works and is now installed; stop

So on a CPU without AVX512 the very first version tried is AVX2_SAFE_CEILING.
The smoke test is a SEPARATE process on purpose: even when 0.3.x surfaces the
illegal instruction as a catchable WinError, older builds may instead hard-abort,
and a hard abort cannot be caught in-process. The verdict is therefore the
child's exit code (see smoke_test_llama.py for the code meanings).

The working version is left installed in the venv, so a successful run also fixes
the machine; nothing needs to be pinned in install.py.

Run it inside the project's activated virtualenv, on the target machine:

    python tools/sweep_llama_versions.py
    python tools/sweep_llama_versions.py --max-tries 10
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SMOKE_TEST = PROJECT_ROOT / "tools" / "smoke_test_llama.py"

# abetlen's PEP 503 listing for the CPU build. INDEX_URL is what pip resolves
# against; LISTING_URL is the same index's HTML page we scrape for the set of
# published versions.
INDEX_URL = "https://abetlen.github.io/llama-cpp-python/whl/cpu"
LISTING_URL = INDEX_URL + "/llama-cpp-python/"

# Mirror the smoke test's exit codes. 3 (INCONCLUSIVE) means the test could not
# run at all (no model, not installed); that is not version-specific, so we stop
# rather than blame every version. 4 (LOAD_FAILED) means the build imported but
# would not load the model on this CPU, which IS a reason to reject the version
# and keep searching. Any other nonzero code is a hard crash (illegal
# instruction), also a rejection.
SMOKE_INCONCLUSIVE = 3
SMOKE_LOAD_FAILED = 4

# Upper bound on one smoke-test run. A deadlock in native code would otherwise
# hang the whole sweep forever; 30 minutes is far beyond any legitimate model
# load plus the tiny test generation, even on a slow machine.
SMOKE_TIMEOUT_S = 30 * 60

# Newest CPU wheel known to use only an AVX2 baseline (no AVX512). Wheels above
# this (0.3.29+) are built with AVX512 and crash with 0xC000001D on CPUs that
# lack it, so on such a CPU the sweep starts here instead of wasting time on
# newer ones. Determined empirically with this tool; bump it if a newer wheel is
# confirmed AVX2-safe.
AVX2_SAFE_CEILING = "0.3.28"

# Wheel filenames look like llama_cpp_python-0.3.30-py3-none-win_amd64.whl; we
# only need the version segment.
_VERSION_RE = re.compile(r"llama_cpp_python-([0-9]+\.[0-9]+\.[0-9]+)-")

LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "sweep_llama_versions.log"

logger = logging.getLogger("sweep_llama_versions")


def _setup_logging() -> None:
    """Mirror console output to logs/sweep_llama_versions.log (per run)."""
    LOG_DIR.mkdir(exist_ok=True)
    handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False


def _say(message: str) -> None:
    """Print to console and log in one call."""
    print(message)
    logger.info(message)


def _version_tuple(version: str) -> tuple[int, ...]:
    """Parse "0.3.28" into (0, 3, 28) for numeric comparison and sorting."""
    return tuple(int(part) for part in version.split("."))


def discover_versions() -> list[str]:
    """Return all published versions, newest first.

    Pip still does the per-interpreter compatibility filtering at install time;
    here we only need the full set of version strings to walk through. Versions
    are sorted by numeric (major, minor, patch) so "0.3.30" beats "0.3.9".
    """
    with urllib.request.urlopen(LISTING_URL, timeout=30) as response:
        html = response.read().decode("utf-8", errors="replace")
    versions = set(_VERSION_RE.findall(html))
    return sorted(versions, key=_version_tuple, reverse=True)


def cpu_has_avx512() -> bool | None:
    """Whether this CPU supports AVX512F, read from numpy (already a dependency).

    numpy exposes a CPUID-based feature map, which is the cleanest cross-platform
    way to read this without a new dependency or fragile Windows API calls.
    Returns None when it cannot be determined, so the caller makes no assumption.
    """
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        # numpy >= 2 moved the internal module to np._core.
        try:
            umath = np._core._multiarray_umath
        except AttributeError:
            umath = np.core._multiarray_umath
        return bool(umath.__cpu_features__.get("AVX512F", False))
    except Exception:  # noqa: BLE001 - any probe failure means "unknown"
        return None


def install_version(version: str, extra_index_url: str) -> bool:
    """Force-reinstall a specific CPU wheel; return True on success.

    --force-reinstall + --no-cache-dir guarantee the previous iteration's build
    is replaced. --only-binary forbids a source build, so a version with no wheel
    for this interpreter fails cleanly (we treat that as "skip", not a crash).
    """
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--force-reinstall", "--no-cache-dir", "--only-binary", ":all:",
        f"llama-cpp-python=={version}",
        "--extra-index-url", extra_index_url,
    ]
    _say(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # Most commonly: no compatible wheel for this Python. Log the tail so a
        # real problem (network/offline) is still visible.
        tail = proc.stdout.strip().splitlines()[-3:] + \
            proc.stderr.strip().splitlines()[-3:]
        for line in tail:
            logger.info("    | %s", line)
        return False
    return True


def run_smoke_test(model: str | None) -> int | None:
    """Run the smoke test in a child process; return its exit code.

    The child's exit code IS the verdict: 0 works, 3 inconclusive (cannot test),
    4 the build would not load on this CPU, any other nonzero a hard crash.
    None means the child hung past SMOKE_TIMEOUT_S and was killed.
    """
    cmd = [sys.executable, str(SMOKE_TEST)]
    if model:
        cmd += ["--model", model]
    _say(f"  $ {' '.join(cmd)}")
    # No capture: let the child stream straight through so a crash and its code
    # are plainly visible in the console and the parent log alike.
    try:
        proc = subprocess.run(cmd, timeout=SMOKE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search abetlen's CPU index for the newest "
                    "llama-cpp-python version that runs on this CPU.",
    )
    parser.add_argument("--max-tries", type=int, default=25,
                        help="how many versions to try, newest first "
                             "(default: 25)")
    parser.add_argument("--model", default=None,
                        help="GGUF passed to the smoke test (defaults to the "
                             "project model, then a tiny downloaded fallback)")
    parser.add_argument("--extra-index-url", default=INDEX_URL,
                        help="wheel index to install from "
                             f"(default: {INDEX_URL})")
    args = parser.parse_args()

    _setup_logging()
    if not SMOKE_TEST.is_file():
        _say(f"Smoke test not found: {SMOKE_TEST}")
        return 2

    _say(f"Python: {sys.version.split()[0]} ({sys.executable})")
    try:
        versions = discover_versions()
    except Exception as exc:  # noqa: BLE001 - network/parse errors end the run
        _say(f"Could not read the wheel index ({LISTING_URL}): {exc}")
        return 2
    if not versions:
        _say("No versions found in the wheel index.")
        return 2

    # CPU-aware ordering: a machine without AVX512 cannot run wheels built with
    # it, so drop everything newer than the AVX2-safe ceiling and start there.
    avx512 = cpu_has_avx512()
    if avx512 is False:
        ceiling = _version_tuple(AVX2_SAFE_CEILING)
        kept = [v for v in versions if _version_tuple(v) <= ceiling]
        _say(f"CPU has no AVX512: skipping {len(versions) - len(kept)} newer "
             f"AVX512 build(s); starting from {AVX2_SAFE_CEILING}.")
        versions = kept
    elif avx512 is True:
        _say("CPU has AVX512: trying the newest versions first.")
    else:
        _say("AVX512 support unknown (numpy unavailable); trying newest first.")

    candidates = versions[:args.max_tries]
    _say(f"Found {len(versions)} versions; trying the newest {len(candidates)}: "
         f"{', '.join(candidates)}")

    rejected: list[str] = []
    skipped: list[str] = []
    # The version currently sitting in the venv: each successful pip install
    # replaces the previous one, so after a failed sweep this is the last
    # REJECTED version - the final message must warn about that.
    left_installed: str | None = None
    for version in candidates:
        _say(f"\n=== llama-cpp-python {version} ===")
        if not install_version(version, args.extra_index_url):
            _say(f"  install failed (no compatible wheel?) -> skip {version}")
            skipped.append(version)
            continue
        left_installed = version

        code = run_smoke_test(args.model)
        if code is None:
            _say(f"  smoke test HUNG for over {SMOKE_TIMEOUT_S // 60} min and "
                 f"was killed -> {version} rejected.")
            rejected.append(version)
            continue
        if code == 0:
            _say(f"\nWORKING: llama-cpp-python {version} runs on this CPU and is "
                 "now installed in this venv.")
            _say("Done: this is the version left installed, so the machine is "
                 "fixed. No change to install.py is needed.")
            return 0
        if code == SMOKE_INCONCLUSIVE:
            # Not version-specific (no model / not installed): trying more
            # versions is pointless until the setup problem is fixed.
            _say("  smoke test INCONCLUSIVE (setup problem, not a CPU verdict); "
                 "fix that and re-run. Stopping.")
            _say(f"NOTE: llama-cpp-python {version} is left installed in this "
                 "venv UNTESTED - the sweep could not check whether it runs "
                 "on this CPU.")
            return 2
        if code == SMOKE_LOAD_FAILED:
            _say(f"  model would not load -> {version} rejected (build likely "
                 "needs CPU features this machine lacks).")
        else:
            _say(f"  CRASHED (exit {code}) -> {version} rejected "
                 "(illegal instruction).")
        rejected.append(version)

    _say("\nNo working version found in the range tried.")
    if rejected:
        _say(f"  rejected: {', '.join(rejected)}")
    if skipped:
        _say(f"  skipped:  {', '.join(skipped)}")
    if left_installed:
        _say(f"WARNING: llama-cpp-python {left_installed} - a version that FAILED "
             "the test - is still installed in this venv. Reinstall a chosen "
             "version explicitly, e.g.:")
        _say("  pip install --force-reinstall --only-binary :all: "
             f"llama-cpp-python==<version> --extra-index-url {args.extra_index_url}")
    _say("Try a larger --max-tries, or build from source with AVX512 disabled.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
