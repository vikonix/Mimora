#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Find the newest llama-cpp-python CPU wheel that actually runs on this machine.

Why this exists
---------------
abetlen's prebuilt CPU index ships exactly ONE wheel per version (no AVX2 vs
AVX512 variants to choose from), and recent wheels are built for an instruction
set some CPUs lack, so they crash with 0xC000001D on first model load. There is
nothing to "select" at install time; the only reliable fix without a source build
is to install an older version whose wheel uses a lower CPU baseline.

This tool automates that search so nobody has to try versions by hand:

    for each version, newest first:
        pip install --force-reinstall llama-cpp-python==<version>  (CPU index)
        run tools/smoke_test_llama.py in a separate process
        if it exits 0 -> this version works on this CPU; stop and report it

The smoke test is a SEPARATE process on purpose: an illegal-instruction crash is
a hard abort that cannot be caught in-process, so its verdict is the child's exit
code (see smoke_test_llama.py for the code meanings).

Once a working version is found, pin it in install.py (step_cpu_llama) as
``llama-cpp-python==<version>`` so fresh installs on similar CPUs skip the search.

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

# Mirrors smoke_test_llama.EXIT_INCONCLUSIVE: a child exit of 3 means the test
# could not run (no model, etc.), which is not version-specific, so we stop
# rather than blame every version.
SMOKE_INCONCLUSIVE = 3

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


def discover_versions() -> list[str]:
    """Return all published versions, newest first.

    Pip still does the per-interpreter compatibility filtering at install time;
    here we only need the full set of version strings to walk through. Versions
    are sorted by numeric (major, minor, patch) so "0.3.30" beats "0.3.9".
    """
    with urllib.request.urlopen(LISTING_URL, timeout=30) as response:
        html = response.read().decode("utf-8", errors="replace")
    versions = set(_VERSION_RE.findall(html))
    return sorted(versions, key=lambda v: tuple(int(p) for p in v.split(".")),
                  reverse=True)


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


def run_smoke_test(model: str | None) -> int:
    """Run the smoke test in a child process; return its exit code.

    The child's exit code IS the verdict: 0 works, 3 inconclusive, anything else
    means it crashed (incompatible build) and was killed by the OS.
    """
    cmd = [sys.executable, str(SMOKE_TEST)]
    if model:
        cmd += ["--model", model]
    _say(f"  $ {' '.join(cmd)}")
    # No capture: let the child stream straight through so a crash and its code
    # are plainly visible in the console and the parent log alike.
    proc = subprocess.run(cmd)
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

    candidates = versions[:args.max_tries]
    _say(f"Found {len(versions)} versions; trying the newest {len(candidates)}: "
         f"{', '.join(candidates)}")

    crashed: list[str] = []
    skipped: list[str] = []
    for version in candidates:
        _say(f"\n=== llama-cpp-python {version} ===")
        if not install_version(version, args.extra_index_url):
            _say(f"  install failed (no compatible wheel?) -> skip {version}")
            skipped.append(version)
            continue

        code = run_smoke_test(args.model)
        if code == 0:
            _say(f"\nWORKING: llama-cpp-python {version} runs on this CPU.")
            _say("Pin it in install.py (step_cpu_llama), replacing the bare "
                 'package name with:')
            _say(f'    "llama-cpp-python=={version}"')
            return 0
        if code == SMOKE_INCONCLUSIVE:
            # Not version-specific (e.g. no model): trying more versions is
            # pointless until the setup problem is fixed.
            _say("  smoke test INCONCLUSIVE (setup problem, not a CPU verdict); "
                 "fix that and re-run. Stopping.")
            return 2
        _say(f"  CRASHED (exit {code}) -> {version} is incompatible with this CPU.")
        crashed.append(version)

    _say("\nNo working version found in the range tried.")
    if crashed:
        _say(f"  crashed:  {', '.join(crashed)}")
    if skipped:
        _say(f"  skipped:  {', '.join(skipped)}")
    _say("Try a larger --max-tries, or build from source with AVX512 disabled.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
