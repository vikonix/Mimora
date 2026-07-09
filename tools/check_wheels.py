#!/usr/bin/env python3
"""Find which dependencies have no prebuilt wheel for *this* interpreter.

Step 4 of install.py installs requirements with ``--only-binary :all:`` so pip
never source-builds a package. A few pure-Python deps publish no wheel (sdist
only) and must be exempted via ``--no-binary`` (the SOURCE_ONLY_PACKAGES list in
install.py). This script discovers that exact set for the Python you run it with,
instead of guessing.

How it works
------------
It runs pip's real resolver against requirements.txt with the same flags as the
installer, but in ``--dry-run`` mode: pip resolves and reports what it *would*
install without installing or compiling anything. When a package has no
compatible wheel, pip fails with "No matching distribution found for <pkg>".
The script catches that name, adds it to the ``--no-binary`` exemption set, and
retries - repeating until the resolve succeeds. The accumulated set is the list
you need to paste into SOURCE_ONLY_PACKAGES.

Run it with the SAME interpreter you install Mimora with (3.11 or 3.12):

    python check_wheels.py                 # uses ./requirements.txt
    python check_wheels.py path/to/requirements.txt

It does not modify any files and never builds from source.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Mirrors SOURCE_ONLY_PACKAGES in install.py - the known sdist-only exemptions
# we start from. The script extends this set with whatever else turns out to
# lack a wheel on this interpreter.
SEED_EXEMPTIONS = ["fastdtw", "docopt", "unicodecsv"]

# Packages the installer installs SEPARATELY (Step 3, from abetlen's prebuilt
# wheel index) before the requirements step. They have no wheel on PyPI, so a
# plain requirements resolve flags them - but that's expected, not a bug, and
# they must NOT go into SOURCE_ONLY_PACKAGES (that would force a source build).
# We let resolution continue past them but exclude them from the final advice.
EXTERNAL_PREINSTALLED = {"llama-cpp-python"}

# Pulled from pip's failure message, e.g.
#   ERROR: No matching distribution found for unicodecsv
_NO_DIST = re.compile(r"No matching distribution found for ([A-Za-z0-9_.\-]+)")

# pip prints the same "No matching distribution" line for a missing wheel, a
# version conflict and a dead network - only the companion line tells them
# apart: "(from versions: none)" means pip reached the index and found NO
# candidate at all (with --only-binary: no wheel). A version conflict lists
# the candidate versions instead and must NOT become a --no-binary exemption.
_NO_CANDIDATES = re.compile(
    r"Could not find a version that satisfies the requirement "
    r"([A-Za-z0-9_.\-]+)\S* \(from versions: none\)")

# When the index itself is unreachable, "(from versions: none)" appears too -
# but alongside network errors, which make the whole verdict meaningless.
_NETWORK_HINTS = ("connection error", "connection broken", "retrying",
                  "temporary failure", "name resolution", "timed out",
                  "network is unreachable", "proxyerror", "newconnectionerror")


def resolve(requirements: Path, exemptions: list[str]) -> tuple[bool, str]:
    """Dry-run the installer's pip command; return (succeeded, combined output)."""
    cmd = [
        sys.executable, "-m", "pip", "install",
        "-r", str(requirements),
        "--only-binary", ":all:",
        "--dry-run",
        # Resolve as if nothing is installed, so the result reflects a FRESH
        # machine. Without this, pip keeps already-installed versions and can
        # mask a conflict that only appears on a clean install.
        "--ignore-installed",
    ]
    if exemptions:
        cmd += ["--no-binary", ",".join(exemptions)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0, proc.stdout + proc.stderr


def main() -> int:
    # This script lives in tools/; the requirements file is in the repo root one
    # level up. An explicit path argument still overrides the default.
    requirements = Path(sys.argv[1]) if len(sys.argv) > 1 \
        else Path(__file__).resolve().parent.parent / "requirements.txt"
    if not requirements.exists():
        print(f"requirements file not found: {requirements}")
        return 2

    print(f"Python      : {sys.version.split()[0]} ({sys.executable})")
    print(f"Requirements: {requirements}\n")

    exemptions = list(SEED_EXEMPTIONS)
    # Re-resolving after each new exemption; capped so a genuinely unsatisfiable
    # requirement (network/version conflict, not a missing wheel) can't loop.
    for _ in range(40):
        ok, output = resolve(requirements, exemptions)
        if ok:
            seed = set(SEED_EXEMPTIONS)
            advice = [p for p in exemptions
                      if p not in seed and p not in EXTERNAL_PREINSTALLED]
            external = [p for p in exemptions if p in EXTERNAL_PREINSTALLED]
            print("Resolve succeeded - every requirement has a compatible wheel "
                  "(given the exemptions below).\n")
            if external:
                print("Installed separately by the installer (ignore here): "
                      f"{', '.join(external)}\n")
            if advice:
                print("Packages with NO wheel that need a --no-binary exemption:")
                for p in advice:
                    print(f"  - {p}")
                final = [p for p in exemptions if p not in EXTERNAL_PREINSTALLED]
                full = ", ".join(f'"{p}"' for p in final)
                print("\nSet this in install.py:")
                print(f"  SOURCE_ONLY_PACKAGES = ({full})")
            else:
                print("The current SOURCE_ONLY_PACKAGES already covers everything.")
            return 0

        lower = output.lower()
        if any(hint in lower for hint in _NETWORK_HINTS):
            print("Resolve failed with signs of NETWORK trouble - the missing-"
                  "wheel detection is unreliable offline. pip said:\n")
            print(output.strip())
            return 1

        missing = _NO_DIST.findall(output)
        no_candidates = {p.lower() for p in _NO_CANDIDATES.findall(output)}
        conflicts = [p for p in missing if p.lower() not in no_candidates]
        if conflicts:
            print("Resolve failed on a VERSION CONFLICT (candidates exist, none "
                  f"satisfies the constraint): {', '.join(conflicts)}. That is "
                  "not a missing wheel - fix the requirement instead of adding "
                  "an exemption. pip said:\n")
            print(output.strip())
            return 1

        new = [p for p in missing if p not in exemptions]
        if not new:
            # Failed for a reason other than a missing wheel - show pip's output
            # so the real cause is visible.
            print("Resolve failed, but not due to a missing wheel. pip said:\n")
            print(output.strip())
            return 1

        for p in new:
            print(f"No wheel for '{p}' -> adding --no-binary exemption, retrying.")
            exemptions.append(p)

    print("Gave up after too many iterations - check the requirements manually.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
