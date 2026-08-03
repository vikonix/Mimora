"""Validate the PEP 508 environment markers in the project's dependency lists.

The project list and the pronunciation/* requirements carry per-platform markers
so that Intel macOS (x86_64) gets a relaxed stack (torch==2.2.2, NumPy<2,
transformers<5) while every other platform keeps the hardened pins. This script
parses each marked line and prints which simulated environments it activates in,
so a mistake (an overlapping or missing marker) is obvious without running pip.

There are three lists rather than one because the two subpackages under
pronunciation/ are reusable GUI-agnostic libraries and must stay installable on
their own. They therefore repeat torch, transformers and numpy with the same
markers as the application - which is the duplication this script exists to
police. (The root requirements.txt that used to be the first of the three is
gone: the application's list lives in pyproject.toml now.)

Pure parsing - no network, no installs - safe to run on any OS. The paths are
resolved against the repository root derived from this file's location, so the
working directory does not matter:

    python tools/check_markers.py
"""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

# Simulated pip marker environments (a subset of the fields pip exposes).
ENVIRONMENTS = {
    "intel_mac": {"platform_system": "Darwin",  "platform_machine": "x86_64", "sys_platform": "darwin"},
    "apple_sil": {"platform_system": "Darwin",  "platform_machine": "arm64",  "sys_platform": "darwin"},
    "windows":   {"platform_system": "Windows", "platform_machine": "AMD64",  "sys_platform": "win32"},
    "linux":     {"platform_system": "Linux",   "platform_machine": "x86_64", "sys_platform": "linux"},
}

# The packages that carry platform-conditional markers.
MARKED_PACKAGES = ("numpy", "torch", "torchaudio", "transformers")

DEPENDENCY_SOURCES = (
    "pyproject.toml",
    "pronunciation/acoustic/requirements.txt",
    "pronunciation/phoneme/requirements.txt",
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def requirement_lines(path: Path) -> list[str]:
    """The requirement strings held by *path*, whichever format it uses.

    Two formats, one job: `[project.dependencies]` in pyproject.toml for the
    application, and a plain requirements file for each subpackage. TOML entries
    arrive already stripped of comments; the text files are stripped here, and
    the caller filters both the same way.
    """
    if path.suffix == ".toml":
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        return [str(item)
                for item in data.get("project", {}).get("dependencies", [])]
    return [line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()]


def active_environments(req: Requirement) -> list[str]:
    """Names of the simulated environments in which this requirement applies."""
    return [
        name
        for name, env in ENVIRONMENTS.items()
        if req.marker is None or req.marker.evaluate(env)
    ]


def main() -> int:
    ok = True
    for rel_path in DEPENDENCY_SOURCES:
        path = REPO_ROOT / rel_path
        print(f"\n== {rel_path} ==")
        # package name -> environment -> number of active lines. Each marked
        # package must resolve to exactly ONE line per environment: zero means
        # pip installs nothing there, two+ means conflicting specifiers.
        coverage: dict[str, dict[str, int]] = {}
        for line in requirement_lines(path):
            if not any(line.startswith(pkg) for pkg in MARKED_PACKAGES):
                continue
            req = Requirement(line)  # raises on an invalid marker/specifier
            envs = active_environments(req)
            counts = coverage.setdefault(req.name, dict.fromkeys(ENVIRONMENTS, 0))
            for env in envs:
                counts[env] += 1
            print(f"  {req.name:14} {str(req.specifier):14} -> {envs}")

        for pkg, counts in coverage.items():
            missing = sorted(env for env, n in counts.items() if n == 0)
            overlapping = sorted(env for env, n in counts.items() if n > 1)
            if missing:
                ok = False
                print(f"  !! {pkg}: no line active in {missing}")
            if overlapping:
                ok = False
                print(f"  !! {pkg}: more than one line active in {overlapping}")

    print("\nOK: every marked package resolves to exactly one line per environment."
          if ok else "\nProblems found - see the !! lines above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
