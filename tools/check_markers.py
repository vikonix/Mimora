"""Validate the PEP 508 environment markers in the requirements files.

The root and pronunciation/* requirements carry per-platform markers so that
Intel macOS (x86_64) gets a relaxed stack (torch==2.2.2, NumPy<2, transformers
<5) while every other platform keeps the hardened pins. This script parses each
marked line and prints which simulated environments it activates in, so a
mistake (an overlapping or missing marker) is obvious without running pip.

Pure string parsing - no network, no installs - safe to run on any OS. Run from
the repository root so the relative requirements paths resolve:

    python tools/check_markers.py
"""

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

REQUIREMENTS_FILES = (
    "requirements.txt",
    "pronunciation/acoustic/requirements.txt",
    "pronunciation/phoneme/requirements.txt",
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def active_environments(req: Requirement) -> list[str]:
    """Names of the simulated environments in which this requirement applies."""
    return [
        name
        for name, env in ENVIRONMENTS.items()
        if req.marker is None or req.marker.evaluate(env)
    ]


def main() -> int:
    ok = True
    for rel_path in REQUIREMENTS_FILES:
        path = REPO_ROOT / rel_path
        print(f"\n== {rel_path} ==")
        # package name -> set of environments it is active in (to spot gaps/overlaps).
        coverage: dict[str, set[str]] = {}
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not any(line.startswith(pkg) for pkg in MARKED_PACKAGES):
                continue
            req = Requirement(line)  # raises on an invalid marker/specifier
            envs = active_environments(req)
            coverage.setdefault(req.name, set()).update(envs)
            print(f"  {req.name:14} {str(req.specifier):14} -> {envs}")

        # Each marked package must resolve to exactly one line in every environment.
        for pkg, covered in coverage.items():
            missing = set(ENVIRONMENTS) - covered
            if missing:
                ok = False
                print(f"  !! {pkg}: no line active in {sorted(missing)}")

    print("\nAll requirement lines parsed successfully." if ok
          else "\nProblems found - see the !! lines above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
