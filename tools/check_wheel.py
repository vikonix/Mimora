# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Check a freshly built wheel before it is uploaded.

    python tools/check_wheel.py                 # the wheel in dist/
    python tools/check_wheel.py <path.whl>      # a specific file

Called by wheel.bat right after the build. Nothing here installs anything or
touches the network: every answer comes from the archive, the working tree and
pyproject.toml. A file name in the index is spent once and for all, so the
artifact is checked before the upload, never after it.

What is checked, and why each check earns its place
---------------------------------------------------
* **One wheel only.** Every other check reads `dist/*.whl`. Two files there
  (a leftover from the previous number) would make the whole run describe the
  wrong artifact. wheel.bat deletes dist/ first; this is the backstop.
* **One version in four places** - the working tree, the copy inside the
  wheel, the METADATA field and the file name. They come apart when the build
  runs before the version bump, and the result is a number in the index that
  reports itself as the previous one.
* **The wheel matches the working tree, byte for byte.** This is the check
  that replaced a growing list of string searches ("is get_lock in there",
  "is the sweep in there", one per past finding). Each of those answered the
  same question - was the wheel built from the current code - for one line of
  one release. Comparing the files answers it for all of them at once, and
  needs no maintenance when the next fix lands. pip copies sources verbatim,
  line endings included, so an exact comparison is safe and reports a real
  difference rather than a formatting one.
* **Nothing that should ship is missing.** The comparison above only looks at
  what the archive already holds; a module or a data file that never got in
  would pass it silently. The expectation is derived from pyproject.toml, not
  restated here: every `.py` under the two packages `packages.find` collects,
  plus every file the `package-data` patterns match. A wheel that lost
  `mimora/languages/` still imports until the first read of LANGUAGE_PROFILES.
* **Layout and metadata that PyPI or the installer would honour** - no
  top-level module (the root `main.py` must not take the name `main` in
  site-packages), the console script as declared, the dependency count against
  pyproject, and `Requires-Python`.

The METADATA is parsed with email.parser rather than by splitting the text: the
header ends with a CRLF pair, so a split on two plain newlines does not find it
and prints the whole README instead. That cost two hundred lines of output once.

Exit code is 0 when every check passes and 1 otherwise, so wheel.bat can stop
before the upload step.
"""

from __future__ import annotations

import argparse
import configparser
import re
import sys
import tomllib
import zipfile
from email.message import Message
from email.parser import Parser
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The two top-level packages `[tool.setuptools.packages.find]` collects. Named
# here because the expectation "every module of these ships" cannot be read out
# of the include patterns without reimplementing setuptools' own discovery.
SOURCE_PACKAGES = ("mimora", "pronunciation")

VERSION_RE = re.compile(r"""^__version__\s*=\s*["'](?P<version>.+?)["']""",
                        re.MULTILINE)

_failures = 0


def report(ok: bool, label: str, detail: str = "") -> bool:
    """Print one result line and remember a failure. Returns *ok* unchanged."""
    global _failures
    if not ok:
        _failures += 1
    tail = f": {detail}" if detail else ""
    print(f"{'[ok]  ' if ok else '[FAIL]'} {label}{tail}")
    return ok


def clauses(specifier: str) -> set[str]:
    """A version specifier as an unordered set of its parts.

    ">=3.11,<3.13" and "<3.13,>=3.11" mean the same range, and setuptools packs
    whichever order it likes into METADATA. Comparing the parts keeps the check
    about the range rather than about the spelling. No dependency on
    `packaging` is taken for this: the strings come from our own pyproject and
    from the wheel built out of it, so the parts match one for one.
    """
    return {part.strip() for part in specifier.split(",") if part.strip()}


def find_wheel(explicit: Optional[Path]) -> Path:
    """The wheel to check: the given one, or the only file in dist/."""
    if explicit is not None:
        return explicit
    wheels = sorted((PROJECT_ROOT / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(
            f"Expected exactly one wheel in dist/, found {len(wheels)}: "
            f"{[w.name for w in wheels]}. Delete dist/ and build again.")
    return wheels[0]


def read_version(text: str) -> Optional[str]:
    """The `__version__` string held by an __init__.py, read as text.

    Read rather than imported: importing the package would pull in its
    dependencies, and this script must run on a checkout where nothing is
    installed.
    """
    match = VERSION_RE.search(text)
    return None if match is None else match.group("version")


def check_version(archive: zipfile.ZipFile, wheel: Path,
                  metadata: Message) -> None:
    """The same version in the tree, the archive, METADATA and the file name."""
    tree = read_version(
        (PROJECT_ROOT / "mimora" / "__init__.py").read_text(encoding="utf-8"))
    packed = read_version(archive.read("mimora/__init__.py").decode("utf-8"))
    declared = metadata["Version"]
    # mimora-1.1.0rc7-py3-none-any.whl -> the second field.
    from_name = wheel.name.split("-")[1]

    seen = {"tree": tree, "wheel": packed, "METADATA": declared,
            "file name": from_name}
    agree = len(set(seen.values())) == 1
    report(agree, "one version everywhere",
           str(tree) if agree else f"they differ: {seen}")


def check_metadata(metadata: Message, config: dict) -> None:
    """The fields an installer reads, against what pyproject declares."""
    print(f"       Name: {metadata['Name']}   Version: {metadata['Version']}")

    requires = metadata.get_all("Requires-Dist") or []
    declared = config["project"]["dependencies"]
    report(len(requires) == len(declared),
           f"Requires-Dist count: {len(requires)}",
           "" if len(requires) == len(declared)
           else f"pyproject declares {len(declared)}")
    for line in requires:
        print(f"       {line}")

    # Compared as a set of clauses, not as text: setuptools rewrites the
    # specifier in its own order (">=3.11,<3.13" is packed as "<3.13,>=3.11"),
    # and a plain string comparison reports that reordering as a difference.
    # Nothing here reads the running interpreter - the question is only whether
    # the wheel carries the range pyproject declares.
    expected_python = clauses(config["project"]["requires-python"])
    actual_python = clauses(metadata["Requires-Python"] or "")
    report(actual_python == expected_python,
           f"Requires-Python matches pyproject: {metadata['Requires-Python']}",
           "" if actual_python == expected_python
           else f"pyproject declares {config['project']['requires-python']}")

    # The licence belongs in the field as an expression and in a separate file.
    # A build that inlines the whole licence text into the field is wrong in a
    # way that only shows on the project page.
    expression = metadata["License-Expression"]
    report(bool(expression) and "\n" not in expression,
           f"License-Expression: {expression}")
    report(bool(metadata.get_all("License-File")),
           "License-File present",
           "" if metadata.get_all("License-File") else "no licence file shipped")


def check_layout(archive: zipfile.ZipFile, config: dict) -> None:
    """No top-level module, and the console script pyproject declares."""
    names = archive.namelist()

    top_level = [n for n in names if n.endswith(".py") and "/" not in n]
    report(not top_level, "no top-level module in the wheel",
           "" if not top_level else f"{top_level} would shadow that name in "
                                    "site-packages")

    entry_points = [n for n in names if n.endswith("dist-info/entry_points.txt")]
    if not report(bool(entry_points), "entry_points.txt present"):
        return
    parser = configparser.ConfigParser()
    parser.read_string(archive.read(entry_points[0]).decode("utf-8"))
    packed = dict(parser["console_scripts"]) if parser.has_section(
        "console_scripts") else {}
    declared = config["project"].get("scripts", {})
    report(packed == declared, f"console scripts: {packed}",
           "" if packed == declared else f"pyproject declares {declared}")


def wheel_sources(archive: zipfile.ZipFile) -> list[str]:
    """Archive members that come from the tree, so without the .dist-info."""
    return [name for name in archive.namelist()
            if not name.endswith("/")
            and not name.split("/")[0].endswith(".dist-info")]


def check_matches_tree(archive: zipfile.ZipFile) -> None:
    """Every packed file is the file on disk, byte for byte."""
    missing: list[str] = []
    differing: list[str] = []
    for name in wheel_sources(archive):
        source = PROJECT_ROOT / name
        if not source.is_file():
            missing.append(name)
        elif source.read_bytes() != archive.read(name):
            differing.append(name)

    report(not missing, "every packed file exists in the tree",
           "" if not missing else f"not in the tree: {missing}")
    report(not differing, "every packed file matches the tree",
           "" if not differing
           else f"built before the last edit? differs: {differing}")


def expected_modules() -> list[str]:
    """Every module of the shipped packages, as archive-style paths."""
    found = []
    for package in SOURCE_PACKAGES:
        for path in sorted((PROJECT_ROOT / package).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            found.append(path.relative_to(PROJECT_ROOT).as_posix())
    return found


def expected_data(config: dict) -> list[str]:
    """Every file the package-data patterns match, as archive-style paths.

    Derived from pyproject rather than listed here, so adding a pattern there
    extends this check by itself. A wheel carries no data file that is not
    named by one of these patterns, which is the trap the comment above that
    table in pyproject.toml describes.
    """
    table = config["tool"]["setuptools"]["package-data"]
    found = []
    for package, patterns in table.items():
        package_dir = PROJECT_ROOT.joinpath(*package.split("."))
        for pattern in patterns:
            for path in sorted(package_dir.glob(pattern)):
                found.append(path.relative_to(PROJECT_ROOT).as_posix())
    return found


def check_nothing_missing(archive: zipfile.ZipFile, config: dict) -> None:
    """Everything the tree offers for shipping is in the wheel."""
    packed = set(archive.namelist())

    modules = expected_modules()
    lost = [name for name in modules if name not in packed]
    report(not lost, f"all {len(modules)} modules packed",
           "" if not lost else f"missing: {lost}")

    data = expected_data(config)
    lost_data = [name for name in data if name not in packed]
    report(not lost_data, f"all {len(data)} data files packed",
           "" if not lost_data else f"missing: {lost_data}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check a built wheel before uploading it.")
    parser.add_argument("wheel", nargs="?", type=Path,
                        help="wheel to check (default: the one in dist/)")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    wheel = find_wheel(args.wheel)
    print(f"Checking {wheel.name}\n")

    with open(PROJECT_ROOT / "pyproject.toml", "rb") as handle:
        config = tomllib.load(handle)

    with zipfile.ZipFile(wheel) as archive:
        names = [n for n in archive.namelist()
                 if n.endswith("dist-info/METADATA")]
        if not names:
            raise SystemExit("The wheel carries no METADATA - not a wheel?")
        metadata = Parser().parsestr(archive.read(names[0]).decode("utf-8"))

        check_version(archive, wheel, metadata)
        check_metadata(metadata, config)
        check_layout(archive, config)
        check_matches_tree(archive)
        check_nothing_missing(archive, config)

    print()
    if _failures:
        print(f"{_failures} check(s) FAILED - do not upload this wheel.")
        return 1
    print("All checks passed. The wheel may be uploaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
