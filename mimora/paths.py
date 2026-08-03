# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Where Mimora's files live - the single answer for the whole project.

Two roots, deliberately separate, because the same tree serves two purposes
that only coincide when the app runs from a clone:

* :func:`data_root` - everything written on the user's machine: settings,
  themes they add, downloaded models, the llama-server binary, logs. Running
  from the repository this is the repository; installed as a package it is the
  OS user-data directory, because a package's own directory belongs to the tool
  that installed it (``uv tool upgrade`` rebuilds that environment, which would
  carry off both the downloads and settings.json).
* :func:`resource_root` - files that ship WITH the code and are only read, in
  the packages BESIDE this one: the committed per-language model calibrations
  under ``pronunciation/``. Always the parent of this package, so they are
  found next to the code in both modes.
* :func:`shipped_root` - the same kind of file, but inside this package: the
  starter practice texts and the built-in theme schemas. They live here rather
  than at the top of the source tree for one reason - a wheel carries only what
  belongs to a package, and a directory at the root of the tree belongs to no
  package and is simply absent from an installed copy.

Conflating the two is the bug this module exists to prevent: a single
``BASE_DIR`` used to answer both questions, and moving it wholesale to the
user-data directory would have sent the app looking for committed resources in
a directory that only ever holds downloads.

The layout inside :func:`data_root` is identical in both modes on purpose. It
means instructions, paths printed in logs and advice in error messages read the
same everywhere, and a user can carry the directory from one machine to
another.

**Only the standard library may be imported here.** ``install.py`` reads this
module before the requirements are installed, which rules out ``platformdirs``
and makes the three OS branches worth writing by hand. The same rule (and the
same reason) as ``mimora/models_info.py``: modules forbidden to import
``config`` - the three fetchers and the hardware detector - may import this one.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# This file lives in mimora/, so the parent of the package is one level up.
# In a clone that is the project root; in an installed package, site-packages.
_PACKAGE_DIR = Path(__file__).resolve().parent
_PARENT_OF_PACKAGE = _PACKAGE_DIR.parent

# Overrides both modes below. It exists for tests, for moving gigabytes of
# downloads to another drive, for escaping the Windows roaming profile (see
# _os_data_root), and as the way out if the automatic choice is ever wrong.
HOME_ENV_VAR = "MIMORA_HOME"

# Presence of this file next to the package means "we are running from the
# source tree". Deliberately NOT a "site-packages" test on __file__: that one
# lies about editable installs, which point at a source tree from inside
# site-packages and must keep behaving like the clone they are.
_REPO_MARKER = "pyproject.toml"

# Directory name under the OS user-data root. Capitalised on Windows and macOS
# and lowercase on Linux, matching what each platform's own applications do.
_APP_DIR_NAME_WINDOWS = "Mimora"
_APP_DIR_NAME_MACOS = "Mimora"
_APP_DIR_NAME_LINUX = "mimora"


def repo_mode() -> bool:
    """True when the code is running from a source tree rather than a package."""
    return (_PARENT_OF_PACKAGE / _REPO_MARKER).is_file()


def _env_root() -> Path | None:
    """The MIMORA_HOME override, or None when unset or empty.

    ``~`` is expanded so the variable can be written the way a shell would
    accept it, and the result is made absolute so a relative value cannot make
    the app's files depend on the working directory at launch.

    One matching pair of surrounding quotes is dropped: ``set MIMORA_HOME="D:\\x"``
    in cmd keeps the quotes inside the value, unlike a POSIX shell, and a quote
    is not a legal character in a Windows filename - so ``ensure_dirs`` would
    raise OSError while ``config`` was still being imported, showing a traceback
    where the whole point of the variable is to be an easy way out.
    """
    value = os.environ.get(HOME_ENV_VAR, "").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    if not value:
        return None
    return Path(value).expanduser().absolute()


def _os_data_root() -> Path:
    """The per-user data directory of the current OS.

    Windows uses ``%APPDATA%`` (Roaming) rather than ``%LOCALAPPDATA%``: it is
    where an application's user data conventionally goes. The known cost is a
    domain network with roaming profiles, where several gigabytes of downloads
    would travel with the profile at login - that is precisely what
    MIMORA_HOME is for.

    A cache directory was considered for the downloads and rejected: they are
    formally reproducible, but reproducing them means four gigabytes over the
    network, and cache directories exist to be cleaned.
    """
    if sys.platform == "win32":
        # APPDATA is absent in some service contexts; the home directory is the
        # documented fallback location for the same data.
        base = os.environ.get("APPDATA", "").strip()
        root = Path(base) if base else Path.home() / "AppData" / "Roaming"
        return root / _APP_DIR_NAME_WINDOWS
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / _APP_DIR_NAME_MACOS
    # Linux and other POSIX systems: the XDG base directory specification.
    base = os.environ.get("XDG_DATA_HOME", "").strip()
    root = Path(base) if base else Path.home() / ".local" / "share"
    return root / _APP_DIR_NAME_LINUX


def data_root() -> Path:
    """Root of everything written on this machine.

    Resolution order, highest priority first:

    1. ``MIMORA_HOME`` when set;
    2. the source tree, when running from a clone (see :func:`repo_mode`);
    3. the OS user-data directory.

    A function rather than a constant so tests can point the environment
    variable somewhere else without reloading the module. ``config.py`` binds
    its own constants from it once at import, which is where the value is
    frozen for a run.
    """
    return _env_root() or (_PARENT_OF_PACKAGE if repo_mode() else _os_data_root())


def resource_root() -> Path:
    """Root of the read-only files shipped in the packages BESIDE this one.

    Always the parent of this package: the source tree in a clone,
    site-packages in an installed package. MIMORA_HOME does NOT apply - it
    redirects what the machine writes, and cannot move files that arrive with
    the code.

    Its one remaining tenant is ``pronunciation/phoneme/``'s committed model
    calibrations, which belong to a sibling package. Resources owned by
    ``mimora`` itself are under :func:`shipped_root` instead.
    """
    return _PARENT_OF_PACKAGE


def shipped_root() -> Path:
    """Root of the read-only files shipped INSIDE this package.

    The package directory itself, which is what makes these files survive
    packaging: setuptools puts a non-Python file into the wheel only when it
    sits inside a package and is named in ``package-data``. The practice texts
    and theme schemas used to live at the top of the source tree, where both
    conditions fail silently - the wheel built fine and the installed app found
    neither.

    Callers join the same relative path they always did (``texts/...``), so
    the layout under this root matches the one the source tree used to have.
    """
    return _PACKAGE_DIR


# ---------------------------------------------------------------------------
# Named locations
# ---------------------------------------------------------------------------
# Every directory the project globs or writes to is named here once. Callers
# bind to these instead of joining their own strings onto a root, for the same
# reason models_info holds the repo ids: a second spelling of the same path is
# free to drift from the first.


def config_dir() -> Path:
    """Hand-edited configuration: settings.json, hardware_config.json, themes/."""
    return data_root() / "config"


def themes_dir() -> Path:
    """The user's own theme schemas; see :func:`shipped_themes_dir` for the rest.

    Read-only as far as the app is concerned - nothing here writes a theme -
    but created by :func:`ensure_dirs` all the same, because it is the
    documented place to drop one and an instruction that starts with "create
    this directory" is an instruction nobody follows.
    """
    return config_dir() / "themes"


def shipped_themes_dir() -> Path:
    """The theme schemas that travel with the code.

    Searched after :func:`themes_dir`, so a user file of the same name wins.
    Named here rather than joined at the call site because config.py both globs
    this directory (to list the selectable themes) and reads one file out of it.
    """
    return shipped_root() / "themes"


def models_dir() -> Path:
    """The GGUF chat model downloaded by gguf_fetch."""
    return data_root() / "models"


def model_cache_dir() -> Path:
    """HF hub cache (HF_HOME) plus the Supertonic weights, filled by model_fetch."""
    return data_root() / "model_cache"


def llama_dir() -> Path:
    """The llama-server binary installed by llama_server_fetch."""
    return data_root() / "bin" / "llama"


def log_dir() -> Path:
    """Application log, engine sample logs and the diagnostic dumps in records/.

    Kept next to the settings rather than in ~/Library/Logs or $XDG_STATE_HOME:
    a separate convention for logs exists on only two of the three platforms,
    and logs are wanted exactly when somebody has come to investigate - at
    which point everything worth looking at being in one place is worth more
    than the formal distinction.
    """
    return data_root() / "logs"


def ensure_dirs() -> None:
    """Create the directories a working installation needs, idempotently.

    Called once by ``config.py`` at import. ``parents=True`` is load-bearing
    rather than defensive: in package mode the data root itself does not exist
    on a first run, and creating a child of a missing parent would fail.

    The criterion for being listed here is that a documented scenario would
    otherwise require the user to run mkdir first. Three of the four are places
    the app writes to - and ``config/`` earns it twice over, because
    ``loader.save_setting`` reports an unwritable settings.json only on stderr
    and returns False, so without the directory no preference would ever
    persist and nothing would say why. ``config/themes/`` is the exception the
    criterion is worded for: nothing writes a theme, but it is where the user
    is told to put one.
    """
    for directory in (config_dir(), themes_dir(), model_cache_dir(), log_dir()):
        directory.mkdir(parents=True, exist_ok=True)
