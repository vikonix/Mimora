# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Where Mimora's files live - the single answer for the whole project.

Two roots, deliberately separate, because the same tree serves two purposes
that only coincide when the app runs from a clone:

* :func:`data_root` - everything written on the user's machine: settings,
  themes they add, downloaded models, the llama-server binary, logs. Running
  from the repository this is the repository; installed as a package it is the
  OS user-data directory, because a package's own directory belongs to the tool
  that installed it (``uv tool upgrade`` rebuilds that environment, which would
  carry off both the downloads and settings.json).
* :func:`resource_root` - files that ship WITH the code and are only read:
  practice texts, the committed per-language model calibrations, the built-in
  theme schemas. Always the parent of this package, so they are found next to
  the code in both modes.

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
    """
    value = os.environ.get(HOME_ENV_VAR, "").strip()
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
    """Root of the read-only files shipped alongside the code.

    Always the parent of this package: the source tree in a clone,
    site-packages in an installed package. MIMORA_HOME does NOT apply - it
    redirects what the machine writes, and cannot move files that arrive with
    the code.
    """
    return _PARENT_OF_PACKAGE


# ---------------------------------------------------------------------------
# Named locations
# ---------------------------------------------------------------------------
# Every directory the project writes to is named here once. Callers bind to
# these instead of joining their own strings onto data_root(), for the same
# reason models_info holds the repo ids: a second spelling of the same path is
# free to drift from the first.


def config_dir() -> Path:
    """Hand-edited configuration: settings.json, hardware_config.json, themes/."""
    return data_root() / "config"


def themes_dir() -> Path:
    """The user's own theme schemas (the shipped ones come from the package)."""
    return config_dir() / "themes"


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
    """Create the directories the app writes to, idempotently.

    Called once by ``config.py`` at import. ``parents=True`` is load-bearing
    rather than defensive: in package mode the data root itself does not exist
    on a first run, and creating a child of a missing parent would fail. The
    config directory is created here for a failure that is otherwise silent -
    ``loader.save_setting`` reports an unwritable settings.json and returns
    False, so without this directory no preference would ever persist and
    nothing would say why.
    """
    for directory in (config_dir(), model_cache_dir(), log_dir()):
        directory.mkdir(parents=True, exist_ok=True)
