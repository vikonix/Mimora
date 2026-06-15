"""Configuration loading machinery — pure, stateless helpers.

This module holds the *mechanics* of building Mimora's configuration: reading
JSON files, validating individual settings, creating directories, probing the
cache and the compute device. None of it runs at import time and none of it
keeps global state — every function takes what it needs as arguments and returns
a value. That keeps the rules (range checks, type checks, fallbacks) unit-testable
in isolation, without a filesystem or the heavy ML stack.

The actual configuration values live in ``config.py``, which calls these
functions at import time to build its constants. Validation problems are reported
to stderr (never raised) so a hand-edited settings.json cannot crash startup;
``config.py`` decides what to do with the returned fallback.
"""

import json
import sys
from pathlib import Path


def read_json(path: Path) -> dict:
    """Parse a JSON object from *path*; returns {} when absent or invalid.

    A missing file is silent (the caller treats it as "no overrides"); a broken
    or non-object file is reported to stderr and also yields {}.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[config] cannot read {path.name} ({exc}); using defaults",
              file=sys.stderr)
        return {}
    if not isinstance(data, dict):
        print(f"[config] {path.name} must contain a JSON object; using defaults",
              file=sys.stderr)
        return {}
    return data


def user_number(user_data: dict, key: str, default, minimum=None, maximum=None):
    """Numeric setting from *user_data*.

    Returns *default* on a non-numeric or out-of-range value: e.g.
    max_record_seconds=0 would cut off every take instantly, and a threshold
    above 100 would make passing impossible — a typo must not break the app.
    """
    value = user_data.get(key, default)
    # bool is a subclass of int — exclude it so `true` is not accepted silently.
    if not (isinstance(value, (int, float)) and not isinstance(value, bool)):
        print(f"[config] settings.json: {key} must be a number, got {value!r}; "
              f"using {default}", file=sys.stderr)
        return default
    if (minimum is not None and value < minimum) or \
            (maximum is not None and value > maximum):
        lo = "-inf" if minimum is None else minimum
        hi = "+inf" if maximum is None else maximum
        print(f"[config] settings.json: {key} must be in range {lo}..{hi}, "
              f"got {value!r}; using {default}", file=sys.stderr)
        return default
    return value


def user_path(user_data: dict, base_dir: Path, key: str, default: Path) -> str:
    """Path setting from *user_data*; *default* on a non-string value.

    A relative value is resolved against *base_dir* (pathlib keeps an absolute
    value as-is when joined), so settings.json works regardless of the working
    directory at launch.
    """
    value = user_data.get(key)
    if value is None:
        return str(default)
    if isinstance(value, str) and value.strip():
        return str(base_dir / value)
    print(f"[config] settings.json: {key} must be a non-empty string, got "
          f"{value!r}; using {default}", file=sys.stderr)
    return str(default)


def user_bool(user_data: dict, key: str, default: bool) -> bool:
    """Boolean setting from *user_data*; *default* on a non-boolean value."""
    value = user_data.get(key, default)
    if not isinstance(value, bool):
        print(f"[config] settings.json: {key} must be true or false, got "
              f"{value!r}; using {default}", file=sys.stderr)
        return default
    return value


def save_setting(path: Path, key: str, value, memory_dict: dict) -> bool:
    """Write one setting back to *path*, keeping every other key.

    The file is re-read first so hand-edited values and the "_" comment keys
    are preserved. On success the in-memory *memory_dict* is updated too, so the
    running app sees the new value without a reload. Failures are reported, never
    raised — saving a preference must not crash the app. Returns True on success.
    """
    data = read_json(path)
    data[key] = value
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    except OSError as exc:
        print(f"[config] cannot write {path.name} ({exc}); {key} not saved",
              file=sys.stderr)
        return False
    memory_dict[key] = value  # keep the in-memory view consistent for this run
    return True


def ensure_dir(path: Path) -> None:
    """Create *path* if missing (parents assumed to exist), idempotently."""
    path.mkdir(exist_ok=True)


def models_cached(hub_dir: Path, repos) -> bool:
    """True only when every repo in *repos* is fully present under *hub_dir*.

    Besides a non-empty snapshots dir, the blobs dir must hold no *.incomplete
    files — those are partial downloads left by an interrupted first run, and
    flipping to offline mode with one present would crash model loading.
    """
    for repo in repos:
        repo_dir = hub_dir / ("models--" + repo.replace("/", "--"))
        snapshots = repo_dir / "snapshots"
        if not snapshots.is_dir() or not any(snapshots.iterdir()):
            return False
        if any(repo_dir.glob("blobs/*.incomplete")):
            return False
    return True


def detect_device(hw_value) -> str:
    """Resolve the compute device: 'cuda' or 'cpu'.

    A valid *hw_value* (written by hwconfig) wins and short-circuits — torch is
    not imported in that case, so callers that already know the device (and unit
    tests) never pay the ~1s torch import. Otherwise probe torch directly,
    falling back to 'cpu' when torch is absent.
    """
    if hw_value in ("cuda", "cpu"):
        return hw_value
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
