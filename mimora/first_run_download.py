# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Fetch the components of a first-run plan, reporting bytes as they arrive.

Two halves that never touch Tk:

* :class:`ProgressState` - the one variable the download thread writes and the
  Tk thread reads, under a lock. The alternative, calling root.after() per
  callback, would flood the event queue: llama_server_fetch reports every MiB
  (about 600 calls for the CUDA build) and huggingface_hub reports far more
  often than that. Instead the worker only writes, and one root.after(100, ...)
  loop on the Tk side reads and repaints.
* :func:`download` - runs the plan's components in order on that worker thread,
  translating each fetcher's own progress convention into ProgressState.

Progress is counted in bytes across the WHOLE selection, with the denominator
fixed before the first byte arrives (from models_info and llama_server_fetch).
That is what keeps the percentage from jumping backwards when one file finishes
and the next starts.

How the byte counts are obtained
--------------------------------
* **llama-server**: ensure_llama_server takes a ProgressFn directly. It reports
  per asset, and the CUDA variant is two archives, so :func:`_binary_progress`
  keeps a running sum across them.
* **Hugging Face**: snapshot_download and hf_hub_download both take a public
  ``tqdm_class`` argument, and huggingface_hub drives that object with real byte
  counts (inside snapshot_download every per-file bar forwards its updates into
  the one instance our class is used for). So no monkeypatching of private
  names is needed - :func:`make_tqdm_class` builds a stand-in that records the
  bytes and discards the drawing.

  Monkeypatching file_download.hf_tqdm, the obvious alternative, would not work
  at all: that name no longer exists (huggingface_hub 1.24.0 imports tqdm.auto
  directly and builds bars through _get_progress_bar_context), and a patch by
  name would have failed in the worst way - silently, with no updates.
* **Supertonic**: its package downloads through its own loader with no hook at
  all, so that component contributes nothing until it finishes and then jumps by
  its whole size. It is 404 MB and only reached with the Spanish TTS backend.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from typing import NamedTuple, Optional, Sequence

from . import (config, first_run, gguf_fetch, llama_server_fetch, model_fetch,
               models_info)

log = logging.getLogger(__name__)

# Decimal, matching models_info.size_mb and llama_server_fetch._human.
BYTES_PER_MB = 1_000_000

# Which fetcher owns a model component. Built from the catalogue so a repo added
# there cannot be forgotten here - _fetch raises on an unknown key rather than
# silently skipping it.
_HF_REPOS_BY_ID = {repo.repo_id: repo for repo in models_info.HF_REPOS}


class Snapshot(NamedTuple):
    """What the Tk side reads. A plain value, safe to hold and compare.

    completed_keys names the components that finished, in order. The bar does
    not need it - the byte counts already cover that - but the dialog does:
    after a failure it has to know which parts of its plan are now on disk,
    because the plan itself was measured before any of this started and never
    changes. It lives here rather than in the window because this is the only
    side that watches components finish.
    """

    label: str
    done_bytes: int
    total_bytes: int
    error: Optional[str]
    finished: bool
    completed_keys: tuple[str, ...]

    @property
    def fraction(self) -> float:
        """0.0 to 1.0; 0.0 when there is nothing to download."""
        if self.total_bytes <= 0:
            return 0.0
        return min(self.done_bytes / self.total_bytes, 1.0)


class ProgressState:
    """Download progress, written by the worker and read by the Tk loop.

    Every method takes the lock; the worker calls advance() thousands of times
    and the reader once per 100 ms, so contention is not a concern.
    """

    def __init__(self, total_bytes: int) -> None:
        self._lock = threading.Lock()
        self._total = total_bytes
        self._completed = 0      # bytes of components already finished
        self._current = 0        # bytes reported within the current component
        self._label = ""
        self._error: Optional[str] = None
        self._finished = False
        self._overflow_logged = False
        # The component being fetched right now, and the keys of the ones that
        # finished. complete() reads the first to bank the second, which is why
        # it needs no argument of its own.
        self._component: Optional[first_run.Component] = None
        self._completed_keys: list[str] = []

    def begin(self, component: first_run.Component) -> None:
        """Start a new component; its own byte count restarts at zero."""
        with self._lock:
            self._component = component
            self._label = component.label
            self._current = 0

    def advance(self, done_in_component: int) -> None:
        """Report the ABSOLUTE byte count reached within the current component.

        Absolute rather than a delta, and taken as a maximum, because more than
        one reporter can describe the same download: on the xet path
        huggingface_hub keeps a "transferred over the network" bar and a
        "written to disk" bar, both of which end up as instances of our
        stand-in. Summing them would double the progress; the larger of the two
        is the honest answer, and taking a maximum also makes the bar monotonic
        for free, which matters because huggingface_hub subtracts the resumed
        size when it retries a file.
        """
        with self._lock:
            if done_in_component > self._current:
                self._current = done_in_component

    def complete(self) -> None:
        """Bank the component begin() named, at its PLANNED size.

        The plan, not the bytes actually observed: the denominator was fixed
        from the same planned numbers, so banking anything else would let the
        bar drift away from 100%. A real disagreement is logged by snapshot().

        Its key is recorded as well, which is what lets the dialog tell "this
        was downloaded just now" from "this was already here" after a later
        component fails.
        """
        with self._lock:
            component = self._component
            if component is None:
                return
            self._completed += (component.size_mb or 0) * BYTES_PER_MB
            self._current = 0
            self._completed_keys.append(component.key)
            self._component = None

    def fail(self, message: str) -> None:
        with self._lock:
            self._error = message
            self._finished = True

    def finish(self) -> None:
        with self._lock:
            self._finished = True

    def snapshot(self) -> Snapshot:
        with self._lock:
            done = self._completed + self._current
            if done > self._total and not self._overflow_logged:
                # The measured sizes are a pin, not a live lookup, so a real
                # download can exceed them (a re-uploaded repo, an asset
                # rebuilt upstream). Recomputing the denominator mid-flight
                # would make the percentage jump, so the plan wins on screen
                # and the disagreement goes to the log instead.
                self._overflow_logged = True
                log.info("Downloaded more than planned (%d MB against %d MB) - "
                         "the recorded sizes are stale; re-snap them with "
                         "tools/measure_model_sizes.py.",
                         done // BYTES_PER_MB, self._total // BYTES_PER_MB)
            return Snapshot(label=self._label,
                            done_bytes=min(done, self._total),
                            total_bytes=self._total,
                            error=self._error,
                            finished=self._finished,
                            completed_keys=tuple(self._completed_keys))


def make_tqdm_class(state: ProgressState) -> type:
    """Build a tqdm stand-in that feeds *state* instead of drawing a bar.

    Passed to huggingface_hub as ``tqdm_class``. It is called in three shapes,
    all of which have to work:

    * ``cls(total=..., initial=..., unit="B", ...)`` for the byte bars - the one
      we care about (_create_progress_bar passes keywords only for a class that
      is not an hf_tqdm subclass, so no positional surprises);
    * ``cls(iterable, desc=...)`` by tqdm.contrib.concurrent.thread_map, which
      then iterates the result - hence __iter__;
    * attribute reads and writes from huggingface_hub's own helpers, which set
      ``total`` and read ``n`` and ``format_dict``.

    Only instances created with ``unit="B"`` contribute bytes; the thread_map
    one counts files and would otherwise add a handful of bogus "bytes".

    Deliberately permissive: an unknown attribute returns a no-op rather than
    raising. This runs inside a multi-gigabyte download, and losing it because
    a future huggingface_hub calls one more display method would trade the
    whole download for cosmetics.
    """

    def _noop(*_args, **_kwargs):
        return None

    class _ProgressTqdm:
        """Minimal tqdm-shaped sink; see make_tqdm_class."""

        def __init__(self, iterable=None, **kwargs):
            self._iterable = iterable
            self._counts_bytes = kwargs.get("unit") == "B"
            self.n = int(kwargs.get("initial") or 0)
            self.total = kwargs.get("total")
            # Read by huggingface_hub's rate formatting; an empty mapping makes
            # it report "no rate yet" rather than crash.
            self.format_dict: dict = {}
            if self._counts_bytes and self.n:
                state.advance(self.n)

        def update(self, n=1) -> None:
            if not n:
                return
            self.n += n
            if self._counts_bytes:
                state.advance(self.n)

        def __iter__(self):
            for item in self._iterable or ():
                yield item
                self.update(1)

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def close(self) -> None:
            pass

        def refresh(self, *_args, **_kwargs) -> None:
            pass

        def set_description(self, *_args, **_kwargs) -> None:
            pass

        def set_description_str(self, *_args, **_kwargs) -> None:
            pass

        def set_postfix_str(self, *_args, **_kwargs) -> None:
            pass

        def __getattr__(self, name):
            # Only reached for names not set in __init__ or defined above.
            log.debug("Progress stand-in: ignoring tqdm attribute %r.", name)
            return _noop

    return _ProgressTqdm


def _binary_progress(state: ProgressState) -> llama_server_fetch.ProgressFn:
    """Adapt ensure_llama_server's per-asset reports to a per-component total.

    ProgressFn gets (asset name, bytes so far IN THAT ASSET, its size). The CUDA
    variant is two archives fetched one after another, so the count restarts
    when the name changes; what the component has downloaded is the sum of the
    finished archives plus the current one.
    """
    finished_bytes = 0
    current_name: Optional[str] = None
    current_bytes = 0

    def report(name: str, downloaded: int, _total: Optional[int]) -> None:
        nonlocal finished_bytes, current_name, current_bytes
        if name != current_name:
            finished_bytes += current_bytes
            current_name = name
            current_bytes = 0
        current_bytes = downloaded
        state.advance(finished_bytes + downloaded)

    return report


# The two variables that put huggingface_hub into offline mode. config.py sets
# both while it is imported.
_OFFLINE_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def _hub_constants():
    """huggingface_hub's constants module, or None when it is not installed.

    A named function only so that _hub_online's ordering can be asserted in a
    test: this must be called while the offline environment variables are
    still in place. See the warning in _hub_online.
    """
    try:
        from huggingface_hub import constants
    except ImportError:
        return None
    return constants


@contextlib.contextmanager
def _hub_online():
    """Take huggingface_hub out of offline mode for the length of a download.

    config.py flips HF_HUB_OFFLINE=1 during its own import as soon as every
    repo the run needs is cached, so that later starts load from disk without
    revalidating over the network. The GGUF chat model is not part of that
    decision: it is a plain file outside the hub cache, and neither is the
    llama-server binary. A machine with all four repos cached and no GGUF
    therefore arrives here with the network already switched off - precisely
    the case this module exists to fix.

    This is the same hazard the fetchers avoid by never importing config, in
    the one shape that rule cannot prevent: the flag reaches us through the
    process environment rather than through an import, because main.py imports
    config long before the first-run window opens.

    Both halves are needed, and the second is the one that actually works
    today: huggingface_hub freezes the flag into constants.HF_HUB_OFFLINE at
    import time and is_offline_mode() reads that, so clearing the environment
    alone would change nothing for an already imported library. The
    environment is cleared anyway, for anything that reads it later.

    Restored on the way out. The flag was a correct description of the models
    that are cached, and putting it back keeps the app's own loading offline.

    ORDER MATTERS, and getting it wrong is silent. The library has to be
    reached while the environment still says what config decided, because
    constants.py computes HF_HUB_OFFLINE at import time - and at this point in
    a run it may well not be imported yet (transformers and kokoro pull it in
    lazily, during the loading that happens after this window). Clearing the
    environment first made this function's own import the first one, freezing
    the constant as False, so "restoring" it left the whole process online for
    good: seventy HTTP round-trips during model loading and tts.py taking its
    online branch.
    """
    hub_constants = _hub_constants()   # first, with the environment intact
    previous = None if hub_constants is None else hub_constants.HF_HUB_OFFLINE
    saved = {name: os.environ.pop(name, None) for name in _OFFLINE_VARS}
    if hub_constants is not None:
        hub_constants.HF_HUB_OFFLINE = False
    if previous or any(saved.values()):
        log.info("Offline mode was on; enabling network access for the "
                 "download and restoring it afterwards.")
    try:
        yield
    finally:
        if hub_constants is not None:
            hub_constants.HF_HUB_OFFLINE = previous
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value


def _fetch(component: first_run.Component, state: ProgressState) -> None:
    """Run the one downloader that owns *component*."""
    key = component.key
    if key == first_run.KEY_LLAMA_SERVER:
        llama_server_fetch.ensure_llama_server(progress=_binary_progress(state))
    elif key == first_run.KEY_GGUF:
        # config.EXTERNAL_MODEL_PATH, not the fetcher's default, for the same
        # reason the plan looked there: gguf_fetch cannot read settings.json.
        gguf_fetch.ensure_gguf(config.EXTERNAL_MODEL_PATH,
                               tqdm_class=make_tqdm_class(state))
    elif key == models_info.SUPERTONIC.name:
        # No progress hook exists for this one - see the module docstring.
        model_fetch.ensure_supertonic()
    elif key in _HF_REPOS_BY_ID:
        model_fetch.ensure_hf_models([_HF_REPOS_BY_ID[key]],
                                     tqdm_class=make_tqdm_class(state))
    else:
        raise RuntimeError(
            f"No downloader knows how to fetch component {key!r}. A component "
            f"was added to the plan without a branch here.")


def download(components: Sequence[first_run.Component],
             state: ProgressState) -> None:
    """Fetch every component in order, recording the outcome in *state*.

    Meant to be the target of a daemon thread. Never raises: a failure is put
    into the state so the dialog can offer "Retry", because there is nobody
    above this to catch it.

    Interruption needs no support here. Both downloaders survive being killed
    mid-flight - ensure_llama_server builds in a staging directory and swaps at
    the very end, hf_hub_download writes .incomplete files and resumes from
    them - so the app can exit at any moment without leaving a half-install.
    """
    with _hub_online():
        for component in components:
            state.begin(component)
            log.info("Fetching %s (%s MB) ...", component.label,
                     component.size_mb)
            try:
                _fetch(component, state)
            except Exception as exc:  # noqa: BLE001 - the dialog reports anything
                log.exception("Fetching %s failed.", component.label)
                state.fail(str(exc))
                return
            state.complete()
            log.info("-> done: %s", component.label)
    state.finish()


def start(components: Sequence[first_run.Component]) -> tuple[ProgressState,
                                                              threading.Thread]:
    """Kick off the downloads on a daemon thread; return the state and thread.

    Daemon on purpose: quit_app ends in lifecycle.hard_exit(), so this thread is
    expected to die mid-call, and that is safe (see download()).
    """
    state = ProgressState(first_run.total_mb(components) * BYTES_PER_MB)
    thread = threading.Thread(target=download, args=(components, state),
                              name="first-run-download", daemon=True)
    thread.start()
    return state, thread
