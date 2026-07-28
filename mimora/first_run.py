# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""What this run still has to download, as data a dialog can render.

The single place the first-run dialog and the progress bar read their plan
from: one record per component - label, download size, and whether this machine
already has it - split into the two levels that differ in what refusing them
means.

* **Required**: the active TTS backend's model plus the active engine's
  recognizer. Refusing leaves no reference audio or no scoring, i.e. no
  product, so this level is a notice with "Download" and "Quit" rather than a
  question.
* **Optional**: the llama-server binary and the GGUF chat model, and only when
  llm_backend is "llama-server". Refusing is a working configuration
  (llm_backend "off" takes practice phrases from the source text verbatim), so
  this is the one real choice.

NLLB is in neither. Translation is off by default and the app runs fine without
its 2483 MB, so asking for it would nearly double the required level.
config._CACHED_REPOS does require it unconditionally, but that constant answers
a different question - what must be cached before the run can go offline - and
that is why the required set is built here the same WAY rather than taken from
it. model_fetch.missing_models() is wrong here for the same kind of reason: it
checks all four repos plus Supertonic, 5776 MB, most of which a given run never
loads.

This module reads config, so the fetchers must never import it: they are
forbidden config (which flips HF_HUB_OFFLINE=1 once the models are cached,
switching the network off exactly when a download is wanted). The dependency
runs one way - this module reads config, models_info and the three fetchers,
and nothing but the GUI reads this module.

Presence is asked of the app, not of our downloaders
----------------------------------------------------
"Is it there?" means "does this machine have something the app would use",
which is not the same as "did our fetcher put it in its own directory". All
three of the predicates that suggest themselves answer the second question:

* the binary comes from config.resolve_llama_server_path(), which also honours
  "llama_server_path" from settings.json and a llama-server on PATH.
  llama_server_fetch.installed_exe() only sees bin/llama/ and would offer a
  641 MB download on a machine that already works;
* the GGUF is looked for at config.EXTERNAL_MODEL_PATH, because gguf_fetch
  cannot read settings.json (it may not import config) and its own default is
  models/<name>;
* hub repos go through loader.models_cached, the predicate config's own offline
  gate uses, so the two answers cannot disagree.

For the binary even that is not the whole question. What the dialog needs is
not "is it there" but "would downloading it help", and the two come apart when
settings.json names a binary that is not there: the download lands in
bin/llama/, the resolver goes on preferring the named path, and the user has
paid 641 MB for nothing. Hence llama_server_status()'s three answers and
Plan.llama_server_blocked.

model_fetch.hf_repo_cached() now answers the same way (it delegates to the same
helper), and calling it here would be equally correct on the answer - but it
also runs prepare_hf_env(), which disables hf-xet on Windows. That is a
download-time workaround, and config declines to apply it for a cache the app
merely reads (mimora/config.py, the MODEL_CACHE_DIR section). Merely asking what
is on disk must not change how the process would download, so the plan is built
without side effects and the fetchers arm their own environment when the
downloading starts.

See tasks/first-run-fetch.md (work 1) for the reasoning in full.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple, Optional, Sequence, Union

from . import config, gguf_fetch, llama_server_fetch, loader, model_fetch, models_info

log = logging.getLogger(__name__)

# Keys for the two components that are not Hugging Face models and so have no
# repo id to be identified by. A model's key is its repo id (or, for Supertonic,
# the name its own package knows it as), which is what the download step
# dispatches on.
KEY_LLAMA_SERVER = "llama-server"
KEY_GGUF = "gguf-chat"

# What llama_server_status() found. "Misconfigured" is its own answer rather
# than a kind of "absent" because the two want opposite reactions: an absent
# binary can be downloaded, one named by a broken setting cannot be helped by
# downloading anything.
SERVER_PRESENT = "present"
SERVER_ABSENT = "absent"
SERVER_MISCONFIGURED = "misconfigured"

# Why the optional level ended up empty, for Plan.llama_server_blocked. Both
# mean "downloading cannot produce a llama-server this app would launch", and
# first_run_window renders one note per reason.
BLOCKED_NO_BUILD = "no-build"          # no pinned build for this platform
BLOCKED_BAD_SETTING = "bad-setting"    # llama_server_path names nothing

# The binary has no display name of its own: llama_server_fetch.Variant holds
# assets and probe patterns, not text. The variant is deliberately not part of
# the label either - the size already distinguishes a CUDA build (641 MB) from
# a CPU one (18 MB), and the variant is logged where it is resolved.
#
# Public for the same reason as the keys above: tools/preview_first_run.py
# fabricates plans, and a preview that spelled the label out again would show
# something the app does not.
LLAMA_SERVER_LABEL = "llama.cpp server (local LLM backend)"

# Either kind of catalogue record can end up in the required level: the English
# TTS model is a hub repo, the Spanish one is not.
Model = Union[models_info.HfRepo, models_info.PackagedModel]

# The recognizer each engine loads. "none" is absent on purpose: it loads no
# recognizer at all, so it requires no model. config validates ENGINE before we
# see it, so a miss can only be that engine.
_ENGINE_RECOGNIZER: dict[str, Model] = {
    "phoneme": models_info.WAV2VEC2_PHONEME,   # default engine
    "acoustic": models_info.WAV2VEC2_ACOUSTIC,
}

# The model each TTS backend loads. Exactly one backend runs per session, so
# exactly one of these is ever required; config validates TTS_BACKEND the same
# way it validates ENGINE.
_TTS_MODEL: dict[str, Model] = {
    "kokoro": models_info.KOKORO,
    "supertonic": models_info.SUPERTONIC,
}


class Component(NamedTuple):
    """One downloadable piece of the run, plus its state on this machine.

    size_mb is the download size in decimal MB - the unit models_info states
    its numbers in and llama_server_fetch._human renders progress in. It is
    None for exactly one case: a llama-server binary that is already installed.
    Naming its size would mean resolving the build variant, which shells out to
    nvidia-smi, and no caller would read the result (both the dialog and the
    progress bar only ever sum what is missing).
    """

    key: str
    label: str
    size_mb: Optional[int]
    present: bool


class Plan(NamedTuple):
    """Everything the first-run dialog and the progress bar need.

    llama_server_blocked names the reason downloading cannot produce a working
    llama-server, or is None when it can. Either way the optional level is then
    empty rather than "the GGUF only" - one model without a server does not
    start the backend - and the window shows the matching note instead of an
    offer. main.py says the same thing again when the server later fails to
    start, which is what a machine in this state does at every launch.

    The distinction is the whole point of the field: the two reasons want
    different advice, and offering a download for either would be an offer that
    cannot help. BLOCKED_BAD_SETTING is the sharper of the two, because there
    the binary is genuinely absent and a download would look reasonable - but
    it would land in bin/llama/ while the resolver keeps honouring the
    settings.json path, so the user would pay for it and change nothing.
    """

    required: tuple[Component, ...]
    optional: tuple[Component, ...]
    llama_server_blocked: Optional[str]

    @property
    def llama_server_available(self) -> bool:
        """Could downloading give this machine a llama-server it would use?"""
        return self.llama_server_blocked is None

    @property
    def missing_required(self) -> tuple[Component, ...]:
        return tuple(c for c in self.required if not c.present)

    @property
    def missing_optional(self) -> tuple[Component, ...]:
        return tuple(c for c in self.optional if not c.present)

    @property
    def missing_required_mb(self) -> int:
        """Bytes over the network the required level still costs, in MB."""
        return total_mb(self.missing_required)

    @property
    def missing_optional_mb(self) -> int:
        """Bytes over the network the optional level still costs, in MB."""
        return total_mb(self.missing_optional)


def total_mb(components: Sequence[Component]) -> int:
    """Download size of *components* together, in decimal MB.

    Also the progress bar's denominator: the bar counts bytes across everything
    it was asked to fetch, so the percentage never jumps backwards when one
    file finishes and the next begins.
    """
    total = 0
    for component in components:
        if component.size_mb is None:
            # Only an already-present component may lack a size (see
            # Component), and every caller sums missing ones, so this cannot
            # fire today. It is here so that a second sizeless case, if one is
            # ever added, fails loudly instead of quietly making the bar's
            # denominator too small - which nothing else would catch.
            raise RuntimeError(
                f"component {component.key!r} has no download size, so it "
                f"cannot be part of a total")
        total += component.size_mb
    return total


# ---------------------------------------------------------------------------
# The required level
# ---------------------------------------------------------------------------

def required_models(engine: str, tts_backend: str) -> tuple[Model, ...]:
    """Models a run cannot start without, for this engine and TTS backend.

    Pure: no filesystem, no config, no imports of the fetchers - which is the
    point, because the risky decision in this module is WHICH models the level
    contains (the NLLB question above), not the plumbing around it.

    TTS comes first because that is the order the levels table in
    tasks/first-run-fetch.md lists them in; nothing depends on it beyond the
    order the dialog and the progress bar walk the components in.
    """
    models: list[Model] = []
    # .get rather than [] for both: "none" legitimately has no recognizer, and
    # a total function is what lets the tests enumerate every combination.
    tts_model = _TTS_MODEL.get(tts_backend)
    if tts_model is not None:
        models.append(tts_model)
    recognizer = _ENGINE_RECOGNIZER.get(engine)
    if recognizer is not None:
        models.append(recognizer)
    return tuple(models)


def model_present(model: Model) -> bool:
    """Is this model already on the machine?"""
    if isinstance(model, models_info.PackagedModel):
        # Supertonic keeps its weights in its own directory rather than the hub
        # cache, and its package downloads them atomically (a temp directory
        # renamed onto the cache dir on success), so a non-empty directory is a
        # finished download.
        return model_fetch.supertonic_cached()
    return loader.models_cached(model_fetch.hf_home() / "hub", (model.repo_id,))


def _model_component(model: Model) -> Component:
    key = (model.name if isinstance(model, models_info.PackagedModel)
           else model.repo_id)
    return Component(key, model.label, model.size_mb, model_present(model))


def _required_components() -> tuple[Component, ...]:
    return tuple(_model_component(model) for model
                 in required_models(config.ENGINE, config.TTS_BACKEND))


# ---------------------------------------------------------------------------
# The optional level
# ---------------------------------------------------------------------------

def llama_server_status() -> str:
    """What this machine has to say about the llama-server binary.

    Asked of config rather than of llama_server_fetch.installed_exe(), which
    only knows about bin/llama/: config.resolve_llama_server_path() also
    resolves "llama_server_path" from settings.json and a binary on PATH, and
    offering a 641 MB download to somebody who has one of those would be
    plainly wrong.

    Three answers rather than a yes/no, because "absent" and "the setting names
    nothing" call for opposite reactions - one is downloadable, the other is
    not. They are told apart without reading settings.json: the resolver's
    other two branches verify existence themselves (installed_exe and
    shutil.which both do), so a non-empty answer that is not a file can only
    have come from the settings branch.
    """
    path = config.resolve_llama_server_path()
    if not path:
        return SERVER_ABSENT
    try:
        exists = Path(path).is_file()
    except OSError:
        exists = False
    return SERVER_PRESENT if exists else SERVER_MISCONFIGURED


def _optional_components() -> tuple[tuple[Component, ...], Optional[str]]:
    """The llama-server level, plus why it is empty when it is."""
    if config.LLM_BACKEND != "llama-server":
        # "lm-studio" talks to a server somebody else runs and "off" loads no
        # LLM at all, so neither needs the binary or the GGUF. Asking either of
        # them to agree to 2.6 GB would be a plain mistake. Not "blocked"
        # either: nothing is missing, the level simply does not apply.
        return (), None

    gguf = models_info.GGUF_CHAT
    gguf_component = Component(
        KEY_GGUF, gguf.label, gguf.size_mb,
        # config.EXTERNAL_MODEL_PATH, not the fetcher's default: gguf_fetch may
        # not import config and so cannot see an overridden path.
        gguf_fetch.gguf_present(config.EXTERNAL_MODEL_PATH))

    status = llama_server_status()
    if status == SERVER_PRESENT:
        # Size deliberately not resolved: select_variant() shells out to
        # nvidia-smi, and nothing would read the number (see Component).
        return (Component(KEY_LLAMA_SERVER, LLAMA_SERVER_LABEL, None, True),
                gguf_component), None

    if status == SERVER_MISCONFIGURED:
        # A download would be honest about the binary being absent and useless
        # all the same: it lands in bin/llama/, while the resolver goes on
        # honouring the settings.json path, so the server would fail to start
        # exactly as before. Silently preferring the download instead is worse
        # - the user named a binary, and quietly running a different one turns
        # a typo into "it works, but not with what I asked for".
        log.info("settings.json 'llama_server_path' names a file that is not "
                 "there: the optional download is not offered, because it "
                 "would not be the binary the app then looks for.")
        return (), BLOCKED_BAD_SETTING

    try:
        variant = llama_server_fetch.select_variant()
    except llama_server_fetch.UnsupportedPlatformError as exc:
        # No build for this platform and nothing installed. The GGUF alone
        # would not start the backend, so there is nothing to offer at all.
        log.info("No llama-server for this machine (%s) and none installed: "
                 "the optional download is not offered.", exc)
        return (), BLOCKED_NO_BUILD

    log.info("llama-server would be fetched as the %s build.", variant)
    return (Component(KEY_LLAMA_SERVER, LLAMA_SERVER_LABEL,
                      llama_server_fetch.variant_size_mb(variant), False),
            gguf_component), None


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def build_plan() -> Plan:
    """Inspect the machine against the active configuration.

    Cheap enough for the Tk main thread before the window is built: a handful
    of stat calls, plus one nvidia-smi only on a machine that is actually
    missing the binary. Only the downloading afterwards needs a background
    thread.
    """
    required = _required_components()
    optional, llama_server_blocked = _optional_components()
    plan = Plan(required=required, optional=optional,
                llama_server_blocked=llama_server_blocked)

    log.info("Startup plan: %d of %d required components missing (%d MB), "
             "%d of %d optional (%d MB).",
             len(plan.missing_required), len(plan.required),
             plan.missing_required_mb,
             len(plan.missing_optional), len(plan.optional),
             plan.missing_optional_mb)
    for component in plan.missing_required + plan.missing_optional:
        log.info("    missing: %s, %d MB", component.label, component.size_mb)
    return plan
