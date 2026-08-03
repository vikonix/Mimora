# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Show the first-run window in a chosen state, without a first run.

    python tools/preview_first_run.py --state both
    python tools/preview_first_run.py --list

The window (mimora/first_run_window.py) appears only on a machine that is
missing something, which is precisely not the machine it is written on. Every
defect found in it so far was found by looking at it, and getting it to appear
meant deleting a model or editing settings.json first. This builds the plan by
hand instead, so each state is one command.

Seven states:

    required      the level that has no working refusal: notice plus Quit
    optional      the one real choice: checkbox plus Skip
    both          what a bare machine sees
    half-failure  the required level completes, then the optional one fails
    no-build      no llama.cpp build for this platform (BLOCKED_NO_BUILD)
    bad-setting   llama_server_path names nothing (BLOCKED_BAD_SETTING)
    real          this machine's actual plan, via first_run.build_plan()

The two blocked states pair their note with a missing required level on
purpose. That is the only way the window is ever reached in that condition: with
nothing else missing, ensure_ready() logs the reason and starts the app without
showing anything. On a fully installed machine --state real hits the same
asymmetry from the other side: the plan is empty, the app would show nothing at
all, and this shows the window regardless and says so.

half-failure is the one state that exists for the *download* rather than for
the layout, and the only cheap way to see a case a Plan cannot describe:
components run in order, so a failure on the optional level can leave a machine
that is already startable. The window has to notice - its second button becomes
"Skip" instead of "Quit", and pressing it is a refusal rather than an exit
(first_run_window.still_missing). Every other state fails on its FIRST
component, where nothing has been fetched and the new behaviour is
indistinguishable from the old.

**Nothing here calls ensure_ready(), and that is the point.** ensure_ready
writes llm_backend "off" into config/settings.json when the optional level is
declined - a real decision, taken by a real user, which a preview must never
fake. This drives FirstRunWindow directly, so pressing Skip changes nothing on
disk; the Outcome it would have acted on is printed instead.

Pressing Download is safe everywhere except --state real. The fabricated
states' component keys are deliberately not real, so first_run_download._fetch
refuses them, download() turns that into a failure, and the window shows its
"Download failed" branch - which is otherwise reachable only by unplugging the
network mid-run. half-failure is the one mixed case: its first component is a
real repo, chosen because this machine already has it, so fetching it returns
without touching the network; only the second is fabricated. In --state real
the keys are real and Download downloads.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The tool runs as a script, so sys.path starts at tools/ and the package next
# door is not importable without this (same as tools/measure_model_sizes.py).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mimora import first_run, llama_server_fetch, models_info  # noqa: E402
from mimora.first_run_window import FirstRunWindow  # noqa: E402

# A key no downloader claims, so Download lands in the failure branch instead of
# fetching gigabytes because somebody previewed a window. The prefix is what
# makes that visible in the error text the window then shows.
_FAKE = "preview:"

# The largest variant rather than the one this machine would pick: resolving
# that means select_variant() and an nvidia-smi subprocess, and the biggest
# number is the one worth looking at anyway (it is the widest the line gets).
_BINARY_MB = max(llama_server_fetch.variant_size_mb(name)
                 for name in llama_server_fetch.VARIANTS)


def _model(model, present: bool = False) -> first_run.Component:
    """A component carrying a catalogue model's real label and size."""
    return first_run.Component(f"{_FAKE}{model.repo_id}", model.label,
                               model.size_mb, present)


def _required() -> tuple[first_run.Component, ...]:
    """The default configuration's required level: Kokoro plus the phoneme ASR."""
    return (_model(models_info.KOKORO), _model(models_info.WAV2VEC2_PHONEME))


def _fake_binary() -> first_run.Component:
    return first_run.Component(f"{_FAKE}{first_run.KEY_LLAMA_SERVER}",
                               first_run.LLAMA_SERVER_LABEL, _BINARY_MB, False)


def _optional() -> tuple[first_run.Component, ...]:
    return (_fake_binary(), _model(models_info.GGUF_CHAT))


def _cached_model():
    """The smallest catalogue repo this machine already has, or None.

    Asked through first_run.model_present rather than model_fetch, for the same
    reason the plan does: the latter runs prepare_hf_env(), and a preview must
    not change how the process would download just by deciding what to show.
    """
    for model in sorted(models_info.HF_REPOS, key=lambda m: m.size_mb):
        if first_run.model_present(model):
            return model
    return None


def _half_failure() -> first_run.Plan | None:
    """Required level that succeeds, optional level that fails.

    The first component carries a REAL repo id, so first_run_download._fetch
    routes it to model_fetch.ensure_hf_models - which finds the repo cached and
    returns without a single request. That is the whole trick: the fetch has to
    genuinely succeed for ProgressState.complete() to bank its key, and banking
    that key is what the window then reacts to.

    present=False although the repo IS present: the flag decides whether the
    component enters the selection at all, and here it has to. None when no
    catalogue repo is cached, because then the "cheap success" is not available
    and the state would start a real multi-hundred-megabyte download.
    """
    model = _cached_model()
    if model is None:
        return None
    required = (first_run.Component(model.repo_id, model.label,
                                    model.size_mb, False),)
    return first_run.Plan(required, (_fake_binary(),), None)


def _plan(state: str) -> first_run.Plan | None:
    if state == "real":
        return first_run.build_plan()
    if state == "required":
        return first_run.Plan(_required(), (), None)
    if state == "optional":
        return first_run.Plan((), _optional(), None)
    if state == "both":
        return first_run.Plan(_required(), _optional(), None)
    if state == "half-failure":
        return _half_failure()
    # The blocked states: an empty optional level plus the reason, alongside a
    # required level that gives the window a reason to be on screen at all.
    blocked = (first_run.BLOCKED_NO_BUILD if state == "no-build"
               else first_run.BLOCKED_BAD_SETTING)
    return first_run.Plan(_required(), (), blocked)


STATES = ("required", "optional", "both", "half-failure", "no-build",
          "bad-setting", "real")

_NO_CACHED_REPO = (
    "half-failure needs one catalogue repo already downloaded, so that "
    "fetching it can succeed without touching the network. This machine has "
    "none cached - run 'python -m mimora.model_fetch --hf' first, or use "
    "another state.")


def _describe(plan: first_run.Plan) -> str:
    return (f"required: {len(plan.missing_required)} missing, "
            f"{plan.missing_required_mb} MB | "
            f"optional: {len(plan.missing_optional)} missing, "
            f"{plan.missing_optional_mb} MB | "
            f"blocked: {plan.llama_server_blocked or 'no'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show the first-run window in a chosen state.")
    parser.add_argument("--state", choices=STATES, default="both",
                        help="which situation to fabricate (default: both)")
    parser.add_argument("--list", action="store_true",
                        help="print each state's plan and exit, showing nothing")
    args = parser.parse_args()

    # The window's own log lines (and, in --state real, the plan's) are half the
    # reason to run this, so let them through.
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.list:
        for state in STATES:
            plan = _plan(state)
            summary = _describe(plan) if plan else "unavailable here"
            print(f"  {state:<13} {summary}")
        return 0

    plan = _plan(args.state)
    if plan is None:
        print(_NO_CACHED_REPO, file=sys.stderr)
        return 1

    print(f"State '{args.state}': {_describe(plan)}")
    if args.state == "real":
        print("This is the real plan with real component keys: pressing "
              "Download really downloads.")
    elif args.state == "half-failure":
        print(f"Press Download. '{plan.required[0].label}' is real and "
              f"already on disk, so it completes without a request; the "
              f"second component is fabricated and fails. Watch the second "
              f"button: it must read 'Skip', not 'Quit', and pressing it must "
              f"report a decline rather than a quit.")
    else:
        print("Fabricated keys: pressing Download shows the failure branch and "
              "fetches nothing.")
    if not plan.missing_required and not plan.missing_optional:
        # Reachable through --state real on a machine that has everything, i.e.
        # the first thing anyone tries. The window below is shown regardless
        # because looking at it is the whole point, but saying so keeps the
        # preview from implying the app would ever put it on screen.
        print("Note: with nothing missing, ensure_ready() returns immediately "
              "and shows no window at all. This one is shown anyway, to look "
              "at.")
    print("Nothing is written to settings.json whatever you press.\n")

    outcome = FirstRunWindow(plan).run()
    print(f"\n{outcome}")
    # Guarded by the plan, not just by the outcome. _on_secondary reports a
    # decline whenever the required level is empty, which is right in the app -
    # ensure_ready cannot reach the window with both levels empty - but here it
    # can, and then the line below would claim a settings.json write that would
    # never have happened.
    if outcome.optional_declined and plan.missing_optional:
        print("ensure_ready() would have written llm_backend \"off\" here.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
