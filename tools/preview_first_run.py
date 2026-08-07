# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Show the first-run window in a chosen state, without a first run.

    python tools/preview_first_run.py --state both
    python tools/preview_first_run.py --tour
    python tools/preview_first_run.py --list

The window (mimora/first_run_window.py) appears only on a machine that is
missing something, which is precisely not the machine it is written on. Every
defect found in it so far was found by looking at it, and getting it to appear
meant deleting a model or editing settings.json first. This builds the plan by
hand instead, so each state is one command.

Nine states:

    required      the level that has no working refusal: notice plus Quit
    optional      one real choice: the chat model, checkbox plus Skip
    translator    the other real choice, and the only one that appears alone
    both          what a bare machine sees
    all-levels    every block at once, which no first run produces
    half-failure  the required level completes, then the optional one fails
    no-build      no llama.cpp build for this platform (BLOCKED_NO_BUILD)
    bad-setting   llama_server_path names nothing (BLOCKED_BAD_SETTING)
    real          this machine's actual plan, via first_run.build_plan()

translator is the state to look at after changing anything about the levels,
because it is the one the app reaches on a fully installed machine: turning
translation on restarts into this window with everything else already present,
so that block has to read correctly with nothing above it. all-levels is its
opposite - three blocks stacked, which only a settings.json carried onto a bare
machine would produce - and it exists because the leading padding of each block
depends on what was rendered before it, and only this state exercises all of
that at once.

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

--tour shows the eight fabricated states one after another, each announced by
name and by what has to be pressed and seen in it, waiting for Enter between
them. It exists because --state answers "show me this one" and the question
before a release is the other one, "show me all of them": nine separate
commands are nine chances to skip the state nobody remembers the point of, and
the point of each is written down here rather than remembered. The states run
in one process, which the window supports by construction - it owns a
short-lived root and Tk takes a second one after the first is destroyed (see
mimora/first_run_window.py) - and no ttk widget, cached image or registered
font survives a window to make the second showing differ from the first.

Two things that sentence got wrong, both found by the first tour of 2026-08-07
and both worth keeping written down.

The harmless one is the log: model_fetch.prepare_hf_env() probes for symlink
privileges once per process, so its INFO line appears in the first state that
presses Download and in none of the others. Under one command per state it
would be in every transcript. Nothing depends on it, but a line that moves
between runs is worth knowing about before it is read as a finding.

The other one aborted the process at state six. Tk VARIABLES do survive a
window: FirstRunWindow sits in a reference cycle (a bound method per button in
Tcl's callback registry, the registry on the root, the root on self), so its
graph waits for a cyclic collection rather than being freed on the spot - and
that collection runs on whichever thread allocates at the time, which in a
state that presses Download is the worker thread. Variable.__del__ then calls
into a destroyed interpreter from the wrong thread and Tcl aborts the process:
"Tcl_AsyncDelete: async handler deleted by the wrong thread". The variables
that killed it had been waiting since state one. The fix is in the window
itself (_close() drops them while the interpreter is alive), because the app
runs the same sequence - window, then a loader thread - and would abort the
same way; the gc.collect() in the tour below only keeps the rest of the graph
from crossing a state boundary.

--state real is deliberately NOT in the tour. Its keys are real, so one stray
click downloads gigabytes, and a walkthrough is exactly where a stray click
happens. It stays a single command, run on purpose.
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The tool runs as a script, so sys.path starts at tools/ and the package next
# door is not importable without this (same as tools/measure_model_sizes.py).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mimora import first_run, llama_server_fetch, models_info  # noqa: E402
from mimora.first_run_window import FirstRunWindow, Outcome  # noqa: E402

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


def _translator() -> tuple[first_run.Component, ...]:
    return (_model(models_info.NLLB),)


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
    if state == "translator":
        return first_run.Plan((), (), None, _translator())
    if state == "both":
        return first_run.Plan(_required(), _optional(), None)
    if state == "all-levels":
        return first_run.Plan(_required(), _optional(), None, _translator())
    if state == "half-failure":
        return _half_failure()
    # The blocked states: an empty optional level plus the reason, alongside a
    # required level that gives the window a reason to be on screen at all.
    blocked = (first_run.BLOCKED_NO_BUILD if state == "no-build"
               else first_run.BLOCKED_BAD_SETTING)
    return first_run.Plan(_required(), (), blocked)


STATES = ("required", "optional", "translator", "both", "all-levels",
          "half-failure", "no-build", "bad-setting", "real")

# Everything except real, whose keys are real and whose Download button costs
# gigabytes. Order follows STATES: the two blocked states last, because they
# are the only ones whose point is a paragraph of text rather than a layout.
TOUR_STATES = tuple(state for state in STATES if state != "real")

# Per state: what to press, and what has to be true. Two fields rather than one
# paragraph, because the paragraph is read after the window is already on
# screen and the button has already been pressed - by which time the only line
# that mattered is the first one. Whatever explains WHY a check exists belongs
# in the docstring above; what is printed is the instruction alone.
#
# One table rather than a branch per state: --state and --tour say the same
# thing, and a second copy drifts on the first edit.
#
# LOOK comes before PRESS in the window's own order of events. Pressing
# Download rewrites the secondary button to "Quit" (first_run_window
# _on_primary), so in the states whose point is that it reads "Skip", a
# download taken first destroys the evidence. That is why "press Download
# everywhere first" is not the shortcut it looks like.
CHECKS = {
    "required": (
        "Quit",
        "It must read 'Quit', never 'Skip': refusing the required level "
        "leaves nothing to run. Outcome: quit_requested=True, nothing "
        "declined."),
    "optional": (
        "Skip",
        "It must read 'Skip', not 'Quit'. Outcome: optional_declined=True. "
        "Clearing the checkbox and pressing Download is the same path, worth "
        "one extra showing."),
    "translator": (
        "Skip",
        "The only state an installed machine reaches, so read it as the first "
        "thing on screen: the title may not claim a first run, and the "
        "block's top padding has to look right with nothing above it. "
        "Outcome: translator_declined=True, optional_declined False."),
    "both": (
        "Download with the box checked, then Quit",
        "The primary button must become 'Retry' and the failure must name the "
        "component. This is the 'Download failed' branch, otherwise reachable "
        "only by pulling the network mid-run."),
    "all-levels": (
        "both checkboxes on and off, then Quit",
        "'Total download' must follow every toggle. Three blocks stacked is "
        "the only case where all the leading padding is exercised at once, so "
        "read the vertical rhythm rather than the words."),
    "half-failure": (
        "Download, then the second button",
        "'{label}' is real and on disk, so it completes without a request; "
        "the second component is fabricated and fails. The second button must "
        "turn from 'Quit' into 'Skip' once the required level is banked. "
        "Outcome: quit_requested=False, optional_declined=True."),
    "no-build": (
        "Quit",
        "No checkbox at all: there must be no way to ask for a chat model "
        "here. The note has to name a way out (a self-built llama-server, or "
        "LM Studio), neither being guessable. optional_declined must be "
        "False - a level never offered cannot be declined."),
    "bad-setting": (
        "Quit",
        "Same shape as no-build, and the difference is the point: the note "
        "must say a download is not offered BECAUSE it would land in "
        "bin/llama/ while \"llama_server_path\" kept winning, not merely that "
        "the file is missing."),
    "real": (
        "Quit, unless you mean it",
        "Real keys: Download really downloads. This is the plan this machine "
        "would produce."),
}

# Printed ONCE per run rather than before every state: it says the same thing
# each time, and repeating it is what buried the two lines that differ.
#
# The last sentence is here because the failure the fabricated keys arrange
# arrives as a traceback reading "A component was added to the plan without a
# branch here", which sounds like a defect in the code rather than the intended
# path. The one thing telling them apart is the "preview:" prefix in the key.
_PREFACE = (
    "Nothing is written to settings.json whatever you press. Outside "
    "--state real the component keys are fabricated, so Download fails by "
    "design and fetches nothing: the traceback it logs names a key starting "
    "with 'preview:', which is what tells it from a real defect.")

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
            f"translator: {len(plan.missing_translator)} missing, "
            f"{plan.missing_translator_mb} MB | "
            f"blocked: {plan.llama_server_blocked or 'no'}")


def _field(label: str, text: str) -> str:
    """One labelled line, wrapped so continuations hang under the text."""
    return textwrap.fill(text, width=78, initial_indent=f"  {label:<7}",
                         subsequent_indent=" " * 9)


def _announce(state: str, plan: first_run.Plan) -> None:
    """Everything printed before the window goes up."""
    press, look = CHECKS[state]
    if state == "half-failure":
        look = look.format(label=plan.required[0].label)
    print(_field("PLAN", _describe(plan)))
    print(_field("LOOK", look))
    # Last, and in that order on purpose: it is the line to still have in view
    # when the window appears, and looking comes before pressing anyway.
    print(_field("PRESS", press))
    if (not plan.missing_required and not plan.missing_optional
            and not plan.missing_translator):
        # Reachable through --state real on a machine that has everything, i.e.
        # the first thing anyone tries. The window below is shown regardless
        # because looking at it is the whole point, but saying so keeps the
        # preview from implying the app would ever put it on screen.
        print(_field("NOTE", "With nothing missing, ensure_ready() returns "
                             "immediately and shows no window at all. This "
                             "one is shown anyway, to look at."))
    print()


def _report(outcome: Outcome) -> None:
    """Everything printed once the window is gone."""
    print(f"\n{outcome}")
    # The flags are already per level and already guarded against a level that
    # was never offered (_on_secondary asks what is still missing rather than
    # assuming), so these two lines can be printed straight off the outcome.
    if outcome.optional_declined:
        print("ensure_ready() would have written llm_backend \"off\" here.")
    if outcome.translator_declined:
        print("ensure_ready() would have written translation_language \"\" here.")


def _pause(position: str) -> None:
    """Wait for Enter between states, unless there is no console to wait on."""
    try:
        input(f"\n[{position}] Enter for the next state, Ctrl+C to stop. ")
    except EOFError:
        # Output redirected to a file: the windows still have to appear, so
        # this walks on rather than treating a closed stdin as a refusal.
        print("\n(stdin is closed, continuing without pauses)")


def _tour() -> int:
    """Show every fabricated state in turn, announcing each one."""
    total = len(TOUR_STATES)
    print(f"Walking {total} states in order. --state real is not among them: "
          f"its keys are real, so its Download button downloads.")
    print(textwrap.fill(_PREFACE, width=78))
    skipped = []
    for number, state in enumerate(TOUR_STATES, start=1):
        position = f"{number}/{total} {state}"
        print(f"\n{'=' * 72}\n  {position}\n{'=' * 72}")
        plan = _plan(state)
        if plan is None:
            # Only half-failure can land here, and only on a machine with no
            # catalogue repo cached. Skipping is the whole handling: the state
            # would otherwise start a real download, which is what the tour
            # exists to avoid.
            print(f"Skipped. {_NO_CACHED_REPO}")
            skipped.append(state)
            continue
        _announce(state, plan)
        _report(FirstRunWindow(plan).run())
        # Collect the closed window here, on the main thread, instead of
        # letting it happen inside the next state - possibly on a download
        # worker, which is where a stale Tk object aborts the process (see the
        # note in the module docstring). first_run_window._close() now drops
        # the two Tcl variables itself, so this is belt and braces for the rest
        # of the graph; a tour that dies at state six loses the transcript of
        # the five that passed, which is worth one collection per window.
        gc.collect()
        if number < total:
            _pause(position)
    print(f"\n{'=' * 72}")
    if skipped:
        print(f"Not shown: {', '.join(skipped)}.")
    print("Tour done. The ninth state is 'real' and runs on its own: "
          "python tools/preview_first_run.py --state real")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show the first-run window in a chosen state.")
    parser.add_argument("--state", choices=STATES, default="both",
                        help="which situation to fabricate (default: both)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true",
                      help="print each state's plan and exit, showing nothing")
    mode.add_argument("--tour", action="store_true",
                      help="show every state except 'real' in turn, saying "
                           "what to press in each; ignores --state")
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

    if args.tour:
        return _tour()

    plan = _plan(args.state)
    if plan is None:
        print(_NO_CACHED_REPO, file=sys.stderr)
        return 1

    print(textwrap.fill(_PREFACE, width=78))
    print(f"\n{'=' * 72}\n  {args.state}\n{'=' * 72}")
    _announce(args.state, plan)
    _report(FirstRunWindow(plan).run())
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
