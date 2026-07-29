# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""The first-run download dialog: one window, asked before the app is built.

`ensure_ready()` is the whole public surface. It builds the plan
(mimora/first_run.py), returns immediately when nothing is missing, and
otherwise shows one window that states the volumes in numbers, downloads what
the user agreed to, and records a refusal where the user will later look for it.

Why its own Tk root
-------------------
The question has to be answered BEFORE PronunciationTrainerGUI.__init__ picks
between LLMManager and SourceTextPhraseProvider (main.py, the llm_backend
branch), because a refusal changes that choice. Running inside __init__ would
mean aborting a half-built object when the user quits; a separate, short-lived
root avoids that entirely and keeps main.py's constructor untouched. Tk is
happy to have a second root after the first is destroyed, as long as the two do
not overlap - and they cannot, because this returns before the app starts.

The palette and FlatButton come from mimora/ui_theme.py rather than mimora/ui.py
so this window depends on the theme, not on the whole view stack.

**No ttk widget may be built here.** ttkbootstrap overrides the ttk widget
constructor to fetch its Style singleton, and that singleton caches the root it
was first created under. One ttk.Progressbar in this window was therefore
enough to bind ttkbootstrap to a root that is destroyed moments later, and the
app's own `ttk.Style(theme=...)` in ui.py then died with "application has been
destroyed". Classic Tk widgets are unaffected (ui_theme disables the classic
autostyle hook on import), so the progress bar here is drawn on a Canvas -
which is what ProgressRing and FaceWidget do anyway.

What the two levels mean here
-----------------------------
* The required level is a statement, not a question: without it there is no
  reference audio or no scoring, so the alternatives are "Download" and "Quit".
* The optional level is the one real choice, and refusing it is a working
  configuration. The refusal is written as ``llm_backend: "off"`` in
  settings.json - the outcome itself rather than a "the user said no" marker,
  so the settings window shows exactly how the app now behaves and the same
  control that already edits llm_backend is where the user changes their mind.
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from tkinter import messagebox
from typing import Container, NamedTuple, Optional, Sequence

from mimora import config, first_run, first_run_download
from mimora.ui_theme import (FONT_FAMILY, FONT_SIZE_BODY, FONT_SIZE_CAPTION,
                             FONT_SIZE_SMALL, FONT_SIZE_TITLE, THEME,
                             FlatButton)

log = logging.getLogger(__name__)

# How often the Tk side reads the shared progress variable. The worker writes
# thousands of times a second; this is what decouples the two.
_POLL_MS = 100

# Progress bar geometry. The width is only the requested size - the canvas is
# packed with fill=X and the real width is read back when the bar is drawn.
_BAR_WIDTH = 420
_BAR_HEIGHT = 10

_INTERRUPT_TEXT = (
    "The download is not finished. What has been downloaded is kept, but part "
    "of it will have to be fetched again on the next start.\n\nQuit anyway?")

# Shown instead of the optional checkbox when that level is empty because
# downloading cannot produce a usable llama-server (Plan.llama_server_blocked).
# The app would otherwise say nothing about a backend it cannot start, and
# neither way out of either state is guessable. One text per reason: they have
# nothing in common except that the checkbox is missing.
_BLOCKED_TEXT = {
    first_run.BLOCKED_NO_BUILD: (
        "No llama.cpp server build is available for this machine, so the "
        "local chat model cannot be offered here. Either point "
        "\"llama_server_path\" in config/settings.json at a llama-server you "
        "build yourself, or set \"llm_backend\" to \"lm-studio\" and generate "
        "phrases with LM Studio."),
    first_run.BLOCKED_BAD_SETTING: (
        "\"llama_server_path\" in config/settings.json points at a file that "
        "is not there, so the local chat model cannot start. Downloading one "
        "is not offered because it would be installed elsewhere and this "
        "setting would still win: fix the path, or clear it to use the build "
        "Mimora installs itself."),
}


class Outcome(NamedTuple):
    """What the window ended up doing, for ensure_ready() to act on.

    Deliberately only the two facts a caller acts on. The list of components
    actually fetched is not among them: nothing downstream needs it, because
    where the binary ended up is answered by config.resolve_llama_server_path()
    at the moment the command line is built, not by remembering the download.
    """

    quit_requested: bool
    optional_declined: bool


def _format_size(size_mb: Optional[int]) -> str:
    """Human size. Gigabytes once the number stops being readable in MB."""
    if not size_mb:
        return "0 MB"
    if size_mb >= 1000:
        return f"{size_mb / 1000:.1f} GB"
    return f"{size_mb} MB"


def still_missing(
        components: Sequence[first_run.Component],
        fetched: Container[str],
) -> tuple[first_run.Component, ...]:
    """*components* minus the ones whose key is in *fetched*.

    A first_run.Plan describes the machine as it was when it was built and is
    never rebuilt, so Plan.missing_required does NOT shrink as components
    arrive. Every question the window asks after a download has started is
    about what is missing *now*: which label the second button carries, what a
    Retry should fetch, and whether leaving is a refusal or a quit.

    Free function rather than a method because that difference is the whole of
    the logic and it is worth being able to test without a Tk root - a download
    that half succeeded and then failed is otherwise reachable only by pulling
    the network out at the right second.
    """
    return tuple(c for c in components if c.key not in fetched)


class FirstRunWindow:
    """The dialog. Owns its root and returns an :class:`Outcome` from run()."""

    def __init__(self, plan: first_run.Plan) -> None:
        self.plan = plan
        self._state: Optional[first_run_download.ProgressState] = None
        self._thread: Optional[threading.Thread] = None
        # Default outcome: quit. Every way of dismissing this window that is
        # not an explicit answer - the close button, Escape, a killed window
        # manager - has to end in "do not start", never in a silent start with
        # half the models missing.
        self._outcome = Outcome(quit_requested=True, optional_declined=False)
        # Keys of the components fetched since this window opened. Accumulated
        # across attempts rather than read from the current ProgressState,
        # because Retry starts a fresh one: without this, a component that
        # arrived on the first attempt would look missing again on the second.
        self._fetched: set[str] = set()
        # Set before destroy() so the pending 100 ms poll does not fire into a
        # dead interpreter and print a TclError traceback on the way out.
        self._closed = False

        self.root = tk.Tk()
        self.root.title("Mimora - first run")
        self.root.configure(bg=THEME["bg_main"])
        self.root.resizable(False, False)
        # Closing the window is a supported action at every moment, including
        # mid-download: a modal that swallows WM_DELETE_WINDOW would turn a
        # ten-minute download into a ten-minute hang.
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Escape>", lambda _e: self._on_close())

        self._wants_optional = tk.BooleanVar(
            # Checked by default: llm_backend is "llama-server", so this is
            # what the configuration already says the app should do. Refusing
            # stays a deliberate act rather than the result of not noticing a
            # checkbox.
            value=bool(plan.missing_optional))

        self._build()
        self._centre()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _label(self, parent, text, *, size=FONT_SIZE_BODY, fg="text",
               bold=False, **kwargs):
        weight = "bold" if bold else "normal"
        return tk.Label(parent, text=text, bg=THEME["bg_main"], fg=THEME[fg],
                        font=(FONT_FAMILY, size, weight), justify=tk.LEFT,
                        anchor="w", **kwargs)

    def _build(self) -> None:
        outer = tk.Frame(self.root, bg=THEME["bg_main"], padx=24, pady=20)
        outer.pack(fill=tk.BOTH, expand=True)

        self._label(outer, "Mimora needs to download some files",
                    size=FONT_SIZE_TITLE, fg="text_bright",
                    bold=True).pack(fill=tk.X)

        self._question = tk.Frame(outer, bg=THEME["bg_main"])
        self._question.pack(fill=tk.X, pady=(14, 0))
        self._build_question(self._question)

        # Progress lives in its own frame so the question can be replaced
        # wholesale when the download starts, instead of growing the window.
        self._progress_frame = tk.Frame(outer, bg=THEME["bg_main"])
        self._status = self._label(self._progress_frame, "", fg="text_muted",
                                   size=FONT_SIZE_SMALL)
        self._status.pack(fill=tk.X)
        # Hand-drawn rather than a ttk.Progressbar - see the module docstring.
        self._bar = tk.Canvas(self._progress_frame, height=_BAR_HEIGHT,
                              width=_BAR_WIDTH, bg=THEME["bg_panel"],
                              highlightthickness=0, bd=0)
        self._bar.pack(fill=tk.X, pady=(6, 0))
        self._bar_fill = self._bar.create_rectangle(
            0, 0, 0, _BAR_HEIGHT, fill=THEME["accent"], width=0)
        # Wrapped, because this line carries two very different things: the
        # short "N of M MB" during a download, and an exception's own text
        # after a failure. The latter is arbitrary - a URL, a stack of nested
        # messages - and an unwrapped label makes a non-resizable window as
        # wide as its longest line, which on a first failure meant a window
        # wider than the screen is comfortable with.
        self._amount = self._label(self._progress_frame, "",
                                   fg="text_dim", size=FONT_SIZE_CAPTION,
                                   wraplength=_BAR_WIDTH)
        self._amount.pack(fill=tk.X, pady=(4, 0))

        # Colours follow config.py's palette, where bg_button is the primary
        # action and bg_accent the quiet one, as in the settings window. Worth
        # stating because the two were swapped here at first, and the button
        # that ended up looking primary was the one that writes
        # llm_backend "off".
        buttons = tk.Frame(outer, bg=THEME["bg_main"])
        buttons.pack(fill=tk.X, pady=(18, 0))
        self._secondary = FlatButton(
            buttons, text=self._secondary_text(), command=self._on_secondary,
            bg=THEME["bg_accent"], fg=THEME["text_accent"],
            activebackground=THEME["bg_accent_active"],
            activeforeground=THEME["text_bright"],
            font=(FONT_FAMILY, FONT_SIZE_SMALL), padx=18, pady=7)
        self._secondary.pack(side=tk.RIGHT)
        self._primary = FlatButton(
            buttons, text="Download", command=self._on_primary,
            bg=THEME["bg_button"], fg=THEME["text_button"],
            activebackground=THEME["bg_button_active"],
            activeforeground=THEME["text"],
            font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"), padx=18, pady=7)
        self._primary.pack(side=tk.RIGHT, padx=(0, 10))
        self._primary.focus_set()

    def _build_question(self, parent) -> None:
        plan = self.plan
        if plan.missing_required:
            self._label(parent,
                        f"Required before the app can start: "
                        f"{_format_size(plan.missing_required_mb)}",
                        bold=True).pack(fill=tk.X)
            self._bullets(parent, plan.missing_required)

        if plan.missing_optional:
            pad = (14, 0) if plan.missing_required else (0, 0)
            box = tk.Frame(parent, bg=THEME["bg_main"])
            box.pack(fill=tk.X, pady=pad)
            tk.Checkbutton(
                box, variable=self._wants_optional, command=self._update_total,
                text=("Also download the local chat model "
                      f"({_format_size(plan.missing_optional_mb)})"),
                bg=THEME["bg_main"], fg=THEME["text"],
                activebackground=THEME["bg_main"],
                activeforeground=THEME["text_bright"],
                selectcolor=THEME["bg_panel"], anchor="w",
                highlightthickness=0, bd=0,
                font=(FONT_FAMILY, FONT_SIZE_BODY, "bold")).pack(fill=tk.X)
            self._label(box,
                        "Without it, practice phrases are taken from your own "
                        "text, sentence by sentence.",
                        fg="text_muted", size=FONT_SIZE_CAPTION,
                        padx=22).pack(fill=tk.X)
            self._bullets(box, plan.missing_optional)
        elif plan.llama_server_blocked:
            # Not an alternative to the block above so much as its absence
            # explained: the optional level is empty because no download would
            # give this machine a working server, and staying silent would
            # leave the user with an LLM backend that fails at every start for
            # no stated reason.
            pad = (14, 0) if plan.missing_required else (0, 0)
            self._label(parent, _BLOCKED_TEXT[plan.llama_server_blocked],
                        fg="text_muted", size=FONT_SIZE_CAPTION,
                        wraplength=_BAR_WIDTH).pack(fill=tk.X, pady=pad)

        self._total = self._label(parent, "", bold=True, fg="text_bright")
        self._total.pack(fill=tk.X, pady=(14, 0))
        self._update_total()

    def _bullets(self, parent, components: Sequence[first_run.Component]) -> None:
        for component in components:
            self._label(parent,
                        f"•  {component.label} - "
                        f"{_format_size(component.size_mb)}",
                        fg="text_muted", size=FONT_SIZE_CAPTION,
                        padx=10).pack(fill=tk.X)

    def _centre(self) -> None:
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() - width) // 2
        y = (self.root.winfo_screenheight() - height) // 3
        self.root.geometry(f"+{x}+{y}")

    # ------------------------------------------------------------------
    # Question state
    # ------------------------------------------------------------------

    def _missing_required(self) -> tuple[first_run.Component, ...]:
        """Required components this window has not managed to fetch yet."""
        return still_missing(self.plan.missing_required, self._fetched)

    def _secondary_text(self) -> str:
        # Refusing the required level leaves nothing to run, so there the only
        # other action is leaving. With only the optional level missing,
        # refusing is a working configuration and must not read like an exit.
        #
        # Read through _missing_required, not off the plan: a run that fetched
        # the required level and then failed on the optional one has a working
        # configuration to offer, and offering "Quit" there would throw the
        # finished download away.
        return "Quit" if self._missing_required() else "Skip"

    def _selected(self) -> tuple[first_run.Component, ...]:
        components = self._missing_required()
        if self._wants_optional.get():
            components += still_missing(self.plan.missing_optional,
                                        self._fetched)
        return components

    def _update_total(self) -> None:
        total = first_run.total_mb(self._selected())
        self._total.config(text=f"Total download: {_format_size(total)}")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_primary(self) -> None:
        """"Download", or "Retry" after a failure."""
        selection = self._selected()
        if not selection:
            # Everything unchecked: the same thing as pressing Skip.
            self._on_secondary()
            return
        self._question.pack_forget()
        self._progress_frame.pack(fill=tk.X, pady=(14, 0))
        self._primary.config(state=tk.DISABLED, text="Download")
        self._secondary.config(text="Quit")
        self._status.config(fg=THEME["text_muted"])   # cleared after a retry
        self._state, self._thread = first_run_download.start(selection)
        self.root.after(_POLL_MS, self._poll)

    def _on_secondary(self) -> None:
        """"Quit" or "Skip", depending on what is missing.

        Pressing it after a failed download counts as a refusal, not as the
        failure: the user was shown the error and chose to go on without the
        optional level. That is the one path where a failure ends in
        llm_backend "off", and it ends there because of the choice.
        """
        if self._downloading():
            self._on_close()
            return
        if self._missing_required():
            # No working outcome without it, so the only other action is
            # leaving; nothing is recorded, and the question returns next time.
            self._outcome = Outcome(True, False)
        else:
            # Everything the run needs is on disk, whether it was there when
            # this window opened or arrived just now. Leaving without the
            # optional level is a working configuration, so it is recorded as
            # the refusal it is instead of quitting.
            self._outcome = Outcome(False, True)
        self._close()

    def run(self) -> Outcome:
        """Show the window and block until it closes."""
        self.root.mainloop()
        return self._outcome

    def _on_close(self) -> None:
        """The window button and Escape. Confirms only mid-download."""
        if self._downloading() and not messagebox.askyesno(
                "Mimora", _INTERRUPT_TEXT, parent=self.root):
            return
        self._outcome = Outcome(quit_requested=True, optional_declined=False)
        self._close()

    def _close(self) -> None:
        self._closed = True
        self.root.destroy()

    def _downloading(self) -> bool:
        return (self._thread is not None and self._thread.is_alive()
                and self._state is not None
                and not self._state.snapshot().finished)

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    def _poll(self) -> None:
        if self._closed or self._state is None:
            return
        snapshot = self._state.snapshot()
        # Banked before anything else looks at the snapshot: a failure below
        # ends this poll, and by then the components that did arrive have to be
        # known, or the error branch would still believe the original plan.
        self._fetched.update(snapshot.completed_keys)
        # winfo_width() rather than the requested width: the canvas is packed
        # with fill=X, so its real width is whatever the window settled on.
        width = self._bar.winfo_width() or _BAR_WIDTH
        self._bar.coords(self._bar_fill, 0, 0,
                         snapshot.fraction * width, _BAR_HEIGHT)
        self._status.config(text=snapshot.label)
        self._amount.config(
            text=f"{snapshot.done_bytes // first_run_download.BYTES_PER_MB} "
                 f"of {snapshot.total_bytes // first_run_download.BYTES_PER_MB} MB")

        if not snapshot.finished:
            self.root.after(_POLL_MS, self._poll)
            return

        if snapshot.error:
            self._show_error(snapshot.error)
            return

        self._outcome = Outcome(
            quit_requested=False,
            # Unchecked optional level: the refusal is recorded even though the
            # required part was downloaded successfully.
            optional_declined=(bool(self.plan.missing_optional)
                               and not self._wants_optional.get()))
        self._close()

    def _show_error(self, message: str) -> None:
        """A failure is not a refusal: offer the same choice again.

        Nothing is written to settings.json here. Somebody whose wifi dropped
        must not find their configuration permanently switched to another mode;
        a refusal is a choice, a failure is an accident, and only the first is
        persisted.
        """
        self._status.config(text="Download failed", fg=THEME["bad"])
        self._amount.config(text=message[:300])
        self._primary.config(state=tk.NORMAL, text="Retry")
        # Recomputed rather than restored: components run in order, so a
        # failure on the optional level can leave the required one complete,
        # and then the way out is "Skip" (start anyway, llm_backend "off")
        # rather than "Quit".
        self._secondary.config(text=self._secondary_text())
        # Retrying builds a fresh ProgressState, and the fetchers skip what is
        # already on disk, so a retry only pays for what actually failed.


def ensure_ready() -> bool:
    """Make the machine runnable, asking first. False means "the user quit".

    Called before the GUI exists, so it may block: the checks are a handful of
    stat calls and the window runs its own event loop.
    """
    plan = first_run.build_plan()

    if plan.llama_server_blocked:
        # No download would give this machine a server it would launch, so
        # there is nothing this window could offer for the LLM backend. Logged
        # before the early return below rather than after it: a blocked backend
        # is exactly what makes the optional level empty, so a check placed
        # after "is anything missing?" could only ever run when something else
        # was missing too - and would say nothing in the commonest case, a
        # machine with every model already cached.
        #
        # Only logged. The advice a user can act on is given where it bites -
        # in main.py._server_failure_message() - because this state repeats at
        # every start, and a window that repeats with it would be a modal
        # standing between the user and an app that does start.
        log.info("Nothing can be offered for the llama-server backend (%s); "
                 "starting without it.", plan.llama_server_blocked)

    if not plan.missing_required and not plan.missing_optional:
        return True

    outcome = FirstRunWindow(plan).run()

    if outcome.optional_declined:
        # The result, not the refusal: one fact in one place, visible and
        # reversible in the settings window where llm_backend already lives.
        log.info("Optional download declined - switching llm_backend to 'off'.")
        config.save_user_setting("llm_backend", "off")
        config.LLM_BACKEND = "off"

    # A binary downloaded just now needs no announcement: llm_server_ctl
    # resolves the path when it builds the command line, so it sees whatever is
    # on disk by then (config.resolve_llama_server_path).
    return not outcome.quit_requested
