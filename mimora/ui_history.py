# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Attempt-history panel for the Mimora view layer.

A scrollable list of per-attempt rows (score chip, trend arrow, phrase, and an
expandable coloured word breakdown). The panel is pure rendering: the bounded
history model lives in the controller, which hands the full list to
:meth:`HistoryPanel.render` after every change (see TrainerView.render_history).
"""
import tkinter as tk

from mimora.ui_theme import (
    FONT_FAMILY,
    FONT_FAMILY_MONO,
    FONT_SIZE_BODY,
    FONT_SIZE_SUBTITLE,
    THEME,
    WHEEL_EVENTS,
    wheel_scroll_units,
)


class HistoryPanel:
    """Scrollable attempt-history list (canvas + inner frame of rows).

    Owns the history canvas, scrollbar and rows; rebuilt wholesale by
    :meth:`render` from the controller-owned entry list.
    """

    def __init__(self, parent):
        """Build and pack the panel into ``parent`` (fills remaining space).

        Packed with expand=True, so it must be created AFTER every fixed-height
        sibling that packs to the same side - it consumes the leftover space.
        """
        # A scrollable Canvas hosts one inner frame of per-attempt rows
        # (mirrors the SettingsWindow scroll pattern): the inner frame is kept
        # as wide as the canvas and the scrollregion as tall as the content.
        outer = tk.Frame(parent, bg=THEME["bg_main"])
        outer.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=(8, 16))

        self.canvas = tk.Canvas(
            outer, bg=THEME["bg_panel"], highlightthickness=1,
            highlightbackground=THEME["border"], bd=0)
        scrollbar = tk.Scrollbar(outer, orient=tk.VERTICAL,
                                 command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.rows_frame = tk.Frame(self.canvas, bg=THEME["bg_panel"])
        self._rows_window = self.canvas.create_window(
            (0, 0), window=self.rows_frame, anchor="nw")
        self.rows_frame.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(
            self._rows_window, width=e.width))
        # Wheel scrolling is bound directly on the canvas and (in render) on
        # every row widget, rather than via a global bind_all toggled on
        # Enter/Leave: the latter flickers off whenever the pointer crosses into
        # a child row, so the wheel would stop working over the rows themselves.
        for sequence in WHEEL_EVENTS:
            self.canvas.bind(sequence, self._on_mousewheel)
            self.rows_frame.bind(sequence, self._on_mousewheel)

        # Empty-state hint shown until the first attempt lands.
        self._show_placeholder()

    def _show_placeholder(self):
        self._placeholder = tk.Label(
            self.rows_frame, text="Your attempts will appear here.",
            font=(FONT_FAMILY, FONT_SIZE_BODY, "italic"), fg=THEME["text_muted"],
            bg=THEME["bg_panel"], anchor="w")
        self._placeholder.pack(fill=tk.X, padx=12, pady=12)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _on_mousewheel(self, event):
        """Scroll the history list with the wheel (see wheel_scroll_units)."""
        units = wheel_scroll_units(event)
        if units:
            self.canvas.yview_scroll(units, "units")

    def _bind_wheel(self, widget):
        """Bind wheel scrolling on ``widget`` and all its descendants.

        Rows and their labels are separate widgets, so the wheel is bound on each
        one; otherwise it would only scroll when the pointer sat on the bare
        canvas between rows.
        """
        for sequence in WHEEL_EVENTS:
            widget.bind(sequence, self._on_mousewheel)
        for child in widget.winfo_children():
            self._bind_wheel(child)

    def render(self, entries: list):
        """Rebuild the attempt-history rows from the controller-owned list.

        ``entries`` is the full bounded history (oldest first); each is a dict with
        a ``kind`` of "attempt", "unscored" or "error". Only the most recent take
        (attempt/unscored) is expanded by default; errors are flat red rows. The
        list is small (<= 10), so a full rebuild per update is cheap and keeps the
        render a pure function of the model.
        """
        for child in self.rows_frame.winfo_children():
            child.destroy()
        if not entries:
            self._show_placeholder()
            return
        # Newest first: the most recent take sits at the top, so a short list
        # fills from the top of the panel instead of leaving it blank above the
        # single bottom row.
        ordered = list(reversed(entries))
        # Expand the latest take (skip leading errors) by default: it is now the
        # first attempt/unscored entry from the top.
        expand_index = None
        for i, entry in enumerate(ordered):
            if entry.get("kind") in ("attempt", "unscored"):
                expand_index = i
                break
        for i, entry in enumerate(ordered):
            self._build_row(entry, expanded=(i == expand_index))
        # Wheel scrolling over any part of the freshly built rows.
        self._bind_wheel(self.rows_frame)
        # Keep the newest row (at the top) in view.
        self.canvas.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.yview_moveto(0.0)

    def _build_row(self, entry: dict, expanded: bool):
        """Build one history row: [score chip][trend][phrase] plus a detail area."""
        kind = entry.get("kind")
        row = tk.Frame(self.rows_frame, bg=THEME["bg_panel"])
        row.pack(fill=tk.X, padx=8, pady=(4, 0))

        if kind == "error":
            tk.Label(row, text=entry.get("text", ""), fg=THEME["bad"],
                     bg=THEME["bg_panel"], font=(FONT_FAMILY, FONT_SIZE_BODY), anchor="w",
                     justify=tk.LEFT, wraplength=520).pack(fill=tk.X, padx=4, pady=2)
            return

        header = tk.Frame(row, bg=THEME["bg_panel"], cursor="hand2")
        header.pack(fill=tk.X)

        # Score chip: a 1px outline in the verdict colour with the mark inside.
        # "score_text" is the mark exactly as the hero card showed it (the
        # "4+"-style grade); older/percent entries fall back to the numeric score.
        if kind == "attempt":
            chip_text = entry.get("score_text") or f"{entry.get('score', 0):.0f}"
            chip_color = entry.get("color") or THEME["text_dim"]
        else:  # unscored ("none" engine): no number, no verdict colour.
            chip_text, chip_color = "--", THEME["text_dim"]
        chip = tk.Label(header, text=chip_text, fg=chip_color, bg=THEME["bg_panel"],
                        font=(FONT_FAMILY, FONT_SIZE_SUBTITLE, "bold"), padx=8, pady=1,
                        highlightthickness=1, highlightbackground=chip_color)
        chip.pack(side=tk.LEFT)

        # Trend arrow vs the previous attempt of the same phrase (controller-set).
        trend = entry.get("trend")
        if trend == "up":
            arrow_text, arrow_color = "▲", THEME["good"]
        elif trend == "down":
            arrow_text, arrow_color = "▼", THEME["bad"]
        else:  # "same", or no earlier attempt to compare against.
            arrow_text, arrow_color = "–", THEME["text_dim"]
        tk.Label(header, text=arrow_text, fg=arrow_color, bg=THEME["bg_panel"],
                 font=(FONT_FAMILY, FONT_SIZE_SUBTITLE)).pack(side=tk.LEFT, padx=(8, 8))

        tk.Label(header, text=entry.get("phrase") or "-", fg=THEME["text_emph"],
                 bg=THEME["bg_panel"], font=(FONT_FAMILY, FONT_SIZE_SUBTITLE), anchor="w",
                 justify=tk.LEFT).pack(side=tk.LEFT, fill=tk.X, expand=True)

        detail = tk.Frame(row, bg=THEME["bg_panel"])
        detail_text = self._fill_detail(detail, entry)
        if expanded:
            detail.pack(fill=tk.X, padx=(4, 4), pady=(2, 4))
            self.canvas.after_idle(lambda t=detail_text: self._fit_text(t))

        # Clicking anywhere on the header toggles the detail below it.
        def toggle(event=None, d=detail, t=detail_text):
            if d.winfo_manager() == "pack":
                d.pack_forget()
            else:
                d.pack(fill=tk.X, padx=(4, 4), pady=(2, 4))
                self.canvas.after_idle(lambda: self._fit_text(t))
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        for widget in (header, *header.winfo_children()):
            widget.bind("<Button-1>", toggle)

    def _fill_detail(self, parent, entry: dict):
        """Fill a row's detail with the coloured word breakdown and problem sounds.

        Returns the inner read-only Text so its height can be fitted once visible.
        """
        txt = tk.Text(parent, bg=THEME["bg_panel"], fg=THEME["text"], bd=0,
                      highlightthickness=0, wrap=tk.WORD, height=1, cursor="arrow",
                      font=(FONT_FAMILY, FONT_SIZE_BODY), padx=4, pady=0, spacing3=2)
        txt.tag_configure("label", foreground=THEME["text_dim"])
        txt.tag_configure("good", foreground=THEME["good"])
        txt.tag_configure("ok", foreground=THEME["text_dim"])
        txt.tag_configure("bad", foreground=THEME["bad"], font=(FONT_FAMILY, FONT_SIZE_BODY, "bold"))
        txt.tag_configure("text", foreground=THEME["text_emph"])
        txt.tag_configure("mono", foreground=THEME["text_dim"], font=(FONT_FAMILY_MONO, FONT_SIZE_BODY))

        if entry.get("kind") == "unscored":
            txt.insert(tk.END, "Heard: ", "label")
            txt.insert(tk.END, "scoring is off - compare the takes by ear", "text")
        else:
            # "Words": the target phrase coloured by per-word correctness
            # (good/ok/bad, three-level "level" or the acoustic engine's boolean
            # "correct"); falls back to the plain phrase if tags are empty.
            txt.insert(tk.END, "Words: ", "label")
            ref_words = entry.get("reference_words") or []
            if ref_words:
                for word in ref_words:
                    level = word.get("level")
                    if level in ("good", "ok", "bad"):
                        tag = level
                    else:
                        tag = "good" if word.get("correct") else "bad"
                    txt.insert(tk.END, word["word"] + " ", tag)
            else:
                txt.insert(tk.END, entry.get("phrase") or "-", "text")
            phonemes = entry.get("phonemes") or []
            if phonemes:
                txt.insert(tk.END, "\nWork on: ", "label")
                for phoneme in phonemes:
                    txt.insert(tk.END, f"/{phoneme}/ ", "mono")
        txt.configure(state=tk.DISABLED)
        txt.pack(fill=tk.X)
        return txt

    def _fit_text(self, txt):
        """Size a detail Text to its number of logical lines.

        Height is the logical line count (the coloured "Words" line plus an
        optional "Work on" line), not the wrapped display-line count: a
        display-line count measured before the canvas-embedded Text gets its real
        width balloons (every word wraps onto its own line), which would leave a
        tall gap under the expanded row and push the older rows to the bottom.
        Logical lines are width-independent, so the row height stays stable.
        """
        if not txt.winfo_exists():
            return
        lines = int(txt.index("end-1c").split(".")[0])
        txt.configure(height=max(1, lines))
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
