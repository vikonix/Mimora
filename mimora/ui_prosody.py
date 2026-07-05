# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Prosody panel for the Mimora view layer.

Pitch (F0) and energy sparklines, you vs reference, folded under one collapse
header. One flag (show_prosody) shows/hides the whole body and (in the
controller) gates the expensive prosody computation. The last analysis result
is cached here so the canvases can redraw on resize.
"""
import tkinter as tk
from typing import Callable

from mimora import config, prosody_utils
from mimora.ui_theme import FONT_FAMILY, THEME


class ProsodyPanel:
    """Collapsible pitch/energy sparkline panel.

    Owns the caption toggle, the body frame and both canvases, plus the cached
    prosody dict they redraw from. The collapse flag routes through
    ``on_prosody_toggled`` so the controller can apply and persist it exactly
    like the other visibility toggles.
    """

    def __init__(self, root, on_prosody_toggled: Callable[[], None]):
        self.root = root
        self._on_prosody_toggled = on_prosody_toggled
        # Last analysis prosody, cached so the canvases can redraw on resize.
        self._last_prosody = None

        self.frame = tk.Frame(root, bg=THEME["bg_main"])
        self.frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(0, 8))

        # Header caption doubles as the collapse toggle (same idiom as the
        # practice-text caption): clicking it expands/collapses the body; the
        # arrow prefix mirrors the state.
        self.show_prosody = tk.BooleanVar(value=config.SHOW_PROSODY)
        self._caption = tk.Button(
            self.frame, text="▾ Intonation & stress",
            command=self._on_caption_clicked,
            font=(FONT_FAMILY, 9, "bold"), fg=THEME["accent"], bg=THEME["bg_main"],
            activebackground=THEME["bg_main"], activeforeground=THEME["accent"],
            bd=0, padx=0, pady=0, cursor="hand2")
        self._caption.pack(anchor=tk.W)

        # Body: everything hidden while collapsed. toggle() packs/forgets it as
        # a whole, so the individual children are packed once here.
        self.body = tk.Frame(self.frame, bg=THEME["bg_main"])

        # Pitch chart: plain title label above its canvas, sharing its row with
        # the line legend at the right edge. The legend lives inside the body
        # (only meaningful when the charts are visible), and this row saves it
        # a line of vertical space.
        pitch_title_row = tk.Frame(self.body, bg=THEME["bg_main"])
        pitch_title_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(pitch_title_row, text="Pitch - intonation (semitones vs your median)",
                 font=(FONT_FAMILY, 8), fg=THEME["text_dim"], bg=THEME["bg_main"]).pack(side=tk.LEFT)
        # Packed right-to-left: "reference" first so "you" lands left of it,
        # keeping the reading order "● you ● reference".
        tk.Label(pitch_title_row, text="● reference", font=(FONT_FAMILY, 8),
                 fg=THEME["reference"], bg=THEME["bg_main"]).pack(side=tk.RIGHT)
        tk.Label(pitch_title_row, text="● you", font=(FONT_FAMILY, 8),
                 fg=THEME["info"], bg=THEME["bg_main"]).pack(side=tk.RIGHT, padx=(0, 8))
        self.f0_canvas = tk.Canvas(self.body, height=46, bg=THEME["bg_panel"],
                                   highlightthickness=1, highlightbackground=THEME["border"])
        self.f0_canvas.pack(fill=tk.X, pady=(0, 4))

        # Energy chart: same, below the pitch chart.
        tk.Label(self.body, text="Energy - stress pattern",
                 font=(FONT_FAMILY, 8), fg=THEME["text_dim"], bg=THEME["bg_main"]).pack(anchor=tk.W)
        self.en_canvas = tk.Canvas(self.body, height=46, bg=THEME["bg_panel"],
                                   highlightthickness=1, highlightbackground=THEME["border"])
        self.en_canvas.pack(fill=tk.X)

        # Reading hint: horizontal axis is time (stretched to equal width for both),
        # so the goal is matching the *shape* of the reference, not exact overlap.
        tk.Label(self.body,
                 text="Time runs (stretched to equal width). Aim to match the reference shape.",
                 font=(FONT_FAMILY, 8), fg=THEME["text_dim"], bg=THEME["bg_main"],
                 wraplength=540, justify=tk.LEFT).pack(anchor=tk.W, pady=(3, 0))

        # fill=X canvases change width on resize, so redraw from the cached prosody.
        self.f0_canvas.bind("<Configure>", lambda e: self.redraw())
        self.en_canvas.bind("<Configure>", lambda e: self.redraw())

    # ------------------------------------------------------------------
    # Collapse toggle
    # ------------------------------------------------------------------
    def _on_caption_clicked(self):
        """Flip the prosody collapse flag and route it through the controller.

        Mirrors the practice-text caption: the Button owns no variable, so the
        panel flips show_prosody here; the controller then applies (toggle) and
        persists it, and updates the worker-visible compute flag.
        """
        self.show_prosody.set(not self.show_prosody.get())
        self._on_prosody_toggled()

    def toggle(self):
        """Show/hide the whole prosody body to match the show_prosody flag."""
        # Return focus to the window so the spacebar record toggle keeps working.
        self.root.focus_set()
        expanded = self.show_prosody.get()
        self._caption.config(
            text="▾ Intonation & stress" if expanded else "▸ Intonation & stress")
        shown = self.body.winfo_manager() == "pack"
        if expanded and not shown:
            self.body.pack(fill=tk.X)
        elif not expanded and shown:
            self.body.pack_forget()

    def set_show(self, flag: bool):
        self.show_prosody.set(bool(flag))

    def get_show(self) -> bool:
        return bool(self.show_prosody.get())

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def set_prosody(self, prosody: dict):
        """Cache a new analysis result's prosody (may be empty) for drawing."""
        self._last_prosody = prosody

    def clear(self):
        """Drop the cached prosody and erase both canvases."""
        self._last_prosody = None
        self.f0_canvas.delete("all")
        self.en_canvas.delete("all")

    def _draw(self, canvas, series):
        """Draw contours onto ``canvas``. ``series`` is a list of (values, color).

        All series share one vertical scale so they are directly comparable.
        No-op until the canvas has been laid out (winfo_width > 1).
        """
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width <= 1 or height <= 1:
            return
        pad_x, pad_y = 4, 4
        plot_w = max(1, width - 2 * pad_x)
        plot_h = max(1, height - 2 * pad_y)

        all_values = [v for values, _ in series for v in values]
        if not all_values:
            return
        lo, hi = min(all_values), max(all_values)
        span = (hi - lo) or 1.0

        for values, color in series:
            points = prosody_utils.resample_series(values)
            if len(points) < 2:
                continue
            coords = []
            for i, value in enumerate(points):
                x = pad_x + (i / (len(points) - 1)) * plot_w
                y = pad_y + (1 - (value - lo) / span) * plot_h
                coords.extend((x, y))
            canvas.create_line(*coords, fill=color, width=3, smooth=True)

    def redraw(self):
        """Redraw both canvases from the cached result (e.g. after resize)."""
        prosody = self._last_prosody
        if not prosody:
            return
        # Normalize each pitch contour to semitones vs its own median so the
        # reference and the user (different vocal registers) become directly
        # comparable in shape; _draw's shared scale then centres both on 0 ST.
        # Energy is already per-utterance scaled, so it is left as-is.
        self._draw(self.f0_canvas, [
            (prosody_utils.to_semitones(prosody.get("ref_f0", [])), THEME["reference"]),  # reference
            (prosody_utils.to_semitones(prosody.get("f0", [])), THEME["info"]),           # you
        ])
        self._draw(self.en_canvas, [
            (prosody.get("ref_energy", []), THEME["reference"]),
            (prosody.get("energy", []), THEME["info"]),
        ])
