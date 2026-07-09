# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Circular session-progress widget for the Mimora hero card.

A ring gauge that shows the running session average (e.g. 3.8 out of 5) with
the distinct-phrase count underneath. It sits on the right of the hero score
row, mirroring the talking face on the left.

The file is named generically on purpose: ``ProgressRing`` is the *current*
shape of this indicator. If the design later moves to a bar or a different
gauge, the class inside changes while the import path (and the widget's role in
the layout) stays the same - so callers in ``ui_hero.py`` are not disturbed.

Rendering split (why this is a Frame, not a bare Canvas like FaceWidget):

  * The *ring* (track + coloured fill arc) is drawn by Pillow at
    ``_SUPERSAMPLE`` times the widget size and downscaled with LANCZOS, so the
    arc is antialiased - Tk Canvas arc primitives are not. Same technique as
    ``face_widget.py``.

  * The *numbers* (the big average, the "/ 5" suffix, the phrase count) are
    plain Tk text, not baked into the Pillow image: Tk renders them with the
    app font and the OS text antialiaser, which avoids shipping a TrueType font
    file just to draw a couple of digits. The average/suffix are Canvas text
    items centred in the ring; the count is a Label under the canvas.

The widget is passive: the hero card calls :meth:`set_progress` whenever the
session tally changes; there is no animation loop.
"""

from __future__ import annotations

import math
import tkinter as tk

from PIL import Image, ImageDraw, ImageTk

from mimora.ui_theme import FONT_FAMILY, FONT_SIZE_CAPTION, THEME

# Frames are drawn this many times larger than the widget and downscaled with
# LANCZOS - this is where the ring's antialiasing comes from (see face_widget).
_SUPERSAMPLE = 4


class ProgressRing(tk.Frame):
    """A ring gauge: session average inside, phrase count beneath.

    Pack/grid it like any widget. Starts in the empty state ("0 phrases", no
    fill) until the first :meth:`set_progress` call.
    """

    def __init__(
        self,
        parent: tk.Misc,
        size: int = 82,
        *,
        bg: str = THEME["bg_card"],
        track_color: str = THEME["bg_accent"],
        fill_color: str = THEME["accent"],
        value_color: str = THEME["text_bright"],
        sub_color: str = THEME["text_dim"],
    ) -> None:
        """Create the ring.

        Args:
            parent: parent Tk widget.
            size: side of the square ring canvas, in pixels. The stroke and the
                centred text scale from this.
            bg: background colour behind the ring; Pillow frames are drawn on it
                so the antialiased arc edges blend into the panel exactly.
            track_color: colour of the full background ring.
            fill_color: colour of the progress arc. A single neutral brand
                colour on purpose - the ring shows a cumulative session average,
                not a pass/fail verdict, so it does not switch good/bad.
            value_color: colour of the big average number.
            sub_color: colour of the "/ max" suffix and the phrase count.
        """
        super().__init__(parent, bg=bg)
        self._size = size
        self._c_bg = self._rgb(bg)
        self._c_track = self._rgb(track_color)
        self._c_fill = self._rgb(fill_color)

        # Ring + centred numbers live on the canvas; the count sits below it.
        self.canvas = tk.Canvas(self, width=size, height=size, bg=bg,
                                highlightthickness=0, bd=0)
        self.canvas.pack()
        self._img_item = self.canvas.create_image(size / 2, size / 2)
        self._photo: ImageTk.PhotoImage | None = None  # keep-alive reference

        # Big average number, with the "/ max" suffix just below it. Font sizes
        # scale from the widget so the ring stays legible at other sizes.
        value_font = (FONT_FAMILY, max(10, int(size * 0.25)), "bold")
        sub_font = (FONT_FAMILY, max(7, int(size * 0.13)))
        self._value_item = self.canvas.create_text(
            size / 2, size * 0.44, text="", fill=value_color, font=value_font)
        self._max_item = self.canvas.create_text(
            size / 2, size * 0.66, text="", fill=sub_color, font=sub_font)

        self.count_label = tk.Label(self, text="0 phrases",
                                    font=(FONT_FAMILY, FONT_SIZE_CAPTION),
                                    fg=sub_color, bg=bg)
        self.count_label.pack(pady=(2, 0))

        # State: no average yet. ``_maximum`` decides the number format
        # (one decimal on the 0-5 grade axis, whole numbers on a 0-100 scale).
        self._value: float | None = None
        self._maximum: float = 5.0
        self._count: int = 0

        # Canvas size in px; updated on <Configure>. Named _cw/_ch (not _w/_h -
        # tkinter reserves _w for the widget's Tcl path name).
        self._cw = size
        self._ch = size
        self.canvas.bind("<Configure>", self._on_resize)
        self._render()

    # -- colours ----------------------------------------------------------

    def _rgb(self, color: str) -> tuple:
        """Resolve a Tk colour string to an 8-bit RGB tuple."""
        r, g, b = self.winfo_rgb(color)
        return (r // 257, g // 257, b // 257)

    # -- public API -------------------------------------------------------

    def set_progress(self, value: float | None, maximum: float,
                     count: int) -> None:
        """Update the ring to a new session average and phrase count.

        Args:
            value: the running average, or ``None`` for the empty state (no
                fill, a dim placeholder in the centre).
            maximum: the top of the scale (5 for the graded phoneme engine, 100
                for a raw-percent engine). Also selects the number format.
            count: distinct phrases practised this run, shown under the ring.
        """
        self._value = value
        self._maximum = maximum if maximum > 0 else 5.0
        self._count = max(0, count)
        self._render()

    def reset(self) -> None:
        """Return to the empty state (no average, zero phrases)."""
        self.set_progress(None, self._maximum, 0)

    # -- geometry / resize ------------------------------------------------

    def _on_resize(self, event: "tk.Event") -> None:
        if event.width == self._cw and event.height == self._ch:
            return
        self._cw, self._ch = event.width, event.height
        self.canvas.coords(self._img_item, self._cw / 2, self._ch / 2)
        self._render()

    # -- rendering --------------------------------------------------------

    def _render(self) -> None:
        """Redraw the ring image and refresh the centred text and count.

        No-op for the ring image before the canvas is laid out (width <= 1);
        the <Configure> binding calls back once a real size is known. The text
        items are cheap and always updated.
        """
        graded = self._maximum <= 5
        if self._value is None:
            self.canvas.itemconfigure(self._value_item, text="–")  # en dash
            self.canvas.itemconfigure(self._max_item, text="")
        else:
            fmt = f"{self._value:.1f}" if graded else f"{self._value:.0f}"
            self.canvas.itemconfigure(self._value_item, text=fmt)
            self.canvas.itemconfigure(self._max_item, text=f"/ {self._maximum:g}")

        plural = "" if self._count == 1 else "s"
        self.count_label.configure(text=f"{self._count} phrase{plural}")

        dim = min(self._cw, self._ch)
        if dim <= 1:  # not laid out yet
            return
        fraction = 0.0
        if self._value is not None and self._maximum > 0:
            fraction = max(0.0, min(1.0, self._value / self._maximum))
        self._photo = self._render_ring(dim, fraction)
        self.canvas.itemconfigure(self._img_item, image=self._photo)

    def _render_ring(self, dim: int, fraction: float) -> ImageTk.PhotoImage:
        """Render the track + fill arc for ``fraction`` (0..1), antialiased.

        Drawn at ``_SUPERSAMPLE`` times ``dim`` on the opaque ``bg`` colour, then
        downscaled with LANCZOS. The fill arc starts at 12 o'clock and sweeps
        clockwise; its ends get round caps (small discs) to match the mock.
        """
        s = dim * _SUPERSAMPLE
        img = Image.new("RGB", (s, s), self._c_bg)
        d = ImageDraw.Draw(img)

        stroke = max(2, int(s * 0.085))
        r = s * 0.42
        cx = cy = s / 2
        box = (cx - r, cy - r, cx + r, cy + r)

        # Pillow draws arc width *inward* from the bounding box: the band runs
        # from radius r-stroke (inner) out to r (outer), so its centreline is at
        # r-stroke/2. The round caps must sit on that centreline, not on r, or
        # they bulge outside the ring.
        r_center = r - stroke / 2.0

        # Full background ring, then the progress arc on top of it.
        d.arc(box, 0, 360, fill=self._c_track, width=stroke)
        if fraction > 0:
            start = -90.0                       # 12 o'clock
            end = start + 360.0 * fraction      # clockwise sweep
            d.arc(box, start, end, fill=self._c_fill, width=stroke)
            self._round_cap(d, cx, cy, r_center, start, stroke)
            self._round_cap(d, cx, cy, r_center, end, stroke)

        img = img.resize((dim, dim), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)

    def _round_cap(self, d: "ImageDraw.ImageDraw", cx: float, cy: float,
                   r: float, angle_deg: float, stroke: int) -> None:
        """Draw a filled disc at the arc point for ``angle_deg`` (round cap).

        ``r`` is the stroke *centreline* radius (r_bbox - stroke/2), so the disc
        of radius stroke/2 exactly fills the band thickness. Angles follow
        Pillow's convention: degrees clockwise from 3 o'clock, with y pointing
        down, so the point is ``(cx + r cos, cy + r sin)``.
        """
        rad = math.radians(angle_deg)
        px = cx + r * math.cos(rad)
        py = cy + r * math.sin(rad)
        cap = stroke / 2
        d.ellipse((px - cap, py - cap, px + cap, py + cap), fill=self._c_fill)


if __name__ == "__main__":
    # Manual eyeball test, no data needed. Run: python -m mimora.progress_widget
    # A new random average target is picked every second and the shown value
    # eases toward it each frame, so the ring fill and the number move smoothly
    # rather than jumping. The phrase count ticks up once per second too.
    import random

    root = tk.Tk()
    root.title("ProgressRing demo")
    root.configure(bg=THEME["bg_card"])
    ring = ProgressRing(root, size=160)
    ring.pack(padx=32, pady=32)

    maximum = 5.0
    step_ms = 40  # ~25 fps, fast enough to read as smooth motion
    state = {
        "current": 0.0,                             # value shown right now
        "target": random.uniform(0.0, maximum),     # value we ease toward
        "count": 0,
        "since_pick_ms": 0,                         # time on the current target
    }

    def _drive() -> None:
        # Ease the displayed value a fraction of the way to the target each
        # frame (exponential smoothing), then re-pick a new random target once
        # a full second has elapsed.
        state["current"] += 0.12 * (state["target"] - state["current"])
        ring.set_progress(state["current"], maximum, state["count"])

        state["since_pick_ms"] += step_ms
        if state["since_pick_ms"] >= 1000:
            state["since_pick_ms"] = 0
            state["target"] = random.uniform(0.0, maximum)
            state["count"] += 1
        root.after(step_ms, _drive)

    _drive()
    root.mainloop()
