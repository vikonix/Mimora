# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Schematic articulation face for Mimora.

A minimal "talking head" drawn on a Tk Canvas: a circle for the face and two
dot eyes, plus a mouth that has two mutually exclusive modes:

  * Talking -- a simple ellipse whose height tracks playback loudness. This is
    live feedback *during* TTS playback (it shows how open the mouth is, not
    tongue position), and is the original behaviour.

  * Paused -- when nothing is playing, the mouth becomes a smiley curve
    (``:)`` / ``:|`` / ``:(``) reflecting the current state. The smiley is never
    drawn on top of the talking ellipse: ``set_level`` switches to talking,
    ``rest`` switches back to the smiley.

Design constraints this module is built around:

  * No pre-rendering. Shapes are created once in ``__init__``; each animation
    frame only moves the existing mouth oval via ``Canvas.coords`` -- a few
    microseconds. Nothing is generated ahead of playback.

  * Thread-safe hand-off. Audio plays on a sounddevice thread while Tk draws on
    the main thread. ``set_level`` therefore only stores the latest loudness in
    a plain attribute (a single assignment, atomic in CPython); a small
    ``after``-loop on the Tk thread reads it and redraws. The audio callback
    must never touch the Canvas directly.

  * Zero extra dependencies: standard-library ``tkinter`` only.

Typical use::

    face = FaceWidget(parent_frame, size=160, bg=THEME["bg_panel"],
                      ink=THEME["text_bright"])
    face.pack()
    ...
    # inside the audio playback callback (any thread):
    rms = float(np.sqrt(np.mean(chunk.astype("float32") ** 2)))
    face.set_level(FaceWidget.level_from_rms(rms))   # talking ellipse
    ...
    face.set_score(score)   # choose the paused smiley (:) / :| / :( )
    face.rest()             # playback stopped -> show the smiley
"""

from __future__ import annotations

import time
import tkinter as tk


class FaceWidget(tk.Canvas):
    """A Tk Canvas that draws a schematic face and animates the mouth.

    The widget is itself a ``tk.Canvas``, so it can be ``pack``/``grid``-ed like
    any other widget. It starts its own redraw loop immediately and begins in
    the paused (smiley) mode.
    """

    def __init__(
        self,
        parent: tk.Misc,
        size: int = 160,
        *,
        bg: str = "#1e1e1e",
        face_color: str = "#ffffff",
        ink: str = "#1e1e1e",
        fps: int = 30,
        attack: float = 0.6,
        release: float = 0.15,
    ) -> None:
        """Create the face.

        Args:
            parent: parent Tk widget.
            size: side of the square canvas, in pixels. All face geometry scales
                from this, so this is the only sizing knob you need.
            bg: canvas background colour (the panel behind the face).
            face_color: fill colour of the face disc.
            ink: colour of the eyes and mouth, drawn on top of the face.
            fps: redraw rate of the animation loop.
            attack: smoothing factor used while the mouth is *opening* (target
                louder than current). Higher = snappier. Range (0, 1].
            release: smoothing factor used while the mouth is *closing*. Lower
                than ``attack`` on purpose -- mouths close more slowly than they
                open, which reads as natural rather than twitchy.
        """
        super().__init__(parent, width=size, height=size, bg=bg,
                         highlightthickness=0, bd=0)

        self._ink = ink
        self._face_color = face_color
        self._frame_ms = max(1, int(1000 / fps))
        self.fps = fps
        self._attack = attack
        self._release = release

        # Loudness state. ``_target`` is written from any thread by set_level;
        # ``_current`` is the smoothed value the loop actually draws.
        self._target = 0.0
        self._current = 0.0
        # Optional pre-computed loudness track (see play_levels). While set, the
        # loop reads the level for the current time from it instead of from
        # set_level; this is how playback drives the mouth when the audio backend
        # gives no per-frame callback (winsound on Windows).
        self._track: list[float] | None = None
        self._track_fps = float(fps)
        self._track_t0 = 0.0
        # Expression state: mouth-corner height, -1 frown .. +1 smile.
        self._curl_target = 0.0
        self._curl_current = 0.0
        # Mouth mode: "talking" (ellipse) or "paused" (smiley). Starts paused.
        self._mode = "paused"
        self._shown: str | None = None
        self._loop_id: str | None = None
        self._size = size  # fallback dimension before the first <Configure>

        # Draw once at the requested size, then rebuild on resize so the face
        # always fills (and never overflows) whatever box the layout gives it.
        self._rebuild(size, size)
        self.bind("<Configure>", self._on_resize)

        self._tick()  # start the redraw loop

    # -- geometry ---------------------------------------------------------

    def _on_resize(self, event: "tk.Event") -> None:
        self._rebuild(event.width, event.height)

    def _rebuild(self, width: int, height: int) -> None:
        """Recompute geometry for the current canvas size and redraw all parts."""
        dim = min(width, height)
        if dim <= 1:  # not laid out yet; fall back to the requested size
            dim = self._size
        self.delete("all")
        self._build_geometry(dim, width, height)
        self._draw_static()
        self._create_mouth()
        self._shown = None  # force _tick to re-apply the visible mouth

    def _build_geometry(self, dim: float, width: float, height: float) -> None:
        """Pre-compute coordinates so the loop does no layout maths.

        ``dim`` is the limiting side (so the face stays circular); ``width`` and
        ``height`` centre it in the canvas.
        """
        self._cx = width / 2
        self._cy = height / 2
        radius = dim * 0.42

        # Eyes: two dots in the upper third of the face.
        eye_y = self._cy - radius * 0.30
        eye_dx = radius * 0.34
        eye_r = max(2.0, radius * 0.08)
        self._eye_r = eye_r
        self._eye_l = (self._cx - eye_dx, eye_y)
        self._eye_r_pos = (self._cx + eye_dx, eye_y)

        self._head_r = radius

        # Mouth: centred in the lower third. Two independent shapes share this
        # spot but are never shown together (see _tick):
        #   * talking ellipse -- width is fixed; height runs from a near-flat
        #     line (closed) up to ``ry_max`` (loud speech).
        #   * paused smiley -- a curve whose corners rise/fall with ``curl``.
        self._mouth_cy = self._cy + radius * 0.38
        self._mouth_rx = radius * 0.34           # half-width of the mouth
        self._mouth_ry_min = max(1.0, radius * 0.02)  # closed = thin line
        self._mouth_ry_max = radius * 0.28       # loud speech (toned down, §9)
        self._smile_amp = radius * 0.16          # corner/mid offset at full curl
        self._line_w = max(2, int(dim * 0.02))

    def _draw_static(self) -> None:
        """Draw the parts that never move: head outline and eyes."""
        r = self._head_r
        self.create_oval(self._cx - r, self._cy - r, self._cx + r, self._cy + r,
                         outline="", fill=self._face_color)
        er = self._eye_r
        for ex, ey in (self._eye_l, self._eye_r_pos):
            self.create_oval(ex - er, ey - er, ex + er, ey + er,
                             outline="", fill=self._ink)

    def _smile_points(self, curl: float) -> list[float]:
        """Three smooth-curve points for the paused smiley.

        ``curl`` (-1..1) lifts the corners and drops the middle for a smile,
        and the reverse for a frown. At ``curl`` 0 all three points sit on one
        line -- the flat ``:|`` mouth.
        """
        cx, my, rx = self._cx, self._mouth_cy, self._mouth_rx
        amp = self._smile_amp
        return [
            cx - rx, my - curl * amp,   # left corner
            cx, my + curl * amp,        # middle
            cx + rx, my - curl * amp,   # right corner
        ]

    def _create_mouth(self) -> None:
        """Create both mouth shapes once; only one is ever visible."""
        # Talking ellipse (shown during playback). Starts hidden and closed.
        rx, ry = self._mouth_rx, self._mouth_ry_min
        self._mouth_oval = self.create_oval(
            self._cx - rx, self._mouth_cy - ry,
            self._cx + rx, self._mouth_cy + ry,
            outline=self._ink, width=self._line_w, fill="", state="hidden",
        )
        # Paused smiley (shown when idle). Starts visible and flat.
        self._mouth_smile = self.create_line(
            self._smile_points(0.0), fill=self._ink, width=self._line_w,
            smooth=True, capstyle=tk.ROUND,
        )

    def _show_mouth(self, which: str) -> None:
        """Toggle which mouth shape is visible (no-op if already current)."""
        if self._shown == which:
            return
        self._shown = which
        talking = which == "oval"
        self.itemconfigure(self._mouth_oval,
                           state="normal" if talking else "hidden")
        self.itemconfigure(self._mouth_smile,
                           state="hidden" if talking else "normal")

    # -- public API -------------------------------------------------------

    def set_level(self, level: float) -> None:
        """Set the target mouth openness, ``0.0`` (closed) .. ``1.0`` (wide).

        Switches the mouth into talking mode (the ellipse). Safe to call from
        the audio thread: it only stores floats. Call it as often as you have
        fresh loudness (e.g. once per audio block); the widget redraws on its
        own schedule regardless.
        """
        # Clamp without branching surprises; NaN-safe via the comparisons.
        if level < 0.0:
            level = 0.0
        elif level > 1.0:
            level = 1.0
        self._target = level
        self._mode = "talking"

    def play_levels(self, levels: "list[float]", fps: "float | None" = None) -> None:
        """Drive the talking mouth from a pre-computed loudness track.

        ``levels`` is a sequence of openness values in ``[0, 1]`` sampled at
        ``fps`` (defaults to the widget's own fps). The track is indexed by
        wall-clock time from this call, so starting it together with audio
        playback keeps the mouth in sync *without* any per-frame audio callback
        -- the path ``winsound`` cannot provide on Windows. When the track runs
        out the mouth returns to the resting smiley on its own; call ``rest()``
        to stop it early (e.g. on a playback interrupt).

        Call on the Tk thread: it switches the widget into talking mode.
        """
        self._track = list(levels)
        self._track_fps = float(fps) if fps else self.fps
        self._track_t0 = time.perf_counter()
        self._mode = "talking"

    def rest(self) -> None:
        """Stop talking and show the smiley. Call when playback stops."""
        self._track = None
        self._target = 0.0
        self._mode = "paused"

    # Named expressions and their emoticon aliases -> corner curl.
    _EXPRESSIONS = {
        "happy": 1.0, ":)": 1.0,
        "neutral": 0.0, ":|": 0.0,
        "sad": -1.0, ":(": -1.0,
    }

    def set_expression(self, expression: "str | float") -> None:
        """Set the resting mouth curve: smile, flat or frown.

        Accepts a name (``"happy"``/``"neutral"``/``"sad"``), an emoticon
        (``":)"`` / ``":|"`` / ``":("``), or a raw curl float in ``[-1, 1]``
        (``+1`` full smile, ``-1`` full frown). This sets the smiley shown while
        paused; it has no effect on the talking ellipse. The curve eases in, so
        switching expressions looks smooth. Handy for score feedback.
        """
        if isinstance(expression, str):
            try:
                curl = self._EXPRESSIONS[expression]
            except KeyError:
                raise ValueError(
                    f"unknown expression {expression!r}; use one of "
                    f"{sorted(self._EXPRESSIONS)} or a float in [-1, 1]"
                ) from None
        else:
            curl = max(-1.0, min(1.0, float(expression)))
        self._curl_target = curl

    def set_score(self, score: float, *, neutral_at: float = 50.0) -> None:
        """Map a pronunciation score to a smiley.

        Above ``neutral_at`` -> smile, below -> frown, exactly at -> flat. This
        only changes the expression; the loudness animation is untouched. For
        the idle "waiting" state, call ``set_expression(":)")`` directly.
        """
        if score > neutral_at:
            self.set_expression("happy")
        elif score < neutral_at:
            self.set_expression("sad")
        else:
            self.set_expression("neutral")

    @staticmethod
    def level_from_rms(rms: float, *, floor: float = 0.0, gain: float = 8.0) -> float:
        """Map an audio RMS amplitude to a ``0..1`` openness.

        ``rms`` is expected for float audio in roughly ``[-1, 1]`` (so RMS is
        small). ``gain`` scales quiet speech up to a usable range; ``floor``
        subtracts background level before scaling. Tune ``gain`` once by ear --
        higher makes the mouth more reactive. Clamped to ``[0, 1]``.
        """
        level = (rms - floor) * gain
        if level < 0.0:
            return 0.0
        if level > 1.0:
            return 1.0
        return level

    # -- animation loop ---------------------------------------------------

    def _tick(self) -> None:
        """One animation frame: update only the mouth shape that is visible."""
        # A pre-computed loudness track, if present, drives the talking mouth by
        # wall-clock time and hands back to the smiley once it is exhausted.
        if self._track is not None:
            idx = int((time.perf_counter() - self._track_t0) * self._track_fps)
            if idx < len(self._track):
                self._target = self._track[idx]
                self._mode = "talking"
            else:
                self._track = None
                self._target = 0.0
                self._mode = "paused"

        if self._mode == "talking":
            self._show_mouth("oval")
            # Asymmetric smoothing: open fast, close slow.
            target, current = self._target, self._current
            coeff = self._attack if target >= current else self._release
            current += coeff * (target - current)
            self._current = current
            ry = self._mouth_ry_min + current * (self._mouth_ry_max - self._mouth_ry_min)
            rx = self._mouth_rx
            self.coords(self._mouth_oval,
                        self._cx - rx, self._mouth_cy - ry,
                        self._cx + rx, self._mouth_cy + ry)
        else:
            self._show_mouth("smile")
            # Expression eases toward its target at a steady, gentle rate.
            self._curl_current += 0.25 * (self._curl_target - self._curl_current)
            self.coords(self._mouth_smile, *self._smile_points(self._curl_current))

        # Reschedule. winfo_exists guards against a destroyed widget.
        if self.winfo_exists():
            self._loop_id = self.after(self._frame_ms, self._tick)

    def destroy(self) -> None:
        """Cancel the loop before the widget goes away."""
        if self._loop_id is not None:
            try:
                self.after_cancel(self._loop_id)
            except tk.TclError:
                pass
            self._loop_id = None
        super().destroy()


if __name__ == "__main__":
    # Manual eyeball test, no audio needed. Run: python -m mimora.face_widget
    # It alternates ~2.5 s of "talking" (the ellipse driven by a sine envelope)
    # with ~1 s of "paused", cycling through the three smileys.
    import math

    root = tk.Tk()
    root.title("FaceWidget demo")
    root.configure(bg="#1e1e1e")
    face = FaceWidget(root, size=220, bg="#1e1e1e",
                      face_color="#ffffff", ink="#1e1e1e")
    face.pack(padx=24, pady=24)

    expressions = [":)", ":|", ":("]
    state = {"phase": 0.0, "talking": True, "left_ms": 2500, "i": 0}
    step_ms = 40

    def _drive() -> None:
        if state["talking"]:
            state["phase"] += 0.18
            # Two overlaid sines fake a speech-like, uneven envelope.
            p = state["phase"]
            level = 0.5 * (math.sin(p) + 1) * (0.6 + 0.4 * math.sin(p * 0.37))
            face.set_level(level)

        state["left_ms"] -= step_ms
        if state["left_ms"] <= 0:
            if state["talking"]:
                face.set_expression(expressions[state["i"] % len(expressions)])
                state["i"] += 1
                face.rest()
                state["talking"] = False
                state["left_ms"] = 1000
            else:
                state["talking"] = True
                state["left_ms"] = 2500
        root.after(step_ms, _drive)

    _drive()
    root.mainloop()
