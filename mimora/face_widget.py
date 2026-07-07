# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Schematic articulation face for Mimora.

A cartoon "talking head" rendered with Pillow and shown on a Tk Canvas: a face
disc with eyebrows, blinking eyes and a mouth that has two mutually exclusive
modes:

  * Talking -- a plain dark mouth ellipse whose height tracks playback
    loudness. This is live feedback *during* TTS playback; it shows how open
    the mouth is, not tongue position.

  * Paused -- when nothing is playing, the mouth becomes a single round-capped
    stroke: smile / flat / frown (``:)`` / ``:|`` / ``:(``) reflecting the
    current state, with the eyebrows following the expression. The smiley is
    never drawn together with the talking mouth: ``set_level`` switches to
    talking, ``rest`` switches back to the smiley.

Design constraints this module is built around:

  * Antialiased rendering. Tk Canvas primitives have no antialiasing, which is
    why the previous vector face read as clipart at ~78 px. Each frame is now
    drawn by Pillow at 4x size and downscaled with LANCZOS, then shown as a
    single Canvas image item.

  * Lazy frame cache instead of per-frame rendering. Mouth openness, smile
    curl and the blink phase are quantized to a small number of steps; every
    distinct combination is rendered once on first use (~1-2 ms) and cached as
    a ``PhotoImage``. Steady-state animation is just an image swap -- cheaper
    than the old ``Canvas.coords`` path. The cache is cleared on resize.

  * Thread-safe hand-off. Audio plays on a sounddevice thread while Tk draws
    on the main thread. ``set_level`` therefore only stores the latest
    loudness in a plain attribute (a single assignment, atomic in CPython);
    the ``after``-loop on the Tk thread reads it and swaps frames. Pillow /
    ``ImageTk`` objects are only ever touched on the Tk thread.

  * Dependencies: stdlib ``tkinter`` plus Pillow (pinned in requirements.txt).

Typical use::

    face = FaceWidget(parent_frame, size=160, bg=THEME["bg_panel"],
                      face_color=THEME["face"], eye_color=THEME["eyes"],
                      mouth_color=THEME["mouth"])
    face.pack()
    ...
    # inside the audio playback callback (any thread):
    rms = float(np.sqrt(np.mean(chunk.astype("float32") ** 2)))
    face.set_level(FaceWidget.level_from_rms(rms))   # talking mouth
    ...
    face.set_score(score)   # choose the paused smiley (:) / :| / :( )
    face.rest()             # playback stopped -> show the smiley
"""

from __future__ import annotations

import random
import time
import tkinter as tk

from PIL import Image, ImageDraw, ImageTk

# Rendering scale: frames are drawn this many times larger than the widget and
# downscaled with LANCZOS -- this is where the antialiasing comes from.
_SUPERSAMPLE = 4

# Quantization steps for the frame cache. More steps = smoother motion but a
# larger (still lazy) cache. Mouth openness 0..1 -> 0.._MOUTH_STEPS; smile
# curl -1..1 -> -_CURL_STEPS.._CURL_STEPS.
_MOUTH_STEPS = 11
_CURL_STEPS = 4

# Blink choreography, in seconds from blink start: lids half-closed, fully
# closed, half again, then open. Kept short so it reads as a blink, not a nap.
_BLINK_HALF_IN = 0.05
_BLINK_OPEN_AT = 0.17
_BLINK_TOTAL = 0.22


def _blend(c1: tuple, c2: tuple, k: float) -> tuple:
    """Linear mix of two RGB tuples: ``k=0`` -> ``c1``, ``k=1`` -> ``c2``."""
    return tuple(round(a + (b - a) * k) for a, b in zip(c1, c2))


def _bezier_point(p0: tuple, p1: tuple, p2: tuple, t: float) -> tuple:
    """Point at parameter ``t`` on the quadratic Bezier ``p0 -> p1 -> p2``."""
    u = 1.0 - t
    return (u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
            u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1])


def _quad_bezier(p0: tuple, p1: tuple, p2: tuple, n: int = 24) -> list:
    """Sample a quadratic Bezier into ``n + 1`` points (for lines/polygons)."""
    return [_bezier_point(p0, p1, p2, i / n) for i in range(n + 1)]


class FaceWidget(tk.Canvas):
    """A Tk Canvas that shows a Pillow-rendered face and animates the mouth.

    The widget is itself a ``tk.Canvas``, so it can be ``pack``/``grid``-ed
    like any other widget. It starts its own redraw loop immediately and
    begins in the paused (smiley) mode.
    """

    def __init__(
        self,
        parent: tk.Misc,
        size: int = 160,
        *,
        bg: str = "#1e1e1e",
        face_color: str = "#f8f8f2",
        face_outline: str = "",
        eye_color: str = "#1e2a44",
        mouth_color: str = "#8a3a2c",
        fps: int = 30,
        attack: float = 0.6,
        release: float = 0.15,
    ) -> None:
        """Create the face.

        Args:
            parent: parent Tk widget.
            size: side of the square canvas, in pixels. All face geometry
                scales from this, so this is the only sizing knob you need.
            bg: canvas background colour (the panel behind the face). Frames
                are rendered on this colour, so antialiased edges blend into
                it exactly.
            face_color: fill colour of the face disc.
            face_outline: outline colour of the face disc. Empty string (the
                default) draws no outline; pass a colour so the disc stays
                visible when its fill matches the panel behind it (e.g. a
                white face on a white panel in the light theme). Drawn as a
                thin 1 px rim, not the old heavy stroke.
            eye_color: colour of the eyes and eyebrows.
            mouth_color: colour of the resting mouth stroke; the darker
                talking-mouth fill is derived from it.
            fps: redraw rate of the animation loop.
            attack: smoothing factor used while the mouth is *opening* (target
                louder than current). Higher = snappier. Range (0, 1].
            release: smoothing factor used while the mouth is *closing*. Lower
                than ``attack`` on purpose -- mouths close more slowly than
                they open, which reads as natural rather than twitchy.
        """
        super().__init__(parent, width=size, height=size, bg=bg,
                         highlightthickness=0, bd=0)

        # Palette: parse the Tk colour strings once (winfo_rgb also resolves
        # named colours), then derive the accent tones from them so the face
        # follows the theme without extra knobs.
        self._c_bg = self._rgb(bg)
        self._c_face = self._rgb(face_color)
        self._c_outline = self._rgb(face_outline) if face_outline else None
        self._c_eye = self._rgb(eye_color)
        self._c_mouth = self._rgb(mouth_color)
        self._c_mouth_in = _blend(self._c_mouth, (24, 12, 12), 0.5)  # interior
        self._c_shine = (255, 255, 255)  # eye highlight

        self._frame_ms = max(1, int(1000 / fps))
        self.fps = fps
        self._attack = attack
        self._release = release

        # Loudness state. ``_target`` is written from any thread by set_level;
        # ``_current`` is the smoothed value the loop actually draws.
        self._target = 0.0
        self._current = 0.0
        # Optional pre-computed loudness track (see play_levels). While set,
        # the loop reads the level for the current time from it instead of
        # from set_level; this is how playback drives the mouth when the audio
        # backend gives no per-frame callback (winsound on Windows).
        self._track: list[float] | None = None
        self._track_fps = float(fps)
        self._track_t0 = 0.0
        # Expression state: mouth-corner height, -1 frown .. +1 smile.
        self._curl_target = 0.0
        self._curl_current = 0.0
        # Mouth mode: "talking" (ellipse) or "paused" (smiley). Starts paused.
        self._mode = "paused"
        # Next blink start, wall-clock. Blinks repeat at a random 2-5 s pace.
        self._blink_at = time.perf_counter() + random.uniform(1.0, 3.0)

        # Frame cache: quantized state -> rendered PhotoImage. Lazy -- only
        # combinations that actually occur are rendered. Cleared on resize.
        self._cache: dict[tuple, ImageTk.PhotoImage] = {}
        self._photo: ImageTk.PhotoImage | None = None  # keep-alive reference
        self._key_shown: tuple | None = None
        self._loop_id: str | None = None
        self._size = size  # fallback dimension before the first <Configure>
        # Canvas size in px. NB: not ``_w``/``_h`` -- tkinter reserves ``_w``
        # for the widget's Tcl path name; shadowing it breaks every Tk call.
        self._cw = size
        self._ch = size

        # One image item holds the current frame; frames are centred so the
        # face stays circular in whatever box the layout gives it.
        self._img_item = self.create_image(size / 2, size / 2)
        self.bind("<Configure>", self._on_resize)

        self._tick()  # start the redraw loop

    # -- colours ------------------------------------------------------------

    def _rgb(self, color: str) -> tuple:
        """Resolve a Tk colour string to an 8-bit RGB tuple."""
        r, g, b = self.winfo_rgb(color)
        return (r // 257, g // 257, b // 257)

    # -- geometry / resize ----------------------------------------------------

    def _on_resize(self, event: "tk.Event") -> None:
        if event.width == self._cw and event.height == self._ch:
            return
        self._cw, self._ch = event.width, event.height
        self.coords(self._img_item, self._cw / 2, self._ch / 2)
        # Cached frames are the old size; re-render lazily at the new one.
        self._cache.clear()
        self._key_shown = None

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
        (``+1`` full smile, ``-1`` full frown). This sets the smiley shown
        while paused and tilts the eyebrows with it; it has no effect on the
        talking mouth height. The curve eases in, so switching expressions
        looks smooth. Handy for score feedback.
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

    def _blink_phase(self) -> int:
        """Return the current eyelid state: 0 open, 1 half, 2 closed.

        Driven by wall-clock so the pace is independent of fps. Advancing past
        the end of a blink schedules the next one at a random 2-5 s distance.
        """
        now = time.perf_counter()
        if now < self._blink_at:
            return 0
        phase = now - self._blink_at
        if phase >= _BLINK_TOTAL:
            self._blink_at = now + random.uniform(2.2, 5.5)
            return 0
        return 2 if _BLINK_HALF_IN <= phase < _BLINK_OPEN_AT else 1

    def _tick(self) -> None:
        """One animation frame: swap in the cached frame for the new state."""
        # A pre-computed loudness track, if present, drives the talking mouth
        # by wall-clock time and hands back to the smiley once exhausted.
        if self._track is not None:
            idx = int((time.perf_counter() - self._track_t0) * self._track_fps)
            if idx < len(self._track):
                self._target = self._track[idx]
                self._mode = "talking"
            else:
                self._track = None
                self._target = 0.0
                self._mode = "paused"

        # Asymmetric loudness smoothing: open fast, close slow.
        target, current = self._target, self._current
        coeff = self._attack if target >= current else self._release
        self._current = current + coeff * (target - current)
        # Expression eases toward its target at a steady, gentle rate. Updated
        # in both modes: the eyebrows follow the curl even while talking.
        self._curl_current += 0.25 * (self._curl_target - self._curl_current)

        blink = self._blink_phase()
        curl_q = round(self._curl_current * _CURL_STEPS)
        if self._mode == "talking":
            key = ("talk", round(self._current * _MOUTH_STEPS), curl_q, blink)
        else:
            key = ("rest", 0, curl_q, blink)

        if key != self._key_shown:
            photo = self._cache.get(key)
            if photo is None:
                if len(self._cache) > 512:  # safety valve, never hit in practice
                    self._cache.clear()
                photo = self._render_frame(key)
                self._cache[key] = photo
            self.itemconfigure(self._img_item, image=photo)
            self._photo = photo  # PhotoImage must stay referenced or Tk blanks
            self._key_shown = key

        # Reschedule. winfo_exists guards against a destroyed widget.
        if self.winfo_exists():
            self._loop_id = self.after(self._frame_ms, self._tick)

    # -- frame rendering ----------------------------------------------------

    def _render_frame(self, key: tuple) -> ImageTk.PhotoImage:
        """Render one face frame for a quantized state, antialiased.

        Drawn at ``_SUPERSAMPLE`` times the widget size on the opaque ``bg``
        colour (so edge pixels blend into the panel exactly), then downscaled
        with LANCZOS. Tk-thread only (creates a ``PhotoImage``).
        """
        mode, mouth_q, curl_q, blink = key
        dim = min(self._cw, self._ch)
        if dim <= 1:  # not laid out yet; fall back to the requested size
            dim = self._size
        s = dim * _SUPERSAMPLE
        curl = curl_q / _CURL_STEPS
        openness = mouth_q / _MOUTH_STEPS

        img = Image.new("RGB", (s, s), self._c_bg)
        d = ImageDraw.Draw(img)
        cx = cy = s / 2
        r = s * 0.42  # head radius; everything below scales from it

        # Face disc, with an optional thin rim (1 px after downscale).
        if self._c_outline is not None:
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=self._c_face,
                      outline=self._c_outline, width=_SUPERSAMPLE)
        else:
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=self._c_face)

        # Eyes: tall ellipses with a highlight; the blink squashes them.
        ey, edx = cy - r * 0.30, r * 0.34
        erx, ery_open = r * 0.105, r * 0.16
        ery = max(ery_open * (1.0, 0.45, 0.10)[blink], r * 0.02)
        for side in (-1, 1):
            ex = cx + side * edx
            d.ellipse((ex - erx, ey - ery, ex + erx, ey + ery), fill=self._c_eye)
            if blink == 0:
                hr = erx * 0.42
                hx, hy = ex - erx * 0.28, ey - ery_open * 0.38
                d.ellipse((hx - hr, hy - hr, hx + hr, hy + hr), fill=self._c_shine)

        # Eyebrows: gentle arches that rise with a smile; a frown raises the
        # inner ends and drops the outer ones (the classic "worried" tilt).
        brow_y = ey - ery_open - r * 0.13 - curl * r * 0.045
        sad = max(0.0, -curl)
        brow_w = max(2, int(r * 0.055))
        for side in (-1, 1):
            ex = cx + side * edx
            inner = (ex - side * erx * 1.15, brow_y - sad * r * 0.09)
            outer = (ex + side * erx * 1.15, brow_y + sad * r * 0.05)
            ctrl = ((inner[0] + outer[0]) / 2, min(inner[1], outer[1]) - r * 0.05)
            d.line(_quad_bezier(inner, ctrl, outer), fill=self._c_eye,
                   width=brow_w, joint="curve")
            for px, py in (inner, outer):  # round caps
                cr = brow_w / 2
                d.ellipse((px - cr, py - cr, px + cr, py + cr), fill=self._c_eye)

        my = cy + r * 0.38  # mouth centre line
        if mode == "talk":
            self._draw_talking_mouth(d, cx, my, r, openness)
        else:
            self._draw_smiley(d, cx, my, r, curl)

        img = img.resize((dim, dim), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)

    def _draw_talking_mouth(self, d: "ImageDraw.ImageDraw", cx: float,
                            my: float, r: float, openness: float) -> None:
        """Plain dark mouth ellipse; slightly narrows as it opens.

        No lip outline or interior detail on purpose -- at widget size the
        single dark shape reads cleaner.
        """
        mrx = r * 0.30 - openness * r * 0.04
        mry = r * 0.03 + openness * (r * 0.26 - r * 0.03)
        d.ellipse((cx - mrx, my - mry, cx + mrx, my + mry),
                  fill=self._c_mouth_in)

    def _draw_smiley(self, d: "ImageDraw.ImageDraw", cx: float, my: float,
                     r: float, curl: float) -> None:
        """Single round-capped stroke: smile, flat line or frown.

        A quadratic Bezier whose corners sit at ``my - curl * amp`` and whose
        middle dips the opposite way, so ``curl`` (-1..1) bends the stroke
        from a full smile through flat (``:|``) to a frown. Round end caps
        are drawn as small discs (PIL lines have butt caps).
        """
        srx = r * 0.32
        amp = r * 0.16
        left = (cx - srx, my - curl * amp)
        right = (cx + srx, my - curl * amp)
        ctrl = (cx, my + curl * amp * 2.0)
        width = max(2, int(r * 0.075))
        d.line(_quad_bezier(left, ctrl, right), fill=self._c_mouth,
               width=width, joint="curve")
        for px, py in (left, right):
            cr = width / 2
            d.ellipse((px - cr, py - cr, px + cr, py + cr), fill=self._c_mouth)

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
    # It alternates ~2.5 s of "talking" (the mouth driven by a sine envelope)
    # with ~1 s of "paused", cycling through the three smileys.
    import math

    root = tk.Tk()
    root.title("FaceWidget demo")
    root.configure(bg="#1e1e1e")
    face = FaceWidget(root, size=220, bg="#1e1e1e",
                      face_color="#ffffff", eye_color="#1e2a44",
                      mouth_color="#8a3a2c")
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
