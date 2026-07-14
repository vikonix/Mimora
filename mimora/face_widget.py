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

Besides the two mouth modes the face is interactive:

  * The eyes follow the mouse cursor (the global pointer is polled once per
    animation frame, no extra bindings), widen as the cursor closes in, and
    go cross-eyed when it hovers right on the nose.

  * A left click on the face disc (ignored while talking, so it never hides
    the talking mouth) plays a short balloon-pop cartoon: the rim bursts
    into shards, the now-floating eyes drop one by one into the mouth which
    gulps them down, the mouth grins wide open, pops as well, and the canvas
    stays blank while the button is held. Releasing the button re-inflates
    the face with a small overshoot.

  * The cartoon timeline and the gaze math are pure module-level helpers
    (``gaze_state``, ``face_hit``, ``pop_scene``, ``appear_scale``),
    unit-tested without Tk in tests/test_face_interaction.py.

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

import math
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


# -- pure interaction logic (Tk-free, unit-tested) ---------------------------

# Facial-feature layout in head-radius units, shared by the live-face
# renderer and the cartoon timeline so a falling eye lands exactly on the
# mouth the renderer draws.
_EYE_X = 0.34    # eye centre, horizontal distance from the face centre
_EYE_Y = -0.30   # eye centre, vertical offset from the face centre
_MOUTH_Y = 0.38  # mouth centre line, vertical offset

_GAZE_STEPS = 2  # gaze quantization: offsets -2..2 per axis in the cache key
_ANIM_QFPS = 30  # cartoon frames are quantized (and cached) at this rate

# Balloon-pop cartoon: stage lengths in seconds. The whole timeline is a pure
# function of elapsed time (``pop_scene``), so frames are cacheable by
# quantized phase and the tests can probe any instant without waiting.
POP_BURST_S = 0.30       # the rim shatters, the disc is gone
POP_EYE_S = 0.55         # one eye falls into the mouth and is gulped down
POP_GRIN_S = 0.45        # the mouth grins wide open
POP_GRIN_BURST_S = 0.25  # the grin pops as well
POP_TOTAL_S = POP_BURST_S + 2 * POP_EYE_S + POP_GRIN_S + POP_GRIN_BURST_S
APPEAR_S = 0.45          # re-inflate after the button is released


def gaze_state(px: float, py: float, cx: float, cy: float, r: float) -> tuple:
    """Quantize the cursor position into an eye-gaze state.

    All coordinates share one space (the widget passes screen pixels);
    ``(cx, cy)`` is the face centre, ``r`` the head radius. Returns
    ``(gqx, gqy, wide, cross)``:

      * ``gqx``/``gqy`` -- gaze offset quantized to -_GAZE_STEPS.._GAZE_STEPS
        per axis (it becomes part of the frame-cache key, hence the coarse
        steps);
      * ``wide`` -- eye widening 0..2, growing as the cursor closes in;
      * ``cross`` -- True when the cursor sits right on the nose, which makes
        the face go cross-eyed.

    Beyond six head radii the face loses interest and rests. Pure function:
    no Tk, unit-tested in tests/test_face_interaction.py.
    """
    if r <= 0:
        return (0, 0, 0, False)
    dx, dy = px - cx, py - cy
    dist = math.hypot(dx, dy)
    if dist < r * 0.45:  # right on the nose
        return (0, 0, 2, True)
    reach = 6.0 * r
    if dist >= reach:
        return (0, 0, 0, False)
    # Unit direction scaled by interest: full deflection up close, ~60 % at
    # the edge of the reach, so far-away cursors get a subtler glance.
    strength = 1.0 - 0.4 * (dist / reach)
    gqx = round(dx / dist * strength * _GAZE_STEPS)
    gqy = round(dy / dist * strength * _GAZE_STEPS)
    if dist < r * 1.3:
        wide = 2
    elif dist < r * 3.0:
        wide = 1
    else:
        wide = 0
    return (gqx, gqy, wide, False)


def face_hit(px: float, py: float, cx: float, cy: float, r: float) -> bool:
    """True when point (px, py) lies on the face disc centred at (cx, cy)."""
    return math.hypot(px - cx, py - cy) <= r


def pop_scene(t: float) -> dict:
    """Scene of the balloon-pop cartoon at ``t`` seconds from the click.

    Returns a dict describing what to draw. Coordinates are in head-radius
    units relative to the face centre, so the renderer only scales by ``r``:

      * ``"shards"`` -- rim-burst expansion 0..1, or None once the debris
        has dissolved;
      * ``"eyes"`` -- ``[left, right]``; each an ``(x, y)`` position or None
        after the mouth has swallowed it;
      * ``"mouth"`` -- ``("open", openness)`` while catching/gulping (also
        the startled "o" during the burst), ``("grin", k)`` for the growing
        open smile, or None once it popped;
      * ``"mouth_shards"`` -- grin-burst expansion 0..1, or None;
      * ``"done"`` -- True once everything vanished (``t >= POP_TOTAL_S``).

    Pure function: no Tk, unit-tested in tests/test_face_interaction.py.
    """
    eyes = [(-_EYE_X, _EYE_Y), (_EYE_X, _EYE_Y)]
    scene = {"shards": None, "eyes": eyes, "mouth": None,
             "mouth_shards": None, "done": False}

    # Guard the documented "done at t >= POP_TOTAL_S" boundary explicitly:
    # the per-stage subtraction chain below accumulates float rounding, so
    # exactly at the total it could otherwise land a hair inside the last
    # stage instead of finishing.
    if t >= POP_TOTAL_S:
        scene["eyes"] = [None, None]
        scene["done"] = True
        return scene

    if t < POP_BURST_S:  # the rim flies apart; the face is startled ("o")
        k = max(0.0, t) / POP_BURST_S
        scene["shards"] = k
        scene["mouth"] = ("open", 0.25 * (1.0 - k))
        return scene
    t -= POP_BURST_S

    for i in (0, 1):  # the left eye falls first, then the right one
        if t < POP_EYE_S:
            for j in range(i):
                eyes[j] = None  # already swallowed
            u = t / POP_EYE_S
            if u < 0.8:  # falling: linear drift in x, gravity ease-in in y
                v = u / 0.8
                x0, y0 = eyes[i]
                eyes[i] = (x0 * (1.0 - v), y0 + (_MOUTH_Y - y0) * v * v)
                # The mouth opens to catch once the eye is on its way down.
                openness = 0.0 if v < 0.35 else min(1.0, (v - 0.35) / 0.45)
            else:  # landed: the eye is gone, the mouth gulps shut
                eyes[i] = None
                openness = 1.0 - (u - 0.8) / 0.2
            scene["mouth"] = ("open", openness)
            return scene
        t -= POP_EYE_S

    scene["eyes"] = [None, None]
    if t < POP_GRIN_S:  # a satisfied, wide-open grin grows
        scene["mouth"] = ("grin", t / POP_GRIN_S)
        return scene
    t -= POP_GRIN_S

    if t < POP_GRIN_BURST_S:  # ... and pops like the head did
        scene["mouth_shards"] = t / POP_GRIN_BURST_S
        return scene

    scene["done"] = True  # nothing left; blank until the button is released
    return scene


def appear_scale(t: float) -> float:
    """Re-inflate curve: head scale 0..1 with a small balloon overshoot.

    Grows to 1.08 over the first 70 % of ``APPEAR_S``, then settles back to
    exactly 1.0. Pure function, unit-tested.
    """
    if t <= 0.0:
        return 0.0
    if t >= APPEAR_S:
        return 1.0
    k = t / APPEAR_S
    if k < 0.7:
        return 1.08 * (k / 0.7)
    return 1.08 - 0.08 * (k - 0.7) / 0.3


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
        face_offset: float = 0.0,
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
            face_offset: vertical shift of the facial features (eyes, eyebrows
                and mouth) relative to the head disc, as a fraction of the head
                radius. Positive moves the features down, negative up; ``0.0``
                (the default) centres them as before. The disc itself does not
                move, so this repositions the face *within* the circle.
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
        # Vertical shift of the features inside the disc, in units of the head
        # radius (see the face_offset argument). Applied in _render_frame.
        self._face_offset = face_offset

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
        # Click cartoon: None, or the current phase of the balloon-pop show
        # ("pop" -> "gone" while the button is held -> "appear").
        self._anim: str | None = None
        self._anim_t0 = 0.0
        self._button_down = False

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
        self.bind("<Button-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

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

    # -- cursor & click interaction -----------------------------------------

    def _poll_gaze(self) -> tuple:
        """Quantized cursor-gaze state for the frame key (Tk thread only).

        Polls the global pointer instead of binding ``<Motion>``: the eyes
        then follow the cursor anywhere on the screen, not only over the
        canvas. One winfo call per frame is negligible next to the frame
        swap itself.
        """
        try:
            px, py = self.winfo_pointerxy()
            cx = self.winfo_rootx() + self._cw / 2
            cy = self.winfo_rooty() + self._ch / 2
        except tk.TclError:  # racing widget destruction
            return (0, 0, 0, False)
        return gaze_state(px, py, cx, cy, min(self._cw, self._ch) * 0.42)

    def _on_press(self, event: "tk.Event") -> None:
        """A left click on the face disc pops it like a balloon.

        Ignored while talking (the cartoon would hide the talking mouth) and
        while a cartoon is already running. Clicks on the canvas corners,
        outside the disc, do nothing on purpose.
        """
        if self._anim is not None or self._mode == "talking":
            return
        r = min(self._cw, self._ch) * 0.42
        if not face_hit(event.x, event.y, self._cw / 2, self._ch / 2, r):
            return
        self._button_down = True
        self._anim = "pop"
        self._anim_t0 = time.perf_counter()

    def _on_release(self, _event: "tk.Event") -> None:
        """Releasing the button lets the popped face re-inflate."""
        self._button_down = False
        if self._anim == "gone":
            self._anim = "appear"
            self._anim_t0 = time.perf_counter()

    def _advance_anim(self) -> "tuple | None":
        """Step the click cartoon; return its frame key, or None when over.

        "pop" plays to the end even if the button was released early; after
        it the face stays "gone" while the button is still held and starts
        to re-inflate ("appear") otherwise. Returning None hands the frame
        back to the normal face path in ``_tick``.
        """
        now = time.perf_counter()
        t = now - self._anim_t0
        if self._anim == "pop":
            if t < POP_TOTAL_S:
                return ("pop", int(t * _ANIM_QFPS))
            if self._button_down:
                self._anim = "gone"
                return ("gone",)
            self._anim = "appear"
            self._anim_t0 = now
            return ("appear", 0)
        if self._anim == "gone":
            return ("gone",)
        if t < APPEAR_S:  # "appear"
            return ("appear", int(t * _ANIM_QFPS))
        # Fully re-inflated: come back smiling (matching the appear frames)
        # and ease from there to whatever the app sets next.
        self._anim = None
        self._curl_current = 1.0
        return None

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
        try:
            # A pre-computed loudness track, if present, drives the talking
            # mouth by wall-clock time and hands back to the smiley once
            # exhausted.
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
            # Expression eases toward its target at a steady, gentle rate.
            # Updated in both modes: the eyebrows follow the curl even while
            # talking.
            self._curl_current += 0.25 * (self._curl_target - self._curl_current)

            # The click cartoon, when active, overrides the whole face. The
            # loudness track above keeps running on wall-clock regardless, so
            # a playback that starts mid-cartoon is back in sync afterwards.
            key = self._advance_anim() if self._anim is not None else None
            if key is None:
                blink = self._blink_phase()
                curl_q = round(self._curl_current * _CURL_STEPS)
                gaze = self._poll_gaze()
                if self._mode == "talking":
                    key = ("talk", round(self._current * _MOUTH_STEPS),
                           curl_q, blink, *gaze)
                else:
                    key = ("rest", 0, curl_q, blink, *gaze)

            if key != self._key_shown:
                photo = self._cache.get(key)
                if photo is None:
                    # Safety valve; the gaze and cartoon dimensions enlarged
                    # the key space, but the cache is still lazy.
                    if len(self._cache) > 4096:
                        self._cache.clear()
                    photo = self._render_frame(key)
                    self._cache[key] = photo
                self.itemconfigure(self._img_item, image=photo)
                self._photo = photo  # PhotoImage must stay referenced or Tk blanks
                self._key_shown = key
        finally:
            # Reschedule even when this frame failed (a transient render error
            # or a TclError racing widget destruction): without this a single
            # exception would break the after-chain and freeze the face for
            # the rest of the session. Tk still reports the exception itself.
            # winfo_exists guards against a destroyed widget.
            if self.winfo_exists():
                self._loop_id = self.after(self._frame_ms, self._tick)

    # -- frame rendering ----------------------------------------------------

    def _render_frame(self, key: tuple) -> ImageTk.PhotoImage:
        """Render one frame for a quantized state key, antialiased.

        Drawn at ``_SUPERSAMPLE`` times the widget size on the opaque ``bg``
        colour (so edge pixels blend into the panel exactly), then downscaled
        with LANCZOS. Tk-thread only (creates a ``PhotoImage``).

        Key formats: ``("talk" | "rest", mouth_q, curl_q, blink, gqx, gqy,
        wide, cross)`` for the live face; ``("pop", idx)``, ``("gone",)`` and
        ``("appear", idx)`` for the click cartoon.
        """
        dim = min(self._cw, self._ch)
        if dim <= 1:  # not laid out yet; fall back to the requested size
            dim = self._size
        s = dim * _SUPERSAMPLE
        img = Image.new("RGB", (s, s), self._c_bg)
        d = ImageDraw.Draw(img)
        cx = cy = s / 2

        kind = key[0]
        if kind == "pop":
            self._paint_pop(d, cx, cy, s * 0.42, key[1])
        elif kind == "appear":
            r = s * 0.42 * appear_scale(key[1] / _ANIM_QFPS)
            if r >= _SUPERSAMPLE:  # skip sub-pixel discs on the first frames
                self._paint_face(d, cx, cy, r, mode="rest", openness=0.0,
                                 curl=1.0, blink=0, gaze=(0, 0, 0, False))
        elif kind == "gone":
            pass  # popped: background only
        else:
            mode, mouth_q, curl_q, blink, gqx, gqy, wide, cross = key
            self._paint_face(d, cx, cy, s * 0.42, mode=mode,
                             openness=mouth_q / _MOUTH_STEPS,
                             curl=curl_q / _CURL_STEPS, blink=blink,
                             gaze=(gqx, gqy, wide, cross))

        img = img.resize((dim, dim), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)

    def _paint_face(self, d: "ImageDraw.ImageDraw", cx: float, cy: float,
                    r: float, *, mode: str, openness: float, curl: float,
                    blink: int, gaze: tuple) -> None:
        """Draw the whole live face (disc, eyes, brows, mouth) at radius r.

        ``gaze`` is a ``gaze_state`` tuple: the eyes shift toward the cursor,
        widen with ``wide`` and turn to the nose bridge when ``cross``. ``r``
        is a parameter (not always ``s * 0.42``) so the re-inflate animation
        can draw the same face smaller.
        """
        # Features are laid out around face_cy, the disc stays centred on cy,
        # so face_offset shifts the face within the circle without moving it.
        face_cy = cy + self._face_offset * r

        # Face disc, with an optional thin rim (1 px after downscale).
        if self._c_outline is not None:
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=self._c_face,
                      outline=self._c_outline, width=_SUPERSAMPLE)
        else:
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=self._c_face)

        # Eyes: tall ellipses with a highlight; the blink squashes them, the
        # cursor shifts (gaze) and widens (wide) them, and ``cross`` pulls
        # both onto the nose bridge.
        gqx, gqy, wide, cross = gaze
        ey, edx = face_cy + r * _EYE_Y, r * _EYE_X
        wide_k = 1.0 + 0.16 * wide
        erx = r * 0.105 * (1.0 + 0.05 * wide)
        ery_open = r * 0.16 * wide_k
        ery = max(ery_open * (1.0, 0.45, 0.10)[blink], r * 0.02)
        gx = gqx / _GAZE_STEPS * r * 0.075
        gy = gqy / _GAZE_STEPS * r * 0.075
        # The highlight wanders inside the eye toward the cursor a bit
        # further than the eye itself moves: a two-layer parallax that reads
        # as a glossy reflection. _draw_eye clamps it inside the ellipse.
        sx = gqx / _GAZE_STEPS * erx * 0.30
        sy = gqy / _GAZE_STEPS * ery_open * 0.25
        for side in (-1, 1):
            if cross:  # both eyes to the nose bridge, a touch downward
                eye_x, eye_y = cx + side * (edx - r * 0.08), ey + r * 0.02
                # ... and the highlights squeeze toward the nose as well.
                shine = (-side * erx * 0.4, ery_open * 0.10)
            else:
                eye_x, eye_y = cx + side * edx + gx, ey + gy
                shine = (sx, sy)
            self._draw_eye(d, eye_x, eye_y, erx, ery, ery_open, blink, *shine)

        # Eyebrows: gentle arches that rise with a smile (and with surprise
        # at a close cursor); a frown raises the inner ends and drops the
        # outer ones (the classic "worried" tilt). Anchored to the resting
        # eye positions on purpose: brows staying put while the eyes dart
        # around reads funnier.
        brow_y = (ey - ery_open - r * 0.13 - curl * r * 0.045
                  - wide * r * 0.02)
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

        my = face_cy + r * _MOUTH_Y  # mouth centre line
        if mode == "talk":
            self._draw_talking_mouth(d, cx, my, r, openness)
        else:
            self._draw_smiley(d, cx, my, r, curl)

    def _paint_pop(self, d: "ImageDraw.ImageDraw", cx: float, cy: float,
                   r: float, idx: int) -> None:
        """Draw one cartoon frame: whatever ``pop_scene`` says is at idx."""
        scene = pop_scene(idx / _ANIM_QFPS)
        face_cy = cy + self._face_offset * r

        if scene["shards"] is not None:  # the rim flies apart
            rim = self._c_outline if self._c_outline is not None else self._c_face
            self._draw_shards(d, cx, cy, r, scene["shards"], rim)

        erx, ery_open = r * 0.105, r * 0.16
        rest_pos = ((-_EYE_X, _EYE_Y), (_EYE_X, _EYE_Y))
        for i, pos in enumerate(scene["eyes"]):  # eyes float / fall
            if pos is None:
                continue
            # A falling eye (moved off its resting spot) gets a highlight
            # blown up almost to the eye size: a wide-open "whee!" look.
            falling = pos != rest_pos[i]
            self._draw_eye(d, cx + pos[0] * r, face_cy + pos[1] * r,
                           erx, ery_open, ery_open, 0,
                           shine_scale=2.0 if falling else 1.0)

        my = face_cy + r * _MOUTH_Y
        if scene["mouth"] is not None:
            mouth_kind, value = scene["mouth"]
            if mouth_kind == "open":
                self._draw_catch_mouth(d, cx, my, r, value)
            else:  # "grin"
                self._draw_grin(d, cx, my, r, value)
        if scene["mouth_shards"] is not None:  # the grin pops too
            self._draw_shards(d, cx, my, r * 0.35, scene["mouth_shards"],
                              self._c_mouth)

    def _draw_eye(self, d: "ImageDraw.ImageDraw", ex: float, ey: float,
                  erx: float, ery: float, ery_open: float, blink: int,
                  shine_dx: float = 0.0, shine_dy: float = 0.0,
                  shine_scale: float = 1.0) -> None:
        """One eye: the dark ellipse plus, when open, the white highlight.

        ``shine_dx``/``shine_dy`` shift the highlight from its resting
        top-left spot, so the reflection can wander toward the cursor
        independently of the eye itself (see the parallax in _paint_face).
        ``shine_scale`` grows the highlight (the cartoon uses ~2.0 on falling
        eyes); the clamp below then pins it near the eye centre.
        """
        d.ellipse((ex - erx, ey - ery, ex + erx, ey + ery), fill=self._c_eye)
        if blink == 0:  # highlight only on a fully open eye, so ery == ery_open
            hr = erx * 0.42 * shine_scale
            # Clamp the highlight centre in ellipse-normalized coordinates so
            # the whole disc stays inside the eye whatever shift is asked
            # for: per-axis limits are not enough, a diagonal gaze otherwise
            # pushes it over the rim where the ellipse curves in.
            limit = 1.0 - hr / min(erx, ery_open)
            if limit > 0.0:
                nx = (-erx * 0.28 + shine_dx) / erx
                ny = (-ery_open * 0.38 + shine_dy) / ery_open
                norm = math.hypot(nx, ny)
                if norm > limit:
                    nx *= limit / norm
                    ny *= limit / norm
                hx, hy = ex + nx * erx, ey + ny * ery_open
                d.ellipse((hx - hr, hy - hr, hx + hr, hy + hr),
                          fill=self._c_shine)

    def _draw_catch_mouth(self, d: "ImageDraw.ImageDraw", cx: float,
                          my: float, r: float, openness: float) -> None:
        """Half-round open mouth for the cartoon: flat top, round belly.

        A true semicircle (equal semi-axes chord) instead of the talking
        ellipse: it reads as a scoop held up to catch the falling eye.
        """
        rad = r * (0.04 + 0.26 * openness)
        d.chord((cx - rad, my - rad, cx + rad, my + rad), 0, 180,
                fill=self._c_mouth_in)

    def _draw_grin(self, d: "ImageDraw.ImageDraw", cx: float, my: float,
                   r: float, k: float) -> None:
        """Wide-open grin: the bottom half of an ellipse, growing with ``k``.

        The straight chord on top plus the round belly below reads as an
        open-mouthed smile without any extra lip work.
        """
        w = r * (0.14 + 0.24 * k)
        h = r * (0.06 + 0.22 * k)
        d.chord((cx - w, my - h, cx + w, my + h), 0, 180, fill=self._c_mouth_in)

    def _draw_shards(self, d: "ImageDraw.ImageDraw", x0: float, y0: float,
                     base_r: float, k: float, color: tuple) -> None:
        """Balloon-pop debris: rim fragments flying outward and fading.

        Ten short tangent segments start on the circle of ``base_r`` and fly
        out as ``k`` goes 0 -> 1, shrinking and blending into the background,
        so the pop dissolves instead of littering the canvas.
        """
        fly = base_r * (0.15 + 0.85 * k)
        fade = _blend(color, self._c_bg, k)
        seg = base_r * 0.16 * (1.0 - 0.5 * k)
        width = max(2, int(base_r * 0.06))
        for i in range(10):
            a = (i / 10.0) * math.tau + 0.3  # offset: no axis-aligned shards
            px = x0 + math.cos(a) * (base_r + fly)
            py = y0 + math.sin(a) * (base_r + fly)
            tx, ty = -math.sin(a), math.cos(a)
            d.line((px - tx * seg, py - ty * seg, px + tx * seg, py + ty * seg),
                   fill=fade, width=width)

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
    # with ~3 s of "paused", cycling through the three smileys. The eyes
    # follow the mouse the whole time; during a pause, click and hold on the
    # face to watch the balloon-pop cartoon (clicks while talking are
    # ignored by design).
    root = tk.Tk()
    root.title("FaceWidget demo (move the mouse, click the face)")
    root.configure(bg="#1e1e1e")
    face = FaceWidget(root, size=220, bg="#1e1e1e",
                      face_color="#ffffff", eye_color="#1e2a44",
                      mouth_color="#8a3a2c", face_offset=0.0)
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
                state["left_ms"] = 3000  # long pause: time to click the face
            else:
                state["talking"] = True
                state["left_ms"] = 2500
        root.after(step_ms, _drive)

    _drive()
    root.mainloop()
