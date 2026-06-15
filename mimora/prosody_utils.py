"""Pure helpers for prosody visualisation (no heavy dependencies).

These functions used to live as static methods on the UI mixin. They hold no UI
state and only do arithmetic, so they live here instead: that keeps them unit-
testable without tkinter, and keeps this module importable without pulling in
the ML stack (``pronounce.speech`` imports torch/transformers, so the helpers
deliberately do *not* live there).
"""

import math
from typing import List, Sequence


def resample_series(values: Sequence[float], target: int = 160) -> List[float]:
    """Evenly resample a 1-D sequence down to ``target`` points for plotting.

    Prosody contours can be hundreds of frames long; thinning keeps the
    sparkline light without changing its shape. Sequences already at or below
    ``target`` length are returned unchanged (as a list).
    """
    n = len(values)
    if n <= target:
        return list(values)
    step = (n - 1) / (target - 1)
    return [values[int(round(i * step))] for i in range(target)]


def to_semitones(values: Sequence[float]) -> List[float]:
    """Convert an F0 (Hz) contour to semitones relative to its own median.

    The reference and the user voice sit in different registers (a female
    Kokoro voice ~200 Hz vs a male user ~110 Hz), so plotting raw Hz on a
    shared axis puts the two curves in separate bands and hides whether the
    *intonation shape* matches. Re-expressing each contour as
    ``12 * log2(f0 / median)`` centres both on 0 semitones, so identical
    intonation overlaps regardless of pitch register.

    The median is taken over voiced frames (f0 > 0) only; unvoiced/zero frames
    are mapped to 0 ST (the speaker's centre) rather than -inf. An empty or
    fully unvoiced contour is returned unchanged so a silent take cannot crash
    the draw.
    """
    voiced = [v for v in values if v > 0]
    if not voiced:
        return list(values)
    median = sorted(voiced)[len(voiced) // 2]
    if median <= 0:
        return list(values)
    return [12.0 * math.log2(v / median) if v > 0 else 0.0 for v in values]
