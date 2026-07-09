# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Per-run practice-session state: score tally and attempt history.

Extracted from the main controller so the pure bookkeeping (running average,
distinct-phrase count, trend arrows) lives in one Tk-free, unit-testable
place. The controller owns a SessionState instance, feeds it takes and
history records, and pushes the returned values into the view; this module
never touches widgets or threads.
"""

from collections import deque
from typing import Optional

# Entries kept in the attempt history; older entries drop off the top.
HISTORY_LIMIT = 10


class SessionState:
    """Session score tally plus the bounded attempt history for one app run.

    The tally feeds the hero card's progress ring: "Phrases: N" counts the
    distinct phrases practiced this run (a set of phrase texts); the average is
    the running mean over *every* scored attempt (repeats of one phrase each
    add to it).
    The two therefore count different things: unique phrases vs total
    attempts. Empty/zero at construction == reset on app start (no explicit
    reset action).

    The history holds the last ``history_limit`` entries - scored takes,
    unscored ("none" engine) takes and error messages - oldest first. It
    lives here so the trend arrow (this take vs the previous attempt of the
    same phrase) can be computed from the retained entries.
    """

    def __init__(self, history_limit: int = HISTORY_LIMIT):
        self._phrases: set[str] = set()
        self._score_sum: float = 0.0
        self._attempts: int = 0
        self._history: deque = deque(maxlen=history_limit)

    def record_take(self, phrase: str, score: float,
                    graded: bool = False) -> Optional[tuple[int, float, float]]:
        """Fold one scored take into the session tally.

        Returns ``(distinct_phrase_count, average, maximum)`` for the
        progress ring, or None for a blank phrase (nothing to record).
        ``graded`` says which scale ``score`` is on: True for the phoneme
        engine's 0-5 grade axis (maximum 5), False for a raw 0-100 score
        (maximum 100). Plain numbers on purpose: the ring formats them
        itself, and a display string would have to be parsed back.
        """
        phrase = (phrase or "").strip()
        if not phrase:
            return None
        self._phrases.add(phrase)
        self._score_sum += score
        self._attempts += 1
        average = self._score_sum / self._attempts
        return len(self._phrases), average, (5.0 if graded else 100.0)

    def add_history_entry(self, record: dict) -> list:
        """Append *record* to the bounded history and return the full list.

        ``record`` carries a ``kind`` of "attempt", "unscored" or "error".
        For a scored take, the trend arrow is derived here by comparing the
        new score with the most recent earlier attempt of the *same* phrase:
        "up" if higher, "down" if lower, "same" if equal, and left unset (a
        dim dash) when there is no earlier attempt to compare against.
        Errors and unscored takes carry no trend. The deque is capped, so
        old entries drop off the top; the caller re-renders every row from
        the returned list.
        """
        if record.get("kind") == "attempt":
            record["trend"] = self._trend(record.get("phrase", ""),
                                          record.get("score", 0.0),
                                          record.get("score_text"))
        self._history.append(record)
        return list(self._history)

    def _trend(self, phrase: str, score: float,
               score_text: Optional[str] = None) -> Optional[str]:
        """Trend of ``score`` vs the previous attempt of ``phrase``.

        Compared on the *displayed* mark, not the raw float, so the arrow
        always agrees with the two chips the user sees. When both takes
        carry a ``score_text`` (the "4+"-style grade chip), equal texts read
        as "same" and the direction comes from the numeric ``score`` behind
        them (grade texts do not order lexically: "4-" < "4"). Without texts
        the comparison falls back to the rounded numeric score: 82 vs 82 is
        "same", never a stray up/down from a sub-point difference like
        82.4 vs 81.6.
        """
        for past in reversed(self._history):
            if past.get("kind") == "attempt" and past.get("phrase") == phrase:
                previous = past.get("score", 0.0)
                previous_text = past.get("score_text")
                if score_text and previous_text:
                    if score_text == previous_text:
                        return "same"
                    return "up" if score > previous else "down"
                if round(score) > round(previous):
                    return "up"
                if round(score) < round(previous):
                    return "down"
                return "same"
        return None
