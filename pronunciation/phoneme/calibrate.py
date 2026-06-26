# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valery Kovalev

"""Semi-automatic re-anchoring of the phoneme GOOD anchor (on request).

The direct analog of ``pronunciation/acoustic/calibrate.py`` (task §5.3). Every
``analyze()`` call appends its components to ``logs/phoneme_samples.jsonl``; after a
practice session run:

    python pronunciation/phoneme/calibrate.py            # compute and write calibration.json
    python pronunciation/phoneme/calibrate.py --dry-run  # only show what would change

What it does, mirroring the acoustic floor calibration:
  * keeps only the current user's attempts (matched on the logged ``user_name``);
  * **drops reference self-tests** (``is_reference``): the Test button compares the
    TTS reference with itself, so its per-phone distance is ~0 and would drag the
    GOOD anchor toward zero, making every later take score far too low. (The
    acoustic calibrator drops its self-tests for the same reason.) Self-tests stay
    in the log as a ceiling sanity check, but never set the anchor;
  * keeps attempts that clearly matched the phrase -- high phoneme recall and no
    word flagged "bad" -- so their per-phone distance samples "good pronunciation,
    different speaker";
  * sets ``phoneme_good`` to a percentile of those distances (default 75), so a
    typical good attempt lands near the top of the phoneme-quality scale.

Only ``phoneme_good`` is rewritten; the 0-5 ``buckets`` / ``bucket_to_percent`` and
the gates are left untouched (they ship with the engine's model calibration and are
not fit from one user's session).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Run as a standalone script: put the project root on the path so both the
# pronunciation package and the host app resolve.
# parents[2]: calibrate.py -> phoneme -> pronunciation -> project root.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from pronunciation.phoneme import speech
# Composition root: this CLI is the only place in pronunciation/phoneme/ that reads
# the host application's settings, wiring them in through the dispatcher's
# engine.configure("phoneme"), which owns the single app-settings -> AnalyzerConfig
# mapping, so the analyzer core stays app-agnostic and the mapping is not duplicated here.
from mimora import config, engine


# A sample counts as a "good attempt" when the phrase clearly matched.
MIN_RECALL = 0.85          # most reference phones produced close to target
# Only the most recent samples are used: the log grows without bound, and old
# sessions (different mic, different voice habits) would skew the anchor.
MAX_SAMPLES_USED = 300
MIN_SAMPLES = 5
DEFAULT_PERCENTILE = 75    # percentile of good-attempt distances used as the anchor


def load_samples() -> list:
    """Read every JSON line from the sample log (tolerating a torn last line)."""
    path = speech.samples_file()
    if not path.exists():
        return []
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # tolerate a torn last line from a crashed session
    return samples


def _has_bad_word(sample: dict) -> bool:
    """True if any reference word in the sample was flagged "bad" (mispronounced)."""
    return any(w.get("level") == "bad" for w in sample.get("words", []))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-anchor the phoneme GOOD anchor from session samples.")
    parser.add_argument("--dry-run", action="store_true",
                        help="only show what would change, do not write calibration.json")
    parser.add_argument("--voice", default=None,
                        help="only use samples recorded with this Kokoro voice")
    parser.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE,
                        help="percentile of good-attempt distances used as phoneme_good")
    args = parser.parse_args()

    # Inject the application's settings so the sample-log path and user name match
    # the running app. The explicit "phoneme" keeps calibration on this engine
    # regardless of config.ENGINE.
    engine.configure("phoneme")

    samples = load_samples()
    samples = [s for s in samples if s.get("user_name", "") == config.USER_NAME]

    # Filter by voice *before* truncating to the most recent samples.
    if args.voice:
        samples = [s for s in samples if s.get("voice") == args.voice]
    samples = samples[-MAX_SAMPLES_USED:]

    n_reference = sum(1 for s in samples if s.get("is_reference"))
    good = [s for s in samples
            if not s.get("is_reference")
            and s.get("recall", 0.0) >= MIN_RECALL
            and not _has_bad_word(s)]

    print(f"Samples file:   {speech.samples_file()}")
    print(f"Current user:   {config.USER_NAME!r}")
    print(f"Total samples:  {len(samples)} (last {MAX_SAMPLES_USED} max"
          + (f", voice={args.voice}" if args.voice else "") + ")")
    print(f"Reference self-tests dropped: {n_reference}")
    print(f"Good attempts:  {len(good)} (recall>={MIN_RECALL}, no 'bad' word, "
          "excluding self-tests)")

    # The anchor depends on the reference voice; mixing several voices blurs it.
    voices = Counter(s.get("voice") or "<unknown>" for s in good)
    if len(voices) > 1:
        listing = ", ".join(f"{v}: {n}" for v, n in voices.most_common())
        print(f"WARNING: samples mix several voices ({listing}). "
              "Consider rerunning with --voice <name> for a tighter anchor.")

    if len(good) < MIN_SAMPLES:
        print(f"\nNot enough good attempts (need {MIN_SAMPLES}). Practice a few "
              "phrases in the app -- speak clearly so the words match -- then rerun.")
        return 1

    distances = np.array([s["per_phone_distance"] for s in good], dtype=float)
    new_good = float(np.percentile(distances, args.percentile))

    print(f"\nPer-phone distance of good attempts: "
          f"min={distances.min():.4f} p25={np.percentile(distances, 25):.4f} "
          f"median={np.median(distances):.4f} p75={np.percentile(distances, 75):.4f} "
          f"max={distances.max():.4f}")
    print(f"Current phoneme_good: {speech.current_phoneme_good():.4f}")
    print(f"Proposed phoneme_good (p{args.percentile:g}): {new_good:.4f}")

    if args.dry_run:
        print("\nDry run: calibration.json not written.")
        return 0

    speech.save_calibration(new_good, extra={
        "phoneme_good_percentile": args.percentile,
        "phoneme_good_samples_used": len(good),
        "phoneme_good_voice": args.voice,
    })
    print(f"\nWritten {speech.CALIBRATION_FILE}. Restart the app (it reads the new "
          "value at startup) to score with the new anchor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
