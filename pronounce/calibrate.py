"""Semi-automatic calibration of the acoustic scoring floor (on request).

Every ``analyze()`` call appends its raw components to logs/pronounce_samples.jsonl.
After a practice session (10+ honest attempts), run:

    python pronounce/calibrate.py            # compute and write calibration.json
    python pronounce/calibrate.py --dry-run  # only show what would change

How it works:
  * keeps attempts whose *text* matched well — low word/phoneme error rates mean
    the user really said the expected phrase, so its acoustic distance is a
    sample of "good pronunciation, different speaker";
  * drops reference self-tests (the Test button compares the reference with
    itself, acoustic distance ~ 0, which would drag the floor down);
  * sets the acoustic floor to the 10th percentile of the remaining distances,
    so a typical good attempt lands near the top of the acoustic scale.

The ceiling needs no calibration: it is derived per utterance from the
random-pair baseline (see speech.acoustic_bad_for).
"""

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from pronounce import speech

# A sample counts as a "good attempt" when the text clearly matched.
MAX_WORD_ERROR_RATE = 0.20
MAX_PHONEME_ERROR_RATE = 0.25
# Below this acoustic distance the attempt is a reference self-test, not speech.
SELF_TEST_ACOUSTIC = 0.02
MIN_SAMPLES = 5
FLOOR_PERCENTILE = 10


def load_samples() -> list:
    if not speech.SAMPLES_FILE.exists():
        return []
    samples = []
    for line in speech.SAMPLES_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # tolerate a torn last line from a crashed session
    return samples


def main() -> int:
    dry_run = "--dry-run" in sys.argv[1:]

    samples = load_samples()
    good = [s for s in samples
            if s.get("acoustic_per_step", 0) >= SELF_TEST_ACOUSTIC
            and s.get("word_error_rate", 1) <= MAX_WORD_ERROR_RATE
            and s.get("phoneme_error_rate", 1) <= MAX_PHONEME_ERROR_RATE]

    print(f"Samples file:   {speech.SAMPLES_FILE}")
    print(f"Total samples:  {len(samples)}")
    print(f"Good attempts:  {len(good)} (word_err<={MAX_WORD_ERROR_RATE}, "
          f"phoneme_err<={MAX_PHONEME_ERROR_RATE}, excluding self-tests)")

    if len(good) < MIN_SAMPLES:
        print(f"\nNot enough good attempts (need {MIN_SAMPLES}). Practice a few "
              "phrases in the app — speak clearly so the text matches — then rerun.")
        return 1

    distances = np.array([s["acoustic_per_step"] for s in good])
    new_floor = float(np.percentile(distances, FLOOR_PERCENTILE))

    print(f"\nAcoustic per-step distance of good attempts: "
          f"min={distances.min():.4f} p10={np.percentile(distances, 10):.4f} "
          f"median={np.median(distances):.4f} max={distances.max():.4f}")
    print(f"Current floor (acoustic_good): {speech.ACOUSTIC_GOOD:.4f}")
    print(f"Proposed floor:                {new_floor:.4f}")

    # Project how the good attempts would have scored with the new floor.
    before, after = [], []
    for s in good:
        bad_b = speech.acoustic_bad_for(s["acoustic_baseline"])
        bad_a = speech.acoustic_bad_for(s["acoustic_baseline"], acoustic_good=new_floor)
        before.append(speech.compute_pronunciation_score(
            s["acoustic_per_step"], s["phoneme_error_rate"], s["word_error_rate"],
            acoustic_bad=bad_b))
        after.append(speech.compute_pronunciation_score(
            s["acoustic_per_step"], s["phoneme_error_rate"], s["word_error_rate"],
            acoustic_bad=bad_a, acoustic_good=new_floor))
    print(f"Median score of good attempts: {np.median(before):.1f} -> {np.median(after):.1f} "
          f"(pass threshold: {speech.SCORE_THRESHOLD})")

    if dry_run:
        print("\nDry run: calibration.json not written.")
        return 0

    speech.save_calibration(new_floor, extra={"samples_used": len(good)})
    print(f"\nWritten {speech.CALIBRATION_FILE}. Restart the app (or it picks the "
          "value up on next launch) to score with the new floor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
