"""recall_ablation.py -- offline what-if over the recall weight, from the CSV.

WHY THIS EXISTS
---------------
The w2v2 engine's final score is a fixed blend of two axes (see
``w2v2_pronounce_poc.align_and_score``)::

    score = WEIGHT_PHONEME * phoneme_score + WEIGHT_WORD * recall * 100
          = (1 - w) * phoneme_score + w * 100 * recall          # w = WEIGHT_WORD

with ``WEIGHT_PHONEME = 0.7`` and ``WEIGHT_WORD = 0.3`` today. The open question
(raised in feedback/f1-1) is whether the separate ``recall`` axis *adds signal*
or just *adds noise* -- maybe alignment cost already captures missed words.

Because ``eval_results.csv`` logs ``phoneme_score`` and ``recall`` as their own
columns, we can answer that WITHOUT re-running any model: just recompute the
blended score for every recall weight ``w`` in a sweep and see how each choice
tracks the reference engine (Pearson/Spearman) and separates good from bad
speech (ROC-AUC). w=0 is the ablation (recall removed); w=0.30 is today's engine.

It also calibrates the global ``PHONEME_GOOD`` anchor (on by default): from the
native takes' own per-phone distances it recommends the anchor a clean native read
actually hits, shows the good/bad phoneme-mean shift, and writes it to
``calibration.json`` -- which ``w2v2_pronounce_poc`` reads at import (falling back
to its built-in defaults when the file is absent). So the test produces the
config; the POC consumes it.

This is intentionally a thin, reusable calibration tool, not a one-off:
  * Reuses ``eval_core``'s statistics verbatim, so numbers match ``run_eval.py``.
  * The score model is one small linear function (``blended_score``); adding a
    third axis later (duration penalty, posterior confidence) is a local change.
  * ``--out-csv`` dumps the full sweep (one row per weight) for plotting/fitting.
  * ``--good-percentile`` tunes the anchor; ``--no-write-config`` previews without
    touching the JSON; ``--config`` points at a different file.

NO MODELS, NO AUDIO. Reads a CSV, writes a report (+ calibration.json). Safe to
run repeatedly.

USAGE
-----
    # Defaults: logs/eval_results.csv, engine core_w2v2, ref core_prod,
    # good=mic bad=mistakes (the labels used in the last run).
    python prototypes/recall_ablation.py

    # Custom CSV / sweep / class labels, and dump the sweep for plotting:
    python prototypes/recall_ablation.py --csv logs/eval_results.csv \
        --weights 0:1:0.05 --good mic --bad mistakes --out-csv logs/recall_sweep.csv

    # Score the model-take ceiling instead of the user take:
    python prototypes/recall_ablation.py --variant ceiling

    # Calibrate PHONEME_GOOD at a stricter percentile, preview without writing:
    python prototypes/recall_ablation.py --good-percentile 90 --no-write-config
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (puts project root on sys.path; see README)

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from eval_core import _pearson, _spearman, agreement, discrimination

# Mirrors ``w2v2_pronounce_poc.BAD_MIN_SPAN`` -- keep in sync if it changes there.
# Used to replay the POC's phoneme-score mapping offline (see ``_phoneme_score``).
BAD_MIN_SPAN = 0.10


def _setup_logging(log_path: Path) -> None:
    """Tee the report to the screen and to ``log_path``, overwriting it each run.

    Same pattern as ``run_eval._setup_logging`` so the ablation log sits next to
    the eval log in the same format: console gets the bare message, the file gets
    a timestamped copy. We own the root logger's handlers (clearing any existing
    ones) so re-running in the same process never doubles lines.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(file_handler)
    root.addHandler(console)

# The engine's current split, kept here only as the "you are here" marker on the
# sweep. If the engine's weights change, update this constant so the report still
# highlights the live configuration. (The sanity check below verifies it.)
CURRENT_RECALL_WEIGHT = 0.30

# Default decision cutoff for verdict-agreement vs the reference. The score scale
# shifts as the weight changes, so verdict agreement at a *fixed* cutoff is a
# rough secondary signal -- read Spearman and AUC as the headline numbers. Each
# weight's own natural cutoff is reported separately as ``best_threshold``.
DEFAULT_PASS_THRESHOLD = 70.0


# ---------------------------------------------------------------------------
# Score model. One place defines how component axes combine into a final score.
# Generalising later = add a parameter/column here and in `recompute_scores`.
# ---------------------------------------------------------------------------
def blended_score(
    phoneme_score: float, recall: float, recall_weight: float
) -> float:
    """Final 0..100 score for a given recall weight ``w``.

    Mirrors ``w2v2_pronounce_poc.align_and_score`` exactly:
    ``(1 - w) * phoneme_score + w * 100 * recall``. ``phoneme_score`` is already
    on 0..100; ``recall`` is a 0..1 fraction, hence the ``* 100``.
    """
    return (1.0 - recall_weight) * phoneme_score + recall_weight * 100.0 * recall


# ---------------------------------------------------------------------------
# CSV loading. We pull only the columns the sweep needs and parse them safely,
# skipping rows where a required field is blank (e.g. an engine that errored on
# that sample) so a few gaps never poison the whole run.
# ---------------------------------------------------------------------------
@dataclass
class Row:
    dataset: str
    sample_id: str
    text: str
    phoneme_score: float       # <engine>_phoneme_score, 0..100
    recall: float              # <engine>_recall, 0..1
    ref_score: float           # reference engine score, 0..100
    ref_passed: bool           # reference engine pass/fail flag


def _to_float(value: str) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _to_bool(value: str) -> Optional[bool]:
    v = (value or "").strip().lower()
    if v in ("true", "1", "yes", "pass", "passed"):
        return True
    if v in ("false", "0", "no", "fail", "failed"):
        return False
    return None


def load_rows(
    csv_path: Path, engine: str, variant: str, ref_engine: str
) -> Tuple[List[Row], List[str]]:
    """Load the columns needed for the sweep. Returns (rows, skipped_notes)."""
    sub = "_ceiling" if variant == "ceiling" else ""
    col_phon = f"{engine}{sub}_phoneme_score"
    col_recall = f"{engine}{sub}_recall"
    col_ref_score = f"{ref_engine}_score"
    col_ref_passed = f"{ref_engine}_passed"

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in (col_phon, col_recall, col_ref_score, col_ref_passed)
                   if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(
                f"CSV {csv_path} is missing required column(s): {', '.join(missing)}.\n"
                f"Available: {', '.join(reader.fieldnames or [])}"
            )

        rows: List[Row] = []
        skipped: List[str] = []
        for raw in reader:
            phon = _to_float(raw[col_phon])
            rec = _to_float(raw[col_recall])
            ref = _to_float(raw[col_ref_score])
            refp = _to_bool(raw[col_ref_passed])
            ident = f"{raw.get('dataset', '?')}/{raw.get('id', '?')}"
            if None in (phon, rec, ref) or refp is None:
                skipped.append(f"{ident} (blank/non-numeric in a required column)")
                continue
            rows.append(Row(
                dataset=raw.get("dataset", "?"),
                sample_id=raw.get("id", "?"),
                text=raw.get("text", ""),
                phoneme_score=phon,
                recall=rec,
                ref_score=ref,
                ref_passed=refp,
            ))
    return rows, skipped


# ---------------------------------------------------------------------------
# Metrics for one recall weight.
# ---------------------------------------------------------------------------
@dataclass
class WeightMetrics:
    weight: float
    n: int
    # vs reference engine
    pearson: float
    spearman: float
    mae: float
    bias: float
    verdict_agreement: float
    test_mean: float
    # good-vs-bad separability (NaN when good/bad labels not supplied)
    auc: float
    best_threshold: float
    best_accuracy: float
    good_mean: float
    bad_mean: float


def recompute_scores(rows: Sequence[Row], weight: float) -> List[float]:
    return [blended_score(r.phoneme_score, r.recall, weight) for r in rows]


def evaluate_weight(
    rows: Sequence[Row],
    weight: float,
    pass_threshold: float,
    good_sets: Sequence[str],
    bad_sets: Sequence[str],
) -> WeightMetrics:
    test_scores = recompute_scores(rows, weight)
    test_passed = [s >= pass_threshold for s in test_scores]

    agr = agreement(
        reference_scores=[r.ref_score for r in rows],
        test_scores=test_scores,
        reference_passed=[r.ref_passed for r in rows],
        test_passed=test_passed,
    )

    good = [s for s, r in zip(test_scores, rows) if r.dataset in good_sets]
    bad = [s for s, r in zip(test_scores, rows) if r.dataset in bad_sets]
    disc = discrimination(good, bad)

    return WeightMetrics(
        weight=weight,
        n=agr.n,
        pearson=agr.pearson,
        spearman=agr.spearman,
        mae=agr.mae,
        bias=agr.bias,
        verdict_agreement=agr.verdict_agreement,
        test_mean=agr.test_mean,
        auc=disc.auc,
        best_threshold=disc.best_threshold,
        best_accuracy=disc.best_accuracy,
        good_mean=disc.good_mean,
        bad_mean=disc.bad_mean,
    )


# ---------------------------------------------------------------------------
# Sanity check: at the live weight, our recomputed score must match the engine's
# logged score. If it doesn't, the engine's formula has drifted from this script
# and every number below is suspect -- so we say so loudly rather than mislead.
# ---------------------------------------------------------------------------
def sanity_check(
    csv_path: Path, engine: str, variant: str, rows: Sequence[Row], tol: float = 0.15
) -> str:
    sub = "_ceiling" if variant == "ceiling" else ""
    logged_col = f"{engine}{sub}_score"
    with csv_path.open(newline="", encoding="utf-8") as fh:
        logged = {f"{r.get('dataset')}/{r.get('id')}": _to_float(r.get(logged_col, ""))
                  for r in csv.DictReader(fh)}

    worst_id, worst_diff = None, 0.0
    checked = 0
    for r in rows:
        ref = logged.get(f"{r.dataset}/{r.sample_id}")
        if ref is None:
            continue
        recomputed = blended_score(r.phoneme_score, r.recall, CURRENT_RECALL_WEIGHT)
        diff = abs(round(recomputed, 1) - ref)
        checked += 1
        if diff > worst_diff:
            worst_diff, worst_id = diff, f"{r.dataset}/{r.sample_id}"

    if checked == 0:
        return f"  [skip] no '{logged_col}' column to verify against."
    status = "OK" if worst_diff <= tol else "MISMATCH"
    note = "" if status == "OK" else (
        f"\n  WARNING: recomputed score diverges from logged '{logged_col}'. The "
        f"engine formula in w2v2_pronounce_poc may have changed -- update "
        f"blended_score()/CURRENT_RECALL_WEIGHT before trusting the sweep."
    )
    # ``worst_id`` stays None only when every row matched exactly (worst_diff 0);
    # omit the parenthetical then so the line doesn't read "(worst None)".
    worst = f" (worst {worst_id})" if worst_id is not None else ""
    return (f"  [{status}] recomputed vs logged at w={CURRENT_RECALL_WEIGHT}: "
            f"max |delta|={worst_diff:.2f} over {checked} rows{worst}.{note}")


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------
def _fmt(x: float, nd: int = 3) -> str:
    return "  n/a" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.{nd}f}"


def print_report(
    metrics: List[WeightMetrics],
    highlights: Dict[float, str],
    has_classes: bool,
) -> None:
    header = (
        f"{'w':>5} {'phon%':>6} {'rec%':>5} | {'Pear':>6} {'Spear':>6} "
        f"{'MAE':>6} {'bias':>7} {'verd%':>6} {'mean':>6}"
    )
    if has_classes:
        header += f" | {'AUC':>6} {'thr':>6} {'acc%':>6} {'good':>6} {'bad':>6}"
    header += "   note"
    logging.info(header)
    logging.info("-" * len(header))

    for m in metrics:
        phon_w = (1.0 - m.weight) * 100
        rec_w = m.weight * 100
        line = (
            f"{m.weight:>5.2f} {phon_w:>6.0f} {rec_w:>5.0f} | "
            f"{_fmt(m.pearson):>6} {_fmt(m.spearman):>6} "
            f"{_fmt(m.mae, 1):>6} {_fmt(m.bias, 1):>7} "
            f"{_fmt(m.verdict_agreement * 100, 0):>6} {_fmt(m.test_mean, 1):>6}"
        )
        if has_classes:
            line += (
                f" | {_fmt(m.auc):>6} {_fmt(m.best_threshold, 1):>6} "
                f"{_fmt(m.best_accuracy * 100, 0):>6} "
                f"{_fmt(m.good_mean, 1):>6} {_fmt(m.bad_mean, 1):>6}"
            )
        note = highlights.get(round(m.weight, 4), "")
        logging.info(f"{line}   {note}")


def verdict_summary(
    by_weight: Dict[float, WeightMetrics],
    ablated_w: float,
    current_w: float,
    has_classes: bool,
) -> None:
    """Plain-language 'does recall help' read-out: ablated vs current vs best."""
    ablated = by_weight.get(round(ablated_w, 4))
    current = by_weight.get(round(current_w, 4))
    if not ablated or not current:
        return

    logging.info("")
    logging.info("VERDICT  (does the recall axis add signal or noise?)")
    logging.info("-" * 60)

    # Primary: rank correlation with the reference engine.
    d_spear = current.spearman - ablated.spearman
    logging.info(f"  Spearman vs ref : recall-off(w=0) {_fmt(ablated.spearman)}  "
          f"-> current(w={current_w:.2f}) {_fmt(current.spearman)}  "
          f"(delta {d_spear:+.3f})")

    if has_classes:
        d_auc = current.auc - ablated.auc
        logging.info(f"  good/bad AUC    : recall-off(w=0) {_fmt(ablated.auc)}  "
              f"-> current(w={current_w:.2f}) {_fmt(current.auc)}  "
              f"(delta {d_auc:+.3f})")

    # Best weight by each objective (ignoring NaNs).
    def best_by(key, label):
        valid = [m for m in by_weight.values()
                 if not math.isnan(getattr(m, key))]
        if not valid:
            return
        best = max(valid, key=lambda m: getattr(m, key))
        logging.info(f"  best {label:<14}: w={best.weight:.2f}  "
              f"{label}={getattr(best, key):.3f}")

    best_by("spearman", "Spearman")
    if has_classes:
        best_by("auc", "AUC")

    # One-line takeaway on the headline metric (Spearman here).
    if abs(d_spear) < 0.01:
        takeaway = ("recall barely moves rank agreement -- it is close to "
                    "redundant on this set; the simpler phoneme-only score is "
                    "defensible.")
    elif d_spear < 0:
        takeaway = ("recall HURTS rank agreement here -- the ablation (w=0) "
                    "tracks the reference better. Treat recall as suspect.")
    else:
        takeaway = ("recall HELPS rank agreement -- keeping a recall axis is "
                    "justified; check the best-weight row for the sweet spot.")
    logging.info("")
    logging.info(f"  Takeaway: {takeaway}")
    logging.info("  (Small n -- read deltas as directional, confirm on a larger set.)")


def component_diagnostics(
    rows: Sequence[Row],
    good_sets: Sequence[str],
    bad_sets: Sequence[str],
    has_classes: bool,
) -> None:
    """Each raw axis as a *standalone* predictor -- the direct test of "is the
    phoneme axis weak/redundant?".

    A separate phoneme-weight sweep would just mirror the recall sweep (phoneme
    weight = 1 - w), so the endpoints already cover the pure axes: phoneme-only is
    the ``w=0`` row, recall-only the ``w=1`` row. What a sweep can NOT show, and
    this block adds: how correlated the two axes are with each other. If phoneme
    and recall move together (high collinearity), phoneme contributes little
    *independent* signal regardless of its own correlation with the reference.

    Both components are put on a common 0..100 scale (recall * 100) so their
    standalone numbers are directly comparable.
    """
    ref = np.asarray([r.ref_score for r in rows], dtype=float)
    phon = np.asarray([r.phoneme_score for r in rows], dtype=float)
    rec = np.asarray([r.recall * 100.0 for r in rows], dtype=float)

    logging.info("")
    logging.info("COMPONENT DIAGNOSTICS  (each axis alone, vs the reference)")
    logging.info("-" * 60)
    head = f"{'axis':<10} {'Pear(ref)':>10} {'Spear(ref)':>11}"
    if has_classes:
        head += f" | {'AUC':>6} {'good':>6} {'bad':>6}"
    logging.info(head)

    for name, vec in (("phoneme", phon), ("recall", rec)):
        line = f"{name:<10} {_fmt(_pearson(ref, vec)):>10} {_fmt(_spearman(ref, vec)):>11}"
        if has_classes:
            good = [v for v, r in zip(vec, rows) if r.dataset in good_sets]
            bad = [v for v, r in zip(vec, rows) if r.dataset in bad_sets]
            d = discrimination(good, bad)
            line += (f" | {_fmt(d.auc):>6} {_fmt(d.good_mean, 1):>6} "
                     f"{_fmt(d.bad_mean, 1):>6}")
        logging.info(line)

    # Redundancy between the two axes -- the part the sweep can't reveal.
    coll_p = _pearson(phon, rec)
    coll_s = _spearman(phon, rec)
    logging.info("")
    logging.info(f"  axis collinearity: Pearson(phoneme, recall) = {_fmt(coll_p)}, "
                 f"Spearman = {_fmt(coll_s)}")

    # One-line read tying standalone strength + redundancy together.
    p_phon, p_rec = _pearson(ref, phon), _pearson(ref, rec)
    if not (math.isnan(p_phon) or math.isnan(p_rec)):
        weaker = "phoneme" if abs(p_phon) < abs(p_rec) else "recall"
        gap = abs(abs(p_rec) - abs(p_phon))
        redundant = (not math.isnan(coll_p)) and abs(coll_p) >= 0.6
        msg = (f"the {weaker} axis is the weaker standalone predictor of the "
               f"reference (|Pearson| gap {gap:.2f})")
        if weaker == "phoneme" and redundant:
            msg += (" and is fairly collinear with recall -- so it carries little "
                    "*independent* signal here. Worth checking phoneme_score "
                    "quality / calibration before trusting that axis.")
        elif weaker == "phoneme":
            msg += (" but is only weakly collinear with recall -- it may still add "
                    "independent signal the reference doesn't reward.")
        logging.info(f"  Read: {msg}")
    logging.info("  (Reminder: the reference is prod, not human truth -- a weak "
                 "phoneme axis vs prod may mean prod itself is recall-dominated.)")


# ---------------------------------------------------------------------------
# PHONEME_GOOD calibration. The text-only POC path (no reference recording) scores
# the phoneme axis against a single global GOOD anchor, ``PHONEME_GOOD``, which
# defaults to 0.0 -- so only a *perfect* read maps to 100 and real good speech is
# under-scored. We derive a data-driven anchor from the native takes' own per-phone
# distances (the ``_ceiling_per_phone_distance`` column, i.e. each phrase's
# model.wav scored through the same pipeline): a percentile of that distribution is
# the distance a clean native read actually achieves, which is what should map to
# 100. We then replay the POC's mapping offline to show how good/bad phoneme means
# shift -- good should rise toward 100 while bad stays near 0 (separation kept).
# ---------------------------------------------------------------------------
@dataclass
class Calibration:
    ok: bool
    note: str = ""
    good_percentile: float = 0.0
    phoneme_good: float = 0.0           # recommended global PHONEME_GOOD
    n_native: int = 0
    native_p50: float = math.nan        # context: median native distance
    native_p90: float = math.nan
    good_phon_before: float = math.nan  # good-class mean phoneme score, good=0.0
    good_phon_after: float = math.nan   # ... and at the recommended anchor
    bad_phon_before: float = math.nan
    bad_phon_after: float = math.nan


def _phoneme_score(per_phone_distance: float, bad: float, good: float) -> float:
    """Replay ``w2v2_pronounce_poc._score_from_distance`` exactly, offline.

    ``good`` -> 100, ``bad`` -> 0, clamped, with a floor on the span so a tiny
    good/bad gap can't blow up. Lets us recompute phoneme scores for any candidate
    GOOD anchor straight from the CSV's distance + baseline columns.
    """
    span = max(bad - good, BAD_MIN_SPAN)
    accuracy = 1.0 - (per_phone_distance - good) / span
    return round(max(0.0, min(1.0, accuracy)) * 100.0, 1)


def compute_calibration(
    csv_path: Path,
    engine: str,
    good_percentile: float,
    good_sets: Sequence[str],
    bad_sets: Sequence[str],
) -> Calibration:
    """Recommend a global ``PHONEME_GOOD`` from native-take distances + show its effect.

    Always reads the *user-take* distance/baseline columns (the text-only path this
    anchor governs) and the *ceiling* native distances, independent of ``--variant``.
    Best-effort: if a required column is absent, returns ``ok=False`` with a note so
    the rest of the run is unaffected.
    """
    col_native = f"{engine}_ceiling_per_phone_distance"
    col_dist = f"{engine}_per_phone_distance"
    col_bad = f"{engine}_bad_baseline"

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        missing = [c for c in (col_native, col_dist, col_bad) if c not in fields]
        if missing:
            return Calibration(
                ok=False,
                note=f"calibration skipped: CSV lacks column(s) {', '.join(missing)}",
            )
        native: List[float] = []
        takes: List[Tuple[str, float, float]] = []  # (dataset, distance, bad)
        for raw in reader:
            nv = _to_float(raw[col_native])
            if nv is not None:
                native.append(nv)
            d, b = _to_float(raw[col_dist]), _to_float(raw[col_bad])
            if d is not None and b is not None:
                takes.append((raw.get("dataset", "?"), d, b))

    if not native:
        return Calibration(ok=False, note="calibration skipped: no native distances")

    arr = np.asarray(native, dtype=float)
    recommended = float(np.percentile(arr, good_percentile))

    def class_mean(datasets: Sequence[str], good: float) -> float:
        vals = [_phoneme_score(d, b, good) for ds, d, b in takes if ds in datasets]
        return float(np.mean(vals)) if vals else math.nan

    return Calibration(
        ok=True,
        good_percentile=good_percentile,
        phoneme_good=recommended,
        n_native=len(native),
        native_p50=float(np.percentile(arr, 50)),
        native_p90=float(np.percentile(arr, 90)),
        good_phon_before=class_mean(good_sets, 0.0),
        good_phon_after=class_mean(good_sets, recommended),
        bad_phon_before=class_mean(bad_sets, 0.0),
        bad_phon_after=class_mean(bad_sets, recommended),
    )


def print_calibration(cal: Calibration) -> None:
    logging.info("")
    logging.info("PHONEME_GOOD CALIBRATION  (global anchor for the text-only path)")
    logging.info("-" * 60)
    if not cal.ok:
        logging.info(f"  {cal.note}")
        return
    logging.info(f"  native per-phone distance: n={cal.n_native}  "
                 f"p50={_fmt(cal.native_p50)}  "
                 f"p{cal.good_percentile:g}={_fmt(cal.phoneme_good)}  "
                 f"p90={_fmt(cal.native_p90)}")
    logging.info(f"  -> recommended PHONEME_GOOD = {cal.phoneme_good:.4f}  "
                 f"(was 0.0)")
    logging.info(f"  good-class phoneme mean: {_fmt(cal.good_phon_before, 1)} "
                 f"-> {_fmt(cal.good_phon_after, 1)}   "
                 f"(should rise toward 100)")
    logging.info(f"  bad-class  phoneme mean: {_fmt(cal.bad_phon_before, 1)} "
                 f"-> {_fmt(cal.bad_phon_after, 1)}   "
                 f"(should stay near 0)")


def write_calibration_config(
    path: Path,
    cal: Calibration,
    csv_path: Path,
    suggested_weight_word: Optional[float],
) -> None:
    """Merge the recommended anchor into ``calibration.json`` (the POC reads it).

    Loads any existing file first so human-set keys (e.g. a hand-tuned
    ``weight_word``) survive; only ``phoneme_good`` and the informational
    ``_meta``/``_suggested`` blocks are (re)written. ``_suggested.weight_word`` is
    recorded but NOT applied, because the sweep that produced it (a) optimizes
    agreement with prod, not human truth, and (b) was computed under the OLD
    phoneme calibration -- re-run the sweep after this anchor takes effect before
    trusting it.
    """
    existing: dict = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except ValueError:
            existing = {}

    existing["phoneme_good"] = round(cal.phoneme_good, 6)
    existing["_meta"] = {
        "generated_by": "recall_ablation.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_csv": str(csv_path),
        "good_percentile": cal.good_percentile,
        "n_native": cal.n_native,
        "good_phoneme_mean_before": round(cal.good_phon_before, 1)
        if not math.isnan(cal.good_phon_before) else None,
        "good_phoneme_mean_after": round(cal.good_phon_after, 1)
        if not math.isnan(cal.good_phon_after) else None,
    }
    if suggested_weight_word is not None:
        existing["_suggested"] = {
            "weight_word": round(suggested_weight_word, 4),
            "note": "best Spearman vs prod on the current sweep; NOT applied "
                    "automatically -- fit to prod, not human truth, and computed "
                    "under the pre-calibration phoneme anchor. Re-run the sweep "
                    "after phoneme_good takes effect, and validate on an accented "
                    "(right words, bad pronunciation) set, before changing weights.",
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_sweep_csv(path: Path, metrics: List[WeightMetrics]) -> None:
    fields = ["weight", "phoneme_weight", "recall_weight", "n", "pearson",
              "spearman", "mae", "bias", "verdict_agreement", "test_mean",
              "auc", "best_threshold", "best_accuracy", "good_mean", "bad_mean"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for m in metrics:
            w.writerow({
                "weight": m.weight,
                "phoneme_weight": 1.0 - m.weight,
                "recall_weight": m.weight,
                "n": m.n,
                "pearson": m.pearson,
                "spearman": m.spearman,
                "mae": m.mae,
                "bias": m.bias,
                "verdict_agreement": m.verdict_agreement,
                "test_mean": m.test_mean,
                "auc": m.auc,
                "best_threshold": m.best_threshold,
                "best_accuracy": m.best_accuracy,
                "good_mean": m.good_mean,
                "bad_mean": m.bad_mean,
            })


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def parse_weights(spec: str) -> List[float]:
    """``start:stop:step`` (inclusive) or a comma list ``0,0.3,1``."""
    if ":" in spec:
        start, stop, step = (float(x) for x in spec.split(":"))
        if step <= 0:
            raise SystemExit("--weights step must be > 0")
        n = int(round((stop - start) / step))
        weights = [round(start + i * step, 6) for i in range(n + 1)]
    else:
        weights = [round(float(x), 6) for x in spec.split(",") if x.strip()]
    # Always include the live weight so the report can mark "you are here".
    if not any(abs(w - CURRENT_RECALL_WEIGHT) < 1e-9 for w in weights):
        weights.append(CURRENT_RECALL_WEIGHT)
    return sorted(set(weights))


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="Offline recall-weight ablation/sweep over eval_results.csv "
                    "(no models run).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--csv", type=Path, default=here / "logs" / "eval_results.csv",
                    help="eval results CSV produced by run_eval.py")
    ap.add_argument("--engine", default="core_w2v2",
                    help="engine column prefix to ablate")
    ap.add_argument("--variant", choices=("user", "ceiling"), default="user",
                    help="'user' = the scored take; 'ceiling' = the model take")
    ap.add_argument("--reference", default="core_prod",
                    help="reference engine column prefix to correlate against")
    ap.add_argument("--weights", default="0:1:0.05",
                    help="sweep as start:stop:step or comma list (live weight "
                         "is always added)")
    ap.add_argument("--good", nargs="*", default=["mic"],
                    help="dataset name(s) treated as good speech (for AUC)")
    ap.add_argument("--bad", nargs="*", default=["mistakes"],
                    help="dataset name(s) treated as bad speech (for AUC)")
    ap.add_argument("--pass-threshold", type=float, default=DEFAULT_PASS_THRESHOLD,
                    help="fixed cutoff for verdict-agreement vs reference")
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="optional path to dump the full sweep for plotting")
    ap.add_argument("--log", type=Path, default=here / "logs" / "recall_ablation.log",
                    help="report log file, overwritten each run (sits next to "
                         "run_eval's eval_run.log)")
    ap.add_argument("--good-percentile", type=float, default=75.0,
                    help="percentile of native-take per-phone distances used as "
                         "the recommended global PHONEME_GOOD")
    ap.add_argument("--config", type=Path, default=here / "calibration.json",
                    help="calibration JSON the POC reads; recommended phoneme_good "
                         "is written here (next to the scripts)")
    ap.add_argument("--no-write-config", action="store_true",
                    help="compute and print calibration but do not write the JSON")
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    # Set up tee logging before any report line is emitted so screen and file
    # stay in sync. (argparse/CSV-missing errors above go to stderr by design.)
    _setup_logging(args.log)

    rows, skipped = load_rows(args.csv, args.engine, args.variant, args.reference)
    if not rows:
        raise SystemExit("No usable rows after parsing -- check column names/CSV.")

    good_sets = set(args.good)
    bad_sets = set(args.bad)
    present = {r.dataset for r in rows}
    has_classes = bool(good_sets & present) and bool(bad_sets & present)

    weights = parse_weights(args.weights)
    metrics = [
        evaluate_weight(rows, w, args.pass_threshold, good_sets, bad_sets)
        for w in weights
    ]
    by_weight = {round(m.weight, 4): m for m in metrics}

    # Header / context.
    logging.info("=" * 78)
    logging.info("RECALL-WEIGHT ABLATION")
    logging.info("=" * 78)
    logging.info(f"  csv        : {args.csv}")
    logging.info(f"  engine     : {args.engine}  (variant={args.variant})")
    logging.info(f"  reference  : {args.reference}_score / _passed")
    logging.info(f"  samples    : {len(rows)} usable" +
          (f", {len(skipped)} skipped" if skipped else ""))
    logging.info(f"  classes    : good={sorted(good_sets & present) or '-'}  "
          f"bad={sorted(bad_sets & present) or '-'}" +
          ("" if has_classes else "   (AUC disabled: a class is absent)"))
    logging.info(f"  pass thr   : {args.pass_threshold} (verdict-agreement only)")
    logging.info(f"  model      : score(w) = (1-w)*phoneme + w*100*recall   "
          f"[w=recall weight]")
    logging.info(sanity_check(args.csv, args.engine, args.variant, rows))
    if skipped:
        for s in skipped[:5]:
            logging.info(f"    skipped: {s}")
        if len(skipped) > 5:
            logging.info("    ... and %d more", len(skipped) - 5)
    logging.info("")

    # Highlight the meaningful weights on the sweep.
    best_spear = max((m for m in metrics if not math.isnan(m.spearman)),
                     key=lambda m: m.spearman, default=None)
    best_auc = (max((m for m in metrics if not math.isnan(m.auc)),
                    key=lambda m: m.auc, default=None) if has_classes else None)
    highlights: Dict[float, str] = {}
    highlights[0.0] = "<- recall OFF (ablation)"
    highlights[round(CURRENT_RECALL_WEIGHT, 4)] = "<- CURRENT engine"
    if best_spear is not None:
        highlights[round(best_spear.weight, 4)] = (
            highlights.get(round(best_spear.weight, 4), "") + " [best Spearman]"
        ).strip()
    if best_auc is not None:
        highlights[round(best_auc.weight, 4)] = (
            highlights.get(round(best_auc.weight, 4), "") + " [best AUC]"
        ).strip()

    print_report(metrics, highlights, has_classes)
    verdict_summary(by_weight, ablated_w=0.0,
                    current_w=CURRENT_RECALL_WEIGHT, has_classes=has_classes)
    component_diagnostics(rows, good_sets, bad_sets, has_classes)

    cal = compute_calibration(args.csv, args.engine, args.good_percentile,
                              good_sets, bad_sets)
    print_calibration(cal)

    # Final recommended-values block, then the optional config write.
    logging.info("")
    logging.info("RECOMMENDED CALIBRATION")
    logging.info("-" * 60)
    if cal.ok:
        logging.info(f"  phoneme_good = {cal.phoneme_good:.4f}   "
                     f"(p{args.good_percentile:g} of native distances; written below)")
    else:
        logging.info(f"  phoneme_good : {cal.note}")
    if best_spear is not None:
        logging.info(f"  weight_word  = {best_spear.weight:.2f}   "
                     f"(suggested by best Spearman; NOT auto-applied -- see config note)")

    if cal.ok and not args.no_write_config:
        write_calibration_config(
            args.config, cal, args.csv,
            suggested_weight_word=best_spear.weight if best_spear else None,
        )
        logging.info("")
        logging.info(f"wrote calibration to {args.config}")

    if args.out_csv:
        write_sweep_csv(args.out_csv, metrics)
        logging.info("")
        logging.info(f"wrote sweep to {args.out_csv}")

    logging.info("")
    logging.info(f"log written to {args.log}")


if __name__ == "__main__":
    main()
