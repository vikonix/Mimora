"""Multi-engine pronunciation evaluation harness.

Runs the reference engine (``core_prod``) and one or more test engines
(currently ``core_w2v2``) over every recording in one or more datasets, then
reports, per dataset, how closely each test engine tracks the reference.

Pipeline
--------
1. Build the engines and ``init`` each once (models load here, not per sample).
2. For every dataset directory given, walk its numbered subfolders (the
   ``records/`` layout: ``normalized.wav`` user take, ``model.wav`` reference,
   ``phrase.txt`` text). Run ``core_prod`` first (the reference), then each test
   engine, on the same sample.
3. Write every per-sample score to a CSV, and log a per-dataset summary comparing
   each test engine to the reference (correlation, MAE, bias, verdict agreement).

The question this answers: is the light text-only engine a viable **alternative**
to the production core -- i.e. does it score "not worse" and track its verdicts?

English only for now (``--lang en``); the reference core is calibrated on English.

Run
---
    # Point at a top-level directory; datasets inside are found automatically.
    # Naming which datasets are good/bad adds ROC-AUC + best-threshold separability:
    python prototypes/run_eval.py "C:/VOICE_DATASET/ENGLISH/VKO" --good mic --bad mistakes

    # Or name datasets directly; mix freely. GPU + cap for a quick smoke run:
    python prototypes/run_eval.py vko_mic vko_bt --device cuda --limit 5

Datasets are discovered by **content**, not by folder name: a folder holding the
sample files is a sample, a folder holding sample folders is a dataset, and a
folder holding datasets is a collection. So subfolder names may be anything. The
sample *filenames* are overridable (``--user-name`` etc.); defaults match the
project's ``records/`` sample.

Status: prototype evaluation tooling. Not wired into the GUI.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Side-effect import: project root on sys.path + espeak registration + logging.
import _bootstrap  # noqa: F401

from eval_core import (
    Engine, EngineResult, Sample, agreement, discover_datasets, discrimination,
    iter_samples,
)
from core_prod import ProdEngine
from core_w2v2 import W2V2Engine


# One scored row: the sample identity plus, for each engine, its score/verdict.
# Engine results are keyed by engine name; a value of None means that engine
# raised on this sample (logged, then skipped in the statistics).
Row = Dict[str, object]


def _score_dataset(
    name: str,
    path: Path,
    reference: Engine,
    test_engines: List[Engine],
    layout: Dict[str, str],
    limit: Optional[int],
) -> List[Row]:
    """Run every engine over one dataset directory and collect per-sample rows.

    A failure in one engine on one sample is logged and recorded as a ``None``
    score for that engine only -- it never aborts the dataset, so one bad file
    cannot lose the whole run.
    """
    engines = [reference, *test_engines]
    skipped: List[str] = []
    rows: List[Row] = []

    samples = iter_samples(
        path,
        user_name=layout["user"],
        reference_name=layout["reference"],
        text_name=layout["text"],
        skipped=skipped,
    )

    for count, sample in enumerate(samples):
        if limit is not None and count >= limit:
            break

        row: Row = {"dataset": name, "id": sample.id, "text": sample.text}
        results: Dict[str, Optional[EngineResult]] = {}
        for engine in engines:
            result = _safe_parse(engine, sample)
            results[engine.name] = result
            row[f"{engine.name}_score"] = None if result is None else result.score
            row[f"{engine.name}_passed"] = None if result is None else result.passed
        _log_sample(name, sample, engines, results)
        rows.append(row)

    if skipped:
        logging.info("  skipped %d folder(s): %s", len(skipped), "; ".join(skipped))
    if not rows:
        logging.warning("  no scorable samples found in %s", path)
    return rows


def _setup_logging(log_path: Path) -> None:
    """Tee output to the screen and to ``log_path``, **overwriting** it each run.

    Deliberately not ``_bootstrap.setup_logging`` (which *appends* to the shared
    ``prototype.log`` so successive POC runs accumulate): an eval run wants a clean,
    self-contained log it can hand over alongside the CSV. We own the root logger's
    handlers here, so any pre-existing ones are cleared first to avoid duplicate
    lines, and library logs (e.g. ``pronounce``'s per-phrase line) are still
    captured because they propagate to the root.
    """
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


def _log_sample(
    dataset: str,
    sample: Sample,
    engines: List[Engine],
    results: Dict[str, Optional[EngineResult]],
) -> None:
    """Log a per-phrase block: every engine's score, verdict and own details.

    This is the detail the early single-file POCs printed (IPA reference/spoken,
    transcription, sub-scores) and which the summary-only harness had dropped.
    Logging both engines side by side per sample is what makes outliers (a clean
    take that one engine tanks) eyeballable straight from the log.
    """
    logging.info("[sample] %s/%s  %r", dataset, sample.id, sample.text)
    for engine in engines:
        result = results.get(engine.name)
        if result is None:
            logging.info("  %-10s: FAILED", engine.name)
            continue
        verdict = "pass" if result.passed else "fail"
        extras = "  ".join(f"{k}={v:.3g}" for k, v in result.extra.items())
        logging.info("  %-10s: %5.1f (%s)  %s", engine.name, result.score, verdict, extras)
        if result.detail:
            logging.info("              %s", result.detail)


def _safe_parse(engine: Engine, sample: Sample) -> Optional[EngineResult]:
    """Run one engine on one sample, turning any failure into a logged ``None``."""
    try:
        return engine.parse(sample)
    except Exception as exc:  # noqa: BLE001 -- a bad file must not kill the run
        logging.warning("  %s failed on sample %s: %s", engine.name, sample.id, exc)
        return None


def _report_dataset(
    dataset_name: str, rows: List[Row], reference: Engine, test_engines: List[Engine]
) -> None:
    """Log the per-dataset comparison of each test engine against the reference."""
    ref_score_key = f"{reference.name}_score"
    ref_pass_key = f"{reference.name}_passed"

    logging.info("=" * 60)
    logging.info("dataset: %s  (%d samples)", dataset_name, len(rows))

    for engine in test_engines:
        score_key = f"{engine.name}_score"
        pass_key = f"{engine.name}_passed"

        # Pair only samples where BOTH the reference and this engine succeeded.
        ref_scores, test_scores, ref_passed, test_passed = [], [], [], []
        for row in rows:
            if row[ref_score_key] is None or row[score_key] is None:
                continue
            ref_scores.append(float(row[ref_score_key]))
            test_scores.append(float(row[score_key]))
            ref_passed.append(bool(row[ref_pass_key]))
            test_passed.append(bool(row[pass_key]))

        stats = agreement(ref_scores, test_scores, ref_passed, test_passed)
        logging.info("-" * 60)
        logging.info("%s  vs  %s  (paired n=%d)", engine.name, reference.name, stats.n)
        logging.info("  mean score     : %s %.1f   vs   %s %.1f",
                     engine.name, stats.test_mean, reference.name, stats.ref_mean)
        logging.info("  Pearson r      : %.3f", stats.pearson)
        logging.info("  Spearman rho   : %.3f", stats.spearman)
        logging.info("  MAE            : %.1f points", stats.mae)
        logging.info("  bias (test-ref): %+.1f points", stats.bias)
        logging.info("  verdict agree  : %.0f%% (pass/fail match at threshold)",
                     stats.verdict_agreement * 100)
    logging.info("=" * 60)
    logging.info("")


def _report_pooled(
    rows: List[Row], reference: Engine, test_engines: List[Engine]
) -> None:
    """Compare each test engine to the reference over ALL samples pooled together.

    Pooling the datasets widens the score range (good + noisy + bad), so the
    correlation is meaningful -- unlike a within-class correlation (e.g. mic
    only), where every score is bunched high and the number is dominated by noise.
    """
    logging.info("#" * 60)
    logging.info("POOLED across all datasets (%d samples)", len(rows))
    _report_dataset("ALL (pooled)", rows, reference, test_engines)


def _scores_for(rows: List[Row], engine_name: str, datasets: List[str]) -> List[float]:
    """Successful scores of one engine restricted to the named datasets."""
    wanted = set(datasets)
    key = f"{engine_name}_score"
    return [
        float(row[key])
        for row in rows
        if row["dataset"] in wanted and row[key] is not None
    ]


def _report_discrimination(
    rows: List[Row], engines: List[Engine], good: List[str], bad: List[str]
) -> None:
    """For each engine, how well its scores separate the good from the bad sets.

    Computed identically for the reference and every test engine, so their AUCs
    are directly comparable -- the core question being "does the alternative rank
    good-above-bad as well as core". AUC ignores the score offset, isolating
    ranking quality from calibration.
    """
    logging.info("#" * 60)
    logging.info("CLASS SEPARABILITY  good=%s  vs  bad=%s", good, bad)
    logging.info("(AUC: 1.0 perfect ranking, 0.5 chance; threshold = best split)")
    for engine in engines:
        stats = discrimination(
            _scores_for(rows, engine.name, good),
            _scores_for(rows, engine.name, bad),
        )
        logging.info("-" * 60)
        logging.info("%s  (good n=%d, bad n=%d)", engine.name, stats.n_good, stats.n_bad)
        logging.info("  good mean %.1f  vs  bad mean %.1f  (gap %.1f)",
                     stats.good_mean, stats.bad_mean,
                     stats.good_mean - stats.bad_mean)
        logging.info("  ROC-AUC        : %.3f", stats.auc)
        logging.info("  best threshold : %.1f  -> accuracy %.0f%%",
                     stats.best_threshold, stats.best_accuracy * 100)
    logging.info("#" * 60)
    logging.info("")


def _write_csv(path: Path, rows: List[Row], engines: List[Engine]) -> None:
    """Write all per-sample rows to ``path`` (one row per sample, all datasets)."""
    fieldnames = ["dataset", "id", "text"]
    for engine in engines:
        fieldnames += [f"{engine.name}_score", f"{engine.name}_passed"]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "datasets",
        nargs="+",
        help="one or more paths. Each may be a dataset (its subfolders are sample "
             "folders) OR a top-level directory whose subfolders are datasets "
             "(e.g. VKO/ containing mic/, bt/, mistakes/). Folder names are "
             "irrelevant -- datasets are found by content, not by name.",
    )
    parser.add_argument("--lang", default="en",
                       help="language key for the test engine (default: en)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                       help="device for the wav2vec2 models (default: cpu)")
    parser.add_argument("--threshold", type=float, default=70.0,
                       help="pass threshold for the test engine verdict (default: 70)")
    parser.add_argument("--limit", type=int, default=None,
                       help="score at most N samples per dataset (smoke testing)")
    parser.add_argument("--good", nargs="*", default=[], metavar="DATASET",
                       help="dataset names that are GOOD speech (for AUC/separability)")
    parser.add_argument("--bad", nargs="*", default=[], metavar="DATASET",
                       help="dataset names that are BAD speech (for AUC/separability)")
    parser.add_argument("--csv", default=str(_bootstrap.PROJECT_ROOT
                                             / "prototypes" / "eval_results.csv"),
                       help="per-sample CSV (overwritten each run)")
    parser.add_argument("--log", default=str(_bootstrap.PROJECT_ROOT
                                            / "prototypes" / "eval_run.log"),
                       help="run log (overwritten each run)")
    parser.add_argument("--user-name", default="normalized.wav",
                       help="user-take filename inside each subfolder")
    parser.add_argument("--reference-name", default="model.wav",
                       help="reference-take filename inside each subfolder")
    parser.add_argument("--text-name", default="phrase.txt",
                       help="phrase-text filename inside each subfolder")
    args = parser.parse_args()

    # Fresh log every run (overwrites), separate from the shared prototype.log.
    _setup_logging(Path(args.log))

    layout = {"user": args.user_name,
              "reference": args.reference_name,
              "text": args.text_name}

    reference: Engine = ProdEngine()
    test_engines: List[Engine] = [
        W2V2Engine(lang=args.lang, device=args.device, threshold=args.threshold),
    ]
    all_engines = [reference, *test_engines]

    # Initialize once -- this is the heavy model-loading step.
    logging.info("initializing engines: %s", ", ".join(e.name for e in all_engines))
    for engine in all_engines:
        engine.init()

    # Expand each given path into the datasets it contains: a path may itself be
    # a dataset (subfolders are samples) or a collection (subfolders are datasets,
    # e.g. VKO/{mic,bt,mistakes}). Dedupe by resolved path, keep discovery order.
    datasets: List[tuple] = []
    seen = set()
    for given in args.datasets:
        root = Path(given).expanduser().resolve()
        found = discover_datasets(
            root,
            user_name=layout["user"],
            reference_name=layout["reference"],
            text_name=layout["text"],
        )
        if not found:
            logging.warning("no datasets found under %s", root)
        for name, path in found:
            if path in seen:
                continue
            seen.add(path)
            datasets.append((name, path))

    logging.info("discovered %d dataset(s): %s",
                 len(datasets), ", ".join(name for name, _ in datasets))

    all_rows: List[Row] = []
    try:
        for name, path in datasets:
            logging.info("scoring dataset: %s  (%s)", name, path)
            rows = _score_dataset(name, path, reference, test_engines, layout, args.limit)
            _report_dataset(name, rows, reference, test_engines)
            all_rows.extend(rows)
    finally:
        for engine in all_engines:
            engine.close()

    if all_rows:
        _report_pooled(all_rows, reference, test_engines)
        if args.good and args.bad:
            _report_discrimination(all_rows, all_engines, args.good, args.bad)
        else:
            logging.info("(pass --good/--bad dataset names for AUC/separability)")

        csv_path = Path(args.csv)
        _write_csv(csv_path, all_rows, all_engines)
        logging.info("wrote %d rows to %s", len(all_rows), csv_path)


if __name__ == "__main__":
    main()
