"""Shared contracts for the multi-engine pronunciation evaluation harness.

This module is the small, dependency-light core that the engines (``core_*.py``)
and the runner (``run_eval.py``) build on. It defines:

* ``Sample``      -- one recording folder (a copy of ``records/``: user take,
                     reference take, phrase text).
* ``EngineResult``-- the normalized output every engine must return.
* ``Engine``      -- the common ``init`` / ``parse`` / ``close`` protocol.
* ``iter_samples``-- walk a dataset directory of numbered subfolders.
* statistics helpers (Pearson, Spearman, MAE, bias, verdict agreement) used to
  compare each test engine against the ``core_prod`` reference.

Why a separate module: the runner wires engines and datasets together, but the
contracts here are what keeps engines interchangeable. Keeping them free of heavy
imports (only stdlib + numpy) means importing an engine never drags in the whole
harness, and vice versa.

Status: prototype evaluation tooling. Not wired into the GUI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Dataset element. One folder == one (user, reference, text) triple, mirroring
# the layout of the project's ``records/`` sample so existing recordings drop in
# unchanged. The exact filenames are configurable in ``iter_samples`` because the
# final dataset layout is not fixed yet; these are the ``records/`` defaults.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Sample:
    id: str                       # folder name, e.g. "001"
    text: str                     # phrase the speaker read (from phrase.txt)
    user_audio: Path              # the attempt to score (normalized.wav)
    reference_audio: Path         # reference take of the same phrase (model.wav)


# ---------------------------------------------------------------------------
# Normalized engine output. Every engine returns this regardless of its internals
# so the runner can tabulate them uniformly. ``detail`` is free-form (IPA strings,
# transcription, alignment) and is only logged, never parsed.
# ---------------------------------------------------------------------------
@dataclass
class EngineResult:
    score: float                  # 0..100 overall pronunciation score
    passed: bool                  # score >= the engine's pass threshold
    detail: str = ""              # human-readable extras for the log
    extra: Dict[str, float] = field(default_factory=dict)  # numeric sub-scores


# ---------------------------------------------------------------------------
# The engine contract. ``init`` is called once before any ``parse`` (load the
# model here -- it is heavy), ``parse`` scores one sample, ``close`` releases
# resources after the last sample. ``runtime_checkable`` lets the runner assert
# an object satisfies the protocol without inheritance.
# ---------------------------------------------------------------------------
@runtime_checkable
class Engine(Protocol):
    name: str

    def init(self) -> None:
        """Load models / warm caches. Called once, before the first ``parse``."""
        ...

    def parse(self, sample: Sample) -> EngineResult:
        """Score one ``Sample`` and return a normalized ``EngineResult``."""
        ...

    def close(self) -> None:
        """Release any resources. Called once, after the last ``parse``."""
        ...


# ---------------------------------------------------------------------------
# Dataset iteration. A dataset is a directory whose immediate subfolders each
# hold one recording (the ``records/`` layout). Subfolders missing a required
# file are skipped with a note in ``skipped`` so a half-populated dataset does
# not crash the run. Folders are yielded in sorted name order for reproducibility.
# ---------------------------------------------------------------------------
def is_sample_folder(
    folder: Path,
    *,
    user_name: str = "normalized.wav",
    reference_name: str = "model.wav",
    text_name: str = "phrase.txt",
) -> bool:
    """True if ``folder`` directly holds the files of one recording (records/ layout)."""
    return all(
        (folder / name).is_file()
        for name in (user_name, reference_name, text_name)
    )


def discover_datasets(
    root: Path,
    *,
    user_name: str = "normalized.wav",
    reference_name: str = "model.wav",
    text_name: str = "phrase.txt",
) -> List[Tuple[str, Path]]:
    """Find the datasets under ``root``, returning ``(name, path)`` pairs.

    Two shapes are supported so the caller can point at either level:

    * ``root`` *is* a dataset -- its immediate subfolders are sample folders
      (e.g. ``vko_mic/001/...``). Returns a single ``(root.name, root)``.
    * ``root`` is a **collection** of datasets -- its subfolders are not sample
      folders themselves but each contains sample folders (e.g.
      ``VKO/{mic,bt,mistakes}/001/...``). Returns one pair per such subfolder,
      named after the subfolder.

    The check is "does this folder have at least one sample-folder child", so an
    empty or half-populated subfolder (e.g. ``bt/004`` with no files) neither
    triggers nor blocks detection. Returned in sorted name order.
    """
    if not root.is_dir():
        raise NotADirectoryError(f"path is not a directory: {root}")

    names = dict(user_name=user_name, reference_name=reference_name, text_name=text_name)
    subdirs = sorted(p for p in root.iterdir() if p.is_dir())

    # Shape 1: root itself is a dataset (some child is a sample folder).
    if any(is_sample_folder(d, **names) for d in subdirs):
        return [(root.name, root)]

    # Shape 2: root is a collection; a child is a dataset if it has sample children.
    datasets: List[Tuple[str, Path]] = []
    for child in subdirs:
        grandchildren = (g for g in child.iterdir() if g.is_dir())
        if any(is_sample_folder(g, **names) for g in grandchildren):
            datasets.append((child.name, child))
    return datasets


def iter_samples(
    root: Path,
    *,
    user_name: str = "normalized.wav",
    reference_name: str = "model.wav",
    text_name: str = "phrase.txt",
    skipped: Optional[List[str]] = None,
) -> Iterator[Sample]:
    """Yield one ``Sample`` per valid subfolder of ``root``.

    Args:
        root: dataset directory containing numbered subfolders (001, 002, ...).
        user_name / reference_name / text_name: filenames inside each subfolder.
            Defaults match the project's ``records/`` sample; override once the
            final dataset layout is known.
        skipped: if given, the names of subfolders skipped for missing files are
            appended to it (the caller can log them).

    A subfolder is skipped (not an error) when the user audio or phrase text is
    missing. The reference audio is required too, since ``core_prod`` needs it;
    text-only engines simply ignore it.
    """
    if not root.is_dir():
        raise NotADirectoryError(f"dataset root is not a directory: {root}")

    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        user_audio = folder / user_name
        reference_audio = folder / reference_name
        text_file = folder / text_name

        missing = [
            f.name for f in (user_audio, reference_audio, text_file) if not f.is_file()
        ]
        if missing:
            if skipped is not None:
                skipped.append(f"{folder.name} (missing: {', '.join(missing)})")
            continue

        text = text_file.read_text(encoding="utf-8").strip()
        if not text:
            if skipped is not None:
                skipped.append(f"{folder.name} (empty {text_name})")
            continue

        yield Sample(
            id=folder.name,
            text=text,
            user_audio=user_audio,
            reference_audio=reference_audio,
        )


# ---------------------------------------------------------------------------
# Statistics. The goal is "is the test engine a viable alternative to core" --
# i.e. does it track the reference scores. So every metric below compares a test
# engine's scores against the reference (core_prod) scores, per dataset.
#
# Implemented with numpy only (no scipy) to avoid a new dependency; Spearman is
# Pearson on average ranks, which handles ties correctly.
# ---------------------------------------------------------------------------
@dataclass
class AgreementStats:
    n: int                        # number of paired samples
    pearson: float                # linear correlation of scores (NaN if n < 2)
    spearman: float               # rank correlation of scores  (NaN if n < 2)
    mae: float                    # mean |test - reference|
    bias: float                   # mean (test - reference); + means test scores higher
    verdict_agreement: float      # fraction of samples where passed flags match
    ref_mean: float               # mean reference score (context for bias/mae)
    test_mean: float              # mean test score


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two equal-length vectors; NaN if undefined.

    Undefined (returns NaN) when fewer than two points or when either vector is
    constant (zero variance) -- ``np.corrcoef`` would emit a warning and NaN there
    anyway, so we short-circuit cleanly.
    """
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) of ``values``, ties sharing their mean rank.

    A minimal stand-in for ``scipy.stats.rankdata(method="average")`` so Spearman
    needs no scipy. ``argsort`` of ``argsort`` gives ordinal ranks; we then average
    the ranks within each group of equal values so ties do not bias the rank
    correlation.
    """
    order = values.argsort()
    ordinal = order.argsort().astype(float)  # 0-based ordinal ranks
    ranks = ordinal + 1.0                     # 1-based

    # Average ranks within tied groups.
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    group_rank_sum = np.zeros(len(counts))
    np.add.at(group_rank_sum, inverse, ranks)
    averaged = group_rank_sum[inverse] / counts[inverse]
    return averaged


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho = Pearson r on average ranks; NaN if undefined."""
    if len(a) < 2:
        return math.nan
    return _pearson(_rankdata(a), _rankdata(b))


def agreement(
    reference_scores: List[float],
    test_scores: List[float],
    reference_passed: List[bool],
    test_passed: List[bool],
) -> AgreementStats:
    """Summarize how closely a test engine tracks the reference engine.

    All four lists are parallel (one entry per scored sample, same order). Raises
    ``ValueError`` on a length mismatch -- that would mean the runner failed to
    pair results, which must not pass silently.
    """
    n = len(reference_scores)
    if not (len(test_scores) == len(reference_passed) == len(test_passed) == n):
        raise ValueError("reference/test score and verdict lists must be equal length")
    if n == 0:
        return AgreementStats(0, math.nan, math.nan, math.nan, math.nan,
                              math.nan, math.nan, math.nan)

    ref = np.asarray(reference_scores, dtype=float)
    test = np.asarray(test_scores, dtype=float)
    diff = test - ref

    verdict_match = sum(
        1 for r, t in zip(reference_passed, test_passed) if r == t
    ) / n

    return AgreementStats(
        n=n,
        pearson=_pearson(ref, test),
        spearman=_spearman(ref, test),
        mae=float(np.mean(np.abs(diff))),
        bias=float(np.mean(diff)),
        verdict_agreement=verdict_match,
        ref_mean=float(np.mean(ref)),
        test_mean=float(np.mean(test)),
    )


# ---------------------------------------------------------------------------
# Class separability. The decisive question for "is this engine a viable
# alternative" is not how close its numbers are to core, but whether it
# *separates* good speech from bad as well as core does. We measure that two
# ways, both computed identically for every engine so they can be compared:
#
#   * ROC-AUC -- threshold-independent: the probability a random good sample
#     scores above a random bad one. 1.0 = perfect ranking, 0.5 = coin flip.
#     Robust to the systematic score offset we saw (it only uses ordering), so
#     it answers "does the engine rank correctly" separately from "is it
#     calibrated", which the bias/MAE numbers already cover.
#   * Best-threshold accuracy -- the single cutoff that best splits good from
#     bad, and the accuracy there. Shows where this engine's natural decision
#     boundary sits (e.g. far below the prod 70 default), informing recalibration.
# ---------------------------------------------------------------------------
@dataclass
class DiscriminationStats:
    n_good: int
    n_bad: int
    auc: float                    # ROC-AUC, good vs bad (NaN if a class is empty)
    best_threshold: float         # cutoff maximizing accuracy (NaN if undefined)
    best_accuracy: float          # accuracy at that cutoff
    good_mean: float
    bad_mean: float


def roc_auc(good_scores: List[float], bad_scores: List[float]) -> float:
    """ROC-AUC via the Mann-Whitney rank statistic, with tie-correct ranks.

    Ranking all scores together and summing the ranks of the "good" group gives
    AUC = (rank_sum_good - n_good*(n_good+1)/2) / (n_good * n_bad). Using average
    ranks (``_rankdata``) makes tied good/bad scores contribute 0.5 each, the
    standard tie handling. NaN if either group is empty.
    """
    n_good, n_bad = len(good_scores), len(bad_scores)
    if n_good == 0 or n_bad == 0:
        return math.nan
    all_scores = np.asarray(list(good_scores) + list(bad_scores), dtype=float)
    ranks = _rankdata(all_scores)
    rank_sum_good = float(ranks[:n_good].sum())
    return (rank_sum_good - n_good * (n_good + 1) / 2.0) / (n_good * n_bad)


def discrimination(
    good_scores: List[float], bad_scores: List[float]
) -> DiscriminationStats:
    """Summarize how well one engine's scores separate good from bad samples.

    ``best_threshold`` is searched over midpoints between adjacent observed scores
    (predict "good" when ``score >= threshold``); the one with the highest
    accuracy wins, ties broken toward the lower threshold. NaN threshold when
    either class is empty.
    """
    n_good, n_bad = len(good_scores), len(bad_scores)
    good = np.asarray(good_scores, dtype=float)
    bad = np.asarray(bad_scores, dtype=float)

    if n_good == 0 or n_bad == 0:
        return DiscriminationStats(
            n_good, n_bad, math.nan, math.nan, math.nan,
            float(np.mean(good)) if n_good else math.nan,
            float(np.mean(bad)) if n_bad else math.nan,
        )

    combined = np.concatenate([good, bad])
    labels = np.concatenate([np.ones(n_good, bool), np.zeros(n_bad, bool)])

    # Candidate cutoffs: midpoints between sorted unique scores, plus the extremes
    # so an all-pass / all-fail boundary is reachable.
    uniq = np.unique(combined)
    midpoints = (uniq[:-1] + uniq[1:]) / 2.0 if len(uniq) > 1 else uniq
    candidates = np.concatenate([[uniq[0] - 1.0], midpoints, [uniq[-1] + 1.0]])

    best_thr, best_acc = math.nan, -1.0
    total = n_good + n_bad
    for thr in candidates:
        predicted_good = combined >= thr
        accuracy = float(np.sum(predicted_good == labels)) / total
        if accuracy > best_acc:
            best_acc, best_thr = accuracy, float(thr)

    return DiscriminationStats(
        n_good=n_good,
        n_bad=n_bad,
        auc=roc_auc(good_scores, bad_scores),
        best_threshold=best_thr,
        best_accuracy=best_acc,
        good_mean=float(np.mean(good)),
        bad_mean=float(np.mean(bad)),
    )
