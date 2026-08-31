"""Nested LOGO selection of Run0 epoch and per-head rank-fusion weights."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .fusion import EmpiricalCDF, FAKE_HEADS
from .metrics import ads_from_eers, equal_error_rate


LABEL_BY_HEAD = {
    "file_fake": "file_fake",
    "voice_fake": "voice_fake",
    "music_fake": "music_fake",
}
SCORE_COLUMN_BY_HEAD = {
    "file_fake": "FILE_FAKE_PROB",
    "voice_fake": "VOICE_FAKE_PROB",
    "music_fake": "MUSIC_FAKE_PROB",
}


def _relevant(row: Mapping[str, str], head: str) -> bool:
    if head == "file_fake":
        return True
    if row["axis"] == "codec":
        return True
    return row["task"] == head.removesuffix("_fake")


def _has_label(row: Mapping[str, str], head: str) -> bool:
    return row[LABEL_BY_HEAD[head]].strip().lower() not in {
        "", "none", "null", "nan"
    }


def _arrays(
    rows: Iterable[Mapping[str, str]], head: str, score_prefix: str
) -> tuple[np.ndarray, np.ndarray]:
    label_name = LABEL_BY_HEAD[head]
    score_name = score_prefix + SCORE_COLUMN_BY_HEAD[head]
    labels: list[int] = []
    scores: list[float] = []
    for row in rows:
        if not _has_label(row, head):
            continue
        raw_label = row[label_name].strip().lower()
        labels.append(int(raw_label))
        scores.append(float(row[score_name]))
    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64)


def _eer(rows: Iterable[Mapping[str, str]], head: str, prefix: str) -> float:
    labels, scores = _arrays(rows, head, prefix)
    return equal_error_rate(labels, scores)


def _choose_weight(
    rows: list[Mapping[str, str]], head: str, baseline_cdf: EmpiricalCDF, run0_cdf: EmpiricalCDF
) -> float:
    labels, baseline = _arrays(rows, head, "baseline_")
    _, run0 = _arrays(rows, head, "run0_")
    baseline_rank = np.asarray([baseline_cdf.transform(value) for value in baseline])
    run0_rank = np.asarray([run0_cdf.transform(value) for value in run0])
    candidates: list[tuple[float, float]] = []
    for step in range(21):
        weight = step / 20.0
        scores = weight * baseline_rank + (1.0 - weight) * run0_rank
        candidates.append((equal_error_rate(labels, scores), weight))
    best_eer = min(value[0] for value in candidates)
    tied = [weight for eer, weight in candidates if abs(eer - best_eer) <= 1e-12]
    return max(tied)


def _nested_head_predictions(
    rows: list[dict[str, str]], head: str
) -> tuple[list[dict[str, object]], float]:
    relevant = [row for row in rows if _relevant(row, head)]
    fold_ids = sorted({row["fold_id"] for row in relevant})
    if len(fold_ids) < 2:
        raise ValueError(f"{head} needs at least two relevant folds")
    held_predictions: list[dict[str, object]] = []
    held_weights: list[float] = []
    for held_fold in fold_ids:
        inner = [row for row in relevant if row["fold_id"] != held_fold]
        held = [row for row in relevant if row["fold_id"] == held_fold]
        _, inner_baseline = _arrays(inner, head, "baseline_")
        _, inner_run0 = _arrays(inner, head, "run0_")
        baseline_cdf = EmpiricalCDF.fit(inner_baseline)
        run0_cdf = EmpiricalCDF.fit(inner_run0)
        weight = _choose_weight(inner, head, baseline_cdf, run0_cdf)
        held_weights.append(weight)
        label_name = LABEL_BY_HEAD[head]
        column = SCORE_COLUMN_BY_HEAD[head]
        for row in held:
            label = row[label_name].strip()
            if not label:
                continue
            baseline = float(row["baseline_" + column])
            run0 = float(row["run0_" + column])
            held_predictions.append({
                "fold_id": row["fold_id"],
                "axis": row["axis"],
                "task": row["task"],
                "sample_id": row["sample_id"],
                "label": int(label),
                "baseline": baseline_cdf.transform(baseline),
                "run0": run0_cdf.transform(run0),
                "hybrid": weight * baseline_cdf.transform(baseline)
                + (1.0 - weight) * run0_cdf.transform(run0),
                "inner_weight": weight,
            })
    return held_predictions, float(np.median(held_weights))


def _axis_ads(
    predictions: Mapping[str, list[dict[str, object]]], score_name: str, axis: str
) -> tuple[float, dict[str, float]]:
    eers: dict[str, float] = {}
    for head in FAKE_HEADS:
        selected = [row for row in predictions[head] if row["axis"] == axis]
        labels = [int(row["label"]) for row in selected]
        scores = [float(row[score_name]) for row in selected]
        eers[head] = equal_error_rate(labels, scores)
    return ads_from_eers(eers["file_fake"], eers["voice_fake"], eers["music_fake"]), eers


def _final_weight(rows: list[dict[str, str]], head: str) -> tuple[float, dict[str, object]]:
    relevant = [
        row for row in rows if _relevant(row, head) and _has_label(row, head)
    ]
    _, baseline = _arrays(relevant, head, "baseline_")
    _, run0 = _arrays(relevant, head, "run0_")
    baseline_cdf = EmpiricalCDF.fit(baseline)
    run0_cdf = EmpiricalCDF.fit(run0)
    labels, _ = _arrays(relevant, head, "baseline_")
    baseline_rank = np.asarray([baseline_cdf.transform(value) for value in baseline])
    run0_rank = np.asarray([run0_cdf.transform(value) for value in run0])
    axes = np.asarray([row["axis"] for row in relevant])
    candidates: list[tuple[float, float, dict[str, float]]] = []
    for step in range(21):
        weight = step / 20.0
        fused = weight * baseline_rank + (1.0 - weight) * run0_rank
        axis_eers = {
            axis: equal_error_rate(labels[axes == axis], fused[axes == axis])
            for axis in ("generator", "codec")
        }
        candidates.append((max(axis_eers.values()), weight, axis_eers))
    best = min(value[0] for value in candidates)
    tied = [value for value in candidates if abs(value[0] - best) <= 1e-12]
    chosen = max(tied, key=lambda value: value[1])
    return chosen[1], {"robust_eer": chosen[0], "axis_eers": chosen[2]}


def _fold_target_score(
    fold_rows: Mapping[str, list[dict[str, object]]], score_name: str, task: str
) -> float:
    file_rows = fold_rows["file_fake"]
    file_eer = equal_error_rate(
        [int(row["label"]) for row in file_rows],
        [float(row[score_name]) for row in file_rows],
    )
    if task == "voice":
        component, file_weight, component_weight = "voice_fake", 0.5, 0.2
    elif task == "music":
        component, file_weight, component_weight = "music_fake", 0.5, 0.3
    else:
        eers = {}
        for head, values in fold_rows.items():
            eers[head] = equal_error_rate(
                [int(row["label"]) for row in values],
                [float(row[score_name]) for row in values],
            )
        return ads_from_eers(eers["file_fake"], eers["voice_fake"], eers["music_fake"])
    values = fold_rows[component]
    component_eer = equal_error_rate(
        [int(row["label"]) for row in values],
        [float(row[score_name]) for row in values],
    )
    return (
        file_weight * (1.0 - file_eer) + component_weight * (1.0 - component_eer)
    ) / (file_weight + component_weight)


def _worst_fold_delta(predictions: Mapping[str, list[dict[str, object]]]) -> float:
    fold_ids = sorted({str(row["fold_id"]) for row in predictions["file_fake"]})
    deltas: list[float] = []
    for fold_id in fold_ids:
        fold_rows = {
            head: [row for row in values if row["fold_id"] == fold_id]
            for head, values in predictions.items()
        }
        task = str(fold_rows["file_fake"][0]["task"])
        hybrid = _fold_target_score(fold_rows, "hybrid", task)
        baseline = _fold_target_score(fold_rows, "baseline", task)
        deltas.append(hybrid - baseline)
    return min(deltas)


def _bootstrap_lower_bound(
    predictions: Mapping[str, list[dict[str, object]]], comparator: str, *, seed: int,
    iterations: int,
) -> float:
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(iterations):
        axis_scores: dict[str, dict[str, float]] = {"hybrid": {}, comparator: {}}
        valid = True
        for axis in ("generator", "codec"):
            eers_by_score: dict[str, dict[str, float]] = {
                "hybrid": {}, comparator: {}
            }
            for head in FAKE_HEADS:
                pool = [row for row in predictions[head] if row["axis"] == axis]
                # One index draw is shared by both systems: this is a paired
                # bootstrap over the same labeled examples.
                indices = rng.integers(0, len(pool), size=len(pool))
                labels = [int(pool[index]["label"]) for index in indices]
                if len(set(labels)) < 2:
                    valid = False
                    break
                for score_name in ("hybrid", comparator):
                    scores = [float(pool[index][score_name]) for index in indices]
                    eers_by_score[score_name][head] = equal_error_rate(labels, scores)
            if not valid:
                break
            for score_name in ("hybrid", comparator):
                axis_scores[score_name][axis] = ads_from_eers(
                    eers_by_score[score_name]["file_fake"],
                    eers_by_score[score_name]["voice_fake"],
                    eers_by_score[score_name]["music_fake"],
                )
        if valid:
            differences.append(
                min(axis_scores["hybrid"].values())
                - min(axis_scores[comparator].values())
            )
    if len(differences) < max(100, iterations // 2):
        raise ValueError("too few valid paired bootstrap samples")
    return float(np.quantile(np.asarray(differences), 0.025))


def select_epoch_and_fusion(
    rows: list[dict[str, str]], *, bootstrap_iterations: int = 1000, seed: int = 2608
) -> dict[str, object]:
    if not rows:
        raise ValueError("prediction ledger is empty")
    epochs = sorted({int(row["epoch"]) for row in rows})
    epoch_reports: list[dict[str, object]] = []
    nested_by_epoch: dict[int, dict[str, list[dict[str, object]]]] = {}
    for epoch in epochs:
        epoch_rows = [row for row in rows if int(row["epoch"]) == epoch]
        predictions: dict[str, list[dict[str, object]]] = {}
        median_weights: dict[str, float] = {}
        for head in FAKE_HEADS:
            predictions[head], median_weights[head] = _nested_head_predictions(epoch_rows, head)
        nested_by_epoch[epoch] = predictions
        scores: dict[str, dict[str, object]] = {}
        for name in ("hybrid", "baseline", "run0"):
            generator_ads, generator_eers = _axis_ads(predictions, name, "generator")
            codec_ads, codec_eers = _axis_ads(predictions, name, "codec")
            scores[name] = {
                "generator_ads": generator_ads,
                "codec_ads": codec_ads,
                "robust_ads": min(generator_ads, codec_ads),
                "generator_eers": generator_eers,
                "codec_eers": codec_eers,
            }
        epoch_reports.append({
            "epoch": epoch,
            "nested": scores,
            "median_inner_weights": median_weights,
        })
    selected_report = max(
        epoch_reports,
        key=lambda report: (float(report["nested"]["hybrid"]["robust_ads"]), -int(report["epoch"])),
    )
    selected_epoch = int(selected_report["epoch"])
    selected_rows = [row for row in rows if int(row["epoch"]) == selected_epoch]
    predictions = nested_by_epoch[selected_epoch]
    final_weights: dict[str, float] = {}
    weight_reports: dict[str, object] = {}
    for head in FAKE_HEADS:
        final_weights[head], weight_reports[head] = _final_weight(selected_rows, head)
    branch_scores = selected_report["nested"]
    comparator = max(
        ("baseline", "run0"), key=lambda name: float(branch_scores[name]["robust_ads"])
    )
    improvement = float(branch_scores["hybrid"]["robust_ads"]) - float(
        branch_scores[comparator]["robust_ads"]
    )
    bootstrap_lower = _bootstrap_lower_bound(
        predictions, comparator, seed=seed, iterations=bootstrap_iterations
    )
    worst_delta = _worst_fold_delta(predictions)
    max_epoch = max(epochs)
    convergence_incomplete = False
    if selected_epoch == max_epoch and len(epochs) > 1:
        prior = max(epoch for epoch in epochs if epoch < selected_epoch)
        current_score = float(branch_scores["hybrid"]["robust_ads"])
        prior_report = next(report for report in epoch_reports if int(report["epoch"]) == prior)
        convergence_incomplete = (
            current_score - float(prior_report["nested"]["hybrid"]["robust_ads"]) >= 0.001
        )
    reasons: list[str] = []
    if improvement < 0.003:
        reasons.append("robust ADS improvement is below 0.003")
    if bootstrap_lower <= 0:
        reasons.append("paired bootstrap 95% lower bound is not positive")
    if worst_delta < -0.01:
        reasons.append("worst-fold ADS delta is below -0.01")
    if convergence_incomplete:
        reasons.append("selected epoch is still improving at the search boundary")
    return {
        "format_version": 1,
        "selection_metric": "robust_nested_logo_ads",
        "selected_epoch": selected_epoch,
        "final_baseline_weights": final_weights,
        "weight_search": weight_reports,
        "epoch_reports": epoch_reports,
        "comparator": comparator,
        "robust_ads_improvement": improvement,
        "paired_bootstrap_lower_95": bootstrap_lower,
        "worst_fold_ads_delta": worst_delta,
        "convergence_incomplete": convergence_incomplete,
        "gate_reasons": reasons,
        "status": "passed" if not reasons else "failed",
    }


def read_prediction_ledger(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "fold_id", "axis", "task", "epoch", "sample_id",
        "file_fake", "voice_fake", "music_fake",
        *("baseline_" + value for value in SCORE_COLUMN_BY_HEAD.values()),
        *("run0_" + value for value in SCORE_COLUMN_BY_HEAD.values()),
    }
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"prediction ledger misses columns: {sorted(missing)}")
    return rows


def write_selection_report(path: str | Path, report: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
