#!/usr/bin/env python3
"""Align fixed Baseline and final Run0 scores on the held calibration set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PREDICTIONS = (
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
)


def read_unique(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["sample_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate sample_id in {path}")
    missing_columns = [
        name for name in ("sample_id", *PREDICTIONS)
        if not rows or name not in rows[0]
    ]
    if missing_columns:
        raise ValueError(f"missing columns in {path}: {missing_columns}")
    return indexed


def build_rows(
    plan_path: Path, baseline_path: Path, run0_path: Path
) -> list[dict[str, str]]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    calibration_ids = [str(value) for value in plan["calibration_sample_ids"]]
    if len(calibration_ids) != len(set(calibration_ids)):
        raise ValueError("duplicate calibration IDs in plan")
    baseline = read_unique(baseline_path)
    run0 = read_unique(run0_path)
    expected = set(calibration_ids)
    if not expected <= baseline.keys() or set(run0) != expected:
        raise ValueError(
            "calibration alignment failed: "
            f"baseline_missing={len(expected - baseline.keys())}, "
            f"run0_missing={len(expected - run0.keys())}, "
            f"run0_extra={len(set(run0) - expected)}"
        )
    rows: list[dict[str, str]] = []
    for sample_id in sorted(calibration_ids):
        row = {"sample_id": sample_id}
        for name in PREDICTIONS:
            row["baseline_" + name] = baseline[sample_id][name]
            row["run0_" + name] = run0[sample_id][name]
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_rows(args.plan, args.baseline, args.run0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} fixed calibration rows to {args.output}")


if __name__ == "__main__":
    main()
