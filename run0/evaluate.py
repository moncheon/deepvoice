#!/usr/bin/env python3
"""Evaluate labeled OOF/holdout CSV predictions with official metric weights."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

from deepvoice_run0.metrics import official_metrics

TRUTH_COLUMNS = (
    "file_fake",
    "voice_fake",
    "music_fake",
    "voice_present",
    "music_present",
)


def _nullable_int(value: str) -> int | None:
    stripped = value.strip().lower()
    return None if stripped in {"", "null", "none", "nan"} else int(stripped)


def score_rows(rows: Iterable[dict[str, str]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        raise ValueError("prediction rows are empty")
    return official_metrics(
        file_fake=[int(row["file_fake"]) for row in rows],
        file_fake_prob=[float(row["FILE_FAKE_PROB"]) for row in rows],
        voice_present=[int(row["voice_present"]) for row in rows],
        voice_present_prob=[float(row["VOICE_PRESENT_PROB"]) for row in rows],
        music_present=[int(row["music_present"]) for row in rows],
        music_present_prob=[float(row["MUSIC_PRESENT_PROB"]) for row in rows],
        voice_fake=[_nullable_int(row["voice_fake"]) for row in rows],
        voice_fake_prob=[float(row["VOICE_FAKE_PROB"]) for row in rows],
        music_fake=[_nullable_int(row["music_fake"]) for row in rows],
        music_fake_prob=[float(row["MUSIC_FAKE_PROB"]) for row in rows],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--group-column",
        action="append",
        default=[],
        help="Also evaluate each value where both classes make every metric defined.",
    )
    args = parser.parse_args()
    with args.predictions.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = set(reader.fieldnames or [])
    required = set(TRUTH_COLUMNS) | {
        "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    }
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"prediction CSV is missing columns: {missing}")
    report: dict[str, object] = {"overall": score_rows(rows), "groups": {}}
    grouped_report: dict[str, object] = {}
    for column in args.group_column:
        if column not in columns:
            raise ValueError(f"group column is missing: {column}")
        values: dict[str, object] = {}
        for value in sorted({row[column] for row in rows}):
            subset = [row for row in rows if row[column] == value]
            try:
                values[value] = {"count": len(subset), "metrics": score_rows(subset)}
            except ValueError as error:
                values[value] = {"count": len(subset), "metrics": None, "reason": str(error)}
        grouped_report[column] = values
    report["groups"] = grouped_report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
