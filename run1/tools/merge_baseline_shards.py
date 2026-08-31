#!/usr/bin/env python3
"""Merge Baseline cache shards only when they exactly cover the LOGO plan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


FIELDS = (
    "sample_id", "audio_sha256", "FILE_FAKE_PROB", "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def expected_ids(plan_path: Path) -> set[str]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    ids = {str(value) for value in plan["calibration_sample_ids"]}
    for fold in plan["folds"]:
        ids.update(str(value) for value in fold["panel_sample_ids"])
    return ids


def merge(plan_path: Path, shard_paths: list[Path]) -> list[dict[str, str]]:
    if not shard_paths:
        raise ValueError("no Baseline shard CSVs supplied")
    indexed: dict[str, dict[str, str]] = {}
    for path in shard_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"invalid Baseline shard columns: {path}")
            for row in reader:
                sample_id = row["sample_id"]
                if sample_id in indexed:
                    raise ValueError(f"duplicate Baseline sample across shards: {sample_id}")
                if len(row["audio_sha256"]) != 64:
                    raise ValueError(f"invalid audio SHA-256 for {sample_id}")
                for name in FIELDS[2:]:
                    value = float(row[name])
                    if not 0.0 <= value <= 1.0:
                        raise ValueError(f"out-of-range {name} for {sample_id}")
                indexed[sample_id] = row
    expected = expected_ids(plan_path)
    if set(indexed) != expected:
        raise ValueError(
            "Baseline shards do not exactly cover the LOGO plan: "
            f"missing={len(expected - indexed.keys())}, "
            f"extra={len(indexed.keys() - expected)}"
        )
    return [indexed[sample_id] for sample_id in sorted(indexed)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    shards = sorted(args.shard_root.glob("baseline-*.csv"))
    rows = merge(args.plan, shards)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "format_version": 1,
        "status": "completed",
        "rows": len(rows),
        "plan_sha256": sha256(args.plan),
        "output_sha256": sha256(args.output),
        "shards": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in shards
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"merged {len(shards)} Baseline shards with {len(rows)} rows")


if __name__ == "__main__":
    main()
