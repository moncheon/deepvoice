#!/usr/bin/env python3
"""Validate, hash-audit, and split a Run 0 data manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RUN0_DIR = Path(__file__).resolve().parents[1]
if str(RUN0_DIR) not in sys.path:
    sys.path.insert(0, str(RUN0_DIR))

from deepvoice_run0.manifest import (
    assign_group_splits,
    assign_stratified_group_splits,
    read_manifest,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("manifest", type=Path)
    audit = subparsers.add_parser("audit-files")
    audit.add_argument("manifest", type=Path)
    split = subparsers.add_parser("split")
    split.add_argument("manifest", type=Path)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=2608)
    split.add_argument("--holdout-generator", action="append", default=[])
    split.add_argument(
        "--stratify-labels",
        action="store_true",
        help="Balance label signatures while preserving split_group boundaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_manifest(args.manifest)
    if args.command == "validate":
        groups = len({record.split_group for record in records})
        print(f"valid records={len(records)} split_groups={groups}")
        return
    if args.command == "audit-files":
        failures: list[str] = []
        for record in records:
            path = Path(record.path)
            if not path.is_absolute():
                path = args.manifest.parent / path
            if not path.is_file():
                failures.append(f"{record.sample_id}: missing {path}")
            elif sha256_file(path).lower() != record.sha256.lower():
                failures.append(f"{record.sample_id}: SHA-256 mismatch")
        if failures:
            raise SystemExit("file audit failed:\n" + "\n".join(failures[:20]))
        print(f"audited files={len(records)}")
        return
    splitter = assign_stratified_group_splits if args.stratify_labels else assign_group_splits
    assignments = splitter(
        records, seed=args.seed, holdout_generator_families=set(args.holdout_generator)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(assignments, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for split_name in assignments.values():
        counts[split_name] = counts.get(split_name, 0) + 1
    print(f"saved {len(assignments)} assignments to {args.output}: {counts}")


if __name__ == "__main__":
    main()
