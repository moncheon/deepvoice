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
    assign_domain_stratified_group_splits,
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
    split.add_argument("--train-ratio", type=float, default=0.8)
    split.add_argument("--dev-ratio", type=float, default=0.1)
    split.add_argument("--holdout-generator", action="append", default=[])
    split.add_argument(
        "--stratify-labels",
        action="store_true",
        help="Balance label signatures while preserving split_group boundaries.",
    )
    split.add_argument(
        "--stratify-domains",
        action="store_true",
        help="Cover coarse generator domains and labels while preserving split groups.",
    )
    split.add_argument("--summary", type=Path)
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
    if args.stratify_domains and (args.stratify_labels or args.holdout_generator):
        raise SystemExit("--stratify-domains cannot be combined with label-only or forced holdout modes")
    if args.stratify_domains:
        assignments = assign_domain_stratified_group_splits(
            records, seed=args.seed, train_ratio=args.train_ratio, dev_ratio=args.dev_ratio
        )
    else:
        splitter = assign_stratified_group_splits if args.stratify_labels else assign_group_splits
        assignments = splitter(
            records,
            seed=args.seed,
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            holdout_generator_families=set(args.holdout_generator),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(assignments, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for split_name in assignments.values():
        counts[split_name] = counts.get(split_name, 0) + 1
    if args.summary:
        from collections import Counter
        from deepvoice_run0.manifest import coarse_generator_domain, domain_label_signature

        group_splits: dict[str, str] = {}
        leakage: list[str] = []
        domain_counts: Counter[str] = Counter()
        stratum_counts: Counter[str] = Counter()
        for record in records:
            split_name = assignments[record.sample_id]
            prior = group_splits.setdefault(record.split_group, split_name)
            if prior != split_name:
                leakage.append(record.split_group)
            voice_domain, music_domain = coarse_generator_domain(record)
            domain_counts[f"{split_name}:voice:{voice_domain}"] += 1
            domain_counts[f"{split_name}:music:{music_domain}"] += 1
            stratum_counts[f"{split_name}:{domain_label_signature(record)}"] += 1
        if leakage:
            raise RuntimeError(f"split-group leakage: {sorted(set(leakage))[:10]}")
        summary = {
            "format_version": 1,
            "mode": "domain_label_source_group_disjoint" if args.stratify_domains else "standard",
            "seed": args.seed,
            "ratios": {"train": args.train_ratio, "dev": args.dev_ratio, "holdout": 1 - args.train_ratio - args.dev_ratio},
            "records": len(records),
            "groups": len(group_splits),
            "split_counts": counts,
            "split_group_leakage": 0,
            "domain_counts": dict(sorted(domain_counts.items())),
            "stratum_counts": dict(sorted(stratum_counts.items())),
            "assignments_sha256": sha256_file(args.output),
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"saved {len(assignments)} assignments to {args.output}: {counts}")


if __name__ == "__main__":
    main()
