#!/usr/bin/env python3
"""Write a compact, trackable summary of the full ignored LOGO plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    folds = []
    for fold in plan["folds"]:
        folds.append({
            "fold_id": fold["fold_id"],
            "axis": fold["axis"],
            "task": fold["task"],
            "targets": fold["targets"],
            "panel_rows": len(fold["panel_sample_ids"]),
            "excluded_split_groups": len(fold["excluded_split_groups"]),
            "split_counts": fold["split_counts"],
        })
    payload = {
        "format_version": 1,
        "status": "audited",
        "plan_sha256": plan["plan_sha256"],
        "seed": plan["seed"],
        "calibration_rows": len(plan["calibration_sample_ids"]),
        "calibration_split_groups": len(plan["calibration_split_groups"]),
        "settings": plan["settings"],
        "fold_count": len(folds),
        "folds": folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote compact summary for {len(folds)} folds to {args.output}")


if __name__ == "__main__":
    main()
