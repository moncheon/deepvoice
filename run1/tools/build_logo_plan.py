#!/usr/bin/env python3
"""Build and audit the Run1 8-generator plus codec LOGO plan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RUN1_ROOT = Path(__file__).resolve().parents[1]
if str(RUN1_ROOT) not in sys.path:
    sys.path.insert(0, str(RUN1_ROOT))

from deepvoice_hybrid.logo import (
    build_logo_plan,
    read_manifest_rows,
    write_fold_splits,
    write_logo_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2608)
    parser.add_argument("--calibration-size", type=int, default=512)
    parser.add_argument("--panel-size", type=int, default=256)
    args = parser.parse_args()
    rows = read_manifest_rows(args.manifest)
    plan = build_logo_plan(
        rows,
        seed=args.seed,
        calibration_size=args.calibration_size,
        panel_size=args.panel_size,
    )
    write_logo_plan(args.output, plan)
    write_fold_splits(args.splits_dir, plan)
    print(
        f"wrote {len(plan['folds'])} audited folds; "
        f"sha256={plan['plan_sha256']}"
    )


if __name__ == "__main__":
    main()
