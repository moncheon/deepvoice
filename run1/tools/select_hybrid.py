#!/usr/bin/env python3
"""Select Run0 epoch and independent per-head fusion weights with nested LOGO."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RUN1_ROOT = Path(__file__).resolve().parents[1]
if str(RUN1_ROOT) not in sys.path:
    sys.path.insert(0, str(RUN1_ROOT))

from deepvoice_hybrid.selection import (
    read_prediction_ledger,
    select_epoch_and_fusion,
    write_selection_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2608)
    args = parser.parse_args()
    report = select_epoch_and_fusion(
        read_prediction_ledger(args.ledger),
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    write_selection_report(args.output, report)
    print(
        f"status={report['status']} epoch={report['selected_epoch']} "
        f"improvement={report['robust_ads_improvement']:.6f}"
    )


if __name__ == "__main__":
    main()
