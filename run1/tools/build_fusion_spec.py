#!/usr/bin/env python3
"""Build the final file-independent FusionSpec from held calibration predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

RUN1_ROOT = Path(__file__).resolve().parents[1]
if str(RUN1_ROOT) not in sys.path:
    sys.path.insert(0, str(RUN1_ROOT))

from deepvoice_hybrid.fusion import EmpiricalCDF, FAKE_HEADS, FusionSpec, HeadFusion


COLUMNS = {
    "file_fake": "FILE_FAKE_PROB",
    "voice_fake": "VOICE_FAKE_PROB",
    "music_fake": "MUSIC_FAKE_PROB",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--calibration-ledger", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.selection_report.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise ValueError("selection gate did not pass")
    with args.calibration_ledger.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError("calibration ledger is too small")
    heads = {}
    for head in FAKE_HEADS:
        column = COLUMNS[head]
        heads[head] = HeadFusion(
            baseline_weight=float(report["final_baseline_weights"][head]),
            baseline_cdf=EmpiricalCDF.fit(float(row["baseline_" + column]) for row in rows),
            run0_cdf=EmpiricalCDF.fit(float(row["run0_" + column]) for row in rows),
        )
    spec = FusionSpec(
        heads=heads,
        provenance={
            "selection_report_sha256": sha256(args.selection_report),
            "calibration_ledger_sha256": sha256(args.calibration_ledger),
            "logo_plan_sha256": sha256(args.plan),
            "run0_model_sha256": args.model_sha256,
            "calibration_rows": len(rows),
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(spec.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote independent fusion spec to {args.output}")


if __name__ == "__main__":
    main()
