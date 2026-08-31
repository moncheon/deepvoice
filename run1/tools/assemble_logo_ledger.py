#!/usr/bin/env python3
"""Join cached Baseline predictions with per-fold Run0 epoch predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

RUN1_ROOT = Path(__file__).resolve().parents[1]
if str(RUN1_ROOT) not in sys.path:
    sys.path.insert(0, str(RUN1_ROOT))

from deepvoice_hybrid.artifacts import verify_artifact_manifest


PREDICTIONS = (
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def index_unique(
    rows: list[dict[str, str]], *, source: str
) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in indexed:
            raise ValueError(f"duplicate sample_id in {source}: {sample_id}")
        indexed[sample_id] = row
    return indexed


def validate_panel_coverage(
    panel: set[str],
    run0_rows: dict[str, dict[str, str]],
    baseline: dict[str, dict[str, str]],
    *,
    fold_id: str,
) -> None:
    """Require the evaluation panel while allowing group-safe holdout extras.

    A generator/codec panel can contain only part of a split group. Training
    correctly holds out the entire group to prevent leakage, so its prediction
    CSV may include additional rows. Those rows are evidence of the group-safe
    split, not evaluation-panel members, and are ignored below.
    """
    missing = sorted(panel - run0_rows.keys())
    missing_baseline = sorted(panel - baseline.keys())
    if missing or missing_baseline:
        raise ValueError(
            f"{fold_id} alignment failed; "
            f"run0_missing={missing[:5]}, "
            f"baseline_missing={missing_baseline[:5]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--source-bundle-sha256", required=True)
    parser.add_argument("--expected-epochs", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_epochs < 1:
        raise ValueError("--expected-epochs must be positive")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    baseline = index_unique(read_csv(args.baseline), source=str(args.baseline))
    output_rows: list[dict[str, object]] = []
    for fold in plan["folds"]:
        panel = set(fold["panel_sample_ids"])
        evidence_root = args.fold_root / fold["fold_id"]
        verify_artifact_manifest(evidence_root)
        report = json.loads(
            (evidence_root / "FOLD_REPORT.json").read_text(encoding="utf-8")
        )
        expected_report = {
            "status": "completed",
            "fold_id": fold["fold_id"],
            "plan_sha256": plan["plan_sha256"],
            "source_bundle_sha256": args.source_bundle_sha256,
            "epochs_requested": args.expected_epochs,
            "epochs_completed": args.expected_epochs,
        }
        mismatches = {
            name: {"expected": expected, "actual": report.get(name)}
            for name, expected in expected_report.items()
            if report.get(name) != expected
        }
        if mismatches:
            raise ValueError(f"invalid fold evidence for {fold['fold_id']}: {mismatches}")
        epoch_paths = sorted(
            (evidence_root / "epochs").glob(
                "epoch-*-holdout_predictions.csv"
            )
        )
        epochs_found = {int(path.stem.split("-")[1]) for path in epoch_paths}
        expected_epochs = set(range(1, args.expected_epochs + 1))
        if epochs_found != expected_epochs or len(epoch_paths) != args.expected_epochs:
            raise FileNotFoundError(
                f"expected exact epochs 1..{args.expected_epochs} for "
                f"{fold['fold_id']}; "
                f"found={sorted(epochs_found)}"
            )
        for epoch_path in epoch_paths:
            epoch = int(epoch_path.stem.split("-")[1])
            run0_rows = index_unique(read_csv(epoch_path), source=str(epoch_path))
            validate_panel_coverage(
                panel, run0_rows, baseline, fold_id=fold["fold_id"]
            )
            for sample_id in sorted(panel):
                run0 = run0_rows[sample_id]
                base = baseline[sample_id]
                row: dict[str, object] = {
                    "fold_id": fold["fold_id"],
                    "axis": fold["axis"],
                    "task": fold["task"],
                    "epoch": epoch,
                    "sample_id": sample_id,
                    "file_fake": run0["file_fake"],
                    "voice_fake": run0["voice_fake"],
                    "music_fake": run0["music_fake"],
                    "voice_present": run0["voice_present"],
                    "music_present": run0["music_present"],
                }
                for name in PREDICTIONS:
                    row["baseline_" + name] = base[name]
                    row["run0_" + name] = run0[name]
                output_rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} aligned LOGO rows to {args.output}")


if __name__ == "__main__":
    main()
