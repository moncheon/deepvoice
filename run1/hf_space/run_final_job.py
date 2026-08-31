#!/usr/bin/env python3
"""Fit the selected Run0 branch on generator-rich data and held calibration groups."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

try:
    from .run_logo_job import (
        CHECKPOINT_MOUNT,
        DATA_ROOT,
        MODEL_ROOT,
        WORK_ROOT,
        artifact_hashes,
        checkpoint_callback,
        prepare_backbone,
        prepare_corpus,
        required_env,
        restore_checkpoint,
        sha256_file,
    )
except ImportError:  # Direct HF Job script execution.
    from run_logo_job import (
        CHECKPOINT_MOUNT,
        DATA_ROOT,
        MODEL_ROOT,
        WORK_ROOT,
        artifact_hashes,
        checkpoint_callback,
        prepare_backbone,
        prepare_corpus,
        required_env,
        restore_checkpoint,
        sha256_file,
    )


WORKSPACE = Path("/workspace")
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


def stable_position(seed: int, group: str) -> float:
    digest = hashlib.sha256(f"{seed}:final-dev:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def build_final_splits(plan_path: Path, output: Path, seed: int) -> dict[str, object]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("plan_sha256") != required_env("RUN1_PLAN_SHA256"):
        raise RuntimeError("mounted LOGO plan does not match final-fit request")
    calibration_ids = set(str(value) for value in plan["calibration_sample_ids"])
    rows: list[dict[str, object]] = []
    with (DATA_ROOT / "manifest.jsonl").open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    by_id = {str(row["sample_id"]): row for row in rows}
    if not calibration_ids <= set(by_id):
        raise RuntimeError("calibration IDs are missing from the final-fit corpus")
    holdout_groups = {
        str(by_id[sample_id]["split_group"]) for sample_id in calibration_ids
    }
    remaining_groups = {
        str(row["split_group"]) for row in rows
        if str(row["split_group"]) not in holdout_groups
    }
    dev_groups = {
        group for group in remaining_groups if stable_position(seed, group) < 0.05
    }
    if not dev_groups or dev_groups == remaining_groups:
        raise RuntimeError("final-fit dev group selection is degenerate")
    assignments: dict[str, str] = {}
    for row in rows:
        group = str(row["split_group"])
        assignments[str(row["sample_id"])] = (
            "holdout" if group in holdout_groups
            else "dev" if group in dev_groups
            else "train"
        )
    output.write_text(
        json.dumps(assignments, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    counts = {
        split: sum(value == split for value in assignments.values())
        for split in ("train", "dev", "holdout")
    }
    if any(value == 0 for value in counts.values()):
        raise RuntimeError(f"final-fit split is empty: {counts}")
    if any(assignments[sample_id] != "holdout" for sample_id in calibration_ids):
        raise RuntimeError("final-fit calibration leakage")
    return {
        "split_counts": counts,
        "calibration_rows": len(calibration_ids),
        "calibration_split_groups": len(holdout_groups),
        "split_sha256": sha256_file(output),
        "calibration_ids": sorted(calibration_ids),
    }


def write_calibration_predictions(
    epoch_predictions: Path, calibration_ids: set[str], output: Path
) -> None:
    with epoch_predictions.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["sample_id"] in calibration_ids]
    if len(rows) != len(calibration_ids) or {row["sample_id"] for row in rows} != calibration_ids:
        raise RuntimeError("selected-epoch calibration predictions are incomplete")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if os.environ.get("RUN1_LICENSES_REVIEWED", "").lower() != "true":
        raise RuntimeError("RUN1_LICENSES_REVIEWED=true is required")
    epochs = int(required_env("RUN1_SELECTED_EPOCH"))
    if epochs < 1 or epochs > 6:
        raise ValueError("RUN1_SELECTED_EPOCH must be between 1 and 6")
    WORK_ROOT.mkdir(parents=True, exist_ok=False)
    MODEL_ROOT.mkdir(parents=True, exist_ok=False)
    corpus = prepare_corpus()
    plan_path = Path("/dataset") / required_env("RUN1_PLAN_RELATIVE")
    final_split_path = DATA_ROOT / "final_splits.json"
    split = build_final_splits(plan_path, final_split_path, seed=2608)
    backbone = prepare_backbone()
    run_id = required_env("RUN1_TRAINING_RUN_ID")
    if not SAFE_RUN_ID.fullmatch(run_id) or ".." in run_id:
        raise ValueError("RUN1_TRAINING_RUN_ID is not a safe path component")
    remote_root = CHECKPOINT_MOUNT / "run1" / required_env("RUN1_PLAN_SHA256") / run_id / "final"
    resume = restore_checkpoint(remote_root)

    from deepvoice_run0.config import load_config
    from deepvoice_run0.training import train

    config = load_config(WORKSPACE / "config.toml")
    config["model"]["backbone_path"] = str(MODEL_ROOT / "backbone")
    config["training"]["epochs"] = epochs
    config["training"]["min_epochs"] = epochs
    config["training"]["early_stopping_patience"] = epochs + 1
    os.environ["RUN1_FOLD_ID"] = "final"
    started = time.monotonic()
    metadata = train(
        manifest_path=DATA_ROOT / "manifest.jsonl",
        split_path=final_split_path,
        config=config,
        output_dir=MODEL_ROOT,
        device="cuda",
        licenses_reviewed=True,
        resume_checkpoint_dir=resume,
        wall_clock_seconds=int(os.environ.get("RUN1_WALL_SECONDS", "3600")),
        epoch_callback=checkpoint_callback(remote_root),
    )
    selected_predictions = (
        MODEL_ROOT / "epochs" / f"epoch-{epochs:02d}-holdout_predictions.csv"
    )
    evidence = WORK_ROOT / "final-evidence"
    evidence.mkdir()
    shutil.copy2(MODEL_ROOT / "last.safetensors", evidence / "run0.safetensors")
    shutil.copy2(MODEL_ROOT / "model_config.json", evidence / "model_config.json")
    write_calibration_predictions(
        selected_predictions,
        set(split.pop("calibration_ids")),
        evidence / "calibration_run0_predictions.csv",
    )
    report = {
        "format_version": 1,
        "status": "completed",
        "selected_epoch": epochs,
        "weights_role": "last_epoch_selected_by_nested_logo",
        "plan_sha256": required_env("RUN1_PLAN_SHA256"),
        "source_bundle_sha256": required_env("RUN1_SOURCE_BUNDLE_SHA256"),
        "source_git_commit": required_env("RUN1_SOURCE_GIT_COMMIT"),
        "training_run_id": run_id,
        "wall_seconds": time.monotonic() - started,
        "corpus": corpus,
        "split": split,
        "backbone": backbone,
        "training": metadata["training"],
    }
    (evidence / "FINAL_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (evidence / "ARTIFACTS.json").write_text(
        json.dumps(
            {"format_version": 1, "files": artifact_hashes(evidence)},
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    from huggingface_hub import HfApi

    HfApi().upload_folder(
        repo_id=required_env("RUN1_FINAL_MODEL_REPO"),
        repo_type="model",
        folder_path=str(evidence),
        path_in_repo="final/run1-hybrid",
        commit_message=f"Add Run1 final selected epoch {epochs}",
    )
    done = dict(report)
    done["artifact_manifest_sha256"] = sha256_file(evidence / "ARTIFACTS.json")
    temporary = remote_root / "DONE.json.tmp"
    temporary.write_text(
        json.dumps(done, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(remote_root / "DONE.json")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
