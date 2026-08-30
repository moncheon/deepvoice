#!/usr/bin/env python3
"""Download, hash-audit, and promote one private Run0 training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any


RUN0_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODEL_FILES = {
    "run0.safetensors",
    "model_config.json",
    "TRAINING_REPORT.json",
    "backbone/config.json",
    "backbone/model.safetensors",
    "backbone/LICENSE-CC-BY-SA-3.0.txt",
    "backbone/ORIGIN.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def checked_relative(name: str) -> PurePosixPath:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise RuntimeError(f"unsafe artifact path: {name}")
    return relative


def verify_run(
    run_root: Path,
    *,
    run_id: str,
    expected_source_commit: str,
) -> dict[str, Any]:
    """Validate the downloaded run without deserializing model weights."""

    artifact_manifest = load_json(run_root / "ARTIFACTS.json")
    if artifact_manifest.get("format_version") != 1:
        raise RuntimeError("ARTIFACTS.json must use format_version 1")
    entries = artifact_manifest.get("files")
    if not isinstance(entries, dict) or not entries:
        raise RuntimeError("ARTIFACTS.json contains no file ledger")

    expected_paths = {"ARTIFACTS.json"}
    verified_bytes = 0
    for name, evidence in entries.items():
        relative = checked_relative(str(name))
        if not isinstance(evidence, dict):
            raise RuntimeError(f"invalid artifact evidence: {name}")
        path = run_root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing or linked artifact: {name}")
        expected_size = int(evidence["bytes"])
        if path.stat().st_size != expected_size:
            raise RuntimeError(f"artifact size mismatch: {name}")
        if sha256_file(path) != str(evidence["sha256"]):
            raise RuntimeError(f"artifact SHA-256 mismatch: {name}")
        expected_paths.add(relative.as_posix())
        verified_bytes += expected_size

    actual_paths = {
        path.relative_to(run_root).as_posix()
        for path in run_root.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(run_root).parts
    }
    if actual_paths != expected_paths:
        raise RuntimeError(
            "downloaded run file set mismatch; "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - expected_paths)}"
        )
    if not REQUIRED_MODEL_FILES.issubset(expected_paths):
        raise RuntimeError(
            f"required model artifacts are missing: {sorted(REQUIRED_MODEL_FILES - expected_paths)}"
        )

    report = load_json(run_root / "TRAINING_REPORT.json")
    metadata = load_json(run_root / "model_config.json")
    if report.get("status") != "passed" or report.get("gate_reasons"):
        raise RuntimeError(f"training performance gate did not pass: {report.get('gate_reasons')}")
    if report.get("run_id") != run_id:
        raise RuntimeError("training report run ID mismatch")
    if report.get("source_git_commit") != expected_source_commit:
        raise RuntimeError("training report source commit mismatch")
    if metadata.get("format_version") != 2:
        raise RuntimeError("model_config.json must use format_version 2")
    provenance = metadata.get("provenance") or {}
    if provenance.get("run_id") != run_id:
        raise RuntimeError("model provenance run ID mismatch")
    if provenance.get("source_git_commit") != expected_source_commit:
        raise RuntimeError("model provenance source commit mismatch")

    return {
        "format_version": 1,
        "status": "reviewed",
        "run_id": run_id,
        "source_git_commit": expected_source_commit,
        "artifact_count": len(entries),
        "artifact_bytes_verified": verified_bytes,
        "best_dev_weighted_score": report["best_dev_metrics"]["weighted_score"],
        "holdout_weighted_score": report["holdout_metrics"]["weighted_score"],
    }


def copy_verified_run(run_root: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(path for path in run_root.rglob("*") if path.is_file()):
        if ".cache" in source.relative_to(run_root).parts:
            continue
        relative = source.relative_to(run_root)
        target = destination / relative
        if target.exists():
            raise RuntimeError(f"refusing to overwrite promoted artifact: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True, help="Immutable model repository commit")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--destination", type=Path, default=RUN0_ROOT / "model")
    parser.add_argument("--final-path", default="final/run0-v1")
    parser.add_argument("--skip-remote-promotion", action="store_true")
    args = parser.parse_args()
    if len(args.revision) != 40 or any(char not in "0123456789abcdef" for char in args.revision):
        raise SystemExit("--revision must be a lowercase 40-character commit SHA")

    stage = RUN0_ROOT / "build" / "promotions" / args.run_id
    if stage.exists():
        raise SystemExit(f"refusing to merge into existing review stage: {stage}")
    download_root = stage / "download"
    download_root.mkdir(parents=True)
    subprocess.run([
        "hf", "download", args.repo_id,
        "--repo-type", "model",
        "--revision", args.revision,
        "--include", f"runs/{args.run_id}/*",
        "--local-dir", str(download_root),
    ], check=True)
    run_root = download_root / "runs" / args.run_id
    review = verify_run(
        run_root,
        run_id=args.run_id,
        expected_source_commit=args.source_commit,
    )
    review.update({"repo_id": args.repo_id, "repo_revision": args.revision})
    review_path = run_root / "PROMOTION_REVIEW.json"
    review_path.write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    copy_verified_run(run_root, args.destination.resolve())

    remote_result = None
    if not args.skip_remote_promotion:
        completed = subprocess.run([
            "hf", "upload", args.repo_id, str(run_root), args.final_path,
            "--repo-type", "model",
            "--commit-message", f"Promote reviewed Run0 artifacts {args.run_id}",
            "--format", "json",
        ], check=True, capture_output=True, text=True)
        remote_result = completed.stdout.strip()
    print(json.dumps({
        "review": review,
        "local_destination": str(args.destination.resolve()),
        "remote_result": remote_result,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
