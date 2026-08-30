#!/usr/bin/env python3
"""Verify staged inputs, train Run0, gate it, and upload a private run."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath


WORKSPACE = Path("/workspace")
DATASET_MOUNT = Path(os.environ.get("RUN0_DATASET_MOUNT", "/dataset"))
BACKBONE_MOUNT = Path(os.environ.get("RUN0_BACKBONE_MOUNT", "/model-source"))
WORK_ROOT = Path("/tmp/run0-training")
DATA_ROOT = WORK_ROOT / "corpus-v1"
MODEL_ROOT = WORK_ROOT / "model"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive_path, "r:") as archive:
        for member in archive.getmembers():
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts:
                raise RuntimeError(f"unsafe tar path: {member.name}")
            if not (member.isdir() or member.isfile()):
                raise RuntimeError(f"unsupported tar member type: {member.name}")
            target = (destination / Path(*pure.parts)).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise RuntimeError(f"tar member escapes destination: {member.name}")
        archive.extractall(destination, filter="data")


def prepare_corpus() -> dict[str, object]:
    shard_manifest_path = DATASET_MOUNT / "shards" / "SHARDS.json"
    shard_manifest = json.loads(shard_manifest_path.read_text(encoding="utf-8"))
    if shard_manifest.get("format_version") != 1:
        raise RuntimeError("unsupported shard manifest format")
    DATA_ROOT.mkdir(parents=True, exist_ok=False)
    for name in ("manifest.jsonl", "splits.json", "SUMMARY.json", "AUDIT.json"):
        shutil.copy2(DATASET_MOUNT / name, DATA_ROOT / name)
    actual_manifest_hash = sha256_file(DATA_ROOT / "manifest.jsonl")
    if actual_manifest_hash != shard_manifest["corpus_manifest_sha256"]:
        raise RuntimeError("corpus manifest hash does not match shard manifest")
    for record in shard_manifest["shards"]:
        source = DATASET_MOUNT / record["path"]
        if source.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"shard size mismatch: {source}")
        if sha256_file(source) != record["sha256"]:
            raise RuntimeError(f"shard hash mismatch: {source}")
        safe_extract(source, DATA_ROOT)

    from deepvoice_run0.manifest import read_manifest, sha256_file as manifest_sha256

    records = read_manifest(DATA_ROOT / "manifest.jsonl")
    if len(records) != 24_000:
        raise RuntimeError(f"expected 24000 corpus rows, found {len(records)}")
    failures: list[str] = []
    for record in records:
        path = DATA_ROOT / record.path
        if not path.is_file() or manifest_sha256(path) != record.sha256:
            failures.append(record.sample_id)
            if len(failures) >= 10:
                break
    if failures:
        raise RuntimeError(f"corpus audio hash audit failed: {failures}")
    return {
        "dataset_repo": required_env("RUN0_CORPUS_REPO"),
        "dataset_revision": required_env("RUN0_CORPUS_REVISION"),
        "manifest_sha256": actual_manifest_hash,
        "files_verified": len(records),
        "shards_verified": len(shard_manifest["shards"]),
    }


def prepare_backbone() -> dict[str, object]:
    review = json.loads((WORKSPACE / "backbone_source.json").read_text(encoding="utf-8"))
    if review.get("status") != "reviewed":
        raise RuntimeError("backbone source review is not approved")
    expected_repo = "microsoft/wavlm-base-plus"
    expected_revision = "4c66d4806a428f2e922ccfa1a962776e232d487b"
    if review.get("repo_id") != expected_repo or review.get("revision") != expected_revision:
        raise RuntimeError("backbone repo or revision differs from the approved source")
    source_relative = Path(str(review["private_source_path"]))
    source = BACKBONE_MOUNT / source_relative
    reviewed_files = review.get("files") or {}
    if not reviewed_files:
        raise RuntimeError("backbone review contains no files")
    actual_files = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file()
    }
    if actual_files != set(reviewed_files):
        raise RuntimeError(
            f"backbone file set mismatch; missing={sorted(set(reviewed_files) - actual_files)}, "
            f"extra={sorted(actual_files - set(reviewed_files))}"
        )
    for relative, expected in reviewed_files.items():
        path = source / relative
        if path.stat().st_size != int(expected["bytes"]):
            raise RuntimeError(f"backbone size mismatch: {relative}")
        if sha256_file(path) != expected["sha256"]:
            raise RuntimeError(f"backbone hash mismatch: {relative}")

    # Import and deserialize only after every staged file passes the review.
    import torch
    from transformers import AutoModel

    MODEL_ROOT.mkdir(parents=True, exist_ok=False)
    output = MODEL_ROOT / "backbone"
    original = AutoModel.from_pretrained(
        str(source), local_files_only=True, output_hidden_states=True
    ).eval()
    original.save_pretrained(output, safe_serialization=True)
    for name in ("preprocessor_config.json", "LICENSE-CC-BY-SA-3.0.txt", "ORIGIN.json"):
        if (source / name).is_file():
            shutil.copy2(source / name, output / name)
    converted = AutoModel.from_pretrained(
        str(output), local_files_only=True, output_hidden_states=True
    ).eval()
    fixed = torch.linspace(-0.1, 0.1, 32_000, dtype=torch.float32)[None]
    with torch.inference_mode():
        expected = original(fixed).last_hidden_state
        actual = converted(fixed).last_hidden_state
    max_error = float((expected - actual).abs().max())
    if max_error > 1e-5:
        raise RuntimeError(f"converted backbone output mismatch: {max_error}")
    del original, converted, expected, actual
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "repo_id": expected_repo,
        "revision": expected_revision,
        "license_id": review["license_id"],
        "source_review_sha256": sha256_file(WORKSPACE / "backbone_source.json"),
        "conversion_max_abs_error": max_error,
        "converted_files": sorted(
            path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
        ),
    }


def artifacts_hashes(root: Path) -> dict[str, dict[str, object]]:
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def main() -> None:
    if os.environ.get("RUN0_LICENSES_REVIEWED", "").lower() != "true":
        raise RuntimeError("RUN0_LICENSES_REVIEWED=true is required")
    run_id = required_env("RUN0_RUN_ID")
    model_repo = required_env("RUN0_MODEL_REPO")
    source_commit = required_env("RUN0_SOURCE_GIT_COMMIT")
    WORK_ROOT.mkdir(parents=True, exist_ok=False)
    corpus = prepare_corpus()
    backbone = prepare_backbone()

    from deepvoice_run0.config import load_config
    from deepvoice_run0.training import train

    config = load_config(WORKSPACE / "config.toml")
    config["model"]["backbone_path"] = str(MODEL_ROOT / "backbone")
    metadata = train(
        manifest_path=DATA_ROOT / "manifest.jsonl",
        split_path=DATA_ROOT / "splits.json",
        config=config,
        output_dir=MODEL_ROOT,
        device="cuda",
        licenses_reviewed=True,
        wall_clock_seconds=int(os.environ.get("RUN0_TRAIN_WALL_SECONDS", "15000")),
    )
    model_config_path = MODEL_ROOT / "model_config.json"
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    model_config["provenance"] = {
        "run_id": run_id,
        "source_git_commit": source_commit,
        "corpus": corpus,
        "backbone": backbone,
    }
    model_config_path.write_text(
        json.dumps(model_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dev = metadata["best_dev_metrics"]
    holdout = metadata["holdout_metrics"]
    gate_reasons: list[str] = []
    if float(holdout["weighted_score"]) <= 0.55:
        gate_reasons.append("holdout weighted score is not above 0.55")
    if float(dev["weighted_score"]) - float(holdout["weighted_score"]) > 0.15:
        gate_reasons.append("dev-holdout weighted score gap exceeds 0.15")
    for name in (
        "file_fake_prob", "voice_fake_prob", "music_fake_prob",
        "voice_present_prob", "music_present_prob",
    ):
        if float(holdout[f"std_{name}"]) <= 0.01:
            gate_reasons.append(f"holdout output std is too small: {name}")
    report = {
        "format_version": 1,
        "run_id": run_id,
        "status": "passed" if not gate_reasons else "failed",
        "gate_reasons": gate_reasons,
        "corpus": corpus,
        "backbone": backbone,
        "best_dev_metrics": dev,
        "holdout_metrics": holdout,
        "source_git_commit": source_commit,
    }
    (MODEL_ROOT / "TRAINING_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    hashes = artifacts_hashes(MODEL_ROOT)
    (MODEL_ROOT / "ARTIFACTS.json").write_text(
        json.dumps({"format_version": 1, "files": hashes}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    from huggingface_hub import HfApi

    HfApi().upload_folder(
        repo_id=model_repo,
        repo_type="model",
        folder_path=str(MODEL_ROOT),
        path_in_repo=f"runs/{run_id}",
        commit_message=f"Upload Run0 training artifacts {run_id}",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if gate_reasons:
        raise SystemExit("internal performance gate failed after artifact upload")


if __name__ == "__main__":
    main()
