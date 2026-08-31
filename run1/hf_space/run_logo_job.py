#!/usr/bin/env python3
"""Verify mounted inputs, run one resumable LOGO fold, and upload evidence."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import time
from pathlib import Path, PurePosixPath


WORKSPACE = Path("/workspace")
DATASET_MOUNT = Path(os.environ.get("RUN1_DATASET_MOUNT", "/dataset"))
MODEL_SOURCE = Path(os.environ.get("RUN1_MODEL_SOURCE", "/model-source/final/run0-v1"))
CHECKPOINT_MOUNT = Path(os.environ.get("RUN1_CHECKPOINT_MOUNT", "/checkpoints"))
WORK_ROOT = Path("/tmp/run1-logo")
DATA_ROOT = WORK_ROOT / "corpus-v1"
MODEL_ROOT = WORK_ROOT / "model"


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"unsafe relative path: {value}")
    return Path(*pure.parts)


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
    corpus_source = DATASET_MOUNT / "corpus-v1"
    DATA_ROOT.mkdir(parents=True, exist_ok=False)
    for name in ("manifest.jsonl", "SUMMARY.json", "AUDIT.json"):
        shutil.copy2(corpus_source / name, DATA_ROOT / name)
    split_relative = safe_relative(required_env("RUN1_SPLIT_RELATIVE"))
    split_source = DATASET_MOUNT / split_relative
    shutil.copy2(split_source, DATA_ROOT / "splits.json")
    manifest_hash = sha256_file(DATA_ROOT / "manifest.jsonl")
    if manifest_hash != shard_manifest["corpus_manifest_sha256"]:
        raise RuntimeError("corpus manifest hash does not match shard manifest")
    for record in shard_manifest["shards"]:
        source = DATASET_MOUNT / record["path"]
        if source.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"shard size mismatch: {source}")
        if sha256_file(source) != record["sha256"]:
            raise RuntimeError(f"shard hash mismatch: {source}")
        safe_extract(source, DATA_ROOT)
    from deepvoice_run0.manifest import read_manifest, sha256_file as audio_sha256

    records = read_manifest(DATA_ROOT / "manifest.jsonl")
    if len(records) != 24_000:
        raise RuntimeError(f"expected 24000 corpus rows, found {len(records)}")
    failures: list[str] = []
    for record in records:
        path = DATA_ROOT / record.path
        if not path.is_file() or audio_sha256(path) != record.sha256:
            failures.append(record.sample_id)
            if len(failures) == 10:
                break
    if failures:
        raise RuntimeError(f"corpus audio hash audit failed: {failures}")
    return {
        "repo": required_env("RUN1_CORPUS_REPO"),
        "revision": required_env("RUN1_CORPUS_REVISION"),
        "manifest_sha256": manifest_hash,
        "split_relative": split_relative.as_posix(),
        "split_sha256": sha256_file(DATA_ROOT / "splits.json"),
        "files_verified": len(records),
        "shards_verified": len(shard_manifest["shards"]),
    }


def prepare_backbone() -> dict[str, object]:
    ledger_path = MODEL_SOURCE / "ARTIFACTS.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("format_version") != 1:
        raise RuntimeError("invalid promoted Run0 artifact ledger")
    source = MODEL_SOURCE / "backbone"
    expected = {
        name.removeprefix("backbone/"): record
        for name, record in ledger["files"].items()
        if name.startswith("backbone/")
    }
    actual = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file()
    }
    if actual != set(expected):
        raise RuntimeError("promoted backbone file set differs from reviewed ledger")
    for relative, record in expected.items():
        path = source / relative
        if path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"backbone size mismatch: {relative}")
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"backbone hash mismatch: {relative}")
    destination = MODEL_ROOT / "backbone"
    shutil.copytree(source, destination)
    return {
        "source_repo": required_env("RUN1_MODEL_SOURCE_REPO"),
        "source_revision": required_env("RUN1_MODEL_SOURCE_REVISION"),
        "artifact_ledger_sha256": sha256_file(ledger_path),
        "files": sorted(actual),
    }


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def checkpoint_manifest(checkpoint: Path) -> dict[str, object]:
    required = {
        "model.safetensors", "optimizer.safetensors", "best.safetensors", "state.json"
    }
    actual = {
        path.name for path in checkpoint.iterdir()
        if path.is_file() and path.name != "CHECKPOINT.json"
    }
    if actual != required:
        raise RuntimeError(
            f"local resume checkpoint file set differs; expected={sorted(required)}, "
            f"actual={sorted(actual)}"
        )
    return {
        "format_version": 1,
        "files": {
            name: {
                "bytes": (checkpoint / name).stat().st_size,
                "sha256": sha256_file(checkpoint / name),
            }
            for name in sorted(required)
        },
    }


def restore_checkpoint(remote_root: Path) -> Path | None:
    remote_resume = remote_root / "resume"
    manifest_path = remote_resume / "CHECKPOINT.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise RuntimeError("unsupported remote checkpoint manifest")
    expected = manifest.get("files")
    if not isinstance(expected, dict):
        raise RuntimeError("remote checkpoint manifest has no files mapping")
    actual = {
        path.name for path in remote_resume.iterdir()
        if path.is_file() and path.name != "CHECKPOINT.json"
    }
    if actual != set(expected):
        raise RuntimeError("remote checkpoint file set differs from manifest")
    for name, record in expected.items():
        path = remote_resume / name
        if path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"remote checkpoint size mismatch: {name}")
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"remote checkpoint hash mismatch: {name}")
    local_resume = MODEL_ROOT / "resume"
    local_resume.mkdir()
    for name in expected:
        shutil.copy2(remote_resume / name, local_resume / name)
    if (remote_root / "epochs").is_dir():
        shutil.copytree(remote_root / "epochs", MODEL_ROOT / "epochs")
    return local_resume


def checkpoint_callback(remote_root: Path):
    def callback(output: Path, checkpoint: Path, epoch: int) -> None:
        manifest = checkpoint_manifest(checkpoint)
        # The manifest is copied last, so an interrupted multi-file update is
        # never accepted as a complete resumable checkpoint.
        for name in ("model.safetensors", "optimizer.safetensors", "best.safetensors", "state.json"):
            atomic_copy(checkpoint / name, remote_root / "resume" / name)
        local_manifest = checkpoint / "CHECKPOINT.json"
        local_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        atomic_copy(local_manifest, remote_root / "resume" / "CHECKPOINT.json")
        epoch_source = output / "epochs" / f"epoch-{epoch:02d}-holdout_predictions.csv"
        if epoch_source.is_file():
            atomic_copy(epoch_source, remote_root / "epochs" / epoch_source.name)
        progress = {
            "format_version": 1,
            "fold_id": required_env("RUN1_FOLD_ID"),
            "plan_sha256": required_env("RUN1_PLAN_SHA256"),
            "epoch_completed": epoch,
            "resume_state_sha256": sha256_file(checkpoint / "state.json"),
            "updated_unix": time.time(),
        }
        temporary = remote_root / "PROGRESS.json.tmp"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(remote_root / "PROGRESS.json")
        print(json.dumps({"event": "checkpoint_uploaded", **progress}, sort_keys=True), flush=True)

    return callback


def artifact_hashes(root: Path) -> dict[str, dict[str, object]]:
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def main() -> None:
    if os.environ.get("RUN1_LICENSES_REVIEWED", "").lower() != "true":
        raise RuntimeError("RUN1_LICENSES_REVIEWED=true is required")
    fold_id = required_env("RUN1_FOLD_ID")
    plan_sha256 = required_env("RUN1_PLAN_SHA256")
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in fold_id):
        raise ValueError("unsafe fold ID")
    if len(plan_sha256) != 64:
        raise ValueError("invalid plan SHA-256")
    epochs = int(required_env("RUN1_EPOCHS"))
    if epochs < 1 or epochs > 8:
        raise ValueError("RUN1_EPOCHS must be between 1 and 8")
    WORK_ROOT.mkdir(parents=True, exist_ok=False)
    MODEL_ROOT.mkdir(parents=True, exist_ok=False)
    corpus = prepare_corpus()
    backbone = prepare_backbone()
    run_id = required_env("RUN1_TRAINING_RUN_ID")
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in run_id):
        raise ValueError("unsafe training run ID")
    remote_root = CHECKPOINT_MOUNT / "run1" / plan_sha256 / run_id / fold_id
    resume = restore_checkpoint(remote_root)

    from deepvoice_run0.config import load_config
    from deepvoice_run0.training import train

    config = load_config(WORKSPACE / "config.toml")
    config["model"]["backbone_path"] = str(MODEL_ROOT / "backbone")
    config["training"]["epochs"] = epochs
    config["training"]["min_epochs"] = min(4, epochs)
    config["training"]["early_stopping_patience"] = epochs + 1
    started = time.monotonic()
    metadata = train(
        manifest_path=DATA_ROOT / "manifest.jsonl",
        split_path=DATA_ROOT / "splits.json",
        config=config,
        output_dir=MODEL_ROOT,
        device="cuda",
        licenses_reviewed=True,
        resume_checkpoint_dir=resume,
        wall_clock_seconds=int(os.environ.get("RUN1_WALL_SECONDS", "7200")),
        epoch_callback=checkpoint_callback(remote_root),
    )
    evidence = WORK_ROOT / "evidence"
    evidence.mkdir()
    for name in (
        "model_config.json", "training_state.json", "dev_predictions.csv",
        "holdout_predictions.csv",
    ):
        if (MODEL_ROOT / name).is_file():
            shutil.copy2(MODEL_ROOT / name, evidence / name)
    shutil.copytree(MODEL_ROOT / "epochs", evidence / "epochs")
    report = {
        "format_version": 1,
        "status": "completed",
        "fold_id": fold_id,
        "plan_sha256": plan_sha256,
        "epochs_requested": epochs,
        "epochs_completed": metadata["training"]["epochs_completed"],
        "wall_seconds": time.monotonic() - started,
        "source_git_commit": required_env("RUN1_SOURCE_GIT_COMMIT"),
        "source_bundle_sha256": required_env("RUN1_SOURCE_BUNDLE_SHA256"),
        "training_run_id": run_id,
        "corpus": corpus,
        "backbone": backbone,
    }
    (evidence / "FOLD_REPORT.json").write_text(
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

    storage_repo = required_env("RUN1_STORAGE_REPO")
    remote_path = f"storage/run1-hybrid/logo/{plan_sha256}/{fold_id}"
    HfApi().upload_folder(
        repo_id=storage_repo,
        repo_type="model",
        folder_path=str(evidence),
        path_in_repo=remote_path,
        commit_message=f"Add Run1 LOGO evidence {fold_id}",
    )
    done = dict(report)
    done["evidence_artifacts_sha256"] = sha256_file(evidence / "ARTIFACTS.json")
    temporary = remote_root / "DONE.json.tmp"
    temporary.write_text(json.dumps(done, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(remote_root / "DONE.json")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
