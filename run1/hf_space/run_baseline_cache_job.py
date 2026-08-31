#!/usr/bin/env python3
"""Compute the exact organizer Baseline once with per-audio resumable caches."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from run_logo_job import DATA_ROOT, WORK_ROOT, prepare_corpus, required_env, sha256_file


WORKSPACE = Path("/workspace")
BASELINE_STORAGE = Path(os.environ.get("RUN1_BASELINE_STORAGE", "/baseline-storage"))
CHECKPOINT_MOUNT = Path(os.environ.get("RUN1_CHECKPOINT_MOUNT", "/checkpoints"))
BASELINE_MODEL = WORK_ROOT / "baseline-model"


def prepare_baseline_model() -> dict[str, object]:
    ledger_path = WORKSPACE / "baseline_artifacts.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("format_version") != 1:
        raise RuntimeError("unsupported Baseline artifact ledger")
    shutil.copytree(WORKSPACE / "baseline_model", BASELINE_MODEL)
    for relative, record in ledger["files"].items():
        source = BASELINE_STORAGE / record["storage_path"]
        if source.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"Baseline artifact size mismatch: {relative}")
        if sha256_file(source) != record["sha256"]:
            raise RuntimeError(f"Baseline artifact hash mismatch: {relative}")
        destination = BASELINE_MODEL / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return {
        "ledger_sha256": sha256_file(ledger_path),
        "source_repo": ledger["source_repo"],
        "source_revision": ledger["source_revision"],
        "files": sorted(ledger["files"]),
    }


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def cache_path(root: Path, branch: str, audio_sha256: str) -> Path:
    return root / branch / audio_sha256[:2] / f"{audio_sha256}.json"


def read_cache(
    root: Path, branch: str, audio_sha256: str, fields: tuple[str, ...]
) -> dict[str, float] | None:
    path = cache_path(root, branch, audio_sha256)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("format_version") != 1
        or payload.get("branch") != branch
        or payload.get("audio_sha256") != audio_sha256
        or set(payload.get("values", {})) != set(fields)
    ):
        raise RuntimeError(f"invalid Baseline cache record: {path}")
    values = {name: float(payload["values"][name]) for name in fields}
    if any(value < 0.0 or value > 1.0 for value in values.values()):
        raise RuntimeError(f"out-of-range Baseline cache record: {path}")
    return values


def write_cache(
    root: Path, branch: str, audio_sha256: str, values: dict[str, float]
) -> None:
    atomic_json(cache_path(root, branch, audio_sha256), {
        "format_version": 1,
        "branch": branch,
        "audio_sha256": audio_sha256,
        "values": values,
    })


def target_rows() -> list[dict[str, object]]:
    plan_path = Path("/dataset") / required_env("RUN1_PLAN_RELATIVE")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("plan_sha256") != required_env("RUN1_PLAN_SHA256"):
        raise RuntimeError("mounted LOGO plan SHA does not match requested plan")
    target_ids = set(plan["calibration_sample_ids"])
    for fold in plan["folds"]:
        target_ids.update(fold["panel_sample_ids"])
    rows: list[dict[str, object]] = []
    with (DATA_ROOT / "manifest.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["sample_id"] in target_ids:
                rows.append(row)
    found = {str(row["sample_id"]) for row in rows}
    if found != target_ids:
        raise RuntimeError(f"Baseline target IDs missing from corpus: {len(target_ids - found)}")
    rows.sort(key=lambda row: str(row["sample_id"]))
    shard_count = int(os.environ.get("RUN1_CACHE_SHARD_COUNT", "1"))
    shard_index = int(os.environ.get("RUN1_CACHE_SHARD_INDEX", "0"))
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid Baseline cache shard")
    selected = [
        row for row in rows
        if int(str(row["sha256"]), 16)
        % shard_count == shard_index
    ]
    limit = int(os.environ.get("RUN1_CACHE_LIMIT", "0"))
    if limit < 0:
        raise ValueError("RUN1_CACHE_LIMIT cannot be negative")
    return selected[:limit] if limit else selected


def main() -> None:
    if os.environ.get("RUN1_LICENSES_REVIEWED", "").lower() != "true":
        raise RuntimeError("RUN1_LICENSES_REVIEWED=true is required")
    started = time.monotonic()
    wall_seconds = int(os.environ.get("RUN1_WALL_SECONDS", "7200"))
    corpus = prepare_corpus()
    baseline_artifacts = prepare_baseline_model()
    fingerprint_inputs = {
        "artifact_ledger_sha256": baseline_artifacts["ledger_sha256"],
        "baseline_script_sha256": sha256_file(WORKSPACE / "baseline_script.py"),
        "adapter_sha256": sha256_file(
            WORKSPACE / "deepvoice_hybrid" / "baseline.py"
        ),
        "requirements_sha256": sha256_file(
            WORKSPACE / "baseline_requirements.txt"
        ),
    }
    baseline_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_inputs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    rows = target_rows()
    cache_root = (
        CHECKPOINT_MOUNT / "run1" / required_env("RUN1_PLAN_SHA256")
        / "baseline-cache" / baseline_fingerprint[:16]
    )
    from deepvoice_hybrid.baseline import BaselineAdapter

    adapter = BaselineAdapter(
        WORKSPACE / "baseline_script.py", BASELINE_MODEL, "cuda"
    )
    presence_fields = ("voice_present", "music_present")
    fake_fields = ("file_fake", "voice_fake", "music_fake")
    presence: dict[str, dict[str, float]] = {}
    missing_presence = [
        row for row in rows
        if read_cache(cache_root, "presence", str(row["sha256"]), presence_fields) is None
    ]
    if missing_presence:
        adapter.load_presence()
    for index, row in enumerate(rows, start=1):
        audio_sha = str(row["sha256"])
        values = read_cache(cache_root, "presence", audio_sha, presence_fields)
        if values is None:
            values = adapter.predict_presence(DATA_ROOT / str(row["path"]))
            write_cache(cache_root, "presence", audio_sha, values)
        presence[audio_sha] = values
        if index % 50 == 0:
            print(json.dumps({"event": "presence_progress", "done": index, "total": len(rows)}), flush=True)
        if time.monotonic() - started >= wall_seconds:
            break
    adapter.close_presence()
    gc.collect()

    # Only start the much larger fake branch after every shard item has a
    # presence score; interrupted jobs therefore resume at an unambiguous phase.
    presence_complete = len(presence) == len(rows)
    fake_by_sha: dict[str, dict[str, float]] = {}
    if presence_complete and time.monotonic() - started < wall_seconds:
        missing_fake = [
            row for row in rows
            if read_cache(cache_root, "baseline_fake", str(row["sha256"]), fake_fields) is None
        ]
        if missing_fake:
            adapter.load_fake()
        for index, row in enumerate(rows, start=1):
            audio_sha = str(row["sha256"])
            values = read_cache(cache_root, "baseline_fake", audio_sha, fake_fields)
            if values is None:
                values = adapter.predict_fake(
                    DATA_ROOT / str(row["path"]), presence[audio_sha]
                )
                write_cache(cache_root, "baseline_fake", audio_sha, values)
            fake_by_sha[audio_sha] = values
            if index % 20 == 0:
                print(json.dumps({"event": "fake_progress", "done": index, "total": len(rows)}), flush=True)
            if time.monotonic() - started >= wall_seconds:
                break
    adapter.close_fake()

    complete_rows: list[dict[str, object]] = []
    for row in rows:
        audio_sha = str(row["sha256"])
        p = presence.get(audio_sha)
        f = fake_by_sha.get(audio_sha)
        if p is None or f is None:
            continue
        complete_rows.append({
            "sample_id": row["sample_id"],
            "audio_sha256": audio_sha,
            "FILE_FAKE_PROB": f["file_fake"],
            "VOICE_FAKE_PROB": f["voice_fake"],
            "MUSIC_FAKE_PROB": f["music_fake"],
            "VOICE_PRESENT_PROB": p["voice_present"],
            "MUSIC_PRESENT_PROB": p["music_present"],
        })
    report = {
        "format_version": 1,
        "plan_sha256": required_env("RUN1_PLAN_SHA256"),
        "target_rows": len(rows),
        "completed_rows": len(complete_rows),
        "status": "completed" if len(complete_rows) == len(rows) else "partial",
        "wall_seconds": time.monotonic() - started,
        "corpus": corpus,
        "baseline_artifacts": baseline_artifacts,
        "baseline_fingerprint": baseline_fingerprint,
        "baseline_fingerprint_inputs": fingerprint_inputs,
        "source_bundle_sha256": required_env("RUN1_SOURCE_BUNDLE_SHA256"),
    }
    atomic_json(cache_root / "PROGRESS.json", report)
    if complete_rows:
        output = WORK_ROOT / "baseline_predictions.csv"
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(complete_rows[0]))
            writer.writeheader()
            writer.writerows(complete_rows)
        destination = cache_root / "shards" / (
            f"baseline-{os.environ.get('RUN1_CACHE_SHARD_INDEX', '0')}.csv"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".csv.tmp")
        shutil.copy2(output, temporary)
        temporary.replace(destination)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
