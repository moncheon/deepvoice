#!/usr/bin/env python3
"""Benchmark a promoted Run0 on L4 and run the three-file submission smoke."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from pathlib import PurePosixPath

import numpy as np

from deepvoice_run0.constants import PREDICTION_COLUMNS
from deepvoice_run0.inference import Predictor
from deepvoice_run0.metrics import official_metrics
from deepvoice_run0.submission import read_submission_template, validate_prediction


WORKSPACE = Path("/workspace")
DATASET_MOUNT = Path(os.environ.get("RUN0_DATASET_MOUNT", "/dataset"))
MODEL_MOUNT = Path(os.environ.get("RUN0_MODEL_MOUNT", "/model-final"))
MODEL_PATH = MODEL_MOUNT / os.environ.get("RUN0_FINAL_MODEL_RELATIVE", "final/run0-v1")
OUTPUT = Path("/tmp/run0-l4-benchmark")
CORPUS_METADATA = DATASET_MOUNT / "corpus-v1"


def split_path() -> Path:
    relative = PurePosixPath(os.environ.get("RUN0_SPLIT_RELATIVE", "corpus-v1/splits.json"))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise RuntimeError(f"unsafe dataset-relative split path: {relative}")
    return DATASET_MOUNT / Path(*relative.parts)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def select_records(limit: int = 100) -> list[dict[str, object]]:
    splits = json.loads(split_path().read_text(encoding="utf-8"))
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    with (CORPUS_METADATA / "manifest.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if splits[record["sample_id"]] != "holdout":
                continue
            signature = (
                record["voice_present"], record["voice_fake"],
                record["music_present"], record["music_fake"],
            )
            groups[signature].append(record)
    for records in groups.values():
        records.sort(key=lambda record: (record["sha256"], record["sample_id"]))
    selected: list[dict[str, object]] = []
    while len(selected) < limit:
        changed = False
        for signature in sorted(groups, key=str):
            if groups[signature] and len(selected) < limit:
                selected.append(groups[signature].pop(0))
                changed = True
        if not changed:
            break
    if len(selected) != limit:
        raise RuntimeError(f"could not select {limit} stratified holdout records")
    return selected


def extract_records(records: list[dict[str, object]]) -> Path:
    test_dir = OUTPUT / "test"
    test_dir.mkdir(parents=True, exist_ok=False)
    by_code: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        code = Path(str(record["path"])).parts[1]
        by_code[code].append(record)
    for code, values in by_code.items():
        archive_path = DATASET_MOUNT / "shards" / f"audio-{code}.tar"
        with tarfile.open(archive_path, "r:") as archive:
            for record in values:
                member_name = str(record["path"]).replace("\\", "/")
                member = archive.getmember(member_name)
                if not member.isfile():
                    raise RuntimeError(f"benchmark member is not a file: {member_name}")
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"could not read benchmark member: {member_name}")
                destination = test_dir / f"{record['sample_id']}.wav"
                with source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
                if sha256_file(destination) != record["sha256"]:
                    raise RuntimeError(f"benchmark extraction hash mismatch: {member_name}")
    return test_dir


def labels_and_metrics(records: list[dict[str, object]], scores: np.ndarray) -> dict[str, float]:
    return official_metrics(
        file_fake=[int(record["file_fake"]) for record in records],
        file_fake_prob=scores[:, 0],
        voice_fake=[record["voice_fake"] for record in records],
        voice_fake_prob=scores[:, 1],
        music_fake=[record["music_fake"] for record in records],
        music_fake_prob=scores[:, 2],
        voice_present=[int(record["voice_present"]) for record in records],
        voice_present_prob=scores[:, 3],
        music_present=[int(record["music_present"]) for record in records],
        music_present_prob=scores[:, 4],
    )


def run_submission_smoke(max_segments: int) -> dict[str, object]:
    output_csv = OUTPUT / "sample-submission.csv"
    smoke_config = OUTPUT / "config.toml"
    source_config = (WORKSPACE / "config.toml").read_text(encoding="utf-8")
    replacements = {
        "max_segments = 16": f"max_segments = {max_segments}",
        "uniform_segments = 12": f"uniform_segments = {max(1, max_segments - 4)}",
        "novelty_segments = 4": f"novelty_segments = {min(4, max_segments - 1)}",
    }
    for original, replacement in replacements.items():
        if source_config.count(original) != 1:
            raise RuntimeError(f"benchmark config contract changed: {original}")
        source_config = source_config.replace(original, replacement)
    smoke_config.write_text(source_config, encoding="utf-8")
    command = [
        sys.executable, str(WORKSPACE / "script.py"),
        "--test-dir", str(WORKSPACE / "sample" / "test"),
        "--sample-submission", str(WORKSPACE / "sample" / "sample_submission.csv"),
        "--output", str(output_csv),
        "--model-dir", str(MODEL_PATH),
        "--config", str(smoke_config),
        "--device", "cuda",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"submission smoke failed: {result.stderr[-1000:]}")
    fieldnames, rows = read_submission_template(output_csv)
    if fieldnames != ["ID", *PREDICTION_COLUMNS] or len(rows) != 3:
        raise RuntimeError("submission smoke output schema mismatch")
    for row in rows:
        validate_prediction(float(row[name]) for name in PREDICTION_COLUMNS)
    return {
        "rows": len(rows),
        "max_segments": max_segments,
        "stdout": result.stdout[-500:],
        "output_sha256": sha256_file(output_csv),
    }


def main() -> None:
    import torch

    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(MODEL_PATH)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    records = select_records(100)
    test_dir = extract_records(records)
    load_started = time.perf_counter()
    predictor = Predictor(MODEL_PATH, WORKSPACE / "config.toml", "cuda")
    load_seconds = time.perf_counter() - load_started
    torch.cuda.reset_peak_memory_stats()
    results: dict[str, dict[str, object]] = {}
    baseline_score: float | None = None
    selected_segments: int | None = None
    for segments in (16, 12, 8):
        predictor.config["audio"]["max_segments"] = segments
        predictor.config["audio"]["uniform_segments"] = max(1, segments - 4)
        predictor.config["audio"]["novelty_segments"] = min(4, segments - 1)
        started = time.perf_counter()
        scores = np.asarray([
            validate_prediction(predictor.predict_file(test_dir / f"{record['sample_id']}.wav"))
            for record in records
        ])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        metrics = labels_and_metrics(records, scores)
        estimated = load_seconds + elapsed / len(records) * 1200
        peak = int(torch.cuda.max_memory_allocated())
        results[str(segments)] = {
            "files": len(records),
            "elapsed_seconds": elapsed,
            "load_seconds": load_seconds,
            "estimated_seconds_for_1200": estimated,
            "peak_vram_bytes": peak,
            "metrics": metrics,
        }
        if baseline_score is None:
            baseline_score = float(metrics["weighted_score"])
        score_drop = baseline_score - float(metrics["weighted_score"])
        if selected_segments is None and estimated <= 2700 and peak <= 20 * 1024**3 and score_drop <= 0.02:
            selected_segments = segments
    if selected_segments is None:
        raise RuntimeError("no segment count passed runtime, VRAM, and score gates")
    smoke = run_submission_smoke(selected_segments)
    report = {
        "format_version": 1,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "selected_max_segments": selected_segments,
        "results": results,
        "submission_smoke": smoke,
        "model_artifacts_sha256": sha256_file(MODEL_PATH / "ARTIFACTS.json"),
        "split_relative": split_path().relative_to(DATASET_MOUNT).as_posix(),
        "split_sha256": sha256_file(split_path()),
    }
    report_path = OUTPUT / "L4_BENCHMARK.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    upload_environment = dict(os.environ)
    upload_environment.pop("HF_HUB_OFFLINE", None)
    upload_environment.pop("TRANSFORMERS_OFFLINE", None)
    upload_code = (
        "from huggingface_hub import HfApi; import sys; "
        "HfApi().upload_file(repo_id=sys.argv[2], repo_type='model', "
        "path_or_fileobj=sys.argv[1], path_in_repo='reports/L4_BENCHMARK.json', "
        "commit_message='Add Run0 L4 benchmark evidence')"
    )
    uploaded = subprocess.run(
        [sys.executable, "-c", upload_code, str(report_path), os.environ["RUN0_MODEL_REPO"]],
        env=upload_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if uploaded.returncode != 0:
        raise RuntimeError(f"benchmark report upload failed: {uploaded.stderr[-1000:]}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
