#!/usr/bin/env python3
"""Benchmark the exact submission archive on 1,200 deterministic corpus files."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path, PurePosixPath


WORK = Path("/tmp/run1-benchmark")
DATASET = Path("/dataset/corpus-v1")
CHECKPOINTS = Path(os.environ.get("RUN1_CHECKPOINT_MOUNT", "/checkpoints"))
ZIP_PATH = Path(os.environ.get("RUN1_SUBMISSION_ZIP", "/submission/submit.zip"))
PREDICTIONS = (
    "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
)


def required(name: str) -> str:
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


def stable_key(sample_id: str) -> bytes:
    return hashlib.sha256(f"2608:benchmark:{sample_id}".encode()).digest()


def safe_extract(archive_path: Path, destination: Path) -> dict[str, int]:
    expected_hash = required("RUN1_SUBMISSION_ZIP_SHA256")
    if sha256_file(archive_path) != expected_hash:
        raise RuntimeError("submission ZIP SHA-256 mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if "script.py" not in names or "requirements.txt" not in names:
            raise RuntimeError("submission root contract is incomplete")
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise RuntimeError(f"unsafe submission ZIP entry: {name}")
            if name not in {"script.py", "requirements.txt"} and not name.startswith("model/"):
                raise RuntimeError(f"unexpected submission ZIP entry: {name}")
        expanded = sum(info.file_size for info in infos)
        if expanded >= 30 * 1024**3:
            raise RuntimeError("submission expands to 30 GB or more")
        archive.extractall(destination)
    return {"entries": len(infos), "expanded_bytes": expanded}


def prepare_inputs(count: int) -> tuple[Path, Path, list[dict[str, object]]]:
    manifest = DATASET / "manifest.jsonl"
    if sha256_file(manifest) != required("RUN1_CORPUS_MANIFEST_SHA256"):
        raise RuntimeError("benchmark corpus manifest SHA-256 mismatch")
    with manifest.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows.sort(key=lambda row: stable_key(str(row["sample_id"])))
    selected = rows[:count]
    if len(selected) != count:
        raise RuntimeError("benchmark corpus is smaller than requested count")
    test_dir = WORK / "data" / "test"
    test_dir.mkdir(parents=True)
    template = WORK / "data" / "sample_submission.csv"
    for row in selected:
        source = DATASET / str(row["path"])
        if sha256_file(source) != str(row["sha256"]):
            raise RuntimeError(f"benchmark audio hash mismatch: {row['sample_id']}")
        destination = test_dir / f"{row['sample_id']}{source.suffix.lower()}"
        destination.symlink_to(source)
    with template.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", *PREDICTIONS])
        writer.writeheader()
        for row in selected:
            writer.writerow({"ID": row["sample_id"], **{name: 0.0 for name in PREDICTIONS}})
    return test_dir, template, selected


def gpu_memory_sampler(stop: threading.Event, peak: dict[str, int]) -> None:
    while not stop.is_set():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi", "--query-compute-apps=used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            values = [
                int(line.strip()) for line in result.stdout.splitlines()
                if line.strip().isdigit()
            ]
            if values:
                peak["used_memory_mib"] = max(peak["used_memory_mib"], max(values))
        except (FileNotFoundError, subprocess.SubprocessError, ValueError):
            peak["sampler_errors"] += 1
        stop.wait(0.5)


def main() -> None:
    count = int(os.environ.get("RUN1_BENCHMARK_COUNT", "1200"))
    if count != 1200:
        raise ValueError("the release gate requires exactly 1,200 files")
    WORK.mkdir(parents=True, exist_ok=False)
    extracted = WORK / "submission"
    extracted.mkdir()
    archive = safe_extract(ZIP_PATH, extracted)
    test_dir, template, selected = prepare_inputs(count)
    output = WORK / "submission.csv"
    environment = dict(os.environ)
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    stop_sampling = threading.Event()
    gpu_peak = {"used_memory_mib": 0, "sampler_errors": 0}
    sampler = threading.Thread(
        target=gpu_memory_sampler, args=(stop_sampling, gpu_peak), daemon=True
    )
    sampler.start()
    started = time.monotonic()
    try:
        subprocess.run(
            [
                sys.executable, str(extracted / "script.py"),
                "--test-dir", str(test_dir),
                "--sample-submission", str(template),
                "--output", str(output),
                "--device", "cuda",
            ],
            cwd=extracted,
            env=environment,
            check=True,
        )
    finally:
        stop_sampling.set()
        sampler.join(timeout=10)
    wall_seconds = time.monotonic() - started
    with output.open("r", encoding="utf-8-sig", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    if len(predictions) != count or {row["ID"] for row in predictions} != {
        str(row["sample_id"]) for row in selected
    }:
        raise RuntimeError("benchmark output does not align with selected inputs")
    for row in predictions:
        if any(not 0.0 <= float(row[name]) <= 1.0 for name in PREDICTIONS):
            raise RuntimeError(f"out-of-range benchmark prediction: {row['ID']}")
    import torch

    report = {
        "format_version": 1,
        "status": "passed" if wall_seconds <= 3000 else "failed_runtime",
        "count": count,
        "wall_seconds": wall_seconds,
        "internal_runtime_limit_seconds": 3000,
        "competition_runtime_limit_seconds": 3600,
        "submission_zip_sha256": required("RUN1_SUBMISSION_ZIP_SHA256"),
        "output_sha256": sha256_file(output),
        "corpus_manifest_sha256": required("RUN1_CORPUS_MANIFEST_SHA256"),
        "selection_rule": "sha256(2608:benchmark:sample_id) first 1200",
        "archive": archive,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "sampled_peak_process_memory_mib": gpu_peak["used_memory_mib"],
            "memory_sampler_errors": gpu_peak["sampler_errors"],
        },
    }
    destination = (
        CHECKPOINTS / "run1" / "benchmark"
        / required("RUN1_SUBMISSION_ZIP_SHA256") / "REPORT.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise RuntimeError("submission exceeded the internal 50-minute runtime gate")


if __name__ == "__main__":
    main()
