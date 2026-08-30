#!/usr/bin/env python3
"""Measure file-independent Run 0 throughput and enforce internal L4 gates."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from deepvoice_run0.inference import Predictor
from deepvoice_run0.submission import aligned_audio_files, read_submission_template, validate_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).with_name("model"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.toml"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--target-files", type=int, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enforce-gates", action="store_true")
    args = parser.parse_args()
    _, rows = read_submission_template(args.sample_submission)
    audio_files = aligned_audio_files(args.test_dir, rows)[: args.limit]
    if not audio_files:
        raise ValueError("no files selected for benchmark")
    predictor = Predictor(args.model_dir, args.config, args.device)
    torch = predictor.torch
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timings: list[float] = []
    for path in audio_files:
        started = time.perf_counter()
        validate_prediction(predictor.predict_file(path))
        if args.device == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
    elapsed = float(sum(timings))
    report = {
        "files_measured": len(timings),
        "total_seconds": elapsed,
        "mean_seconds_per_file": statistics.mean(timings),
        "p95_seconds_per_file": sorted(timings)[max(0, int(0.95 * len(timings)) - 1)],
        "estimated_seconds_for_target": elapsed / len(timings) * args.target_files,
        "target_files": args.target_files,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.enforce_gates:
        if report["estimated_seconds_for_target"] > 3000:
            raise SystemExit("runtime gate failed: estimate exceeds 50 minutes")
        if report["peak_vram_bytes"] is not None and report["peak_vram_bytes"] > 20 * 1024**3:
            raise SystemExit("VRAM gate failed: peak exceeds 20 GiB")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
