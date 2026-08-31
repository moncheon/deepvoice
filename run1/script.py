#!/usr/bin/env python3
"""Generate a DACON submission with the file-independent Run1 hybrid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
for root in (MODEL_DIR, MODEL_DIR / "run0"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from deepvoice_hybrid.inference import HybridPredictor
from deepvoice_run0.constants import PREDICTION_COLUMNS
from deepvoice_run0.submission import (
    aligned_audio_files,
    read_submission_template,
    save_submission,
    validate_prediction,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"))
    parser.add_argument(
        "--sample-submission", type=Path, default=Path("data/sample_submission.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("output/submission.csv"))
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--cache-dir", type=Path, default=Path("output/.hybrid-cache"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    fieldnames, template_rows = read_submission_template(args.sample_submission)
    audio_files = aligned_audio_files(args.test_dir, template_rows)
    predictor = HybridPredictor(
        args.model_dir,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    predictions = predictor.predict_all(audio_files)
    rows = []
    for template, audio_path in zip(template_rows, audio_files):
        values = validate_prediction(predictions[audio_path])
        row: dict[str, object] = {"ID": template["ID"]}
        row.update({name: round(value, 10) for name, value in zip(PREDICTION_COLUMNS, values)})
        rows.append(row)
    save_submission(args.output, fieldnames, rows)
    print(f"Saved {len(rows)} independent Run1 predictions to {args.output}")


if __name__ == "__main__":
    main()
