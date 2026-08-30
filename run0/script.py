#!/usr/bin/env python3
"""Generate the five-column DACON submission with the local Run 0 model."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.dont_write_bytecode = True
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from deepvoice_run0.constants import PREDICTION_COLUMNS
from deepvoice_run0.inference import Predictor
from deepvoice_run0.submission import (
    aligned_audio_files,
    read_submission_template,
    save_submission,
    validate_prediction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"))
    parser.add_argument(
        "--sample-submission", type=Path, default=Path("data/sample_submission.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("output/submission.csv"))
    parser.add_argument("--model-dir", type=Path, default=BASE_DIR / "model")
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config.toml")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--fail-soft",
        action="store_true",
        help="Log a failed file and emit neutral 0.5 probabilities instead of aborting.",
    )
    parser.add_argument("--error-log", type=Path, default=Path("output/errors.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fieldnames, template_rows = read_submission_template(args.sample_submission)
    audio_files = aligned_audio_files(args.test_dir, template_rows)
    predictor = Predictor(args.model_dir, args.config, args.device)
    errors: list[dict[str, str]] = []
    output_rows: list[dict[str, object]] = []
    for template, audio_path in zip(template_rows, audio_files):
        try:
            values = validate_prediction(predictor.predict_file(audio_path))
        except Exception as error:
            if not args.fail_soft:
                raise RuntimeError(f"inference failed for {template['ID']}: {error}") from error
            values = [0.5] * len(PREDICTION_COLUMNS)
            errors.append({"ID": template["ID"], "error": f"{type(error).__name__}: {error}"})
        row: dict[str, object] = {"ID": template["ID"]}
        row.update({name: round(value, 10) for name, value in zip(PREDICTION_COLUMNS, values)})
        output_rows.append(row)
    save_submission(args.output, fieldnames, output_rows)
    if errors:
        args.error_log.parent.mkdir(parents=True, exist_ok=True)
        with args.error_log.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "error"])
            writer.writeheader()
            writer.writerows(errors)
    print(f"Saved {len(output_rows)} predictions to {args.output}; errors={len(errors)}")


if __name__ == "__main__":
    main()
