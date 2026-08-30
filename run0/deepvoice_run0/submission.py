"""Strict test/template alignment and submission schema validation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from .constants import PREDICTION_COLUMNS, SUPPORTED_AUDIO_EXTENSIONS


def read_submission_template(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    template = Path(path)
    if not template.is_file():
        raise FileNotFoundError(template)
    with template.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    required = ["ID", *PREDICTION_COLUMNS]
    if fieldnames != required:
        raise ValueError(f"submission columns must be exactly {required}; got {fieldnames}")
    ids = [str(row["ID"]).strip() for row in rows]
    if not ids or any(not sample_id for sample_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("submission template IDs must be non-empty and unique")
    for row, sample_id in zip(rows, ids):
        row["ID"] = sample_id
    return fieldnames, rows


def aligned_audio_files(test_dir: str | Path, rows: Iterable[dict[str, str]]) -> list[Path]:
    directory = Path(test_dir)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    by_id: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
            if path.stem in by_id:
                raise ValueError(f"duplicate audio ID across extensions: {path.stem}")
            by_id[path.stem] = path
    expected = [row["ID"] for row in rows]
    missing = [sample_id for sample_id in expected if sample_id not in by_id]
    extra = sorted(set(by_id) - set(expected))
    if missing or extra:
        raise ValueError(f"audio/template ID mismatch; missing={missing[:5]}, extra={extra[:5]}")
    return [by_id[sample_id] for sample_id in expected]


def validate_prediction(probabilities: Iterable[float]) -> list[float]:
    values = np.asarray(list(probabilities), dtype=np.float64)
    if values.shape != (len(PREDICTION_COLUMNS),):
        raise ValueError(f"expected {len(PREDICTION_COLUMNS)} predictions")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("predictions must be finite probabilities in [0, 1]")
    return [float(value) for value in values]


def save_submission(
    path: str | Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
