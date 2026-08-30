#!/usr/bin/env python3
"""Fit and save an OOF-only file-level logistic meta-fuser."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

RUN0_DIR = Path(__file__).resolve().parents[1]
if str(RUN0_DIR) not in sys.path:
    sys.path.insert(0, str(RUN0_DIR))

from deepvoice_run0.fusion import LogisticFuser

DEFAULT_FEATURES = [
    "file_model",
    "voice_fake",
    "music_fake",
    "voice_present",
    "music_present",
    "noisy_or",
    "segment_uncertainty",
    "duration_sec",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="file_fake")
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    args = parser.parse_args()
    with args.oof_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("OOF CSV is empty")
    features = np.asarray([[float(row[name]) for name in args.features] for row in rows])
    labels = np.asarray([int(row[args.label]) for row in rows])
    fuser = LogisticFuser().fit(features, labels, oof_predictions=True)
    payload = {"format_version": 1, "features": args.features, "model": fuser.to_dict()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"saved fuser trained on {len(rows)} OOF rows to {args.output}")


if __name__ == "__main__":
    main()
