#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
workspace=/workspace
wheelhouse=/tmp/run0-benchmark-wheelhouse
review=/tmp/run0-benchmark-wheel-review.json
mkdir -p "$wheelhouse"
python -m pip download \
  --only-binary=:all: \
  --dest "$wheelhouse" \
  --requirement "$workspace/train_requirements.txt"
python "$workspace/review_wheels.py" "$wheelhouse" "$review"
python -m pip install \
  --no-index \
  --find-links "$wheelhouse" \
  --requirement "$workspace/train_requirements.txt"
python "$workspace/run_benchmark_job.py"
