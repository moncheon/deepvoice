#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

workspace=/workspace
wheelhouse=/tmp/run1-train-wheelhouse
review=/tmp/run1-train-wheel-review.json
mkdir -p "$wheelhouse"

# Downloads are stored and reviewed before any wheel is imported or installed.
python -m pip download \
  --only-binary=:all: \
  --dest "$wheelhouse" \
  --requirement "$workspace/train_requirements.txt"
python "$workspace/review_wheels.py" "$wheelhouse" "$review"
python -m pip install \
  --no-index \
  --find-links "$wheelhouse" \
  --requirement "$workspace/train_requirements.txt"

python "$workspace/run_logo_job.py"
