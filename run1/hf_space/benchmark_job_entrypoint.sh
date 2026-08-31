#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

workspace=/workspace
wheelhouse=/tmp/run1-benchmark-wheelhouse
review=/tmp/run1-benchmark-wheel-review.json
mkdir -p "$wheelhouse"

# Downloads are stored and reviewed before any wheel is imported or installed.
python -m pip download \
  --only-binary=:all: \
  --no-deps \
  --dest "$wheelhouse" \
  --requirement "$workspace/baseline_requirements.txt"
python "$workspace/review_wheels.py" "$wheelhouse" "$review"
python -m pip install \
  --no-index \
  --no-deps \
  --find-links "$wheelhouse" \
  --requirement "$workspace/baseline_requirements.txt"
python -m pip check

python "$workspace/run_benchmark_job.py"
