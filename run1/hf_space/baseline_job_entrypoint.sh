#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

workspace=/workspace
wheelhouse=/tmp/run1-baseline-wheelhouse
review=/tmp/run1-baseline-wheel-review.json
mkdir -p "$wheelhouse"

# Downloads are stored and reviewed before any wheel is imported or installed.
python -m pip download \
  --no-deps \
  --only-binary=:all: \
  --dest "$wheelhouse" \
  --requirement "$workspace/baseline_requirements.txt"
python "$workspace/review_wheels.py" "$wheelhouse" "$review"
python -m pip install \
  --no-deps \
  --no-index \
  --find-links "$wheelhouse" \
  --requirement "$workspace/baseline_requirements.txt"
python -m pip check

if [[ "${RUN1_BASELINE_DEPENDENCY_SMOKE:-false}" == "true" ]]; then
  python "$workspace/baseline_dependency_smoke.py"
  exit 0
fi

python "$workspace/run_baseline_cache_job.py"
