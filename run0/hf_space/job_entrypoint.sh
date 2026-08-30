#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1

workspace=/workspace
wheelhouse=/tmp/run0-wheelhouse
model_root=/tmp/run0-models
output_root=/tmp/run0-output
mkdir -p "$wheelhouse" "$output_root"

python -m pip download \
  --only-binary=:all: \
  --no-deps \
  --dest "$wheelhouse" \
  --requirement "$workspace/job_requirements.txt"
python "$workspace/review_wheels.py" \
  "$wheelhouse" "$output_root/WHEEL_REVIEW.json"
python -m pip install \
  --no-index \
  --no-deps \
  --find-links "$wheelhouse" \
  --requirement "$workspace/job_requirements.txt"

python "$workspace/generate_open_audio.py" stage-models --root "$model_root"
python "$workspace/generate_open_audio.py" review-models --root "$model_root"

# A three-file CUDA/model/API smoke must pass before the bounded full batch.
python "$workspace/generate_open_audio.py" generate \
  --root "$model_root" \
  --output "$output_root/generated-smoke" \
  --mms-count 1 --musicgen-count 1 --audioldm-count 1

python "$workspace/generate_open_audio.py" generate \
  --root "$model_root" \
  --output "$output_root/generated-v1" \
  --mms-count "${RUN0_MMS_COUNT:-1000}" \
  --musicgen-count "${RUN0_MUSICGEN_COUNT:-256}" \
  --audioldm-count "${RUN0_AUDIOLDM_COUNT:-64}"

cp "$model_root/REVIEW.json" "$output_root/generated-v1/MODEL_REVIEW.json"
cp "$output_root/WHEEL_REVIEW.json" "$output_root/generated-v1/WHEEL_REVIEW.json"
python "$workspace/generate_open_audio.py" upload \
  --output "$output_root/generated-v1" \
  --repo-id "$RUN0_GENERATED_REPO"
