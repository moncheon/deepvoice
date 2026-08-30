# Run 0 model artifact contract

This directory is intentionally incomplete until a license-reviewed training run succeeds.

Required for offline inference:

- `backbone/`: a compact Hugging Face-compatible SSL audio backbone, reviewed and copied locally
- `run0.safetensors`: tensor-only `Run0Detector` weights
- `model_config.json`: architecture settings and experiment metadata
- optional `fuser.json`: OOF-only logistic fusion parameters and feature names

Do not copy the baseline's pickle-backed multi-gigabyte checkpoint into this directory. External artifacts must be saved, then reviewed for source, license, revision, and SHA-256 before any loader executes them.
