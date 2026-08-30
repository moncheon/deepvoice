# Run 0 model artifact contract

This directory is intentionally incomplete until a license-reviewed training run succeeds.

Required for offline inference:

- `backbone/`: reviewed WavLM Base+ safe-tensor snapshot with license and origin ledger
- `run0.safetensors`: tensor-only `Run0Detector` weights
- `model_config.json`: architecture settings and experiment metadata
- optional `fuser.json`: OOF-only logistic fusion parameters and feature names

Do not copy the baseline's pickle-backed multi-gigabyte checkpoint into this directory. The pinned WavLM source may be deserialized only after it matches `backbone_source.json`; the promoted submission backbone must use `model.safetensors` and include `LICENSE-CC-BY-SA-3.0.txt` plus `ORIGIN.json`.
