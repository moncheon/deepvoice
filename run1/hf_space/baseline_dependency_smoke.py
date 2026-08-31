#!/usr/bin/env python3
"""Import the reviewed Baseline runtime without loading external weights."""

import json
import shutil
from pathlib import Path

import accelerate
import demucs
import librosa
import torch
import torchaudio
import transformers

# panns-inference otherwise attempts a network download at import time. Supply
# the organizer-packaged label file first, exactly as baseline_script.py does.
label_target = Path.home() / "panns_data" / "class_labels_indices.csv"
label_target.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(
    "/workspace/baseline_model/panns/class_labels_indices.csv", label_target
)
import panns_inference


from deepvoice_hybrid.baseline import load_baseline_module

module = load_baseline_module(
    "/workspace/baseline_script.py", "/workspace/baseline_model"
)
print(json.dumps({
    "status": "passed",
    "torch": torch.__version__,
    "torchaudio": torchaudio.__version__,
    "librosa": librosa.__version__,
    "transformers": transformers.__version__,
    "accelerate": accelerate.__version__,
    "baseline_functions": all(
        hasattr(module, name)
        for name in ("load_panns_model", "load_htdemucs_model", "load_df_arena_model")
    ),
}, sort_keys=True))
