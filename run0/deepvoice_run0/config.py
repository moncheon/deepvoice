"""TOML configuration loading with explicit, deterministic defaults."""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG: dict[str, Any] = {
    "audio": {
        "sample_rate": 16_000,
        "window_seconds": 4.0,
        "hop_seconds": 2.0,
        "all_windows_until_seconds": 16.0,
        "uniform_segments": 12,
        "novelty_segments": 4,
        "max_segments": 16,
    },
    "model": {
        "backbone_path": "model/backbone",
        "weights_path": "model/run0.safetensors",
        "hidden_size": 256,
        "music_channels": 48,
        "layer_fusion_count": 8,
        "segment_chunk_size": 4,
    },
    "training": {
        "seed": 2608,
        "batch_size": 2,
        "gradient_accumulation_steps": 2,
        "max_segments_per_file": 8,
        "epochs": 10,
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "rank_loss_weight": 0.25,
        "consistency_loss_weight": 0.10,
        "domain_loss_weight": 0.05,
        "num_workers": 0,
    },
    "inference": {
        "batch_size": 1,
        "device": "cuda",
        "strict": True,
    },
}


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a TOML file over defaults and validate safety-critical limits."""

    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        config_path = Path(path)
        with config_path.open("rb") as handle:
            config = _deep_merge(config, tomllib.load(handle))

    audio = config["audio"]
    for key in ("sample_rate", "max_segments", "uniform_segments", "novelty_segments"):
        if int(audio[key]) <= 0:
            raise ValueError(f"audio.{key} must be positive")
    if float(audio["window_seconds"]) <= 0 or float(audio["hop_seconds"]) <= 0:
        raise ValueError("audio window and hop must be positive")
    if int(audio["uniform_segments"]) + int(audio["novelty_segments"]) > int(
        audio["max_segments"]
    ):
        raise ValueError("uniform_segments + novelty_segments exceeds max_segments")
    return config
