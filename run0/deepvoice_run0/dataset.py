"""Manifest-backed lazy audio dataset and variable-segment collation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .audio import decode_audio, to_mid_side
from .manifest import ManifestRecord, coarse_generator_domain
from .segmentation import segment_audio


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for dataset training") from error
    return torch


class ManifestDataset:
    def __init__(
        self,
        records: Iterable[ManifestRecord],
        manifest_dir: str | Path,
        audio_settings: dict[str, object],
    ) -> None:
        self.records = list(records)
        self.manifest_dir = Path(manifest_dir)
        self.audio_settings = dict(audio_settings)
        if not self.records:
            raise ValueError("dataset records are empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = _torch()
        record = self.records[index]
        path = Path(record.path)
        if not path.is_absolute():
            path = self.manifest_dir / path
        decoded = decode_audio(path, int(self.audio_settings["sample_rate"]))
        segments = segment_audio(to_mid_side(decoded.samples), **self.audio_settings)
        labels = {
            "file_fake": float(record.file_fake),
            "voice_fake": float(record.voice_fake or 0),
            "music_fake": float(record.music_fake or 0),
            "voice_present": float(record.voice_present),
            "music_present": float(record.music_present),
        }
        return {
            "sample_id": record.sample_id,
            "audio": torch.from_numpy(segments.samples),
            "valid_mask": torch.from_numpy(segments.valid_mask),
            "labels": {name: torch.tensor(value) for name, value in labels.items()},
            "group": record.generator_family or record.source_dataset,
        }


def collate_records(items: list[dict[str, Any]]) -> dict[str, Any]:
    torch = _torch()
    max_segments = max(item["audio"].shape[0] for item in items)
    _, channels, samples = items[0]["audio"].shape
    audio = torch.zeros((len(items), max_segments, channels, samples), dtype=torch.float32)
    valid = torch.zeros((len(items), max_segments, samples), dtype=torch.bool)
    segment_mask = torch.zeros((len(items), max_segments), dtype=torch.bool)
    for row, item in enumerate(items):
        count = item["audio"].shape[0]
        audio[row, :count] = item["audio"]
        valid[row, :count] = item["valid_mask"]
        segment_mask[row, :count] = True
    labels = {
        name: torch.stack([item["labels"][name] for item in items])
        for name in items[0]["labels"]
    }
    return {
        "sample_id": [item["sample_id"] for item in items],
        "audio": audio,
        "valid_mask": valid,
        "segment_mask": segment_mask,
        "labels": labels,
    }


def inverse_group_weights(records: Iterable[ManifestRecord]) -> np.ndarray:
    records = list(records)
    groups = [record.generator_family or record.source_dataset for record in records]
    counts = Counter(groups)
    return np.asarray([1.0 / counts[group] for group in groups], dtype=np.float64)


def training_stratum(record: ManifestRecord) -> tuple[object, ...]:
    """Balance task labels and broad domains, not individual generator names."""

    return (
        record.voice_present,
        record.music_present,
        record.voice_fake,
        record.music_fake,
        *coarse_generator_domain(record),
    )


def inverse_training_stratum_weights(records: Iterable[ManifestRecord]) -> np.ndarray:
    records = list(records)
    strata = [training_stratum(record) for record in records]
    counts = Counter(strata)
    return np.asarray([1.0 / counts[stratum] for stratum in strata], dtype=np.float64)
