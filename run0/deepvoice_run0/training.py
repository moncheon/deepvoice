"""Bounded training loop with group-balanced sampling and tensor-only output."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import ManifestDataset, collate_records, inverse_group_weights
from .losses import multitask_loss
from .manifest import assign_group_splits, read_manifest
from .model import Run0Detector


def _dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from safetensors.torch import save_file
        from torch.utils.data import DataLoader, WeightedRandomSampler
    except ImportError as error:
        raise RuntimeError("Training requires torch and safetensors") from error
    return torch, save_file, (DataLoader, WeightedRandomSampler)


def train(
    *,
    manifest_path: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
    device: str,
    licenses_reviewed: bool,
    allow_fixture_training: bool = False,
    split_path: str | Path | None = None,
) -> dict[str, object]:
    """Train v1. This function never downloads a backbone or dataset."""

    if not licenses_reviewed:
        raise ValueError("training requires an explicit licensed-data review acknowledgement")
    seed = int(config["training"]["seed"])
    records = read_manifest(manifest_path)
    fixture_records = [
        record.sample_id for record in records
        if "fixture" in record.source_dataset.lower() or "fixture" in record.license_id.lower()
    ]
    if fixture_records and not allow_fixture_training:
        raise ValueError(
            "fixture-only audio cannot be used for a performance model; "
            "use --allow-fixture-smoke-training only for pipeline smoke tests"
        )
    torch, save_file, loader_types = _dependencies()
    DataLoader, WeightedRandomSampler = loader_types
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if split_path is None:
        assignments = assign_group_splits(records, seed=seed)
    else:
        assignments = json.loads(Path(split_path).read_text(encoding="utf-8"))
        expected_ids = {record.sample_id for record in records}
        if set(assignments) != expected_ids:
            raise ValueError("split assignment IDs do not exactly match the manifest")
        allowed_splits = {"train", "dev", "holdout", "generator_holdout"}
        if any(split not in allowed_splits for split in assignments.values()):
            raise ValueError("split assignment contains an unsupported split name")
        group_split: dict[str, str] = {}
        for record in records:
            prior = group_split.setdefault(record.split_group, assignments[record.sample_id])
            if prior != assignments[record.sample_id]:
                raise ValueError(f"split assignment leaks group: {record.split_group}")
    train_records = [record for record in records if assignments[record.sample_id] == "train"]
    if not train_records:
        raise ValueError("group split produced no training records")
    audio_settings = dict(config["audio"])
    training_segment_limit = int(config["training"].get("max_segments_per_file", 8))
    if training_segment_limit <= 0:
        raise ValueError("max_segments_per_file must be positive")
    audio_settings["max_segments"] = min(int(audio_settings["max_segments"]), training_segment_limit)
    audio_settings["novelty_segments"] = min(
        int(audio_settings["novelty_segments"]),
        max(0, int(audio_settings["max_segments"]) - 1),
        max(1, training_segment_limit // 4),
    )
    audio_settings["uniform_segments"] = max(1, int(audio_settings["max_segments"]) - int(audio_settings["novelty_segments"]))
    dataset = ManifestDataset(train_records, Path(manifest_path).parent, audio_settings)
    weights = inverse_group_weights(train_records)
    sampler = WeightedRandomSampler(weights.tolist(), len(weights), replacement=True)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        sampler=sampler,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collate_records,
    )
    model_settings = config["model"]
    model = Run0Detector(
        backbone_path=model_settings["backbone_path"],
        hidden_size=int(model_settings["hidden_size"]),
        music_channels=int(model_settings["music_channels"]),
        layer_fusion_count=int(model_settings["layer_fusion_count"]),
        segment_chunk_size=int(model_settings.get("segment_chunk_size", 4)),
    ).to(device)
    model.freeze_backbone_lower_half()
    if hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    history: list[dict[str, float]] = []
    accumulation_steps = int(config["training"].get("gradient_accumulation_steps", 1))
    if accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    for epoch in range(int(config["training"]["epochs"])):
        model.train()
        totals: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            with torch.autocast(
                device_type=torch.device(device).type,
                dtype=torch.bfloat16,
                enabled=torch.device(device).type == "cuda",
            ):
                outputs = model(
                    batch["audio"].to(device),
                    batch["valid_mask"].to(device),
                    batch["segment_mask"].to(device),
                )
                targets = {name: value.to(device) for name, value in batch["labels"].items()}
                losses = multitask_loss(
                    outputs,
                    targets,
                    rank_weight=float(config["training"]["rank_loss_weight"]),
                    consistency_weight=float(config["training"]["consistency_loss_weight"]),
                    domain_weight=float(config["training"]["domain_loss_weight"]),
                )
            (losses["total"] / accumulation_steps).backward()
            should_step = (step + 1) % accumulation_steps == 0 or step + 1 == len(loader)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            totals.append(float(losses["total"].detach().cpu()))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(totals))})

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in model.state_dict().items()
    }
    save_file(state, str(output / "run0.safetensors"))
    metadata = {
        "format_version": 1,
        "architecture": "Run0Detector",
        "model": model_settings,
        "manifest_sha256_note": "Record the source manifest hash in the experiment ledger.",
        "fixture_smoke_training": bool(fixture_records),
        "split_assignment": str(split_path) if split_path is not None else "deterministic-group-hash",
        "history": history,
    }
    (output / "model_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata
