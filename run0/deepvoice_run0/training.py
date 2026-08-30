"""Bounded GPU training with best-checkpoint and honest holdout evidence."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .constants import PREDICTION_COLUMNS
from .dataset import ManifestDataset, collate_records, inverse_group_weights
from .losses import multitask_loss
from .manifest import ManifestRecord, assign_group_splits, read_manifest
from .metrics import official_metrics
from .model import Run0Detector


def _dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from safetensors.torch import load_file, save_file
        from torch.utils.data import DataLoader, WeightedRandomSampler
    except ImportError as error:
        raise RuntimeError("Training requires torch and safetensors") from error
    return torch, load_file, save_file, (DataLoader, WeightedRandomSampler)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _loader(
    records: list[ManifestRecord],
    *,
    manifest_dir: Path,
    audio_settings: dict[str, object],
    batch_size: int,
    num_workers: int,
    training: bool,
    loader_types: tuple[Any, Any],
) -> Any:
    DataLoader, WeightedRandomSampler = loader_types
    dataset = ManifestDataset(records, manifest_dir, audio_settings)
    options: dict[str, object] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_records,
        "pin_memory": True,
    }
    if num_workers > 0:
        options.update(persistent_workers=True, prefetch_factor=2)
    if training:
        weights = inverse_group_weights(records)
        options["sampler"] = WeightedRandomSampler(
            weights.tolist(), len(weights), replacement=True
        )
    else:
        options["shuffle"] = False
    return DataLoader(**options)


def _loss(
    model: Any,
    batch: dict[str, Any],
    *,
    device: str,
    training: dict[str, object],
) -> tuple[dict[str, Any], dict[str, Any]]:
    outputs = model(
        batch["audio"].to(device, non_blocking=True),
        batch["valid_mask"].to(device, non_blocking=True),
        batch["segment_mask"].to(device, non_blocking=True),
    )
    targets = {
        name: value.to(device, non_blocking=True)
        for name, value in batch["labels"].items()
    }
    losses = multitask_loss(
        outputs,
        targets,
        rank_weight=float(training["rank_loss_weight"]),
        consistency_weight=float(training.get("consistency_loss_weight", 0.0)),
        domain_weight=float(training.get("domain_loss_weight", 0.0)),
    )
    return outputs, losses


def _choose_batch_size(
    model: Any,
    records: list[ManifestRecord],
    *,
    manifest_dir: Path,
    audio_settings: dict[str, object],
    training: dict[str, object],
    device: str,
    candidates: Iterable[int],
    memory_gate_bytes: int,
) -> tuple[int, int]:
    """Select the largest candidate that survives a worst-duration backward pass."""

    torch, _, _, _ = _dependencies()
    if torch.device(device).type != "cuda":
        batch_size = min(int(value) for value in candidates)
        return batch_size, 0
    ordered = sorted(records, key=lambda record: record.duration_sec, reverse=True)
    last_error_message: str | None = None
    for candidate in candidates:
        candidate = int(candidate)
        if candidate <= 0:
            continue
        print(json.dumps({"event": "auto_batch_start", "batch_size": candidate}), flush=True)
        dataset = ManifestDataset(ordered[:candidate], manifest_dir, audio_settings)
        batch = collate_records([dataset[index] for index in range(len(dataset))])
        outputs = None
        losses = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model.zero_grad(set_to_none=True)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs, losses = _loss(model, batch, device=device, training=training)
            losses["total"].backward()
            peak = int(torch.cuda.max_memory_allocated())
            if peak <= memory_gate_bytes:
                model.zero_grad(set_to_none=True)
                print(json.dumps({
                    "event": "auto_batch_selected",
                    "batch_size": candidate,
                    "peak_vram_bytes": peak,
                }, sort_keys=True), flush=True)
                return candidate, peak
        except torch.cuda.OutOfMemoryError as error:
            # Do not retain the CUDA exception traceback: checkpoint frames can
            # otherwise keep the failed graph alive and poison the next candidate.
            last_error_message = str(error).splitlines()[0]
            print(json.dumps({
                "event": "auto_batch_rejected",
                "batch_size": candidate,
                "reason": last_error_message,
            }, sort_keys=True), flush=True)
        finally:
            model.zero_grad(set_to_none=True)
            del outputs, losses, batch, dataset
            gc.collect()
            torch.cuda.empty_cache()
    detail = f"; last CUDA error: {last_error_message}" if last_error_message else ""
    raise RuntimeError(f"no auto-batch candidate passed the CUDA memory gate{detail}")


def _metrics_from_arrays(labels: dict[str, list[int]], scores: np.ndarray) -> dict[str, float]:
    voice_fake = [
        value if present else None
        for value, present in zip(labels["voice_fake"], labels["voice_present"])
    ]
    music_fake = [
        value if present else None
        for value, present in zip(labels["music_fake"], labels["music_present"])
    ]
    return official_metrics(
        file_fake=labels["file_fake"],
        file_fake_prob=scores[:, 0],
        voice_fake=voice_fake,
        voice_fake_prob=scores[:, 1],
        music_fake=music_fake,
        music_fake_prob=scores[:, 2],
        voice_present=labels["voice_present"],
        voice_present_prob=scores[:, 3],
        music_present=labels["music_present"],
        music_present_prob=scores[:, 4],
    )


def evaluate_model(
    model: Any,
    loader: Any,
    *,
    device: str,
) -> tuple[dict[str, float], np.ndarray, list[str]]:
    """Evaluate a sequential loader and return official metrics and probabilities."""

    torch, _, _, _ = _dependencies()
    model.eval()
    labels = {
        name: []
        for name in ("file_fake", "voice_fake", "music_fake", "voice_present", "music_present")
    }
    probabilities: list[np.ndarray] = []
    sample_ids: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(
                device_type=torch.device(device).type,
                dtype=torch.bfloat16,
                enabled=torch.device(device).type == "cuda",
            ):
                outputs = model(
                    batch["audio"].to(device, non_blocking=True),
                    batch["valid_mask"].to(device, non_blocking=True),
                    batch["segment_mask"].to(device, non_blocking=True),
                )
            probabilities.append(torch.sigmoid(outputs["logits"]).float().cpu().numpy())
            sample_ids.extend(batch["sample_id"])
            for name in labels:
                labels[name].extend(int(value) for value in batch["labels"][name].tolist())
    scores = np.concatenate(probabilities, axis=0)
    metrics = _metrics_from_arrays(labels, scores)
    for index, name in enumerate(PREDICTION_COLUMNS):
        metrics[f"std_{name.lower()}"] = float(scores[:, index].std())
    return metrics, scores, sample_ids


def _write_predictions(
    path: Path,
    records: list[ManifestRecord],
    scores: np.ndarray,
    sample_ids: list[str],
) -> None:
    if len(records) != len(sample_ids) or len(records) != scores.shape[0]:
        raise ValueError("prediction rows do not align with records")
    expected = [record.sample_id for record in records]
    if sample_ids != expected:
        raise ValueError("prediction loader changed record order")
    fieldnames = [
        "sample_id", "file_fake", "voice_fake", "music_fake",
        "voice_present", "music_present", *PREDICTION_COLUMNS,
        "generator_family", "source_dataset", "codec_chain", "channel_profile",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record, values in zip(records, scores):
            row: dict[str, object] = {
                "sample_id": record.sample_id,
                "file_fake": record.file_fake,
                "voice_fake": "" if record.voice_fake is None else record.voice_fake,
                "music_fake": "" if record.music_fake is None else record.music_fake,
                "voice_present": record.voice_present,
                "music_present": record.music_present,
                "generator_family": record.generator_family or "real",
                "source_dataset": record.source_dataset,
                "codec_chain": record.codec_chain,
                "channel_profile": record.channel_profile,
            }
            row.update({name: float(value) for name, value in zip(PREDICTION_COLUMNS, values)})
            writer.writerow(row)


def _state_dict(model: Any) -> dict[str, Any]:
    return {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in model.state_dict().items()
    }


def train(
    *,
    manifest_path: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
    device: str,
    licenses_reviewed: bool,
    allow_fixture_training: bool = False,
    split_path: str | Path | None = None,
    resume_weights: str | Path | None = None,
    wall_clock_seconds: int | None = None,
) -> dict[str, object]:
    """Train Run0 without downloading data or a backbone."""

    if not licenses_reviewed:
        raise ValueError("training requires an explicit licensed-data review acknowledgement")
    manifest = Path(manifest_path)
    seed = int(config["training"]["seed"])
    records = read_manifest(manifest)
    fixture_records = [
        record.sample_id for record in records
        if "fixture" in record.source_dataset.lower() or "fixture" in record.license_id.lower()
    ]
    if fixture_records and not allow_fixture_training:
        raise ValueError(
            "fixture-only audio cannot be used for a performance model; "
            "use --allow-fixture-smoke-training only for pipeline smoke tests"
        )
    torch, load_file, save_file, loader_types = _dependencies()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

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
    by_split = {
        split: [record for record in records if assignments[record.sample_id] == split]
        for split in ("train", "dev", "holdout")
    }
    if any(not by_split[split] for split in by_split):
        raise ValueError("train, dev, and holdout must all contain records")

    audio_settings = dict(config["audio"])
    training = dict(config["training"])
    segment_limit = int(training.get("max_segments_per_file", 6))
    audio_settings["max_segments"] = min(int(audio_settings["max_segments"]), segment_limit)
    audio_settings["novelty_segments"] = min(
        int(audio_settings["novelty_segments"]),
        max(0, int(audio_settings["max_segments"]) - 1),
    )
    audio_settings["uniform_segments"] = max(
        1, int(audio_settings["max_segments"]) - int(audio_settings["novelty_segments"])
    )
    model_settings = dict(config["model"])
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
    if resume_weights is not None:
        missing, unexpected = model.load_state_dict(
            load_file(str(resume_weights), device="cpu"), strict=False
        )
        if missing or unexpected:
            raise RuntimeError(f"resume state mismatch; missing={missing}, unexpected={unexpected}")

    memory_gate = int(training.get("auto_batch_memory_gate_gib", 70)) * 1024**3
    batch_candidates = training.get("auto_batch_candidates", [2])
    batch_size, smoke_peak = _choose_batch_size(
        model,
        by_split["train"],
        manifest_dir=manifest.parent,
        audio_settings=audio_settings,
        training=training,
        device=device,
        candidates=batch_candidates,
        memory_gate_bytes=memory_gate,
    )
    effective_batch = int(training.get("effective_batch_size", 8))
    accumulation_steps = max(1, math.ceil(effective_batch / batch_size))
    num_workers = int(training.get("num_workers", 8))
    train_loader = _loader(
        by_split["train"], manifest_dir=manifest.parent, audio_settings=audio_settings,
        batch_size=batch_size, num_workers=num_workers, training=True,
        loader_types=loader_types,
    )
    evaluation_batch = max(1, min(batch_size, int(training.get("evaluation_batch_size", 4))))
    dev_loader = _loader(
        by_split["dev"], manifest_dir=manifest.parent, audio_settings=audio_settings,
        batch_size=evaluation_batch, num_workers=num_workers, training=False,
        loader_types=loader_types,
    )
    holdout_loader = _loader(
        by_split["holdout"], manifest_dir=manifest.parent, audio_settings=audio_settings,
        batch_size=evaluation_batch, num_workers=num_workers, training=False,
        loader_types=loader_types,
    )

    backbone_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": float(training.get("backbone_learning_rate", 2e-5))},
            {"params": head_parameters, "lr": float(training.get("learning_rate", 1e-4))},
        ],
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(training["epochs"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_updates = max(1, updates_per_epoch * epochs)
    warmup_updates = max(1, int(total_updates * float(training.get("warmup_ratio", 0.05))))

    def schedule(step: int) -> float:
        if step < warmup_updates:
            return float(step + 1) / warmup_updates
        progress = min(1.0, (step - warmup_updates) / max(1, total_updates - warmup_updates))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "run0.safetensors"
    last_path = output / "last.safetensors"
    history: list[dict[str, object]] = []
    best_score = -math.inf
    best_epoch = 0
    patience = 0
    min_epochs = int(training.get("min_epochs", 4))
    patience_limit = int(training.get("early_stopping_patience", 2))
    started = time.monotonic()
    trainable_backbone = list(backbone_parameters)

    for epoch in range(epochs):
        if wall_clock_seconds is not None and epoch >= min_epochs:
            if time.monotonic() - started >= wall_clock_seconds:
                break
        for parameter in trainable_backbone:
            parameter.requires_grad_(epoch > 0)
        model.train()
        totals: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            with torch.autocast(
                device_type=torch.device(device).type,
                dtype=torch.bfloat16,
                enabled=torch.device(device).type == "cuda",
            ):
                _, losses = _loss(model, batch, device=device, training=training)
            (losses["total"] / accumulation_steps).backward()
            should_step = (step + 1) % accumulation_steps == 0 or step + 1 == len(train_loader)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            totals.append(float(losses["total"].detach().cpu()))
        save_file(_state_dict(model), str(last_path))
        dev_metrics, dev_scores, dev_ids = evaluate_model(model, dev_loader, device=device)
        epoch_record: dict[str, object] = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(totals)),
            "dev": dev_metrics,
            "elapsed_seconds": float(time.monotonic() - started),
        }
        history.append(epoch_record)
        print(json.dumps({
            "event": "epoch_complete",
            "epoch": epoch + 1,
            "train_loss": epoch_record["train_loss"],
            "dev_weighted_score": dev_metrics["weighted_score"],
            "elapsed_seconds": epoch_record["elapsed_seconds"],
        }, sort_keys=True), flush=True)
        score = float(dev_metrics["weighted_score"])
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            patience = 0
            save_file(_state_dict(model), str(best_path))
            _write_predictions(output / "dev_predictions.csv", by_split["dev"], dev_scores, dev_ids)
        else:
            patience += 1
        (output / "training_state.json").write_text(
            json.dumps({
                "history": history,
                "best_epoch": best_epoch,
                "best_dev_score": best_score,
                "selected_batch_size": batch_size,
                "gradient_accumulation_steps": accumulation_steps,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if epoch + 1 >= min_epochs and patience >= patience_limit:
            break

    if not best_path.is_file() or len(history) < min_epochs:
        raise RuntimeError(f"training did not produce a best checkpoint after {min_epochs} epochs")
    missing, unexpected = model.load_state_dict(load_file(str(best_path), device="cpu"), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"best state mismatch; missing={missing}, unexpected={unexpected}")
    model.to(device).eval()
    holdout_metrics, holdout_scores, holdout_ids = evaluate_model(
        model, holdout_loader, device=device
    )
    _write_predictions(
        output / "holdout_predictions.csv", by_split["holdout"], holdout_scores, holdout_ids
    )
    best_dev_metrics = dict(history[best_epoch - 1]["dev"])
    portable_model_settings = dict(model_settings)
    portable_model_settings["backbone_path"] = "model/backbone"
    metadata: dict[str, object] = {
        "format_version": 2,
        "architecture": "Run0Detector",
        "model": portable_model_settings,
        "training": {
            "seed": seed,
            "selected_batch_size": batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "auto_batch_peak_vram_bytes": smoke_peak,
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "wall_clock_seconds": float(time.monotonic() - started),
            "fixture_smoke_training": bool(fixture_records),
        },
        "data": {
            "manifest_sha256": _sha256(manifest),
            "splits_sha256": _sha256(Path(split_path)) if split_path is not None else None,
            "split_counts": {name: len(values) for name, values in by_split.items()},
        },
        "best_dev_metrics": best_dev_metrics,
        "holdout_metrics": holdout_metrics,
        "history": history,
    }
    (output / "model_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata
