"""Offline, file-independent inference using only tensor-safe local artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .audio import decode_audio, to_mid_side
from .config import load_config
from .fusion import LogisticFuser, noisy_or
from .model import load_detector
from .segmentation import segment_audio

# Competition inference must not reach a model hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class Predictor:
    def __init__(
        self,
        model_dir: str | Path,
        config_path: str | Path,
        device: str = "cuda",
    ) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required for inference") from error
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.torch = torch
        self.device = torch.device(device)
        self.config = load_config(config_path)
        self.model_dir = Path(model_dir)
        model_settings = dict(self.config["model"])
        metadata_path = self.model_dir / "model_config.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            model_settings.update(metadata.get("model") or {})
        self.model = load_detector(self.model_dir, device=device, config=model_settings)
        self.fuser: LogisticFuser | None = None
        self.fuser_features: list[str] = []
        fuser_path = self.model_dir / "fuser.json"
        if fuser_path.is_file():
            payload = json.loads(fuser_path.read_text(encoding="utf-8"))
            self.fuser = LogisticFuser.from_dict(payload["model"])
            self.fuser_features = list(payload["features"])

    def predict_file(self, path: str | Path) -> list[float]:
        decoded = decode_audio(path, int(self.config["audio"]["sample_rate"]))
        batch = segment_audio(to_mid_side(decoded.samples), **self.config["audio"])
        audio = self.torch.from_numpy(batch.samples)[None].to(self.device)
        valid = self.torch.from_numpy(batch.valid_mask)[None].to(self.device)
        segment_mask = self.torch.ones((1, audio.shape[1]), dtype=self.torch.bool, device=self.device)
        with self.torch.inference_mode():
            with self.torch.autocast(
                device_type=self.device.type,
                dtype=self.torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                outputs = self.model(audio, valid, segment_mask)
                probabilities = self.torch.sigmoid(outputs["logits"])[0].float().cpu().numpy()
                segment_prob = self.torch.sigmoid(outputs["segment_logits"])[0, :, 0]
                uncertainty = float((segment_prob * (1.0 - segment_prob)).mean().float().cpu())
        if self.fuser is not None:
            file_model, voice_fake, music_fake, voice_present, music_present = probabilities
            values = {
                "file_model": float(file_model),
                "voice_fake": float(voice_fake),
                "music_fake": float(music_fake),
                "voice_present": float(voice_present),
                "music_present": float(music_present),
                "noisy_or": float(noisy_or(voice_present, voice_fake, music_present, music_fake)),
                "segment_uncertainty": uncertainty,
                "duration_sec": float(decoded.duration_sec),
            }
            missing = [name for name in self.fuser_features if name not in values]
            if missing:
                raise ValueError(f"unsupported fuser feature(s): {missing}")
            matrix = np.asarray([[values[name] for name in self.fuser_features]])
            probabilities[0] = self.fuser.predict_proba(matrix)[0]
        return [float(value) for value in probabilities]
