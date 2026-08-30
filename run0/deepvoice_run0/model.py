"""Compact two-view detector: local SSL voice encoder plus spectral music CNN."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _torch_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError("PyTorch is required for model training and inference") from error
    return torch, nn, functional


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # Lets ledger/metric utilities work in CPU-light environments.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


if nn is not None:

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            groups = max(1, min(8, channels // 8))
            self.net = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.GroupNorm(groups, channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.GroupNorm(groups, channels),
            )

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return F.silu(value + self.net(value))


    class MusicResidualEncoder(nn.Module):
        """Multi-resolution magnitude/residual/phase-coherence encoder."""

        def __init__(self, output_size: int = 256, channels: int = 48) -> None:
            super().__init__()
            self.fft_sizes = (512, 1024, 2048)
            input_channels = len(self.fft_sizes) * 4
            self.network = nn.Sequential(
                nn.Conv2d(input_channels, channels, 5, stride=2, padding=2, bias=False),
                nn.GroupNorm(6 if channels % 6 == 0 else 1, channels),
                nn.SiLU(),
                ResidualBlock(channels),
                nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8 if (channels * 2) % 8 == 0 else 1, channels * 2),
                nn.SiLU(),
                ResidualBlock(channels * 2),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.projection = nn.Linear(channels * 2, output_size)

        def _features(self, audio: torch.Tensor, n_fft: int) -> torch.Tensor:
            window = torch.hann_window(n_fft, dtype=audio.dtype, device=audio.device)
            hop = n_fft // 4
            mid = torch.stft(
                audio[:, 0], n_fft=n_fft, hop_length=hop, window=window,
                return_complex=True, center=True,
            )
            side = torch.stft(
                audio[:, 1], n_fft=n_fft, hop_length=hop, window=window,
                return_complex=True, center=True,
            )
            mid_magnitude = torch.log1p(mid.abs())
            side_magnitude = torch.log1p(side.abs())
            residual = F.pad(torch.diff(mid_magnitude, dim=1).abs(), (0, 0, 1, 0))
            coherence = (mid * side.conj()).real / (mid.abs() * side.abs() + 1e-6)
            features = torch.stack((mid_magnitude, side_magnitude, residual, coherence), dim=1)
            return F.interpolate(features, size=(96, 128), mode="bilinear", align_corners=False)

        def forward(self, audio: torch.Tensor) -> torch.Tensor:
            features = torch.cat([self._features(audio, n_fft) for n_fft in self.fft_sizes], dim=1)
            encoded = self.network(features).flatten(1)
            return self.projection(encoded)


    class Run0Detector(nn.Module):
        """File-independent five-head detector with gated multi-instance pooling."""

        def __init__(
            self,
            backbone_path: str | Path,
            hidden_size: int = 256,
            music_channels: int = 48,
            layer_fusion_count: int = 8,
            segment_chunk_size: int = 4,
            num_domains: int = 0,
        ) -> None:
            super().__init__()
            try:
                from transformers import AutoModel
            except ImportError as error:
                raise RuntimeError("transformers is required to construct Run0Detector") from error
            path = Path(backbone_path)
            if not path.is_dir():
                raise FileNotFoundError(
                    f"Local SSL backbone directory is missing: {path}. Network loading is disabled."
                )
            self.backbone = AutoModel.from_pretrained(
                str(path), local_files_only=True, output_hidden_states=True
            )
            backbone_hidden = int(self.backbone.config.hidden_size)
            available_layers = int(getattr(self.backbone.config, "num_hidden_layers", 12)) + 1
            self.layer_fusion_count = min(int(layer_fusion_count), available_layers)
            self.segment_chunk_size = max(1, int(segment_chunk_size))
            self.layer_logits = nn.Parameter(torch.zeros(self.layer_fusion_count))
            self.voice_projection = nn.Sequential(
                nn.Linear(backbone_hidden, hidden_size), nn.LayerNorm(hidden_size), nn.SiLU()
            )
            self.music_encoder = MusicResidualEncoder(hidden_size, music_channels)
            joint_size = hidden_size * 2
            self.segment_projection = nn.Sequential(
                nn.Linear(joint_size, joint_size), nn.LayerNorm(joint_size), nn.SiLU()
            )
            self.segment_heads = nn.Linear(joint_size, 5)
            self.attention = nn.Sequential(
                nn.Linear(joint_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
            )
            self.file_heads = nn.Sequential(
                nn.Linear(joint_size * 4, joint_size),
                nn.LayerNorm(joint_size),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(joint_size, 5),
            )
            self.domain_head = nn.Linear(joint_size, num_domains) if num_domains > 1 else None

        def freeze_backbone_lower_half(self) -> None:
            parameters = list(self.backbone.parameters())
            boundary = len(parameters) // 2
            for parameter in parameters[:boundary]:
                parameter.requires_grad_(False)

        def _voice_encode(self, mid: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
            output = self.backbone(input_values=mid, attention_mask=valid_mask.long())
            hidden_states = output.hidden_states[-self.layer_fusion_count :]
            weights = torch.softmax(self.layer_logits, dim=0)
            fused = sum(weight * state for weight, state in zip(weights, hidden_states))
            # The SSL encoder already consumes the valid input mask; output lengths vary by model.
            return self.voice_projection(fused.mean(dim=1))

        def _encode_segments(self, audio: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
            embeddings: list[torch.Tensor] = []
            for start in range(0, audio.shape[0], self.segment_chunk_size):
                end = start + self.segment_chunk_size
                piece = audio[start:end]
                piece_mask = valid_mask[start:end]
                voice = self._voice_encode(piece[:, 0], piece_mask)
                music_input = piece * piece_mask[:, None].to(piece.dtype)
                music = self.music_encoder(music_input.float())
                embeddings.append(self.segment_projection(torch.cat((voice, music), dim=-1)))
            return torch.cat(embeddings, dim=0)

        def forward(
            self,
            audio: torch.Tensor,
            valid_mask: torch.Tensor,
            segment_mask: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            if audio.ndim != 4 or audio.shape[2] != 2:
                raise ValueError("audio must have shape [batch, segments, 2, samples]")
            batch, segments, _, samples = audio.shape
            flat_audio = audio.reshape(batch * segments, 2, samples)
            flat_valid = valid_mask.reshape(batch * segments, samples)
            embedding = self._encode_segments(flat_audio, flat_valid)
            embedding = embedding.reshape(batch, segments, -1)
            segment_logits = self.segment_heads(embedding)

            if segment_mask is None:
                segment_mask = torch.ones((batch, segments), dtype=torch.bool, device=audio.device)
            attention_logits = self.attention(embedding).squeeze(-1)
            attention_logits = attention_logits.masked_fill(~segment_mask, -10_000.0)
            attention = torch.softmax(attention_logits, dim=1)
            pooled = (embedding * attention.unsqueeze(-1)).sum(dim=1)
            mask = segment_mask.unsqueeze(-1).to(embedding.dtype)
            count = mask.sum(dim=1).clamp_min(1.0)
            mean = (embedding * mask).sum(dim=1) / count
            variance = ((embedding - mean[:, None]) ** 2 * mask).sum(dim=1) / count
            first_index = torch.zeros(batch, dtype=torch.long, device=audio.device)
            last_index = segment_mask.long().sum(dim=1).sub(1).clamp_min(0)
            first = embedding[torch.arange(batch, device=audio.device), first_index]
            last = embedding[torch.arange(batch, device=audio.device), last_index]
            change = last - first
            file_embedding = torch.cat((pooled, mean, torch.sqrt(variance + 1e-6), change), dim=-1)
            result = {
                "logits": self.file_heads(file_embedding),
                "segment_logits": segment_logits,
                "attention": attention,
                "segment_embedding": embedding,
            }
            if self.domain_head is not None:
                result["domain_logits"] = self.domain_head(mean)
            return result


else:

    class Run0Detector:  # type: ignore[no-redef]
        def __init__(self, *_: object, **__: object) -> None:
            _torch_dependencies()


def load_detector(
    model_dir: str | Path,
    *,
    device: str = "cpu",
    config: dict[str, Any] | None = None,
) -> Any:
    """Load architecture and tensor-only weights from a local model directory."""

    torch_module, _, _ = _torch_dependencies()
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError("safetensors is required to load model weights") from error
    directory = Path(model_dir)
    settings = config or {}
    detector = Run0Detector(
        backbone_path=directory / "backbone",
        hidden_size=int(settings.get("hidden_size", 256)),
        music_channels=int(settings.get("music_channels", 48)),
        layer_fusion_count=int(settings.get("layer_fusion_count", 8)),
        segment_chunk_size=int(settings.get("segment_chunk_size", 4)),
        num_domains=int(settings.get("num_domains", 0)),
    )
    weights_path = directory / "run0.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights are missing: {weights_path}")
    missing, unexpected = detector.load_state_dict(load_file(str(weights_path), device="cpu"), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"model state mismatch; missing={missing}, unexpected={unexpected}")
    return detector.to(torch_module.device(device)).eval()
