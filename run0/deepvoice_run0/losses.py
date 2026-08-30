"""Official-weighted masked losses and generalization regularizers."""

from __future__ import annotations

from typing import Any

from .constants import OFFICIAL_WEIGHTS


def _torch() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError("PyTorch is required to compute training losses") from error
    return torch, functional


def pairwise_rank_loss(logits: Any, labels: Any, margin: float = 0.2) -> Any:
    torch, functional = _torch()
    positive = logits[labels > 0.5]
    negative = logits[labels < 0.5]
    if positive.numel() == 0 or negative.numel() == 0:
        return logits.sum() * 0.0
    differences = positive[:, None] - negative[None, :]
    return functional.softplus(margin - differences).mean()


def multitask_loss(
    outputs: dict[str, Any],
    targets: dict[str, Any],
    *,
    rank_weight: float = 0.25,
    consistency_weight: float = 0.10,
    domain_weight: float = 0.05,
    paired_logits: Any | None = None,
    domain_targets: Any | None = None,
) -> dict[str, Any]:
    """Apply component losses only where the corresponding component exists."""

    torch, functional = _torch()
    logits = outputs["logits"]
    order = ("file_fake", "voice_fake", "music_fake", "voice_present", "music_present")
    index = {name: position for position, name in enumerate(order)}

    def bce(name: str, mask: Any | None = None) -> Any:
        values = functional.binary_cross_entropy_with_logits(
            logits[:, index[name]], targets[name].float(), reduction="none"
        )
        if mask is None:
            return values.mean()
        selected = values[mask]
        return selected.mean() if selected.numel() else values.sum() * 0.0

    voice_mask = targets["voice_present"] > 0.5
    music_mask = targets["music_present"] > 0.5
    components = {
        "file": bce("file_fake"),
        "voice_fake": bce("voice_fake", voice_mask),
        "music_fake": bce("music_fake", music_mask),
        "voice_presence": bce("voice_present"),
        "music_presence": bce("music_present"),
    }
    primary = (
        OFFICIAL_WEIGHTS["file_fake"] * components["file"]
        + OFFICIAL_WEIGHTS["voice_fake"] * components["voice_fake"]
        + OFFICIAL_WEIGHTS["music_fake"] * components["music_fake"]
        + OFFICIAL_WEIGHTS["voice_present"] * components["voice_presence"]
        + OFFICIAL_WEIGHTS["music_present"] * components["music_presence"]
    )
    ranking = pairwise_rank_loss(logits[:, 0], targets["file_fake"].float())
    consistency = logits.sum() * 0.0
    if paired_logits is not None:
        consistency = functional.mse_loss(torch.sigmoid(logits), torch.sigmoid(paired_logits))
    domain = logits.sum() * 0.0
    if domain_targets is not None and "domain_logits" in outputs:
        domain = functional.cross_entropy(outputs["domain_logits"], domain_targets.long())
    total = primary + rank_weight * ranking + consistency_weight * consistency + domain_weight * domain
    return {
        "total": total,
        "primary": primary,
        "rank": ranking,
        "consistency": consistency,
        "domain": domain,
        **components,
    }
