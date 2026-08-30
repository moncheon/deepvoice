"""File-level noisy-OR and a small OOF-only logistic meta-fuser."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def noisy_or(
    voice_present: np.ndarray | float,
    voice_fake: np.ndarray | float,
    music_present: np.ndarray | float,
    music_fake: np.ndarray | float,
) -> np.ndarray:
    vp, vf, mp, mf = np.broadcast_arrays(voice_present, voice_fake, music_present, music_fake)
    for value in (vp, vf, mp, mf):
        if not np.isfinite(value).all() or ((value < 0) | (value > 1)).any():
            raise ValueError("fusion probabilities must be finite and within [0, 1]")
    q_voice = vp * vf
    q_music = mp * mf
    return 1.0 - (1.0 - q_voice) * (1.0 - q_music)


@dataclass
class LogisticFuser:
    weights: np.ndarray | None = None
    bias: float = 0.0
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        oof_predictions: bool,
        learning_rate: float = 0.05,
        steps: int = 1000,
        l2: float = 1e-3,
    ) -> "LogisticFuser":
        """Fit only from out-of-fold predictions to prevent meta leakage."""

        if not oof_predictions:
            raise ValueError("meta-fusion must be fitted on out-of-fold predictions")
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if x.ndim != 2 or y.shape != (x.shape[0],) or x.shape[0] == 0:
            raise ValueError("invalid fuser training shapes")
        if not np.isfinite(x).all() or not np.isin(y, (0, 1)).all():
            raise ValueError("fuser inputs must be finite and labels binary")
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        z = (x - self.mean) / self.scale
        self.weights = np.zeros(x.shape[1], dtype=np.float64)
        self.bias = 0.0
        for _ in range(int(steps)):
            logits = np.clip(z @ self.weights + self.bias, -30.0, 30.0)
            probability = 1.0 / (1.0 + np.exp(-logits))
            residual = probability - y
            self.weights -= learning_rate * ((z.T @ residual) / y.size + l2 * self.weights)
            self.bias -= learning_rate * float(residual.mean())
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None or self.mean is None or self.scale is None:
            raise RuntimeError("fuser has not been fitted")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.weights.size or not np.isfinite(x).all():
            raise ValueError("invalid fuser feature matrix")
        logits = np.clip(((x - self.mean) / self.scale) @ self.weights + self.bias, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def to_dict(self) -> dict[str, object]:
        if self.weights is None or self.mean is None or self.scale is None:
            raise RuntimeError("fuser has not been fitted")
        return {
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "LogisticFuser":
        return cls(
            weights=np.asarray(payload["weights"], dtype=np.float64),
            bias=float(payload["bias"]),
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
        )
