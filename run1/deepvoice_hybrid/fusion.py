"""Per-file independent fixed-ECDF rank fusion for the three fake heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


FAKE_HEADS = ("file_fake", "voice_fake", "music_fake")


@dataclass(frozen=True)
class EmpiricalCDF:
    """A fixed monotone score-to-midrank mapping learned without test data."""

    thresholds: tuple[float, ...]
    quantiles: tuple[float, ...]

    @classmethod
    def fit(cls, scores: Iterable[float]) -> "EmpiricalCDF":
        values = np.asarray(list(scores), dtype=np.float64)
        if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
            raise ValueError("ECDF fitting requires at least two finite scores")
        values.sort(kind="mergesort")
        thresholds, starts, counts = np.unique(
            values, return_index=True, return_counts=True
        )
        midranks = starts.astype(np.float64) + (counts.astype(np.float64) + 1.0) / 2.0
        quantiles = (midranks - 0.5) / float(values.size)
        return cls(
            thresholds=tuple(float(value) for value in thresholds),
            quantiles=tuple(float(value) for value in quantiles),
        )

    def __post_init__(self) -> None:
        if len(self.thresholds) != len(self.quantiles) or not self.thresholds:
            raise ValueError("ECDF thresholds and quantiles must be non-empty and aligned")
        x = np.asarray(self.thresholds, dtype=np.float64)
        q = np.asarray(self.quantiles, dtype=np.float64)
        if not np.isfinite(x).all() or not np.isfinite(q).all():
            raise ValueError("ECDF values must be finite")
        if np.any(np.diff(x) <= 0) or np.any(np.diff(q) < 0):
            raise ValueError("ECDF thresholds must increase and quantiles must not decrease")
        if np.any(q <= 0) or np.any(q >= 1):
            raise ValueError("ECDF quantiles must lie strictly inside (0, 1)")

    def transform(self, score: float) -> float:
        if not np.isfinite(score):
            raise ValueError("score must be finite")
        return float(
            np.interp(
                float(score),
                np.asarray(self.thresholds),
                np.asarray(self.quantiles),
                left=self.quantiles[0],
                right=self.quantiles[-1],
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "thresholds": list(self.thresholds),
            "quantiles": list(self.quantiles),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "EmpiricalCDF":
        return cls(
            thresholds=tuple(float(value) for value in payload["thresholds"]),  # type: ignore[index]
            quantiles=tuple(float(value) for value in payload["quantiles"]),  # type: ignore[index]
        )


@dataclass(frozen=True)
class HeadFusion:
    baseline_weight: float
    baseline_cdf: EmpiricalCDF
    run0_cdf: EmpiricalCDF

    def __post_init__(self) -> None:
        if not 0.0 <= self.baseline_weight <= 1.0:
            raise ValueError("baseline_weight must be in [0, 1]")

    def fuse(self, baseline_score: float, run0_score: float) -> float:
        baseline_rank = self.baseline_cdf.transform(baseline_score)
        run0_rank = self.run0_cdf.transform(run0_score)
        value = self.baseline_weight * baseline_rank + (1.0 - self.baseline_weight) * run0_rank
        return float(np.clip(value, 1e-7, 1.0 - 1e-7))


@dataclass(frozen=True)
class FusionSpec:
    heads: Mapping[str, HeadFusion]
    provenance: Mapping[str, object]
    format_version: int = 1
    method: str = "fixed_oof_ecdf_rank_blend"
    file_independent: bool = True

    def __post_init__(self) -> None:
        if self.format_version != 1 or self.method != "fixed_oof_ecdf_rank_blend":
            raise ValueError("unsupported fusion spec")
        if not self.file_independent:
            raise ValueError("competition fusion must remain file-independent")
        if set(self.heads) != set(FAKE_HEADS):
            raise ValueError(f"fusion spec must define exactly {FAKE_HEADS}")

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "method": self.method,
            "file_independent": self.file_independent,
            "heads": {
                name: {
                    "baseline_weight": head.baseline_weight,
                    "baseline_cdf": head.baseline_cdf.to_dict(),
                    "run0_cdf": head.run0_cdf.to_dict(),
                }
                for name, head in self.heads.items()
            },
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FusionSpec":
        raw_heads = payload["heads"]  # type: ignore[index]
        if not isinstance(raw_heads, Mapping):
            raise ValueError("fusion heads must be an object")
        heads: dict[str, HeadFusion] = {}
        for name, raw in raw_heads.items():
            if not isinstance(raw, Mapping):
                raise ValueError(f"invalid fusion head: {name}")
            heads[str(name)] = HeadFusion(
                baseline_weight=float(raw["baseline_weight"]),
                baseline_cdf=EmpiricalCDF.from_dict(raw["baseline_cdf"]),  # type: ignore[arg-type]
                run0_cdf=EmpiricalCDF.from_dict(raw["run0_cdf"]),  # type: ignore[arg-type]
            )
        provenance = payload.get("provenance", {})
        if not isinstance(provenance, Mapping):
            raise ValueError("fusion provenance must be an object")
        return cls(
            heads=heads,
            provenance=provenance,
            format_version=int(payload.get("format_version", 0)),
            method=str(payload.get("method", "")),
            file_independent=bool(payload.get("file_independent", False)),
        )


def fuse_fake_scores(
    spec: FusionSpec,
    baseline_scores: Mapping[str, float],
    run0_scores: Mapping[str, float],
) -> dict[str, float]:
    return {
        name: spec.heads[name].fuse(baseline_scores[name], run0_scores[name])
        for name in FAKE_HEADS
    }
