"""ADS/CPS metrics matching the competition's normalized components."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _binary_arrays(
    labels: Iterable[int], scores: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(list(labels), dtype=np.int64)
    p = np.asarray(list(scores), dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or y.size != p.size or y.size == 0:
        raise ValueError("labels and scores must be equally-sized non-empty vectors")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(p).all():
        raise ValueError("labels must be binary and scores must be finite")
    if np.unique(y).size != 2:
        raise ValueError("both binary classes are required")
    return y, p


def roc_points(
    labels: Iterable[int], scores: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    y, p = _binary_arrays(labels, scores)
    order = np.argsort(-p, kind="mergesort")
    y = y[order]
    p = p[order]
    ends = np.r_[np.flatnonzero(np.diff(p)), y.size - 1]
    true_positive = np.r_[0, np.cumsum(y)[ends]]
    false_positive = np.r_[0, (ends + 1) - np.cumsum(y)[ends]]
    return false_positive / false_positive[-1], true_positive / true_positive[-1]


def roc_auc(labels: Iterable[int], scores: Iterable[float]) -> float:
    fpr, tpr = roc_points(labels, scores)
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return float(integrate(tpr, fpr))


def equal_error_rate(labels: Iterable[int], scores: Iterable[float]) -> float:
    fpr, tpr = roc_points(labels, scores)
    fnr = 1.0 - tpr
    delta = fpr - fnr
    exact = np.flatnonzero(delta == 0)
    if exact.size:
        return float(fpr[exact[0]])
    crossing = np.flatnonzero(delta[:-1] * delta[1:] < 0)
    if crossing.size:
        index = int(crossing[0])
        fraction = -delta[index] / (delta[index + 1] - delta[index])
        return float(fpr[index] + fraction * (fpr[index + 1] - fpr[index]))
    index = int(np.argmin(np.abs(delta)))
    return float((fpr[index] + fnr[index]) * 0.5)


def ads_from_eers(file_eer: float, voice_fake_eer: float, music_fake_eer: float) -> float:
    return float(
        0.50 * (1.0 - file_eer)
        + 0.20 * (1.0 - voice_fake_eer)
        + 0.30 * (1.0 - music_fake_eer)
    )


def ads_cps_metrics(
    *,
    file_fake: Iterable[int],
    file_fake_prob: Iterable[float],
    voice_present: Iterable[int],
    voice_present_prob: Iterable[float],
    music_present: Iterable[int],
    music_present_prob: Iterable[float],
    voice_fake: Iterable[int | None],
    voice_fake_prob: Iterable[float],
    music_fake: Iterable[int | None],
    music_fake_prob: Iterable[float],
) -> dict[str, float]:
    """Return official components plus normalized ADS, CPS, and total score."""

    vf = np.asarray(list(voice_fake), dtype=object)
    vfp = np.asarray(list(voice_fake_prob), dtype=np.float64)
    mf = np.asarray(list(music_fake), dtype=object)
    mfp = np.asarray(list(music_fake_prob), dtype=np.float64)
    voice_mask = np.asarray([value is not None for value in vf])
    music_mask = np.asarray([value is not None for value in mf])
    metrics = {
        "file_eer": equal_error_rate(file_fake, file_fake_prob),
        "voice_fake_eer": equal_error_rate(vf[voice_mask].astype(int), vfp[voice_mask]),
        "music_fake_eer": equal_error_rate(mf[music_mask].astype(int), mfp[music_mask]),
        "voice_presence_auc": roc_auc(voice_present, voice_present_prob),
        "music_presence_auc": roc_auc(music_present, music_present_prob),
    }
    metrics["ads"] = ads_from_eers(
        metrics["file_eer"], metrics["voice_fake_eer"], metrics["music_fake_eer"]
    )
    metrics["cps"] = 0.5 * (
        metrics["voice_presence_auc"] + metrics["music_presence_auc"]
    )
    metrics["weighted_score"] = 0.9 * metrics["ads"] + 0.1 * metrics["cps"]
    return metrics
