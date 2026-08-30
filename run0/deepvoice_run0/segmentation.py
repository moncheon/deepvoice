"""Length-bounded, content-aware segmentation for file-independent inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmentBatch:
    samples: np.ndarray  # [segments, 2, window]
    valid_mask: np.ndarray  # [segments, window]
    starts: np.ndarray  # sample indices


def _window_starts(length: int, window: int, hop: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, hop))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def _novelty_scores(mid: np.ndarray, starts: list[int], window: int) -> np.ndarray:
    scores: list[float] = []
    previous_spectrum: np.ndarray | None = None
    for start in starts:
        frame = mid[start : start + window]
        if frame.size < window:
            frame = np.pad(frame, (0, window - frame.size))
        # A small fixed analysis FFT keeps selection cheap for one-minute files.
        stride = max(1, frame.size // 2048)
        reduced = frame[::stride][:2048]
        spectrum = np.abs(np.fft.rfft(reduced * np.hanning(reduced.size)))
        spectrum /= float(spectrum.sum() + 1e-8)
        energy = float(np.log(np.mean(frame.astype(np.float64) ** 2) + 1e-10))
        flux = 0.0 if previous_spectrum is None else float(np.maximum(spectrum - previous_spectrum, 0).sum())
        scores.append(energy + 2.0 * flux)
        previous_spectrum = spectrum
    values = np.asarray(scores, dtype=np.float64)
    if values.size > 1:
        values = (values - values.mean()) / (values.std() + 1e-8)
    return values


def select_segment_starts(
    mid: np.ndarray,
    sample_rate: int = 16_000,
    window_seconds: float = 4.0,
    hop_seconds: float = 2.0,
    all_windows_until_seconds: float = 16.0,
    uniform_segments: int = 12,
    novelty_segments: int = 4,
    max_segments: int = 16,
) -> list[int]:
    """Use all short-file windows, otherwise uniform plus novelty windows."""

    mono = np.asarray(mid, dtype=np.float32).reshape(-1)
    window = int(round(window_seconds * sample_rate))
    hop = int(round(hop_seconds * sample_rate))
    candidates = _window_starts(mono.size, window, hop)
    if mono.size <= int(all_windows_until_seconds * sample_rate):
        return candidates[:max_segments]

    uniform_count = min(uniform_segments, len(candidates), max_segments)
    uniform_indices = np.linspace(0, len(candidates) - 1, uniform_count, dtype=int)
    selected = {candidates[int(index)] for index in uniform_indices}
    remaining_budget = min(novelty_segments, max_segments - len(selected))
    if remaining_budget > 0:
        scores = _novelty_scores(mono, candidates, window)
        for index in np.argsort(scores)[::-1]:
            selected.add(candidates[int(index)])
            if len(selected) >= uniform_count + remaining_budget:
                break
    return sorted(selected)[:max_segments]


def _extract_with_padding(audio: np.ndarray, start: int, window: int) -> tuple[np.ndarray, np.ndarray]:
    piece = audio[:, start : start + window]
    valid = piece.shape[1]
    mask = np.zeros(window, dtype=np.bool_)
    mask[:valid] = True
    if valid == window:
        return piece.astype(np.float32, copy=False), mask
    padding = window - valid
    if valid > 1:
        piece = np.pad(piece, ((0, 0), (0, padding)), mode="reflect")
    else:
        piece = np.pad(piece, ((0, 0), (0, padding)), mode="constant")
    return piece[:, :window].astype(np.float32, copy=False), mask


def segment_audio(mid_side: np.ndarray, **settings: object) -> SegmentBatch:
    """Create model windows and explicit valid-sample masks."""

    audio = np.asarray(mid_side, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[0] != 2 or audio.shape[1] == 0:
        raise ValueError("mid_side must have shape [2, nonzero_time]")
    sample_rate = int(settings.get("sample_rate", 16_000))
    window_seconds = float(settings.get("window_seconds", 4.0))
    window = int(round(sample_rate * window_seconds))
    starts = select_segment_starts(audio[0], **settings)
    pieces, masks = zip(*(_extract_with_padding(audio, start, window) for start in starts))
    return SegmentBatch(
        samples=np.stack(pieces),
        valid_mask=np.stack(masks),
        starts=np.asarray(starts, dtype=np.int64),
    )
