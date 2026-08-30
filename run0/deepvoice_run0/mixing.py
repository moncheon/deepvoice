"""Label-safe voice/music mixture construction for controlled Red data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Component:
    samples: np.ndarray  # [channels, time]
    kind: str  # voice | music
    fake: int
    source_id: str


@dataclass(frozen=True)
class Mixture:
    samples: np.ndarray
    voice_present: int
    music_present: int
    voice_fake: int | None
    music_fake: int | None
    file_fake: int
    mix_mode: str


def _stereo(samples: np.ndarray) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[None, :]
    if audio.ndim != 2 or audio.shape[0] not in (1, 2) or audio.shape[1] == 0:
        raise ValueError("component samples must be mono/stereo and non-empty")
    return np.repeat(audio, 2, axis=0) if audio.shape[0] == 1 else audio


def _pan(audio: np.ndarray, pan: float) -> np.ndarray:
    pan = float(np.clip(pan, -1.0, 1.0))
    angle = (pan + 1.0) * np.pi / 4.0
    gains = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)[:, None]
    return audio * gains


def mix_components(
    voice: Component | None,
    music: Component | None,
    mode: str = "simultaneous",
    voice_gain_db: float = 0.0,
    music_gain_db: float = 0.0,
    voice_pan: float = 0.0,
    music_pan: float = 0.0,
    gap_samples: int = 0,
) -> Mixture:
    """Mix components and derive labels instead of accepting caller labels."""

    if voice is None and music is None:
        raise ValueError("at least one component is required")
    if voice is not None and voice.kind != "voice":
        raise ValueError("voice component kind mismatch")
    if music is not None and music.kind != "music":
        raise ValueError("music component kind mismatch")
    if mode not in {"none", "simultaneous", "sequential"}:
        raise ValueError(f"unsupported mix mode: {mode}")
    if (voice is None or music is None) and mode != "none":
        mode = "none"

    prepared: list[np.ndarray] = []
    for component, gain_db, pan in (
        (voice, voice_gain_db, voice_pan),
        (music, music_gain_db, music_pan),
    ):
        if component is not None:
            gain = 10.0 ** (float(gain_db) / 20.0)
            prepared.append(_pan(_stereo(component.samples), pan) * gain)

    if len(prepared) == 1:
        output = prepared[0]
    elif mode == "sequential":
        gap = np.zeros((2, max(0, int(gap_samples))), dtype=np.float32)
        output = np.concatenate((prepared[0], gap, prepared[1]), axis=1)
    else:
        length = max(item.shape[1] for item in prepared)
        output = np.zeros((2, length), dtype=np.float32)
        for item in prepared:
            output[:, : item.shape[1]] += item

    peak = float(np.max(np.abs(output)))
    if peak > 0.999:
        output = output * (0.999 / peak)
    voice_fake = None if voice is None else int(voice.fake)
    music_fake = None if music is None else int(music.fake)
    return Mixture(
        samples=output.astype(np.float32, copy=False),
        voice_present=int(voice is not None),
        music_present=int(music is not None),
        voice_fake=voice_fake,
        music_fake=music_fake,
        file_fake=int(bool(voice_fake) or bool(music_fake)),
        mix_mode=mode,
    )
