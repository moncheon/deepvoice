"""Deterministic channel transforms used equally on real and fake classes."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

CHANNEL_PROFILES = (
    "clean_mono",
    "clean_stereo",
    "telephone_g711_proxy",
    "noisy_room",
    "clipped_asymmetric",
)


def _channels_first(samples: np.ndarray) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[None]
    if audio.ndim != 2 or audio.shape[0] not in (1, 2) or audio.shape[1] == 0:
        raise ValueError("audio must have shape [time], [1,time], or [2,time]")
    return audio.copy()


def _peak_limit(audio: np.ndarray, peak: float = 0.98) -> np.ndarray:
    maximum = float(np.max(np.abs(audio)))
    return audio if maximum <= peak else audio * (peak / maximum)


def _band_limit(audio: np.ndarray, sample_rate: int, low: float, high: float) -> np.ndarray:
    spectrum = np.fft.rfft(audio, axis=1)
    frequencies = np.fft.rfftfreq(audio.shape[1], 1.0 / sample_rate)
    spectrum[:, (frequencies < low) | (frequencies > high)] = 0
    return np.fft.irfft(spectrum, n=audio.shape[1], axis=1).astype(np.float32)


def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    target_length = max(1, int(round(audio.shape[1] * target_rate / source_rate)))
    old_axis = np.linspace(0.0, 1.0, audio.shape[1], endpoint=False)
    new_axis = np.linspace(0.0, 1.0, target_length, endpoint=False)
    return np.stack([np.interp(new_axis, old_axis, channel) for channel in audio]).astype(np.float32)


def _mu_law_round_trip(audio: np.ndarray, levels: int = 256) -> np.ndarray:
    mu = float(levels - 1)
    compressed = np.sign(audio) * np.log1p(mu * np.abs(audio)) / np.log1p(mu)
    quantized = np.round((compressed + 1.0) * 0.5 * mu) / mu * 2.0 - 1.0
    return (np.sign(quantized) * np.expm1(np.abs(quantized) * np.log1p(mu)) / mu).astype(np.float32)


def apply_channel_profile(
    samples: np.ndarray,
    profile: str,
    *,
    sample_rate: int = 16_000,
    seed: int = 2608,
) -> tuple[np.ndarray, str]:
    """Apply a bounded transform and return the auditable transform chain."""

    audio = _channels_first(samples)
    rng = np.random.default_rng(seed)
    if profile == "clean_mono":
        audio = audio.mean(axis=0, keepdims=True)
        chain = "pcm16|mono"
    elif profile == "clean_stereo":
        audio = np.repeat(audio, 2, axis=0) if audio.shape[0] == 1 else audio
        chain = "pcm16|stereo"
    elif profile == "telephone_g711_proxy":
        mono = audio.mean(axis=0, keepdims=True)
        mono = _band_limit(mono, sample_rate, 300.0, 3400.0)
        mono = _resample_linear(mono, sample_rate, 8000)
        mono = _mu_law_round_trip(mono)
        audio = _resample_linear(mono, 8000, sample_rate)
        chain = "bandpass_300_3400|downsample_8k|mulaw8|upsample_16k"
    elif profile == "noisy_room":
        stereo = np.repeat(audio, 2, axis=0) if audio.shape[0] == 1 else audio
        delayed = np.zeros_like(stereo)
        for delay_ms, gain in ((23, 0.28), (61, 0.15), (113, 0.08)):
            delay = int(sample_rate * delay_ms / 1000)
            delayed[:, delay:] += stereo[:, :-delay] * gain
        noise = rng.normal(0.0, 0.006, stereo.shape).astype(np.float32)
        audio = stereo + delayed + noise
        chain = "three_tap_room|white_noise_approx_28db"
    elif profile == "clipped_asymmetric":
        stereo = np.repeat(audio, 2, axis=0) if audio.shape[0] == 1 else audio
        stereo[0] *= 1.25
        stereo[1] *= -0.78
        audio = np.clip(np.tanh(stereo * 1.7), -0.82, 0.82)
        chain = "lr_gain_asymmetry|right_polarity_flip|soft_and_hard_clip"
    else:
        raise ValueError(f"unsupported channel profile: {profile}")
    audio = _peak_limit(audio).astype(np.float32, copy=False)
    if not np.isfinite(audio).all():
        raise ValueError("channel transform produced non-finite audio")
    return audio, chain


def write_pcm16_wav(path: str | Path, samples: np.ndarray, sample_rate: int = 16_000) -> None:
    """Write channel-first float audio as deterministic PCM16 WAV."""

    audio = np.clip(_channels_first(samples), -1.0, 1.0)
    pcm = np.round(audio.T * 32767.0).astype("<i2")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as handle:
        handle.setnchannels(audio.shape[0])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
