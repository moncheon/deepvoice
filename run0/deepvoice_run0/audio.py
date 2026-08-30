"""One-pass ffmpeg decoding and stereo mid/side preparation."""

from __future__ import annotations

import json
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray  # [channels, time], float32
    sample_rate: int
    original_sample_rate: int
    original_channels: int
    duration_sec: float
    codec_name: str


def _require_binary(name: str) -> str:
    binary = shutil.which(name)
    if not binary:
        raise RuntimeError(f"Required executable is not on PATH: {name}")
    return binary


def probe_audio(path: str | Path) -> dict[str, object]:
    """Read stream metadata without trusting filename extensions."""

    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    command = [
        _require_binary("ffprobe"),
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,codec_name,duration:format=duration",
        "-of",
        "json",
        str(audio_path),
    ]
    result = subprocess.run(command, capture_output=True, check=False, text=True)
    if result.returncode != 0:
        raise ValueError(f"ffprobe failed for {audio_path.name}: {result.stderr[-500:]}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError(f"No audio stream found: {audio_path}")
    stream = streams[0]
    duration = stream.get("duration") or (payload.get("format") or {}).get("duration")
    return {
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
        "codec_name": str(stream.get("codec_name") or "unknown"),
        "duration_sec": float(duration or 0.0),
    }


def _decode_native_pcm_wav(path: Path, target_sample_rate: int) -> AudioData | None:
    """Decode reviewed, uncompressed PCM WAV without spawning subprocesses.

    Only mono/stereo PCM already at the model sample rate takes this fast path.
    Every other WAV variant falls back to the common ffmpeg decoder below, so
    resampling and uncommon container details retain ffmpeg's behavior.
    """

    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            if (
                handle.getcomptype() != "NONE"
                or channels not in {1, 2}
                or sample_width not in {1, 2, 3, 4}
                or sample_rate != target_sample_rate
                or frame_count <= 0
            ):
                return None
            payload = handle.readframes(frame_count)
    except (EOFError, wave.Error):
        return None

    if sample_width == 1:
        values = (np.frombuffer(payload, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        values = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        packed = np.frombuffer(payload, dtype=np.uint8)
        if packed.size % 3:
            return None
        triplets = packed.reshape(-1, 3).astype(np.int32)
        signed = triplets[:, 0] | (triplets[:, 1] << 8) | (triplets[:, 2] << 16)
        signed = (signed ^ 0x800000) - 0x800000
        values = signed.astype(np.float32) / 8388608.0
    else:
        values = np.frombuffer(payload, dtype="<i4").astype(np.float32) / 2147483648.0
    usable = values.size - (values.size % channels)
    if usable <= 0:
        return None
    samples = values[:usable].reshape(-1, channels).T.copy()
    if not np.isfinite(samples).all():
        raise ValueError(f"Decoded audio contains non-finite values: {path}")
    return AudioData(
        samples=samples,
        sample_rate=target_sample_rate,
        original_sample_rate=sample_rate,
        original_channels=channels,
        duration_sec=float(samples.shape[1] / target_sample_rate),
        codec_name=f"pcm_s{sample_width * 8}le",
    )


def decode_audio(path: str | Path, target_sample_rate: int = 16_000) -> AudioData:
    """Decode to float32 while retaining mono/stereo information.

    Sources with more than two channels are downmixed to stereo by ffmpeg. Mono
    remains mono, which lets :func:`to_mid_side` create an exact zero side view.
    """

    audio_path = Path(path)
    native = _decode_native_pcm_wav(audio_path, target_sample_rate)
    if native is not None:
        return native
    metadata = probe_audio(audio_path)
    output_channels = 1 if int(metadata["channels"]) == 1 else 2
    command = [
        _require_binary("ffmpeg"),
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-map",
        "0:a:0",
        "-ac",
        str(output_channels),
        "-ar",
        str(target_sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace")[-500:]
        raise ValueError(f"ffmpeg failed for {audio_path.name}: {message}")
    flat = np.frombuffer(result.stdout, dtype="<f4")
    usable = flat.size - (flat.size % output_channels)
    if usable <= 0:
        raise ValueError(f"Decoded audio is empty: {path}")
    samples = flat[:usable].reshape(-1, output_channels).T.copy()
    if not np.isfinite(samples).all():
        raise ValueError(f"Decoded audio contains non-finite values: {path}")
    return AudioData(
        samples=samples,
        sample_rate=target_sample_rate,
        original_sample_rate=int(metadata["sample_rate"]),
        original_channels=int(metadata["channels"]),
        duration_sec=float(samples.shape[1] / target_sample_rate),
        codec_name=str(metadata["codec_name"]),
    )


def to_mid_side(samples: np.ndarray) -> np.ndarray:
    """Convert channel-first mono/stereo input to [mid, side]."""

    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[None, :]
    if audio.ndim != 2 or audio.shape[0] not in (1, 2):
        raise ValueError("samples must have shape [time], [1,time], or [2,time]")
    if audio.shape[0] == 1:
        return np.stack((audio[0], np.zeros_like(audio[0])))
    mid = (audio[0] + audio[1]) * 0.5
    side = (audio[0] - audio[1]) * 0.5
    return np.stack((mid, side)).astype(np.float32, copy=False)
