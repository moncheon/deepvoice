"""Thin adapter around the organizer Baseline branch, preserving its scoring."""

from __future__ import annotations

import gc
import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Mapping


def _install_demucs_401_load_track_compatibility() -> None:
    """Restore the removed 4.0.1 loader without changing separation math.

    The organizer script imports ``load_track`` from Demucs 4.0.1. Demucs
    4.1.0 removed that symbol in favor of its API class. This is the tagged
    4.0.1 implementation, using the still-public 4.1 audio helpers:
    https://github.com/facebookresearch/demucs/blob/v4.0.1/demucs/separate.py
    """

    import demucs.separate as separate

    if hasattr(separate, "load_track"):
        return
    import torchaudio as ta
    from demucs.audio import AudioFile, convert_audio

    def load_track(track: Path, audio_channels: int, samplerate: int):
        errors: dict[str, str] = {}
        waveform = None
        try:
            waveform = AudioFile(track).read(
                streams=0, samplerate=samplerate, channels=audio_channels
            )
        except FileNotFoundError:
            errors["ffmpeg"] = "FFmpeg is not installed."
        except subprocess.CalledProcessError:
            errors["ffmpeg"] = "FFmpeg could not read the file."
        if waveform is None:
            try:
                waveform, source_rate = ta.load(str(track))
            except RuntimeError as error:
                errors["torchaudio"] = str(error.args[0])
            else:
                waveform = convert_audio(
                    waveform, source_rate, samplerate, audio_channels
                )
        if waveform is None:
            details = "; ".join(
                f"{backend}: {error}" for backend, error in errors.items()
            )
            raise RuntimeError(f"could not load audio {track}; {details}")
        return waveform

    separate.load_track = load_track


def load_baseline_module(script_path: str | Path, model_dir: str | Path) -> ModuleType:
    source = Path(script_path)
    if not source.is_file():
        raise FileNotFoundError(f"baseline script is missing: {source}")
    _install_demucs_401_load_track_compatibility()
    spec = importlib.util.spec_from_file_location("run1_packaged_baseline", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load baseline script: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = Path(model_dir)
    module.MODEL_DIR = root
    module.DF_ARENA_DIR = root / "df_arena_1b"
    module.HTDEMUCS_DIR = root / "htdemucs"
    module.PANNS_DIR = root / "panns"
    return module


class BaselineAdapter:
    def __init__(self, script_path: str | Path, model_dir: str | Path, device: str) -> None:
        self.module = load_baseline_module(script_path, model_dir)
        self.device = self.module.select_device(device)
        self.presence_model = None
        self.voice_indices = None
        self.music_indices = None
        self.fake_model = None
        self.fake_label_index = None
        self.separation_model = None

    def load_presence(self) -> None:
        if self.presence_model is None:
            (
                self.presence_model,
                self.voice_indices,
                self.music_indices,
            ) = self.module.load_panns_model(self.device)

    def predict_presence(self, audio_path: str | Path) -> dict[str, float]:
        self.load_presence()
        audio = self.module.load_audio(Path(audio_path))
        voice, music = self.module.predict_presence(
            self.presence_model, self.voice_indices, self.music_indices, audio
        )
        return {"voice_present": float(voice), "music_present": float(music)}

    def close_presence(self) -> None:
        self.presence_model = None
        self.voice_indices = None
        self.music_indices = None
        gc.collect()
        if self.device.type == "cuda":
            self.module.torch.cuda.empty_cache()

    def load_fake(self) -> None:
        if self.fake_model is None:
            self.fake_model, self.fake_label_index = self.module.load_df_arena_model(
                self.device
            )
            self.separation_model = self.module.load_htdemucs_model()

    def predict_fake(
        self, audio_path: str | Path, presence: Mapping[str, float]
    ) -> dict[str, float]:
        self.load_fake()
        voice_audio, music_audio = self.module.separate_voice_and_music(
            Path(audio_path), self.separation_model, self.device
        )
        voice_fake = self.module.predict_fake(
            self.fake_model, self.fake_label_index, voice_audio, self.device
        )
        music_fake = self.module.predict_fake(
            self.fake_model, self.fake_label_index, music_audio, self.device
        )
        file_fake = self.module.combine_file_fake_score(
            voice_fake,
            music_fake,
            float(presence["voice_present"]),
            float(presence["music_present"]),
        )
        return {
            "file_fake": float(file_fake),
            "voice_fake": float(voice_fake),
            "music_fake": float(music_fake),
        }

    def close_fake(self) -> None:
        self.fake_model = None
        self.fake_label_index = None
        self.separation_model = None
        gc.collect()
        if self.device.type == "cuda":
            self.module.torch.cuda.empty_cache()
