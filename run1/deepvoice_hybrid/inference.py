"""Sequential, resumable, and file-independent Run1 hybrid inference."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Iterable

from .artifacts import verify_artifact_manifest
from .baseline import BaselineAdapter
from .cache import PerFileCache, sha256_file
from .fusion import FusionSpec, fuse_fake_scores


PRESENCE_FIELDS = ("voice_present", "music_present")
FAKE_FIELDS = ("file_fake", "voice_fake", "music_fake")


class HybridPredictor:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "cuda",
        cache_dir: str | Path = "output/.hybrid-cache",
        verify_artifacts: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        if verify_artifacts:
            self.model_fingerprint = verify_artifact_manifest(self.model_dir)
        else:
            self.model_fingerprint = sha256_file(self.model_dir / "fusion.json")
        payload = json.loads((self.model_dir / "fusion.json").read_text(encoding="utf-8"))
        self.fusion = FusionSpec.from_dict(payload)
        self.cache = PerFileCache(cache_dir, self.model_fingerprint)

    def _audio_records(self, audio_files: Iterable[Path]) -> list[tuple[Path, str]]:
        records: list[tuple[Path, str]] = []
        for path in audio_files:
            records.append((Path(path), sha256_file(path)))
        return records

    def predict_all(self, audio_files: Iterable[Path]) -> dict[Path, list[float]]:
        records = self._audio_records(audio_files)
        baseline = BaselineAdapter(
            self.model_dir / "baseline_script.py",
            self.model_dir / "baseline",
            self.device,
        )
        presence_by_sha: dict[str, dict[str, float]] = {}
        if any(
            self.cache.load("presence", audio_sha, PRESENCE_FIELDS) is None
            for _, audio_sha in records
        ):
            baseline.load_presence()
        for path, audio_sha in records:
            values = self.cache.load("presence", audio_sha, PRESENCE_FIELDS)
            if values is None:
                values = baseline.predict_presence(path)
                self.cache.save("presence", audio_sha, values)
            presence_by_sha[audio_sha] = values
        baseline.close_presence()

        baseline_fake_by_sha: dict[str, dict[str, float]] = {}
        if any(
            self.cache.load("baseline_fake", audio_sha, FAKE_FIELDS) is None
            for _, audio_sha in records
        ):
            baseline.load_fake()
        for path, audio_sha in records:
            values = self.cache.load("baseline_fake", audio_sha, FAKE_FIELDS)
            if values is None:
                values = baseline.predict_fake(path, presence_by_sha[audio_sha])
                self.cache.save("baseline_fake", audio_sha, values)
            baseline_fake_by_sha[audio_sha] = values
        baseline.close_fake()
        del baseline
        gc.collect()

        from deepvoice_run0.inference import Predictor as Run0Predictor

        run0 = Run0Predictor(
            self.model_dir / "run0",
            self.model_dir / "run0" / "config.toml",
            self.device,
        )
        run0_fake_by_sha: dict[str, dict[str, float]] = {}
        for path, audio_sha in records:
            values = self.cache.load("run0_fake", audio_sha, FAKE_FIELDS)
            if values is None:
                raw = run0.predict_file(path)
                values = {
                    "file_fake": float(raw[0]),
                    "voice_fake": float(raw[1]),
                    "music_fake": float(raw[2]),
                }
                self.cache.save("run0_fake", audio_sha, values)
            run0_fake_by_sha[audio_sha] = values
        del run0
        gc.collect()
        try:
            import torch

            if self.device == "cuda":
                torch.cuda.empty_cache()
        except ImportError:
            pass

        output: dict[Path, list[float]] = {}
        for path, audio_sha in records:
            fake = fuse_fake_scores(
                self.fusion,
                baseline_fake_by_sha[audio_sha],
                run0_fake_by_sha[audio_sha],
            )
            presence = presence_by_sha[audio_sha]
            output[path] = [
                fake["file_fake"],
                fake["voice_fake"],
                fake["music_fake"],
                presence["voice_present"],
                presence["music_present"],
            ]
        return output
