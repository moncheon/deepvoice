"""Atomic per-audio caches that never use another evaluation file's statistics."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class PerFileCache:
    def __init__(self, root: str | Path, model_fingerprint: str) -> None:
        if len(model_fingerprint) != 64:
            raise ValueError("model_fingerprint must be a SHA-256 digest")
        self.root = Path(root) / model_fingerprint

    def _path(self, branch: str, audio_sha256: str) -> Path:
        if not branch.replace("_", "").isalnum() or len(audio_sha256) != 64:
            raise ValueError("unsafe cache key")
        return self.root / branch / f"{audio_sha256}.json"

    def load(
        self,
        branch: str,
        audio_sha256: str,
        expected_fields: tuple[str, ...],
    ) -> dict[str, float] | None:
        path = self._path(branch, audio_sha256)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("branch") != branch or payload.get("audio_sha256") != audio_sha256:
            raise ValueError(f"cache identity mismatch: {path}")
        values = payload.get("values")
        if not isinstance(values, dict) or set(values) != set(expected_fields):
            raise ValueError(f"cache field mismatch: {path}")
        normalized = {name: float(values[name]) for name in expected_fields}
        if any(not 0.0 <= value <= 1.0 for value in normalized.values()):
            raise ValueError(f"cache probability is outside [0, 1]: {path}")
        return normalized

    def save(
        self,
        branch: str,
        audio_sha256: str,
        values: Mapping[str, float],
    ) -> None:
        path = self._path(branch, audio_sha256)
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized = {name: float(value) for name, value in values.items()}
        if any(not 0.0 <= value <= 1.0 for value in normalized.values()):
            raise ValueError("cache probabilities must lie in [0, 1]")
        payload = {
            "format_version": 1,
            "branch": branch,
            "audio_sha256": audio_sha256,
            "values": normalized,
        }
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
