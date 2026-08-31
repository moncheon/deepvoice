"""Review and fingerprint packaged model artifacts before deserialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Mapping

from .cache import sha256_file


def verify_artifact_manifest(model_root: str | Path) -> str:
    root = Path(model_root)
    manifest_path = root / "ARTIFACTS.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1 or not isinstance(payload.get("files"), Mapping):
        raise ValueError("invalid Run1 artifact manifest")
    listed = payload["files"]
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != set(listed):
        raise ValueError(
            "artifact file set mismatch; "
            f"missing={sorted(set(listed) - actual)}, extra={sorted(actual - set(listed))}"
        )
    for raw_name, record in listed.items():
        pure = PurePosixPath(str(raw_name))
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"unsafe artifact path: {raw_name}")
        path = root.joinpath(*pure.parts)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"artifact is missing or linked: {raw_name}")
        if not isinstance(record, Mapping):
            raise ValueError(f"invalid artifact ledger record: {raw_name}")
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"artifact size mismatch: {raw_name}")
        if sha256_file(path) != str(record["sha256"]):
            raise ValueError(f"artifact hash mismatch: {raw_name}")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
