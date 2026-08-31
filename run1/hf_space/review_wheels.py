#!/usr/bin/env python3
"""Hash and structurally review downloaded wheels before installation."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

BLOCKED_SUFFIXES = {".bat", ".cmd", ".com", ".exe", ".msi", ".ps1"}
NATIVE_SUFFIXES = {".dll", ".dylib", ".pyd", ".so"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    wheelhouse, output = map(Path, sys.argv[1:3])
    wheels = sorted(wheelhouse.glob("*.whl"))
    if not wheels:
        raise SystemExit("wheelhouse is empty")
    records: list[dict[str, object]] = []
    blocked: list[str] = []
    for wheel in wheels:
        native: list[str] = []
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts:
                    blocked.append(f"{wheel.name}:{member.filename}:unsafe-path")
                suffix = path.suffix.lower()
                if suffix in BLOCKED_SUFFIXES:
                    blocked.append(f"{wheel.name}:{member.filename}:blocked-suffix")
                elif suffix in NATIVE_SUFFIXES:
                    native.append(member.filename)
        records.append({
            "filename": wheel.name,
            "bytes": wheel.stat().st_size,
            "sha256": sha256_file(wheel),
            "native_members": native,
        })
    report = {
        "format_version": 1,
        "status": "blocked" if blocked else "reviewed",
        "source": "PyPI wheels resolved from pinned direct requirements",
        "blocked_members": blocked,
        "wheel_count": len(records),
        "wheels": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if blocked:
        raise SystemExit(f"wheel review blocked {len(blocked)} archive members")
    print(json.dumps({"status": "reviewed", "wheel_count": len(records)}))


if __name__ == "__main__":
    main()
