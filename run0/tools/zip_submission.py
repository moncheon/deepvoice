#!/usr/bin/env python3
"""Create a deterministic, low-memory Zip64 archive from a reviewed stage."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stage = args.stage.resolve(strict=True)
    output = args.output.resolve(strict=False)
    if not stage.is_dir():
        raise SystemExit(f"stage is not a directory: {stage}")
    if output.exists():
        raise SystemExit(f"refusing to overwrite archive: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in stage.rglob("*") if path.is_file())
    if not files:
        raise SystemExit("submission stage contains no files")
    for path in files:
        if path.is_symlink():
            raise SystemExit(f"submission stage contains a symlink: {path}")
    with zipfile.ZipFile(
        output,
        mode="x",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
        strict_timestamps=False,
    ) as archive:
        for path in files:
            archive.write(path, path.relative_to(stage).as_posix())
    print(f"stored_files={len(files)} archive={output}")


if __name__ == "__main__":
    main()
