#!/usr/bin/env python3
"""Train Run 0 from a license-reviewed manifest and local SSL backbone."""

from __future__ import annotations

import argparse
from pathlib import Path

from deepvoice_run0.config import load_config
from deepvoice_run0.training import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("model"))
    parser.add_argument("--splits", type=Path, help="Exact sample_id-to-split JSON assignment.")
    parser.add_argument(
        "--backbone-path",
        type=Path,
        help="Reviewed local backbone directory; overrides config.toml.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--acknowledge-licenses-reviewed",
        action="store_true",
        help="Confirm that every manifest source was reviewed for this use.",
    )
    parser.add_argument(
        "--allow-fixture-smoke-training",
        action="store_true",
        help="Permit procedural bootstrap audio only for a non-performance smoke run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.backbone_path is not None:
        config["model"]["backbone_path"] = str(args.backbone_path)
    else:
        configured = Path(config["model"]["backbone_path"])
        if not configured.is_absolute():
            config["model"]["backbone_path"] = str(args.config.resolve().parent / configured)
    metadata = train(
        manifest_path=args.manifest,
        config=config,
        output_dir=args.output_dir,
        device=args.device,
        licenses_reviewed=args.acknowledge_licenses_reviewed,
        allow_fixture_training=args.allow_fixture_smoke_training,
        split_path=args.splits,
    )
    print(f"Saved tensor-only weights after {len(metadata['history'])} epoch(s): {args.output_dir}")


if __name__ == "__main__":
    main()
