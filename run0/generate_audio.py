#!/usr/bin/env python3
"""Generate Run 0 bootstrap audio or build mixtures from reviewed sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepvoice_run0.generation import build_corpus_from_sources, generate_bootstrap_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    bootstrap = subparsers.add_parser("bootstrap", help="Create procedural fixture WAV files.")
    bootstrap.add_argument("--output", type=Path, default=Path(__file__).with_name("data") / "bootstrap-v1")
    bootstrap.add_argument("--samples-per-combination", type=int, default=4)
    bootstrap.add_argument("--seed", type=int, default=2608)
    bootstrap.add_argument("--min-duration", type=float, default=4.0)
    bootstrap.add_argument("--max-duration", type=float, default=8.0)
    build = subparsers.add_parser("build", help="Mix license-reviewed source audio.")
    build.add_argument("--source-manifest", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--variants-per-combination", type=int, default=4)
    build.add_argument("--duration-sec", type=float, default=6.0)
    build.add_argument("--seed", type=int, default=2608)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "bootstrap":
        summary = generate_bootstrap_corpus(
            args.output,
            samples_per_combination=args.samples_per_combination,
            seed=args.seed,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )
    else:
        summary = build_corpus_from_sources(
            args.source_manifest,
            args.output,
            variants_per_combination=args.variants_per_combination,
            duration_sec=args.duration_sec,
            seed=args.seed,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
