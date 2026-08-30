#!/usr/bin/env python3
"""Acquire, review, prepare, build, and audit the real Run 0 corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RUN0_DIR = Path(__file__).resolve().parent
if str(RUN0_DIR) not in sys.path:
    sys.path.insert(0, str(RUN0_DIR))

from deepvoice_run0.corpus_pipeline import (
    audit_corpus,
    extract_reviewed_archives,
    fetch_source,
    load_catalog,
    prepare_source_manifest,
    review_generated_manifest,
    review_source,
)
from deepvoice_run0.generation import build_corpus_from_sources


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--catalog", type=Path, default=RUN0_DIR / "corpus_sources.json")
    result.add_argument("--staging", type=Path, default=RUN0_DIR / "data" / "staging-v1")
    sub = result.add_subparsers(dest="command", required=True)
    for command in ("fetch", "review", "extract"):
        item = sub.add_parser(command)
        item.add_argument("source", nargs="+", help="Catalog source name(s).")
    generated = sub.add_parser("review-generated")
    generated.add_argument("manifest", type=Path)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", type=Path, default=RUN0_DIR / "data" / "sources-v1" / "sources.jsonl")
    prepare.add_argument("--generated-manifest", type=Path, action="append", default=[])
    prepare.add_argument("--seed", type=int, default=2608)
    build = sub.add_parser("build")
    build.add_argument("--source-manifest", type=Path, default=RUN0_DIR / "data" / "sources-v1" / "sources.jsonl")
    build.add_argument("--output", type=Path, default=RUN0_DIR / "data" / "corpus-v1")
    build.add_argument("--per-combination", type=int, default=3000)
    build.add_argument("--duration", type=float, default=6.0)
    build.add_argument("--seed", type=int, default=2608)
    audit = sub.add_parser("audit")
    audit.add_argument("--manifest", type=Path, default=RUN0_DIR / "data" / "corpus-v1" / "manifest.jsonl")
    audit.add_argument("--splits", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    catalog = load_catalog(args.catalog)
    if args.command == "review-generated":
        value = review_generated_manifest(args.manifest)
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.command in {"fetch", "review", "extract"}:
        for name in args.source:
            if name not in catalog:
                raise SystemExit(f"unknown catalog source: {name}")
            if args.command == "fetch":
                value = fetch_source(name, catalog[name], args.staging)
            elif args.command == "review":
                value = review_source(name, catalog[name], args.staging)
            else:
                value = extract_reviewed_archives(name, catalog[name], args.staging)
            display = value
            if args.command == "review" and isinstance(value, dict):
                display = {
                    key: value[key]
                    for key in (
                        "source", "revision", "license_id", "reviewed_for_noncommercial_research",
                        "blocked_risky_files", "file_count", "total_bytes",
                    )
                }
                display["review_path"] = str(args.staging / name / "REVIEW.json")
            print(json.dumps(display, ensure_ascii=False, default=str, sort_keys=True))
        return
    if args.command == "prepare":
        value = prepare_source_manifest(
            catalog, args.staging, args.output,
            generated_manifests=args.generated_manifest, seed=args.seed,
        )
    elif args.command == "build":
        value = build_corpus_from_sources(
            args.source_manifest, args.output,
            variants_per_combination=args.per_combination,
            duration_sec=args.duration, seed=args.seed,
        )
    else:
        value = audit_corpus(args.manifest, args.splits)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
