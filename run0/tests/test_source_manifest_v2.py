from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from deepvoice_run0.channels import write_pcm16_wav
from deepvoice_run0.generation import build_corpus_from_sources
from deepvoice_run0.corpus_pipeline import import_generated_manifest, review_generated_manifest
from deepvoice_run0.manifest import read_manifest, sha256_file
from deepvoice_run0.source_manifest import (
    SourceRecord,
    assign_source_splits,
    audit_source_split_leakage,
    write_source_manifest,
)


class SourceManifestV2Tests(unittest.TestCase):
    def _record(
        self, root: Path, name: str, component: str, fake: int, *,
        split: str | None = None, unique_audio: bool = False,
    ) -> SourceRecord:
        path = root / f"{name}.wav"
        amplitude = 0.04 + fake * 0.01
        if unique_audio:
            amplitude += int.from_bytes(hashlib.sha256(name.encode()).digest()[:2], "big") / 10_000_000
        write_pcm16_wav(path, np.full((1, 1600), amplitude, dtype=np.float32))
        return SourceRecord(
            source_id=name, path=path.name, sha256=sha256_file(path), component=component,
            fake=fake, source_dataset=f"dataset-{name}", source_item_id=name,
            license_id="CC-BY-4.0", consent_basis="public-corpus-test",
            identity_group=f"identity-{name}", content_group=f"content-{name}",
            mix_split_group=None, generator_family=f"generator-{name}" if fake else None,
            generator_model=f"model-{name}" if fake else None,
            generator_revision="revision" if fake else None,
            source_revision="source-revision", source_url="https://example.test/source",
            license_url="https://example.test/license", language="en", split=split,
        )

    def test_v2_builder_pools_different_datasets_by_preassigned_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                self._record(root, f"{split}-{name}", component, fake, split=split, unique_audio=True)
                for split in ("train", "dev", "holdout")
                for name, component, fake in (
                    ("real-voice", "voice", 0),
                    ("fake-voice", "voice", 1),
                    ("real-music", "music", 0),
                    ("fake-music", "music", 1),
                )
            ]
            manifest = root / "sources.jsonl"
            write_source_manifest(manifest, records)
            output = root / "corpus"
            summary = build_corpus_from_sources(
                manifest, output, variants_per_combination=1, duration_sec=4.0, seed=5
            )
            built = read_manifest(output / "manifest.jsonl")
            splits = json.loads((output / "splits.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["format_version"], 2)
            self.assertEqual(len(built), 8)
            self.assertEqual(set(splits.values()), {"train"})
            self.assertTrue(all(record.split_group.startswith("parents:") for record in built))
            self.assertEqual(
                [len(list((output / "audio" / f"C{index:02d}").glob("*.wav"))) for index in range(8)],
                [1] * 8,
            )

    def test_production_builder_rejects_an_incomplete_split_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                self._record(root, f"{split}-{name}", component, fake, split=split, unique_audio=True)
                for split in ("train", "dev", "holdout")
                for name, component, fake in (
                    ("real-voice", "voice", 0),
                    ("fake-voice", "voice", 1),
                    ("real-music", "music", 0),
                    ("fake-music", "music", 1),
                )
                if not (split == "dev" and component == "music" and fake == 1)
            ]
            manifest = root / "sources.jsonl"
            write_source_manifest(manifest, records)
            with self.assertRaisesRegex(ValueError, "incomplete by split"):
                build_corpus_from_sources(manifest, root / "corpus", variants_per_combination=1)

    def test_connected_identity_and_parent_sources_never_cross_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._record(root, "first", "voice", 0)
            second = self._record(root, "second", "music", 0)
            second = SourceRecord(**(second.__dict__ | {
                "identity_group": first.identity_group,
                "parent_source_ids": (first.source_id,),
            }))
            assigned = assign_source_splits([first, second], seed=8)
            self.assertEqual(len({record.split for record in assigned}), 1)
            self.assertEqual(audit_source_split_leakage(assigned)["leakage_keys"], 0)

    def test_identical_audio_hashes_are_split_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._record(root, "duplicate-a", "voice", 0)
            second = self._record(root, "duplicate-b", "music", 0)
            self.assertEqual(first.sha256, second.sha256)
            assigned = assign_source_splits([first, second], seed=19)
            self.assertEqual(len({record.split for record in assigned}), 1)
            self.assertEqual(audit_source_split_leakage(assigned)["leakage_keys"], 0)

    def test_generated_sources_require_matching_review_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self._record(root, "generated", "voice", 1, split="train")
            record = SourceRecord(**(record.__dict__ | {
                "consent_basis": "project-authored-generic-prompt-no-identity-cloning",
            }))
            manifest = root / "sources.jsonl"
            write_source_manifest(manifest, [record])
            for name in ("MODEL_REVIEW.json", "WHEEL_REVIEW.json"):
                (root / name).write_text(json.dumps({"status": "reviewed"}), encoding="utf-8")
            report = review_generated_manifest(manifest)
            self.assertEqual(report["records"], 1)
            imported = import_generated_manifest(manifest, root / "combined")
            self.assertEqual(imported[0].source_id, record.source_id)

    def test_forced_generator_holdout_overrides_its_connected_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = self._record(root, "mms", "voice", 1, split="train")
            fake = SourceRecord(**(fake.__dict__ | {"generator_family": "MMS-TTS-KOR"}))
            assigned = assign_source_splits(
                [fake], seed=2608, holdout_generator_families={"MMS-TTS-KOR"}
            )
            self.assertEqual(assigned[0].split, "holdout")


if __name__ == "__main__":
    unittest.main()
