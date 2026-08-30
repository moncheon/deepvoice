from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from deepvoice_run0.channels import CHANNEL_PROFILES, apply_channel_profile, write_pcm16_wav
from deepvoice_run0.generation import build_corpus_from_sources, generate_bootstrap_corpus
from deepvoice_run0.manifest import read_manifest, sha256_file
from deepvoice_run0.training import train
from deepvoice_run0.config import load_config


class GenerationTests(unittest.TestCase):
    def test_every_channel_profile_is_finite(self) -> None:
        source = np.vstack((
            np.linspace(-0.5, 0.5, 3200, dtype=np.float32),
            np.linspace(0.5, -0.5, 3200, dtype=np.float32),
        ))
        for index, profile in enumerate(CHANNEL_PROFILES):
            transformed, chain = apply_channel_profile(source, profile, seed=index)
            self.assertTrue(np.isfinite(transformed).all())
            self.assertTrue(chain)
            self.assertLessEqual(float(np.max(np.abs(transformed))), 0.981)

    def test_bootstrap_writes_all_eight_label_combinations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bootstrap"
            summary = generate_bootstrap_corpus(
                output,
                samples_per_combination=1,
                seed=7,
                min_duration=0.15,
                max_duration=0.15,
            )
            records = read_manifest(output / "manifest.jsonl")
            self.assertEqual(len(records), 8)
            self.assertEqual(summary["files"], 8)
            self.assertTrue(summary["fixture_only"])
            self.assertTrue((output / "splits.json").is_file())
            self.assertTrue(all((output / record.path).is_file() for record in records))
            combinations = {(record.voice_fake, record.music_fake) for record in records}
            self.assertEqual(len(combinations), 8)
            saved = json.loads((output / "SUMMARY.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["manifest_sha256"], summary["manifest_sha256"])

    def test_reviewed_source_builder_keeps_mix_group_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for component in ("voice", "music"):
                for fake in (0, 1):
                    path = root / f"{component}-{fake}.wav"
                    write_pcm16_wav(path, np.full((1, 800), 0.05 + fake * 0.02, dtype=np.float32))
                    rows.append({
                        "source_id": f"{component}-{fake}",
                        "path": path.name,
                        "sha256": sha256_file(path),
                        "component": component,
                        "fake": fake,
                        "source_dataset": "reviewed-test-source",
                        "source_item_id": f"item-{component}-{fake}",
                        "license_id": "test-project-owned",
                        "consent_basis": "test-generated-no-person",
                        "identity_group": f"identity-{component}-{fake}",
                        "content_group": f"content-{component}-{fake}",
                        "mix_split_group": "atomic-group-1",
                        "generator_family": "test-generator" if fake else None,
                        "generator_model": "test-model" if fake else None,
                        "generator_revision": "v1" if fake else None,
                    })
            source_manifest = root / "sources.jsonl"
            source_manifest.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
            )
            output = root / "corpus"
            summary = build_corpus_from_sources(
                source_manifest, output, variants_per_combination=1, duration_sec=4.0, seed=9
            )
            records = read_manifest(output / "manifest.jsonl")
            self.assertEqual(summary["files"], 8)
            self.assertFalse(summary["fixture_only"])
            self.assertEqual({record.split_group for record in records}, {"atomic-group-1"})

    def test_fixture_manifest_is_rejected_for_performance_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bootstrap"
            generate_bootstrap_corpus(
                output, samples_per_combination=1, seed=11, min_duration=0.1, max_duration=0.1
            )
            with self.assertRaisesRegex(ValueError, "fixture-only"):
                train(
                    manifest_path=output / "manifest.jsonl",
                    config=load_config(),
                    output_dir=Path(directory) / "model",
                    device="cpu",
                    licenses_reviewed=True,
                    allow_fixture_training=False,
                )


if __name__ == "__main__":
    unittest.main()
