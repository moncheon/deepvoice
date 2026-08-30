from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from deepvoice_run0.manifest import (
    ManifestRecord,
    assign_domain_stratified_group_splits,
    assign_group_splits,
    assign_stratified_group_splits,
    read_manifest,
    write_manifest,
)
from deepvoice_run0.dataset import inverse_training_stratum_weights, training_stratum
from deepvoice_run0.mixing import Component, mix_components


def make_record(sample_id: str = "sample-1", group: str = "source-1") -> ManifestRecord:
    return ManifestRecord(
        sample_id=sample_id,
        path="audio.wav",
        sha256="0" * 64,
        source_dataset="reviewed-dataset",
        source_item_id=sample_id,
        license_id="reviewed-license",
        consent_basis="documented",
        speaker_or_artist_group="speaker-1",
        content_group="content-1",
        generator_family=None,
        generator_model=None,
        generator_revision=None,
        generation_seed=None,
        voice_present=1,
        music_present=0,
        voice_fake=0,
        music_fake=None,
        file_fake=0,
        mix_mode="none",
        codec_chain="pcm",
        channel_profile="clean",
        sample_rate=16000,
        channels=1,
        duration_sec=4.0,
        split_group=group,
    )


class ManifestMixingTests(unittest.TestCase):
    def test_manifest_round_trip_and_duplicate_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            write_manifest(path, [make_record()])
            self.assertEqual(read_manifest(path), [make_record()])
            with self.assertRaises(ValueError):
                write_manifest(path, [make_record(), make_record()])

    def test_label_logic_rejects_absent_component_label(self) -> None:
        invalid = replace(make_record(), music_present=0, music_fake=1, file_fake=1)
        with self.assertRaisesRegex(ValueError, "must be null"):
            invalid.validate()

    def test_group_split_cannot_leak(self) -> None:
        records = [make_record("a", "shared"), make_record("b", "shared"), make_record("c", "other")]
        assignments = assign_group_splits(records)
        self.assertEqual(assignments["a"], assignments["b"])

    def test_generator_holdout(self) -> None:
        record = replace(
            make_record("fake"), generator_family="held-family", generator_model="m",
            voice_fake=1, file_fake=1,
        )
        assignments = assign_group_splits([record], holdout_generator_families={"held-family"})
        self.assertEqual(assignments["fake"], "generator_holdout")

    def test_stratified_split_covers_small_stratum(self) -> None:
        records = [make_record(f"sample-{index}", f"group-{index}") for index in range(4)]
        assignments = assign_stratified_group_splits(records)
        counts = {name: list(assignments.values()).count(name) for name in set(assignments.values())}
        self.assertEqual(counts, {"train": 2, "dev": 1, "holdout": 1})

    def test_domain_split_covers_generator_and_preserves_groups(self) -> None:
        records = [
            replace(
                make_record(f"mms-{index}", f"group-{index}"),
                generator_family="MMS-TTS-KOR",
                generator_model="facebook/mms-tts-kor",
                voice_fake=1,
                file_fake=1,
            )
            for index in range(9)
        ]
        assignments = assign_domain_stratified_group_splits(records)
        self.assertEqual(set(assignments.values()), {"train", "dev", "holdout"})
        self.assertEqual(len(assignments), len(records))

    def test_training_sampler_collapses_exact_mlaad_families(self) -> None:
        first = replace(
            make_record("a", "a"), generator_family="MLAAD:model-a",
            generator_model="a", voice_fake=1, file_fake=1,
        )
        second = replace(
            make_record("b", "b"), generator_family="MLAAD:model-b",
            generator_model="b", voice_fake=1, file_fake=1,
        )
        real = make_record("c", "c")
        self.assertEqual(training_stratum(first), training_stratum(second))
        weights = inverse_training_stratum_weights([first, second, real])
        self.assertTrue(np.allclose(weights, [0.5, 0.5, 1.0]))

    def test_mixing_derives_component_and_file_labels(self) -> None:
        voice = Component(np.ones((1, 10), dtype=np.float32), "voice", 0, "v")
        music = Component(np.ones((2, 6), dtype=np.float32), "music", 1, "m")
        mixture = mix_components(voice, music, mode="sequential", gap_samples=3)
        self.assertEqual(mixture.samples.shape, (2, 19))
        self.assertEqual(mixture.voice_fake, 0)
        self.assertEqual(mixture.music_fake, 1)
        self.assertEqual(mixture.file_fake, 1)


if __name__ == "__main__":
    unittest.main()
