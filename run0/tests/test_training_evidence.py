from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from deepvoice_run0.config import load_config
from deepvoice_run0.training import _metrics_from_arrays


class TrainingEvidenceTests(unittest.TestCase):
    def test_repository_training_policy_is_bounded(self) -> None:
        root = Path(__file__).parents[1]
        config = load_config(root / "config.toml")
        training = config["training"]
        self.assertEqual(training["auto_batch_candidates"], [8, 4, 2])
        self.assertEqual(training["effective_batch_size"], 8)
        self.assertGreaterEqual(training["min_epochs"], 4)
        self.assertEqual(training["consistency_loss_weight"], 0.0)
        self.assertEqual(training["domain_loss_weight"], 0.0)

    def test_reviewed_backbone_is_pinned_and_allowlisted(self) -> None:
        root = Path(__file__).parents[1]
        payload = json.loads((root / "backbone_source.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "reviewed")
        self.assertEqual(
            payload["revision"], "4c66d4806a428f2e922ccfa1a962776e232d487b"
        )
        self.assertEqual(payload["license_id"], "CC-BY-SA-3.0")
        self.assertEqual(
            set(payload["files"]),
            {
                "README.md", "config.json", "preprocessor_config.json",
                "pytorch_model.bin", "LICENSE-CC-BY-SA-3.0.txt", "ORIGIN.json",
            },
        )
        for record in payload["files"].values():
            self.assertGreater(record["bytes"], 0)
            self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")

    def test_official_metrics_adapter_preserves_perfect_predictions(self) -> None:
        labels = {
            "file_fake": [0, 1, 0, 1],
            "voice_fake": [0, 1, 0, 0],
            "music_fake": [0, 0, 0, 1],
            "voice_present": [1, 1, 0, 0],
            "music_present": [0, 0, 1, 1],
        }
        scores = np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 1.0],
            ]
        )
        metrics = _metrics_from_arrays(labels, scores)
        self.assertAlmostEqual(metrics["weighted_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
