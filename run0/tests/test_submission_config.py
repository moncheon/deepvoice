from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from deepvoice_run0.config import load_config
from deepvoice_run0.constants import PREDICTION_COLUMNS
from deepvoice_run0.submission import aligned_audio_files, read_submission_template, validate_prediction


class SubmissionConfigTests(unittest.TestCase):
    def test_repository_config(self) -> None:
        config = load_config(Path(__file__).parents[1] / "config.toml")
        self.assertEqual(config["audio"]["sample_rate"], 16000)
        self.assertEqual(config["audio"]["max_segments"], 16)

    def test_template_alignment_uses_template_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "sample.csv"
            with template.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["ID", *PREDICTION_COLUMNS])
                writer.writeheader()
                for sample_id in ("b", "a"):
                    writer.writerow({"ID": sample_id, **{name: 0 for name in PREDICTION_COLUMNS}})
            audio_dir = root / "test"
            audio_dir.mkdir()
            (audio_dir / "a.wav").write_bytes(b"not decoded in this test")
            (audio_dir / "b.flac").write_bytes(b"not decoded in this test")
            fields, rows = read_submission_template(template)
            self.assertEqual(fields, ["ID", *PREDICTION_COLUMNS])
            self.assertEqual([path.stem for path in aligned_audio_files(audio_dir, rows)], ["b", "a"])

    def test_prediction_range(self) -> None:
        self.assertEqual(validate_prediction([0, 0.25, 0.5, 0.75, 1]), [0, 0.25, 0.5, 0.75, 1])
        with self.assertRaises(ValueError):
            validate_prediction([0, 0, 0, 0, float("nan")])


if __name__ == "__main__":
    unittest.main()
