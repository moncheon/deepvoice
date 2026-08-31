import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.build_calibration_ledger import PREDICTIONS, build_rows


def write_predictions(path: Path, sample_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", *PREDICTIONS])
        writer.writeheader()
        for index, sample_id in enumerate(sample_ids):
            writer.writerow({
                "sample_id": sample_id,
                **{name: str((index + 1) / 10) for name in PREDICTIONS},
            })


class CalibrationLedgerTests(unittest.TestCase):
    def test_requires_exact_run0_calibration_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root / "plan.json"
            baseline = root / "baseline.csv"
            run0 = root / "run0.csv"
            plan.write_text(
                json.dumps({"calibration_sample_ids": ["b", "a"]}),
                encoding="utf-8",
            )
            write_predictions(baseline, ["a", "b", "outside"])
            write_predictions(run0, ["b", "a"])
            rows = build_rows(plan, baseline, run0)
            self.assertEqual([row["sample_id"] for row in rows], ["a", "b"])
            write_predictions(run0, ["a", "b", "outside"])
            with self.assertRaisesRegex(ValueError, "run0_extra=1"):
                build_rows(plan, baseline, run0)


if __name__ == "__main__":
    unittest.main()
