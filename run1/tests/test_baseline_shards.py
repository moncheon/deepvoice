import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.merge_baseline_shards import FIELDS, merge


def write_shard(path: Path, sample_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        for sample_id in sample_ids:
            writer.writerow({
                "sample_id": sample_id,
                "audio_sha256": "a" * 64,
                **{name: "0.5" for name in FIELDS[2:]},
            })


class BaselineShardTests(unittest.TestCase):
    def test_merge_requires_exact_plan_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root / "plan.json"
            plan.write_text(json.dumps({
                "calibration_sample_ids": ["cal"],
                "folds": [{"panel_sample_ids": ["panel"]}],
            }), encoding="utf-8")
            first = root / "baseline-0.csv"
            second = root / "baseline-1.csv"
            write_shard(first, ["cal"])
            write_shard(second, ["panel"])
            self.assertEqual(
                [row["sample_id"] for row in merge(plan, [first, second])],
                ["cal", "panel"],
            )
            write_shard(second, ["cal"])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                merge(plan, [first, second])


if __name__ == "__main__":
    unittest.main()
