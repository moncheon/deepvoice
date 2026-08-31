import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hf_space import run_final_job


class FinalSplitTests(unittest.TestCase):
    def test_calibration_groups_are_held_out_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [
                {"sample_id": "cal-a", "split_group": "shared-cal"},
                {"sample_id": "cal-relative", "split_group": "shared-cal"},
            ]
            rows.extend(
                {"sample_id": f"sample-{index}", "split_group": f"group-{index}"}
                for index in range(200)
            )
            (root / "manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            plan = root / "plan.json"
            plan.write_text(
                json.dumps({
                    "plan_sha256": "a" * 64,
                    "calibration_sample_ids": ["cal-a"],
                }),
                encoding="utf-8",
            )
            output = root / "splits.json"
            with patch.object(run_final_job, "DATA_ROOT", root), patch.dict(
                os.environ, {"RUN1_PLAN_SHA256": "a" * 64}, clear=False
            ):
                report = run_final_job.build_final_splits(plan, output, seed=2608)
            assignments = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(assignments["cal-a"], "holdout")
            self.assertEqual(assignments["cal-relative"], "holdout")
            self.assertEqual(report["calibration_rows"], 1)
            self.assertGreater(report["split_counts"]["train"], 0)
            self.assertGreater(report["split_counts"]["dev"], 0)


if __name__ == "__main__":
    unittest.main()
