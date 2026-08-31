import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "hf_budget.py"


class HfBudgetTests(unittest.TestCase):
    def run_budget(self, ledger: Path, *arguments: str, check: bool = True):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--ledger", str(ledger), *arguments],
            text=True,
            capture_output=True,
            check=check,
        )

    def test_reservation_settlement_and_hard_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "budget.json"
            reserved = self.run_budget(
                ledger, "reserve", "--name", "pilot", "--flavor", "a100-large",
                "--rate-per-hour", "2.5", "--timeout-minutes", "24",
            )
            reservation_id = json.loads(reserved.stdout)["reservation_id"]
            attached = self.run_budget(
                ledger, "attach", "--reservation-id", reservation_id,
                "--job-id", "job-1",
            )
            self.assertEqual(json.loads(attached.stdout)["job_id"], "job-1")
            blocked = self.run_budget(
                ledger, "reserve", "--name", "too-large", "--flavor", "a100x4",
                "--rate-per-hour", "10", "--timeout-minutes", "60", check=False,
            )
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn("budget blocked", blocked.stderr)
            self.run_budget(
                ledger, "settle", "--reservation-id", reservation_id,
                "--job-id", "job-1", "--runtime-seconds", "60", "--status", "completed",
            )
            status = self.run_budget(ledger, "status")
            self.assertAlmostEqual(json.loads(status.stdout)["spent_usd"], 2.5 / 60)


if __name__ == "__main__":
    unittest.main()
