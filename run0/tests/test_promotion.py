import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.promote_training_run import REQUIRED_MODEL_FILES, verify_run


class PromotionTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[str, str]:
        run_id = "run0-test"
        source_commit = "a" * 40
        report = {
            "status": "passed",
            "gate_reasons": [],
            "run_id": run_id,
            "source_git_commit": source_commit,
            "best_dev_metrics": {"weighted_score": 0.8},
            "holdout_metrics": {"weighted_score": 0.7},
        }
        metadata = {
            "format_version": 2,
            "provenance": {"run_id": run_id, "source_git_commit": source_commit},
        }
        contents = {
            name: b"fixture"
            for name in REQUIRED_MODEL_FILES
            if name not in {"TRAINING_REPORT.json", "model_config.json"}
        }
        contents["TRAINING_REPORT.json"] = (
            json.dumps(report, sort_keys=True) + "\n"
        ).encode()
        contents["model_config.json"] = (
            json.dumps(metadata, sort_keys=True) + "\n"
        ).encode()
        ledger = {}
        for name, data in contents.items():
            path = root.joinpath(*name.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            ledger[name] = {
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        (root / "ARTIFACTS.json").write_text(
            json.dumps({"format_version": 1, "files": ledger}), encoding="utf-8"
        )
        return run_id, source_commit

    def test_verified_run_requires_exact_hashes_and_passed_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_id, source_commit = self._fixture(root)
            review = verify_run(
                root, run_id=run_id, expected_source_commit=source_commit
            )
            self.assertEqual(review["status"], "reviewed")
            self.assertEqual(review["artifact_count"], len(REQUIRED_MODEL_FILES))

            (root / "run0.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                verify_run(root, run_id=run_id, expected_source_commit=source_commit)


if __name__ == "__main__":
    unittest.main()
