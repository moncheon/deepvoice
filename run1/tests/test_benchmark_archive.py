import hashlib
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from hf_space.run_benchmark_job import safe_extract


class BenchmarkArchiveTests(unittest.TestCase):
    def test_safe_extract_checks_hash_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "submission.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("script.py", "print('ok')\n")
                handle.writestr("requirements.txt", "")
                handle.writestr("model/fusion.json", "{}\n")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            destination = root / "safe"
            destination.mkdir()
            with patch.dict(
                os.environ, {"RUN1_SUBMISSION_ZIP_SHA256": digest}, clear=False
            ):
                report = safe_extract(archive, destination)
            self.assertEqual(report["entries"], 3)
            self.assertTrue((destination / "script.py").is_file())

            malicious = root / "malicious.zip"
            with zipfile.ZipFile(malicious, "w") as handle:
                handle.writestr("script.py", "")
                handle.writestr("requirements.txt", "")
                handle.writestr("../escape", "bad")
            digest = hashlib.sha256(malicious.read_bytes()).hexdigest()
            with patch.dict(
                os.environ, {"RUN1_SUBMISSION_ZIP_SHA256": digest}, clear=False
            ), self.assertRaisesRegex(RuntimeError, "unsafe"):
                safe_extract(malicious, root / "blocked")


if __name__ == "__main__":
    unittest.main()
