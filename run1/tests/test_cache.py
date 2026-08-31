from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deepvoice_hybrid.cache import PerFileCache


class CacheTests(unittest.TestCase):
    def test_cache_is_keyed_only_by_model_branch_and_individual_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = PerFileCache(Path(temporary), "a" * 64)
            fields = ("file_fake", "voice_fake", "music_fake")
            self.assertIsNone(cache.load("run0_fake", "b" * 64, fields))
            cache.save(
                "run0_fake",
                "b" * 64,
                {"file_fake": 0.1, "voice_fake": 0.2, "music_fake": 0.3},
            )
            self.assertEqual(
                cache.load("run0_fake", "b" * 64, fields),
                {"file_fake": 0.1, "voice_fake": 0.2, "music_fake": 0.3},
            )
            self.assertIsNone(cache.load("run0_fake", "c" * 64, fields))


if __name__ == "__main__":
    unittest.main()
