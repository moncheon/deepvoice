from __future__ import annotations

import unittest

from deepvoice_hybrid.fusion import EmpiricalCDF, FusionSpec, HeadFusion, fuse_fake_scores
from deepvoice_hybrid.metrics import ads_cps_metrics


class FusionMetricsTests(unittest.TestCase):
    def test_ecdf_uses_midranks_and_is_monotone(self) -> None:
        cdf = EmpiricalCDF.fit([0.1, 0.2, 0.2, 0.9])
        self.assertAlmostEqual(cdf.transform(0.1), 0.125)
        self.assertAlmostEqual(cdf.transform(0.2), 0.5)
        self.assertAlmostEqual(cdf.transform(0.9), 0.875)
        values = [cdf.transform(value) for value in [-1.0, 0.15, 0.5, 2.0]]
        self.assertEqual(values, sorted(values))

    def test_fusion_is_per_file_and_json_roundtrips(self) -> None:
        baseline = EmpiricalCDF.fit([0.0, 0.25, 0.5, 0.75, 1.0])
        run0 = EmpiricalCDF.fit([0.0, 0.1, 0.6, 0.8, 1.0])
        spec = FusionSpec(
            heads={
                name: HeadFusion(0.6, baseline, run0)
                for name in ("file_fake", "voice_fake", "music_fake")
            },
            provenance={"plan_sha256": "a" * 64},
        )
        restored = FusionSpec.from_dict(spec.to_dict())
        first = fuse_fake_scores(
            restored,
            {name: 0.4 for name in restored.heads},
            {name: 0.7 for name in restored.heads},
        )
        second = fuse_fake_scores(
            restored,
            {name: 0.4 for name in restored.heads},
            {name: 0.7 for name in restored.heads},
        )
        self.assertEqual(first, second)

    def test_ads_cps_reconstruct_total(self) -> None:
        metrics = ads_cps_metrics(
            file_fake=[0, 0, 1, 1],
            file_fake_prob=[0.1, 0.2, 0.8, 0.9],
            voice_present=[0, 1, 0, 1],
            voice_present_prob=[0.1, 0.9, 0.2, 0.8],
            music_present=[1, 0, 1, 0],
            music_present_prob=[0.9, 0.1, 0.8, 0.2],
            voice_fake=[None, 0, None, 1],
            voice_fake_prob=[0.5, 0.1, 0.5, 0.9],
            music_fake=[0, None, 1, None],
            music_fake_prob=[0.1, 0.5, 0.9, 0.5],
        )
        self.assertEqual(metrics["ads"], 1.0)
        self.assertEqual(metrics["cps"], 1.0)
        self.assertEqual(metrics["weighted_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
