from __future__ import annotations

import unittest

import numpy as np

from deepvoice_run0.fusion import LogisticFuser, noisy_or
from deepvoice_run0.metrics import equal_error_rate, official_metrics, roc_auc


class MetricsFusionTests(unittest.TestCase):
    def test_perfect_and_reversed_binary_metrics(self) -> None:
        labels = [0, 0, 1, 1]
        self.assertAlmostEqual(equal_error_rate(labels, [0.1, 0.2, 0.8, 0.9]), 0.0)
        self.assertAlmostEqual(roc_auc(labels, [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertAlmostEqual(equal_error_rate(labels, [0.9, 0.8, 0.2, 0.1]), 1.0)
        self.assertAlmostEqual(roc_auc(labels, [0.9, 0.8, 0.2, 0.1]), 0.0)

    def test_official_weighted_score_is_one_for_perfect_predictions(self) -> None:
        values = official_metrics(
            file_fake=[0, 1], file_fake_prob=[0.0, 1.0],
            voice_present=[0, 1], voice_present_prob=[0.0, 1.0],
            music_present=[0, 1], music_present_prob=[0.0, 1.0],
            voice_fake=[0, 1], voice_fake_prob=[0.0, 1.0],
            music_fake=[0, 1], music_fake_prob=[0.0, 1.0],
        )
        self.assertAlmostEqual(values["weighted_score"], 1.0)

    def test_noisy_or_and_validation(self) -> None:
        self.assertAlmostEqual(float(noisy_or(1.0, 0.5, 1.0, 0.5)), 0.75)
        with self.assertRaises(ValueError):
            noisy_or(1.1, 0.5, 1.0, 0.5)

    def test_fuser_requires_oof_and_learns_order(self) -> None:
        x = np.asarray([[0.0], [0.1], [0.9], [1.0]])
        y = np.asarray([0, 0, 1, 1])
        with self.assertRaisesRegex(ValueError, "out-of-fold"):
            LogisticFuser().fit(x, y, oof_predictions=False)
        model = LogisticFuser().fit(x, y, oof_predictions=True, steps=500)
        predictions = model.predict_proba(x)
        self.assertLess(predictions[0], predictions[-1])
        restored = LogisticFuser.from_dict(model.to_dict())
        np.testing.assert_allclose(restored.predict_proba(x), predictions)


if __name__ == "__main__":
    unittest.main()
