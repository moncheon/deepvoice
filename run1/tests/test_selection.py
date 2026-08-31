from __future__ import annotations

import unittest

import numpy as np

from deepvoice_hybrid.selection import select_epoch_and_fusion


class SelectionTests(unittest.TestCase):
    def test_nested_selection_returns_independent_head_weights(self) -> None:
        rng = np.random.default_rng(2608)
        folds = (
            [(f"gen-voice-{index:02d}", "generator", "voice") for index in range(1, 7)]
            + [(f"gen-music-{index:02d}", "generator", "music") for index in range(1, 3)]
            + [(f"codec-{index:02d}", "codec", "all") for index in range(1, 6)]
        )
        rows: list[dict[str, str]] = []
        for epoch in (1, 2):
            for fold_id, axis, task in folds:
                for index in range(24):
                    label = index % 2
                    baseline_noise = float(rng.normal(0, 0.28))
                    run0_noise = float(rng.normal(0, 0.28))
                    base = float(np.clip(0.25 + 0.5 * label + baseline_noise, 0, 1))
                    run0 = float(np.clip(0.25 + 0.5 * label + run0_noise, 0, 1))
                    row = {
                        "fold_id": fold_id,
                        "axis": axis,
                        "task": task,
                        "epoch": str(epoch),
                        "sample_id": f"{epoch}-{fold_id}-{index}",
                        "file_fake": str(label),
                        "voice_fake": str(label),
                        "music_fake": str(label),
                    }
                    # Codec grouping can legitimately include examples where
                    # a component-specific label is not applicable.
                    if axis == "codec" and index < 3:
                        row["voice_fake"] = ""
                    if axis == "codec" and 3 <= index < 6:
                        row["music_fake"] = ""
                    for prefix, value in (("baseline_", base), ("run0_", run0)):
                        row[prefix + "FILE_FAKE_PROB"] = str(value)
                        row[prefix + "VOICE_FAKE_PROB"] = str(value)
                        row[prefix + "MUSIC_FAKE_PROB"] = str(value)
                    rows.append(row)
        report = select_epoch_and_fusion(rows, bootstrap_iterations=100, seed=7)
        self.assertIn(report["selected_epoch"], {1, 2})
        self.assertEqual(
            set(report["final_baseline_weights"]),
            {"file_fake", "voice_fake", "music_fake"},
        )
        self.assertIn(report["status"], {"passed", "failed"})


if __name__ == "__main__":
    unittest.main()
