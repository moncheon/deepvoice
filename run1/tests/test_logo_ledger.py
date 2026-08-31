import unittest

from tools.assemble_logo_ledger import index_unique, validate_panel_coverage


class LogoLedgerTests(unittest.TestCase):
    def test_group_safe_holdout_extras_are_allowed(self) -> None:
        baseline = {"target": {"sample_id": "target"}}
        run0 = {
            "target": {"sample_id": "target"},
            "same-group-extra": {"sample_id": "same-group-extra"},
        }
        validate_panel_coverage({"target"}, run0, baseline, fold_id="fold")

    def test_missing_panel_row_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "run0_missing"):
            validate_panel_coverage(
                {"target"}, {}, {"target": {"sample_id": "target"}}, fold_id="fold"
            )

    def test_duplicate_prediction_row_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            index_unique(
                [{"sample_id": "a"}, {"sample_id": "a"}], source="predictions.csv"
            )


if __name__ == "__main__":
    unittest.main()
