"""Pure checks for isolated prediction export and frozen-matrix comparison."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.reproduce_models import (CONDITIONS, LABELS, SPLITS, compare_reports, join_predictions,
                                      matrix, prerequisites, publish_exports, reproduce)


class ReproductionGateTests(unittest.TestCase):
    def fixture(self):
        rf_rows, ae_rows = [], []
        baseline, autoencoder = {"conditions": {}}, {"conditions": {}, "threshold": 0.5}
        for condition in CONDITIONS:
            baseline["conditions"][condition] = {}
            autoencoder["conditions"][condition] = {}
            for split in SPLITS:
                baseline["conditions"][condition][split] = {
                    "random_forest": {"labels": list(LABELS), "denominator_episodes": 3,
                                      "confusion_rows_true_columns_predicted": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}}
                autoencoder["conditions"][condition][split] = {
                    "autoencoder_only_binary": {"labels": ["background", "intervention"],
                                                "denominator_episodes": 3,
                                                "confusion_rows_true_columns_predicted": [[1, 0], [0, 2]]},
                    "autoencoder_plus_forest": {"labels": list(LABELS), "denominator_episodes": 3,
                                                "confusion_rows_true_columns_predicted": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}}
                for index, label in enumerate(LABELS):
                    probability = {f"rf_prob_{name}": float(name == label) for name in LABELS}
                    rf_rows.append({"condition": condition, "split": split, "episode_id": "base",
                                    "batch": 5 if split == "validation" else 7, "gas_code": 1,
                                    "concentration": 10.0, "family": "background" if index == 0 else "bias",
                                    "true_label": label, "rf_predicted": label, **probability})
                    ae_rows.append({"condition": condition, "split": split, "episode_id": "base",
                                    "true_label": label, "ae_score": float(index),
                                    "ae_binary_predicted": "background" if index == 0 else "intervention",
                                    "gated_predicted": label, "refit_rf_predicted": label,
                                    **{key.replace("rf_prob_", "refit_rf_prob_"): value
                                       for key, value in probability.items()}})
        return rf_rows, ae_rows, baseline, autoencoder

    def test_export_reconstructs_every_frozen_matrix(self):
        rf_rows, ae_rows, baseline, autoencoder = self.fixture()
        rows, comparisons = join_predictions(rf_rows, ae_rows, baseline, autoencoder)
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(comparisons[condition][split]["frozen_match"]
                            for condition in CONDITIONS for split in SPLITS))
        self.assertEqual(matrix(rows[:3], "ae_binary_predicted", binary=True), [[1, 0], [0, 2]])
        ae_rows[0]["gated_predicted"] = "synthetic_attack"
        with self.assertRaisesRegex(ValueError, "gated prediction"):
            join_predictions(rf_rows, ae_rows, baseline, autoencoder)

    def test_invalid_probability_and_extra_row_cannot_be_exported(self):
        rf_rows, ae_rows, baseline, autoencoder = self.fixture()
        rf_rows[0]["rf_prob_background"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid RF probability"):
            join_predictions(rf_rows, ae_rows, baseline, autoencoder)
        rf_rows, ae_rows, baseline, autoencoder = self.fixture()
        extra = {**rf_rows[0], "condition": "unknown"}
        rf_rows.append(extra)
        ae_rows.append({**ae_rows[0], "condition": "unknown"})
        with self.assertRaisesRegex(ValueError, "unknown condition"):
            join_predictions(rf_rows, ae_rows, baseline, autoencoder)

    def test_same_matrix_with_different_label_order_is_rejected(self):
        _, _, frozen, _ = self.fixture()
        frozen["archive_sha256"] = "archive"
        frozen["calibration_digest"] = "calibration"
        generated = deepcopy(frozen)
        generated["conditions"]["matched_primary"]["validation"]["random_forest"]["labels"] = list(reversed(LABELS))
        with self.assertRaisesRegex(ValueError, "label order differs"):
            compare_reports(generated, frozen, ("random_forest",), "baseline")

    def test_export_matrix_mismatch_is_rejected(self):
        rf_rows, ae_rows, baseline, autoencoder = self.fixture()
        baseline["conditions"]["matched_primary"]["validation"]["random_forest"]["confusion_rows_true_columns_predicted"] = [
            [0, 1, 0], [0, 1, 0], [0, 0, 1]]
        with self.assertRaisesRegex(ValueError, "export matrix"):
            join_predictions(rf_rows, ae_rows, baseline, autoencoder)

    def test_existing_or_frozen_output_is_rejected_before_generation(self):
        archive = Path("missing.zip")
        with self.assertRaisesRegex(ValueError, "new directory outside frozen reports"):
            reproduce(archive, Path(__file__).resolve().parents[1] / "reports" / "new-run")
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "new directory outside frozen reports"):
                reproduce(archive, Path(temporary))

    def test_archive_diagnostics_distinguish_missing_and_mismatched(self):
        with TemporaryDirectory() as temporary:
            archive = Path(temporary) / "uci-270.zip"
            self.assertEqual(prerequisites(archive)[0], f"missing pinned archive: {archive}")
            archive.write_bytes(b"incorrect archive")
            self.assertTrue(prerequisites(archive)[0].startswith(f"archive SHA-256 mismatch: {archive}"))

    def test_failed_export_write_never_publishes_prediction_csv(self):
        class BrokenValue:
            def __str__(self):
                raise RuntimeError("simulated serialization failure")

        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "serialization failure"):
                publish_exports(output, "archive", [{"episode_id": "one"},
                                                    {"episode_id": BrokenValue()}], {})
            self.assertFalse((output / "episode_predictions.csv").exists())
            self.assertFalse((output / "matrix_comparison.json").exists())

    def test_complete_export_publishes_csv_and_comparison(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            publish_exports(output, "archive", [{"episode_id": "one"}], {"matched_primary": {}})
            self.assertIn("episode_id", (output / "episode_predictions.csv").read_text(encoding="utf-8"))
            comparison = json.loads((output / "matrix_comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(comparison["rows"], 1)


if __name__ == "__main__":
    unittest.main()
