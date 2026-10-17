import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from drift_attack.episodes import construct_episodes, fit_generator_scales, generate_variants
from drift_attack.features import FEATURE_GROUPS, INPUT_WIDTH, episode_features, fit_feature_scale
from drift_attack.ml_baselines import _binary_metrics, _classification_metrics, select_anomaly_threshold
from drift_attack.model_preflight import exact_copy_signature, inspect as preflight
from test_episodes import fixture


class ModelInputTests(unittest.TestCase):
    def test_exact_copy_sensor_block_signatures(self):
        features = np.ones(INPUT_WIDTH)
        self.assertEqual(exact_copy_signature(features)["zero_active_variation_sensor_blocks"], 0)
        features[512:520] = 0
        features[640 + 8:640 + 16] = 0
        self.assertEqual(exact_copy_signature(features), {
            "zero_active_variation_sensor_blocks": 1,
            "zero_lag_four_sensor_blocks": 1,
        })
        with self.assertRaises(ValueError):
            exact_copy_signature(np.zeros(16))

    def test_exact_numeric_input_and_no_metadata_arguments(self):
        batches = fixture()
        episode = next(item for item in construct_episodes(batches)[0] if item.batch == 2)
        variant = generate_variants(episode, fit_generator_scales(batches))[0]
        scale = fit_feature_scale(batches)
        features = episode_features(variant.rows, scale)
        self.assertEqual(features.shape, (INPUT_WIDTH,))
        self.assertEqual(INPUT_WIDTH, 128 * len(FEATURE_GROUPS))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(episode_features(variant.rows, scale).tobytes(), features.tobytes())
        changed_metadata = replace(variant, batch=10, split="test", label="synthetic_attack",
                                   family="bias", severity=99.0, sensor=15)
        self.assertEqual(episode_features(changed_metadata.rows, scale).tobytes(), features.tobytes())
        with self.assertRaises(TypeError):
            episode_features(variant.rows, scale, episode.gas)

    def test_feature_scale_does_not_depend_on_validation_or_test(self):
        batches = fixture()
        first = fit_feature_scale(batches)
        for batch in range(5, 11):
            batches[batch] = [type(row)(row.batch, row.line, row.gas, row.concentration,
                                        tuple(value * 1e6 for value in row.features))
                              for row in batches[batch]]
        second = fit_feature_scale(batches)
        np.testing.assert_array_equal(first.center, second.center)
        np.testing.assert_array_equal(first.spread, second.spread)

    def test_anomaly_threshold_uses_validation_scores_only(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = ["background", "background", "synthetic_drift", "synthetic_attack"]
        threshold, score = select_anomaly_threshold(scores, labels)
        self.assertEqual(score, 1.0)
        self.assertTrue(0.2 < threshold < 0.8)

    def test_metric_denominators_and_directional_errors(self):
        true = ["background", "synthetic_drift", "synthetic_attack"]
        predicted = ["synthetic_attack", "synthetic_attack", "background"]
        metrics = _classification_metrics(true, predicted)
        self.assertEqual(metrics["confusion_rows_true_columns_predicted"],
                         [[0, 0, 1], [0, 0, 1], [1, 0, 0]])
        self.assertEqual(metrics["rates"]["attack_false_alarm"]["numerator"], 1)
        self.assertEqual(metrics["rates"]["drift_mistaken_for_attack"]["denominator"], 1)
        self.assertEqual(metrics["rates"]["attack_mistaken_for_background_or_drift"]["numerator"], 1)
        binary = _binary_metrics(["background", "intervention", "intervention"],
                                 ["intervention", "intervention", "background"], true)
        self.assertEqual(binary["rates"]["any_intervention_false_alarm"]["denominator"], 1)


class PreflightIntegrationTests(unittest.TestCase):
    def test_actual_calibration_and_static_leakage_gates(self):
        root = Path(__file__).resolve().parents[1]
        archive = root / "data/raw/uci-270.zip"
        if not archive.exists():
            self.skipTest("official archive is needed for integration preflight")
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for name in ("magnitude_calibration.json", "episode_diagnostics_matched_primary.json",
                         "episode_diagnostics_original_stress.json"):
                (folder / name).write_bytes((root / "reports" / name).read_bytes())
            result = preflight(archive, folder)
            self.assertEqual(result["model_input_metadata_fields"], [])
            self.assertEqual(result["model_input_width"], 768)
            self.assertGreater(result["conditions"]["matched_primary"]["finite_pool_signature"]["test"]
                               ["episodes_with_reused_magnitude_within_split"], 0)
            self.assertEqual(json.loads((folder / "ml_preflight.json").read_text())["calibration_digest"],
                             result["calibration_digest"])
            for condition in ("matched_primary", "original_stress"):
                signatures = result["conditions"][condition]["exact_copy_signatures_by_split_family"]
                for split, expected in (("train", 68), ("validation", 56), ("test", 181)):
                    self.assertEqual(signatures[split]["freeze"]["zero_active_variation_sensor_blocks"], expected)
                    self.assertEqual(signatures[split]["replay"]["zero_lag_four_sensor_blocks"], expected)
                    for family, counts in signatures[split].items():
                        if family != "freeze":
                            self.assertEqual(counts["zero_active_variation_sensor_blocks"], 0)
                        if family != "replay":
                            self.assertEqual(counts["zero_lag_four_sensor_blocks"], 0)
