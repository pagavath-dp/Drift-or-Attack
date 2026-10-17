import unittest

import numpy as np

from drift_attack.autoencoder_stage import (_build_autoencoder, _fit_normalizer, _gated_predictions,
                                            _normalize, _source_group_holdout)
from drift_attack.features import INPUT_WIDTH


class AutoencoderProtocolTests(unittest.TestCase):
    def test_approved_architecture_and_loss(self):
        import tensorflow as tf
        model = _build_autoencoder(tf)
        self.assertEqual([layer.units for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)],
                         [64, 16, 64, INPUT_WIDTH])
        self.assertIsInstance(model.loss, tf.keras.losses.Huber)
        self.assertAlmostEqual(float(model.loss(np.array([[0.0]]), np.array([[2.0]]))), 1.5)

    def test_source_group_holdout_is_deterministic_and_batch_local(self):
        metadata = [{"label": "background", "batch": batch, "episode_id": f"{batch}-{i}"}
                    for batch, count in ((2, 135), (3, 192), (4, 15)) for i in range(count)]
        fit, holdout, counts = _source_group_holdout(metadata)
        fit_again, holdout_again, _ = _source_group_holdout(metadata)
        np.testing.assert_array_equal(fit, fit_again)
        np.testing.assert_array_equal(holdout, holdout_again)
        self.assertEqual(len(fit) + len(holdout), 342)
        self.assertFalse(set(fit) & set(holdout))
        self.assertEqual({key: value["holdout"] for key, value in counts.items()},
                         {"2": 27, "3": 38, "4": 3})

    def test_group_reuse_and_intervention_rejected(self):
        rows = [{"label": "background", "batch": 2, "episode_id": "same"}] * 2
        with self.assertRaises(ValueError):
            _source_group_holdout(rows)
        with self.assertRaises(ValueError):
            _source_group_holdout([{"label": "synthetic_attack", "batch": 2, "episode_id": "a"}])

    def test_normalizer_uses_supplied_fit_rows_only(self):
        X = np.tile(np.arange(12, dtype=float).reshape(3, 4), (1, INPUT_WIDTH // 4))
        center, spread = _fit_normalizer(X)
        changed = X.copy()
        changed[1] *= 1000
        self.assertFalse(np.array_equal(center, _fit_normalizer(changed)[0]))
        self.assertTrue(np.isfinite(_normalize(X, center, spread)).all())

    def test_gate_background_and_drift_tie(self):
        predicted = _gated_predictions(np.array([.1, .5, .7]), .2,
                                       np.array([[0, .2, .8], [0, .4, .4], [0, .2, .7]]),
                                       ["background", "synthetic_drift", "synthetic_attack"])
        self.assertEqual(predicted, ["background", "synthetic_drift", "synthetic_attack"])
