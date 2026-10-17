import unittest

from drift_attack.dataset import Measurement
from drift_attack.proxy import fit_reference, percentile


def row(batch, line, gas=1, concentration=10, level=0):
    return Measurement(batch, line, gas, concentration, tuple(float(level + j) for j in range(128)))


class ProxyTests(unittest.TestCase):
    def setUp(self):
        self.reference = [row(1, i + 1, level=i) for i in range(5)]

    def test_exact_matching_and_sparse_exclusion(self):
        fitted = fit_reference(self.reference + [row(1, 6, gas=4)])
        self.assertEqual(fitted.status(row(2, 1)), "eligible")
        self.assertEqual(fitted.status(row(2, 1, gas=4)), "sparse_reference")
        self.assertEqual(fitted.status(row(2, 1, concentration=11)), "no_exact_reference")
        self.assertIsNone(fitted.score(row(2, 1, concentration=11)))

    def test_cutoffs_are_batch1_only(self):
        fitted = fit_reference(self.reference)
        later = row(10, 1, level=10000)
        before = dict(fitted.cutoffs)
        self.assertGreater(fitted.score(later), fitted.cutoffs[0.99])
        self.assertEqual(fitted.cutoffs, before)
        with self.assertRaisesRegex(ValueError, "Batch 1 only"):
            fit_reference(self.reference + [later])

    def test_linear_interpolated_percentile(self):
        self.assertEqual(percentile([0, 10], 0.9), 9)
