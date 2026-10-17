import unittest

from drift_attack.audit import SPLITS


class SplitIntegrityTests(unittest.TestCase):
    def test_proposal_batch_partition_is_disjoint_and_complete(self):
        self.assertEqual(set(SPLITS), set(range(1, 11)))
        self.assertEqual([b for b, s in SPLITS.items() if s == "reference"], [1])
        self.assertEqual([b for b, s in SPLITS.items() if s == "train"], [2, 3, 4])
        self.assertEqual([b for b, s in SPLITS.items() if s == "validation"], [5, 6])
        self.assertEqual([b for b, s in SPLITS.items() if s == "test"], [7, 8, 9, 10])
