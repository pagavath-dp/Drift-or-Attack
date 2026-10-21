"""Generation entry points must reject the checked-in frozen reports tree."""

from pathlib import Path
import unittest

from drift_attack.audit import audit
from drift_attack.episode_audit import inspect as inspect_episodes
from drift_attack.model_preflight import inspect as inspect_preflight
from scripts.build_final_results import build


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "reports"


class FrozenOutputSafetyTests(unittest.TestCase):
    def test_standard_library_generators_reject_frozen_target_before_reading_inputs(self):
        missing_archive = ROOT / "data" / "raw" / "missing.zip"
        with self.assertRaisesRegex(ValueError, "outside frozen reports"):
            audit(missing_archive, FROZEN, ROOT)
        with self.assertRaisesRegex(ValueError, "outside frozen reports"):
            inspect_episodes(missing_archive, FROZEN / "nested", "matched_primary")
        with self.assertRaisesRegex(ValueError, "outside frozen reports"):
            inspect_preflight(missing_archive, FROZEN)
        with self.assertRaisesRegex(ValueError, "outside frozen reports"):
            build(ROOT / "other-reports", FROZEN / "final_results", require_external_inputs=False)


if __name__ == "__main__":
    unittest.main()
