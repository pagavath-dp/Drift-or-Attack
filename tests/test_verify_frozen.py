"""Exercise the read-only verifier against an isolated copy of historical evidence."""

from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from scripts.build_final_results import EXTERNAL_INPUTS
from scripts.verify_frozen import verify


ROOT = Path(__file__).resolve().parents[1]


class FrozenVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.reports = self.root / "reports"
        self.reports.mkdir()
        manifest = json.loads((ROOT / "docs/frozen_report_hashes.json").read_text(encoding="utf-8"))
        for relative in manifest:
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        for relative in ("docs/frozen_report_hashes.json", "docs/experimental_protocol.md",
                         *EXTERNAL_INPUTS):
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)

    def check(self):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = verify(self.reports)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_separately_managed_evaluation2_passes(self):
        report = self.reports / "evaluation2" / "main.tex"
        report.parent.mkdir()
        report.write_text("Separate report source.\n", encoding="utf-8")
        status, output, errors = self.check()
        self.assertEqual((status, errors), (0, ""))
        self.assertIn("34 pinned file hashes", output)

    def test_ds_store_does_not_fail_inventory(self):
        (self.reports / ".DS_Store").write_bytes(b"Finder metadata")
        (self.reports / "final_results" / ".DS_Store").write_bytes(b"Finder metadata")
        self.assertEqual(self.check()[0], 0)

    def test_missing_historical_report_fails(self):
        (self.reports / "ml_baselines.md").unlink()
        status, _, errors = self.check()
        self.assertEqual(status, 1)
        self.assertIn("frozen artifact reports/ml_baselines.md: missing", errors)

    def test_changed_historical_report_fails(self):
        path = self.reports / "ml_baselines.md"
        path.write_bytes(path.read_bytes() + b"\nchanged\n")
        status, _, errors = self.check()
        self.assertEqual(status, 1)
        self.assertIn("frozen artifact reports/ml_baselines.md: mismatch", errors)

    def test_unexpected_file_in_historical_location_fails(self):
        (self.reports / "final_results" / "unexpected.csv").write_text("extra\n", encoding="utf-8")
        status, _, errors = self.check()
        self.assertEqual(status, 1)
        self.assertIn("reports/final_results/unexpected.csv", errors)

    def test_verification_does_not_write_files(self):
        (self.reports / "evaluation2").mkdir()
        (self.reports / "evaluation2" / "main.tex").write_text("Separate report.\n", encoding="utf-8")
        before = self.snapshot()
        self.assertEqual(self.check()[0], 0)
        self.assertEqual(self.snapshot(), before)

    def snapshot(self):
        return {path.relative_to(self.root).as_posix():
                ("file", sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
                if path.is_file() else ("directory", path.stat().st_mtime_ns)
                for path in self.root.rglob("*")}


if __name__ == "__main__":
    unittest.main()
