import tempfile
import unittest
from pathlib import Path

from drift_attack.audit import audit


class AuditProvenanceTests(unittest.TestCase):
    def test_unknown_archive_hash_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unverified.zip"
            archive.write_bytes(b"not the approved archive")
            output = root / "reports"
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                audit(archive, output, root)
            self.assertFalse(output.exists())
