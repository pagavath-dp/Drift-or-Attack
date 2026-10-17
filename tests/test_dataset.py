import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from drift_attack.dataset import load_archive, parse_line


def sample_line(gas=1, concentration=10, bad_index=None):
    features = " ".join(
        f"{bad_index if i == 3 and bad_index is not None else i}:{float(i)}"
        for i in range(1, 129)
    )
    return f"{gas};{concentration:.6f} {features}\n"


class DatasetParsingTests(unittest.TestCase):
    def test_record_preserves_gas_concentration_and_128_indexed_values(self):
        row = parse_line(sample_line(), 3, 7)
        self.assertEqual((row.batch, row.line, row.gas, row.concentration), (3, 7, 1, 10.0))
        self.assertEqual(len(row.features), 128)
        self.assertEqual(row.features[2], 3.0)

    def test_rejects_misnumbered_features(self):
        with self.assertRaisesRegex(ValueError, "expected feature index 3"):
            parse_line(sample_line(bad_index=4), 1, 1)

    def test_rejects_unexpected_zip_entries_before_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.zip"
            with ZipFile(path, "w") as archive:
                for batch in range(1, 11):
                    archive.writestr(f"batch{batch}.dat", sample_line())
                archive.writestr("../unexpected.txt", "x")
            with self.assertRaisesRegex(ValueError, "unexpected ZIP entries"):
                load_archive(path)

    def test_complete_ten_batch_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.zip"
            with ZipFile(path, "w") as archive:
                for batch in range(1, 11):
                    archive.writestr(f"batch{batch}.dat", sample_line())
            rows = load_archive(path)
            self.assertEqual(sorted(rows), list(range(1, 11)))
            self.assertEqual([rows[i][0].batch for i in range(1, 11)], list(range(1, 11)))
