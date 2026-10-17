import copy
import csv
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from scripts.build_final_results import (INPUTS, build, derive_tables, digest,
                                         load_and_validate, validate_model_result)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "final_results"


class FinalResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.batches, cls.provenance = load_and_validate(REPORTS)
        cls.tables = derive_tables(cls.data, cls.batches)

    def test_track_a_exclusions_and_cutoff_denominators(self):
        rows = self.tables["track_a_by_batch"]
        self.assertEqual(len(rows), 10)
        test = [row for row in rows if row["split"] == "test"]
        self.assertEqual(sum(row["rows"] for row in test), 7977)
        self.assertEqual(sum(row["eligible"] for row in test), 3834)
        for row in rows:
            self.assertEqual(row["rows"], row["eligible"] + row["excluded"])
            for key in ("no_exact_reference", "sparse_reference", "excluded"):
                self.assertEqual(row[f"{key}_denominator"], row["rows"])
            for q in (90, 95, 99):
                self.assertEqual(row[f"eligible_q{q}_denominator"], row["eligible"])
        batch_seven = next(row for row in rows if row["batch"] == 7)
        self.assertEqual((batch_seven["eligible"], batch_seven["no_exact_reference"],
                          batch_seven["sparse_reference"], batch_seven["rows"]),
                         (1118, 2082, 413, 3613))

    def test_model_task_separation_and_matrix_orientation_gate(self):
        baseline = self.data["ml_baselines"]["conditions"]["matched_primary"]["test"]
        binary = copy.deepcopy(baseline["anomaly_only"])
        three = copy.deepcopy(baseline["random_forest"])
        validate_model_result(binary, "binary", "binary fixture")
        validate_model_result(three, "three_class", "three-class fixture")
        with self.assertRaisesRegex(ValueError, "task/label/matrix orientation"):
            validate_model_result(binary, "three_class", "cross-task")
        three["confusion_rows_true_columns_predicted"][0][1] += 1
        with self.assertRaisesRegex(ValueError, "denominator"):
            validate_model_result(three, "three_class", "corrupt count")
        transposed = copy.deepcopy(baseline["random_forest"])
        matrix = transposed["confusion_rows_true_columns_predicted"]
        transposed["confusion_rows_true_columns_predicted"] = list(map(list, zip(*matrix)))
        with self.assertRaisesRegex(ValueError, "orientation"):
            validate_model_result(transposed, "three_class", "transposed matrix")

    def test_family_controls_and_saved_denominators(self):
        rows = self.tables["attack_family_severity"]
        self.assertTrue(rows)
        self.assertTrue(all(row["exact_copy_control"] ==
                            (row["family"] in ("freeze", "replay")) for row in rows))
        for row in rows:
            self.assertLessEqual(row["correct_or_detected"], row["denominator"])
            self.assertEqual(row["evaluation_status"],
                             "exploratory" if row["split"] == "test" else "validation")

    def test_vector_outputs_and_source_hashes(self):
        self.assertEqual(len(list(OUTPUT.glob("figures/*.svg"))), 7)
        for path in OUTPUT.glob("figures/*.svg"):
            root = ET.parse(path).getroot()
            self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
            self.assertGreater(path.stat().st_size, 3000)
        saved = json.loads((OUTPUT / "source_hashes.json").read_text())
        self.assertEqual(saved, self.provenance)
        for name in INPUTS:
            self.assertEqual(saved["source_sha256"][name], digest(REPORTS / name))
        self.assertIn("exploratory", (OUTPUT / "final_results.md").read_text().lower())
        self.assertIn("exact-copy controls", (OUTPUT / "final_results.md").read_text())

    def test_derived_csv_schema_and_repeatability(self):
        for name, rows in self.tables.items():
            with (OUTPUT / "tables" / f"{name}.csv").open(newline="") as stream:
                parsed = list(csv.DictReader(stream))
            self.assertEqual(parsed, [{key: str(value) for key, value in row.items()}
                                      for row in rows], name)
        before = {str(path.relative_to(OUTPUT)): digest(path)
                  for path in OUTPUT.rglob("*") if path.is_file()}
        build(REPORTS, OUTPUT)
        after = {str(path.relative_to(OUTPUT)): digest(path)
                 for path in OUTPUT.rglob("*") if path.is_file()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
