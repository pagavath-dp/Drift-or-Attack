"""Reproduce frozen models in a new directory; export predictions only after exact checks.

Historical PDF provenance remains the responsibility of scripts.verify_frozen.
This runner needs the pinned archive, frozen calibration evidence, and locked model environment.
"""

from __future__ import annotations

import argparse
import csv
from importlib.metadata import PackageNotFoundError, version
import json
from math import isfinite
from pathlib import Path
import shutil
import sys

from scripts.build_final_results import digest, input_hash_status


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "reports"
EVIDENCE = ("magnitude_calibration.json", "episode_diagnostics_matched_primary.json",
            "episode_diagnostics_original_stress.json")
CONDITIONS = ("matched_primary", "original_stress")
SPLITS = ("validation", "test")
LABELS = ("background", "synthetic_drift", "synthetic_attack")
BINARY_LABELS = ("background", "intervention")


def frozen_report_problems() -> list[str]:
    pinned = json.loads((ROOT / "docs/frozen_report_hashes.json").read_text(encoding="utf-8"))
    problems = []
    for relative, expected in pinned.items():
        status = input_hash_status(ROOT / relative, expected)
        if status not in ("exact", "crlf_checkout"):
            problems.append(f"frozen report {relative}: {status}")
    return problems


def prerequisites(archive: Path) -> list[str]:
    """Report every blocker without importing model libraries or writing files."""
    problems = []
    manifest = json.loads((FROZEN / "source_manifest.json").read_text(encoding="utf-8"))
    expected_archive = manifest["archive_sha256"]
    if not archive.is_file():
        problems.append(f"missing pinned archive: {archive}")
    elif digest(archive) != expected_archive:
        problems.append(f"archive SHA-256 mismatch: {archive}; expected {expected_archive}")
    if sys.version_info[:2] != (3, 13):
        problems.append(f"Python 3.13 required; running {sys.version.split()[0]} at {sys.executable}")
    for line in (ROOT / "requirements-ae.lock").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        try:
            actual = version(name)
        except PackageNotFoundError:
            problems.append(f"missing package: {name}=={expected}")
        else:
            if actual != expected:
                problems.append(f"package version mismatch: {name} {actual}; expected {expected}")
    problems.extend(frozen_report_problems())
    return problems


def matrix(rows: list[dict], predicted_field: str, *, binary: bool = False) -> list[list[int]]:
    labels = ("background", "intervention") if binary else LABELS
    index = {label: i for i, label in enumerate(labels)}
    result = [[0] * len(labels) for _ in labels]
    for row in rows:
        true = ("background" if row["true_label"] == "background" else "intervention") if binary else row["true_label"]
        result[index[true]][index[row[predicted_field]]] += 1
    return result


def compare_reports(actual: dict, frozen: dict, models: tuple[str, ...], stage: str) -> None:
    for field in ("archive_sha256", "calibration_digest"):
        if actual[field] != frozen[field]:
            raise ValueError(f"{stage} {field} differs from frozen result")
    for condition in CONDITIONS:
        for split in SPLITS:
            for model in models:
                actual_entry = actual["conditions"][condition][split][model]
                frozen_entry = frozen["conditions"][condition][split][model]
                labels = BINARY_LABELS if model in ("anomaly_only", "autoencoder_only_binary") else LABELS
                if actual_entry["labels"] != list(labels) or frozen_entry["labels"] != list(labels):
                    raise ValueError(f"{stage} {condition}/{split}/{model}: label order differs")
                if actual_entry["denominator_episodes"] != frozen_entry["denominator_episodes"]:
                    raise ValueError(f"{stage} {condition}/{split}/{model}: episode count differs")
                current = actual_entry["confusion_rows_true_columns_predicted"]
                expected = frozen_entry["confusion_rows_true_columns_predicted"]
                if current != expected:
                    raise ValueError(f"{stage} {condition}/{split}/{model}: generated matrix {current} != frozen {expected}")


def join_predictions(rf_rows: list[dict], ae_rows: list[dict], frozen_baseline: dict,
                     frozen_ae: dict) -> tuple[list[dict], dict]:
    def key(row: dict) -> tuple[str, str, str, str]:
        return row["condition"], row["split"], row["episode_id"], row["true_label"]

    rf_by_key = {key(row): row for row in rf_rows}
    ae_by_key = {key(row): row for row in ae_rows}
    if len(rf_by_key) != len(rf_rows) or len(ae_by_key) != len(ae_rows) or rf_by_key.keys() != ae_by_key.keys():
        raise ValueError("duplicate or missing episode/variant prediction keys")
    allowed_splits = {(condition, split) for condition in CONDITIONS for split in SPLITS}
    if any((row["condition"], row["split"]) not in allowed_splits for row in rf_rows):
        raise ValueError("prediction row has an unknown condition or split")
    combined = []
    for item in rf_rows:
        other = ae_by_key[key(item)]
        if item["rf_predicted"] != other["refit_rf_predicted"]:
            raise ValueError(f"refitted RF prediction differs: {key(item)}")
        total = 0.0
        for label in LABELS:
            probability = item[f"rf_prob_{label}"]
            refit_probability = other[f"refit_rf_prob_{label}"]
            if (not isfinite(probability) or not isfinite(refit_probability) or
                    not 0 <= probability <= 1 or not 0 <= refit_probability <= 1):
                raise ValueError(f"invalid RF probability: {key(item)}/{label}")
            if abs(probability - refit_probability) > 1e-12:
                raise ValueError(f"refitted RF probability differs: {key(item)}/{label}")
            total += probability
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"RF probabilities do not sum to one: {key(item)}")
        score = other["ae_score"]
        if not isfinite(score):
            raise ValueError(f"invalid AE score: {key(item)}")
        active = score > frozen_ae["threshold"]
        binary = "intervention" if active else "background"
        gated = ("background" if not active else
                 ("synthetic_attack" if item["rf_prob_synthetic_attack"] >
                  item["rf_prob_synthetic_drift"] else "synthetic_drift"))
        if other["ae_binary_predicted"] != binary or other["gated_predicted"] != gated:
            raise ValueError(f"AE binary or gated prediction differs from score and RF probabilities: {key(item)}")
        combined.append({**item, "ae_score": other["ae_score"],
                         "ae_binary_predicted": other["ae_binary_predicted"],
                         "gated_predicted": other["gated_predicted"]})
    comparisons = {}
    for condition in CONDITIONS:
        comparisons[condition] = {}
        for split in SPLITS:
            selected = [row for row in combined if row["condition"] == condition and row["split"] == split]
            expected_count = frozen_baseline["conditions"][condition][split]["random_forest"]["denominator_episodes"]
            if len(selected) != expected_count:
                raise ValueError(f"{condition}/{split}: {len(selected)} exported rows; expected {expected_count}")
            matrices = {
                "random_forest": matrix(selected, "rf_predicted"),
                "autoencoder_only_binary": matrix(selected, "ae_binary_predicted", binary=True),
                "autoencoder_plus_forest": matrix(selected, "gated_predicted"),
            }
            for model, source in (("random_forest", frozen_baseline),
                                  ("autoencoder_only_binary", frozen_ae),
                                  ("autoencoder_plus_forest", frozen_ae)):
                entry = source["conditions"][condition][split][model]
                labels = BINARY_LABELS if model == "autoencoder_only_binary" else LABELS
                if entry["labels"] != list(labels) or entry["denominator_episodes"] != len(selected):
                    raise ValueError(f"{condition}/{split}/{model}: frozen label order or count differs")
                expected = entry["confusion_rows_true_columns_predicted"]
                if matrices[model] != expected:
                    raise ValueError(f"{condition}/{split}/{model}: export matrix {matrices[model]} != frozen {expected}")
            comparisons[condition][split] = {"episodes": len(selected), "matrices": matrices,
                                            "frozen_match": True}
    return combined, comparisons


def publish_exports(output: Path, archive_sha256: str, rows: list[dict], comparisons: dict) -> None:
    """Publish the CSV last, after both complete payloads have been written."""
    if not rows:
        raise ValueError("no prediction rows to export")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError("prediction rows have inconsistent fields")
    csv_temp = output / ".episode_predictions.csv.tmp"
    comparison_temp = output / ".matrix_comparison.json.tmp"
    with csv_temp.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with comparison_temp.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"archive_sha256": archive_sha256, "rows": len(rows),
                                 "conditions": comparisons}, indent=2, sort_keys=True) + "\n")
    comparison_temp.replace(output / "matrix_comparison.json")
    csv_temp.replace(output / "episode_predictions.csv")


def reproduce(archive: Path, output: Path) -> None:
    resolved = output.resolve()
    if resolved.is_relative_to(FROZEN.resolve()) or output.exists():
        raise ValueError("output must be a new directory outside frozen reports/")
    problems = prerequisites(archive)
    if problems:
        raise ValueError("reproduction prerequisites failed: " + "; ".join(problems))
    output.mkdir(parents=True)
    for name in EVIDENCE:
        shutil.copyfile(FROZEN / name, output / name)
    sys.path.insert(0, str(ROOT / "src"))
    from drift_attack.ml_baselines import run as run_baseline
    from drift_attack.autoencoder_stage import run as run_ae

    frozen_baseline = json.loads((FROZEN / "ml_baselines.json").read_text(encoding="utf-8"))
    frozen_ae = json.loads((FROZEN / "ml_autoencoder.json").read_text(encoding="utf-8"))
    rf_rows: list[dict] = []
    baseline = run_baseline(archive, output, prediction_rows=rf_rows)
    generated_preflight = json.loads((output / "ml_preflight.json").read_text(encoding="utf-8"))
    frozen_preflight = json.loads((FROZEN / "ml_preflight.json").read_text(encoding="utf-8"))
    if generated_preflight != frozen_preflight:
        raise ValueError("generated preflight differs from frozen evidence")
    for field in ("input_groups", "input_width", "feature_scale_fit_batches", "model_fit_batches",
                  "selection_batches", "exploratory_evaluation_batches"):
        if baseline[field] != frozen_baseline[field]:
            raise ValueError(f"baseline {field} differs from frozen result")
    if baseline["anomaly_only"]["parameters"] != frozen_baseline["anomaly_only"]["parameters"]:
        raise ValueError("Isolation Forest settings differ from frozen result")
    if baseline["random_forest"]["selected_settings"] != frozen_baseline["random_forest"]["selected_settings"]:
        raise ValueError("selected RF settings differ from frozen result")
    if baseline["anomaly_only"]["threshold"] != frozen_baseline["anomaly_only"]["threshold"]:
        raise ValueError("Isolation Forest threshold differs from frozen result")
    compare_reports(baseline, frozen_baseline, ("anomaly_only", "random_forest"), "baseline")
    ae_rows: list[dict] = []
    autoencoder = run_ae(archive, output, prediction_rows=ae_rows)
    for field in ("seed", "architecture", "objective", "normalization", "gate",
                  "selection_batches", "exploratory_evaluation_batches", "final_fit_background_episodes"):
        if autoencoder[field] != frozen_ae[field]:
            raise ValueError(f"AE {field} differs from frozen result")
    if autoencoder["threshold"] != frozen_ae["threshold"] or autoencoder["early_stop"]["best_epoch"] != frozen_ae["early_stop"]["best_epoch"]:
        raise ValueError("AE threshold or selected epoch differs from frozen result")
    compare_reports(autoencoder, frozen_ae, ("autoencoder_only_binary", "autoencoder_plus_forest"), "AE")
    rows, comparisons = join_predictions(rf_rows, ae_rows, frozen_baseline, frozen_ae)
    problems = frozen_report_problems()
    if problems:
        raise ValueError("frozen reports changed during isolated run: " + "; ".join(problems))
    publish_exports(output, digest(archive), rows, comparisons)
    print(f"Exact frozen matrix matches; exported {len(rows)} episode variants to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/raw/uci-270.zip")
    parser.add_argument("--output", type=Path, help="new isolated directory; required unless --check-only")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    problems = prerequisites(args.archive)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        raise SystemExit(2)
    if args.check_only:
        print("Pinned archive, Python 3.13 environment, packages, and frozen reports: ready")
        return
    if args.output is None:
        parser.error("--output is required for reproduction")
    reproduce(args.archive, args.output)


if __name__ == "__main__":
    main()
