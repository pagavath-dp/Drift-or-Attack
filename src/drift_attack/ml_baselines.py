"""Train the approved anomaly-only and Random Forest baselines, then evaluate once."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

from .audit import APPROVED_ARCHIVE_SHA256, file_sha256, reject_frozen_output
from .dataset import load_archive
from .episodes import (GAIN_LEVELS, construct_episodes, fit_generator_scales,
                       fit_magnitude_calibration, generate_variants)
from .features import FEATURE_GROUPS, INPUT_WIDTH, episode_features, fit_feature_scale
from .model_preflight import inspect as run_preflight


LABELS = ("background", "synthetic_drift", "synthetic_attack")
BINARY_LABELS = ("background", "intervention")
RF_GRID = ((8, 1), (8, 5), (None, 1), (None, 5))


def _rate(numerator: int, denominator: int) -> dict:
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def _classification_metrics(true: list[str], predicted: list[str]) -> dict:
    matrix = confusion_matrix(true, predicted, labels=LABELS)
    result = {
        "labels": list(LABELS),
        "denominator_episodes": len(true),
        "confusion_rows_true_columns_predicted": matrix.tolist(),
        "macro_f1": float(f1_score(true, predicted, labels=LABELS, average="macro", zero_division=0)),
        "per_class": {},
    }
    for index, label in enumerate(LABELS):
        tp = int(matrix[index, index])
        support = int(matrix[index, :].sum())
        predicted_count = int(matrix[:, index].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result["per_class"][label] = {
            "support": support, "predicted": predicted_count,
            "precision": precision, "recall": recall, "f1": f1,
        }
    bg, drift, attack = 0, 1, 2
    result["rates"] = {
        "attack_false_alarm": _rate(int(matrix[bg, attack]), int(matrix[bg, :].sum())),
        "any_intervention_false_alarm": _rate(int(matrix[bg, drift] + matrix[bg, attack]),
                                              int(matrix[bg, :].sum())),
        "drift_mistaken_for_attack": _rate(int(matrix[drift, attack]), int(matrix[drift, :].sum())),
        "attack_mistaken_for_background_or_drift": _rate(int(matrix[attack, bg] + matrix[attack, drift]),
                                                          int(matrix[attack, :].sum())),
    }
    return result


def _binary_metrics(true: list[str], predicted: list[str], original_labels: list[str]) -> dict:
    matrix = confusion_matrix(true, predicted, labels=BINARY_LABELS)
    result = {
        "labels": list(BINARY_LABELS),
        "denominator_episodes": len(true),
        "confusion_rows_true_columns_predicted": matrix.tolist(),
        "macro_f1": float(f1_score(true, predicted, labels=BINARY_LABELS, average="macro", zero_division=0)),
        "per_class": {},
    }
    for index, label in enumerate(BINARY_LABELS):
        tp = int(matrix[index, index])
        support = int(matrix[index, :].sum())
        predicted_count = int(matrix[:, index].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        result["per_class"][label] = {"support": support, "predicted": predicted_count,
                                      "precision": precision, "recall": recall}
    result["rates"] = {
        "any_intervention_false_alarm": _rate(int(matrix[0, 1]), int(matrix[0, :].sum())),
        "synthetic_drift_detection": _rate(sum(p == "intervention" for y, p in zip(original_labels, predicted)
                                               if y == "synthetic_drift"),
                                           original_labels.count("synthetic_drift")),
        "synthetic_attack_detection": _rate(sum(p == "intervention" for y, p in zip(original_labels, predicted)
                                                if y == "synthetic_attack"),
                                            original_labels.count("synthetic_attack")),
    }
    return result


def _binary_macro_f1_at_threshold(scores: np.ndarray, truth: np.ndarray, threshold: float) -> float:
    predicted = scores > threshold
    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    tn = int(np.sum(~predicted & ~truth))
    positive = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    negative = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return (positive + negative) / 2


def select_anomaly_threshold(validation_scores: np.ndarray, validation_labels: list[str]) -> tuple[float, float]:
    """Maximize validation binary macro-F1; higher threshold wins an exact tie."""
    scores = np.asarray(validation_scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) != len(validation_labels) or not np.isfinite(scores).all():
        raise ValueError("invalid validation scores")
    truth = np.asarray([label != "background" for label in validation_labels], dtype=bool)
    unique = np.unique(scores)
    candidates = np.concatenate(([-np.inf], (unique[:-1] + unique[1:]) / 2, [np.inf]))
    return max(((float(threshold), _binary_macro_f1_at_threshold(scores, truth, float(threshold)))
                for threshold in candidates), key=lambda item: (item[1], item[0]))


def _build_datasets(batches: dict, condition: str, scale, generator_scales, calibration) -> dict:
    episodes, _ = construct_episodes(batches)
    rows: dict[str, list] = {split: [] for split in ("train", "validation", "test")}
    for episode in episodes:
        if episode.batch == 1:
            continue
        variants = generate_variants(episode, generator_scales, bias_condition=condition,
                                     calibration=calibration if condition == "matched_primary" else None)
        level = episode.rotation_index % len(GAIN_LEVELS)
        for variant in variants:
            rows[episode.split].append({
                "features": episode_features(variant.rows, scale),
                "label": variant.label,
                "batch": episode.batch,
                "episode_id": episode.episode_id,
                "gas_code": episode.gas,
                "concentration": episode.concentration,
                "family": variant.family,
                "level": level,
                "severity": variant.severity,
            })
    return {split: {"X": np.vstack([row["features"] for row in members]),
                    "labels": [row["label"] for row in members],
                    "metadata": [{key: value for key, value in row.items() if key != "features"}
                                 for row in members]}
            for split, members in rows.items()}


def _breakdowns(true: list[str], predicted: list[str], metadata: list[dict],
                binary: bool, condition: str) -> dict:
    by_batch: dict[int, list[int]] = defaultdict(list)
    by_family_severity: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        by_batch[row["batch"]].append(index)
        if row["label"] != "background":
            severity = (f"gain_level={GAIN_LEVELS[row['level']]}" if row["family"] == "bias" and
                        condition == "matched_primary"
                        else f"severity={row['severity']}")
            key = f"{row['family']}:{severity}"
            by_family_severity[key].append(index)
    by_batch_result = {}
    for batch, indices in sorted(by_batch.items()):
        selected_true = [true[index] for index in indices]
        selected_predicted = [predicted[index] for index in indices]
        if binary:
            selected_binary = ["background" if label == "background" else "intervention"
                               for label in selected_true]
            by_batch_result[str(batch)] = _binary_metrics(selected_binary, selected_predicted, selected_true)
        else:
            by_batch_result[str(batch)] = _classification_metrics(selected_true, selected_predicted)
    family_result = {}
    for key, indices in sorted(by_family_severity.items()):
        family_result[key] = {
            "denominator_episodes": len(indices),
            "correct": sum(predicted[index] == ("intervention" if binary else true[index]) for index in indices),
            "predicted_counts": dict(Counter(predicted[index] for index in indices)),
        }
        family_result[key]["recall"] = family_result[key]["correct"] / len(indices)
    return {"per_batch": by_batch_result, "per_family_severity": family_result}


def run(archive: Path, reports: Path, *, prediction_rows: list[dict] | None = None) -> dict:
    reject_frozen_output(reports)
    preflight = run_preflight(archive, reports)  # mandatory before any fit
    if file_sha256(archive) != APPROVED_ARCHIVE_SHA256:
        raise ValueError("unapproved source archive")
    batches = load_archive(archive)
    scale = fit_feature_scale(batches)
    generator_scales = fit_generator_scales(batches)
    episodes, _ = construct_episodes(batches)
    calibration = fit_magnitude_calibration(episodes, generator_scales)
    if calibration.digest != preflight["calibration_digest"]:
        raise ValueError("preflight calibration mismatch")
    primary = _build_datasets(batches, "matched_primary", scale, generator_scales, calibration)
    stress = _build_datasets(batches, "original_stress", scale, generator_scales, calibration)
    for condition, data in (("matched_primary", primary), ("original_stress", stress)):
        for split, values in data.items():
            if values["X"].shape[1] != INPUT_WIDTH or not np.isfinite(values["X"]).all():
                raise ValueError(f"invalid model input: {condition}/{split}")
    if not np.array_equal(primary["train"]["X"][::3], stress["train"]["X"][::3]):
        raise ValueError("condition backgrounds differ")
    training = primary["train"]
    validation = primary["validation"]
    background_mask = np.asarray([label == "background" for label in training["labels"]])
    anomaly = IsolationForest(n_estimators=200, max_samples="auto", contamination="auto",
                              random_state=1729, n_jobs=4)
    anomaly.fit(training["X"][background_mask])
    validation_scores = -anomaly.score_samples(validation["X"])
    threshold, selection_score = select_anomaly_threshold(validation_scores, validation["labels"])
    rf_candidates = []
    best_model = None
    best_score = -1.0
    best_settings = None
    for depth, leaf in RF_GRID:
        model = RandomForestClassifier(n_estimators=200, max_depth=depth, min_samples_leaf=leaf,
                                       max_features="sqrt", random_state=1729, n_jobs=4)
        model.fit(training["X"], training["labels"])
        score = float(f1_score(validation["labels"], model.predict(validation["X"]),
                               labels=LABELS, average="macro", zero_division=0))
        settings = {"n_estimators": 200, "max_depth": depth, "min_samples_leaf": leaf,
                    "max_features": "sqrt", "random_state": 1729, "n_jobs": 4}
        rf_candidates.append({"settings": settings, "validation_macro_f1": score})
        if score > best_score:
            best_model, best_score, best_settings = model, score, settings
    result = {
        "archive_sha256": APPROVED_ARCHIVE_SHA256,
        "calibration_digest": calibration.digest,
        "python_packages": {name: version(name) for name in ("numpy", "scipy", "scikit-learn")},
        "input_groups": list(FEATURE_GROUPS), "input_width": INPUT_WIDTH,
        "feature_scale_fit_batches": [1, 2, 3, 4],
        "model_fit_batches": [2, 3, 4],
        "selection_batches": [5, 6],
        "exploratory_evaluation_batches": [7, 8, 9, 10],
        "anomaly_only": {"model": "IsolationForest", "fit_class": "background",
                         "threshold": threshold, "validation_binary_macro_f1_at_selection": selection_score,
                         "parameters": {"n_estimators": 200, "max_samples": "auto",
                                        "contamination": "auto", "random_state": 1729, "n_jobs": 4}},
        "random_forest": {"selected_settings": best_settings,
                          "validation_macro_f1_at_selection": best_score,
                          "candidate_validation_results": rf_candidates},
        "conditions": {},
        "limitations": preflight["limitations"],
    }
    for condition, data in (("matched_primary", primary), ("original_stress", stress)):
        condition_results = {}
        for split in ("validation", "test"):
            values = data[split]
            true = values["labels"]
            binary_true = ["background" if label == "background" else "intervention" for label in true]
            scores = -anomaly.score_samples(values["X"])
            anomaly_predicted = ["intervention" if score > threshold else "background" for score in scores]
            rf_predicted = best_model.predict(values["X"]).tolist()
            if prediction_rows is not None:
                probabilities = best_model.predict_proba(values["X"])
                class_index = {label: index for index, label in enumerate(best_model.classes_)}
                for metadata, label, predicted, probability in zip(
                        values["metadata"], true, rf_predicted, probabilities, strict=True):
                    prediction_rows.append({
                        "condition": condition, "split": split,
                        "episode_id": metadata["episode_id"], "batch": metadata["batch"],
                        "gas_code": metadata["gas_code"],
                        "concentration": metadata["concentration"],
                        "family": metadata["family"], "true_label": label,
                        "rf_predicted": predicted,
                        **{f"rf_prob_{name}": float(probability[class_index[name]])
                           for name in LABELS},
                    })
            anomaly_metrics = _binary_metrics(binary_true, anomaly_predicted, true)
            anomaly_metrics.update(_breakdowns(true, anomaly_predicted, values["metadata"],
                                               binary=True, condition=condition))
            rf_metrics = _classification_metrics(true, rf_predicted)
            rf_metrics.update(_breakdowns(true, rf_predicted, values["metadata"],
                                          binary=False, condition=condition))
            condition_results[split] = {"anomaly_only": anomaly_metrics, "random_forest": rf_metrics}
        result["conditions"][condition] = condition_results
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "ml_baselines.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (reports / "ml_baselines.md").write_text(_markdown(result), encoding="utf-8")
    return result


def _markdown(data: dict) -> str:
    lines = [
        "# Anomaly-only and Random Forest results",
        "",
        "These are results for artificial interventions on extracted UCI features. They do not establish actual sensor aging or real cyberattack detection. Batches 7–10 are exploratory because their audit distributions were examined before protocol freeze.",
        "",
        f"- Archive: `{data['archive_sha256']}`; calibration: `{data['calibration_digest']}`.",
        f"- Inputs: {data['input_width']} numeric values from {', '.join(data['input_groups'])}. Laboratory and generator metadata are excluded.",
        f"- Preprocessing fits Batches 1–4; models fit Batches 2–4; threshold and Random Forest setting selection use Batches 5–6 only.",
        f"- Anomaly threshold: {data['anomaly_only']['threshold']:.7g}; selected Random Forest settings: `{data['random_forest']['selected_settings']}`.",
        "",
        "| Condition | Split | Model | Macro-F1 | True-label denominators |",
        "|---|---|---|---:|---|",
    ]
    for condition, splits in data["conditions"].items():
        for split, models in splits.items():
            for name, metrics in models.items():
                if name == "anomaly_only":
                    support_text = (f"background {metrics['per_class']['background']['support']}; "
                                    f"intervention {metrics['per_class']['intervention']['support']}")
                else:
                    support_text = "; ".join(f"{label} {metrics['per_class'][label]['support']}" for label in LABELS)
                lines.append(f"| {condition} | {split} | {name} | {metrics['macro_f1']:.4f} | {support_text} |")
    lines += [
        "",
        "The anomaly baseline is binary (`background` versus any synthetic intervention). Its binary macro-F1 is not directly comparable with the Random Forest's three-class macro-F1. Its directional drift/attack error rates are undefined.",
        "",
        "## Precision, recall, and confusion matrices",
        "",
        "Confusion rows are true labels; columns follow the label order printed in each row below.",
        "",
        "| Condition | Split | Model | Class | Precision | Recall | Support |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for condition, splits in data["conditions"].items():
        for split, models in splits.items():
            for name, metrics in models.items():
                for label in metrics["labels"]:
                    values = metrics["per_class"][label]
                    lines.append(f"| {condition} | {split} | {name} | {label} | {values['precision']:.3f} | {values['recall']:.3f} | {values['support']} |")
    lines += [
        "",
        "| Condition | Split | Model | Label order | Confusion rows |",
        "|---|---|---|---|---|",
    ]
    for condition, splits in data["conditions"].items():
        for split, models in splits.items():
            for name, metrics in models.items():
                lines.append(f"| {condition} | {split} | {name} | {', '.join(metrics['labels'])} | `{metrics['confusion_rows_true_columns_predicted']}` |")
    lines += [
        "",
        "## Three-class directional rates",
        "",
        "| Condition | Split | Attack false alarm | Any-intervention false alarm | Drift → attack | Attack → background/drift |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for condition, splits in data["conditions"].items():
        for split, models in splits.items():
            rates = models["random_forest"]["rates"]
            rendered = [f"{rates[key]['numerator']}/{rates[key]['denominator']} ({rates[key]['rate']:.1%})"
                        for key in ("attack_false_alarm", "any_intervention_false_alarm",
                                    "drift_mistaken_for_attack", "attack_mistaken_for_background_or_drift")]
            lines.append(f"| {condition} | {split} | {' | '.join(rendered)} |")
    lines += [
        "",
        "The binary anomaly-only background false-alarm denominators are:",
        "",
        "| Condition | Split | Background flagged as intervention | Drift detected | Attack detected |",
        "|---|---|---:|---:|---:|",
    ]
    for condition, splits in data["conditions"].items():
        for split, models in splits.items():
            rates = models["anomaly_only"]["rates"]
            rendered = [f"{rates[key]['numerator']}/{rates[key]['denominator']} ({rates[key]['rate']:.1%})"
                        for key in ("any_intervention_false_alarm", "synthetic_drift_detection",
                                    "synthetic_attack_detection")]
            lines.append(f"| {condition} | {split} | {' | '.join(rendered)} |")
    lines += [
        "",
        "## Per-batch performance",
        "",
        "| Condition | Split | Batch | Model | Macro-F1 | Background false alarms / background |",
        "|---|---|---:|---|---:|---:|",
    ]
    for condition, splits in data["conditions"].items():
        for split, models in splits.items():
            for name, metrics in models.items():
                for batch, batch_metrics in sorted(metrics["per_batch"].items(), key=lambda item: int(item[0])):
                    far = batch_metrics["rates"]["any_intervention_false_alarm"]
                    lines.append(f"| {condition} | {split} | {batch} | {name} | {batch_metrics['macro_f1']:.3f} | {far['numerator']}/{far['denominator']} ({far['rate']:.1%}) |")
    lines += [
        "",
        "## Exploratory test recall by intervention and severity",
        "",
        "| Condition | Model | Family and severity | Correct / episodes | Recall |",
        "|---|---|---|---:|---:|",
    ]
    for condition, splits in data["conditions"].items():
        for name, metrics in splits["test"].items():
            for family, values in metrics["per_family_severity"].items():
                lines.append(f"| {condition} | {name} | {family} | {values['correct']}/{values['denominator_episodes']} | {values['recall']:.3f} |")
    lines += [
        "",
        "The exploratory primary test background recall is low, and both models raise many background alarms. These results do not support reliable deployment or attribution of drift versus attack.",
        "",
        "`ml_baselines.json` contains all confusion matrices, per-class precision and recall, binary baseline rates, per-batch metrics, family/severity recall with denominators, and the full validation selection grid. `ml_preflight.json` contains finite-pool reuse and paired magnitude mismatch by split, family, and severity.",
        "",
        "## Scientific limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in data["limitations"])
    lines += [
        "- The same primary-trained models and primary-selected threshold are applied to the original-stress condition; stress results are a separate sensitivity analysis.",
        "- Episode order is seeded and artificial. No real-time detection-delay claim is made.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.archive, args.reports)
    print("Selected RF validation macro-F1:", result["random_forest"]["validation_macro_f1_at_selection"])


if __name__ == "__main__":
    main()
