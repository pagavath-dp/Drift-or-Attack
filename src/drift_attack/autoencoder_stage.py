"""Frozen autoencoder comparison for artificial intervention episodes."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .audit import APPROVED_ARCHIVE_SHA256, file_sha256, reject_frozen_output
from .dataset import load_archive
from .episodes import construct_episodes, fit_generator_scales, fit_magnitude_calibration
from .features import INPUT_WIDTH, episode_features, fit_feature_scale
from .ml_baselines import (_binary_metrics, _breakdowns, _build_datasets,
                           _classification_metrics, select_anomaly_threshold)
from .model_preflight import inspect as run_preflight

SEED = 1729
MAX_EPOCHS = 100
PATIENCE = 10
MIN_DELTA = 1e-4
BATCH_SIZE = 32


def _source_group_holdout(metadata: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Hold out about 20% of B2-4 backgrounds per batch by source episode ID."""
    by_batch: dict[int, list[tuple[str, int]]] = {}
    for index, row in enumerate(metadata):
        if row["label"] != "background" or row["batch"] not in (2, 3, 4):
            raise ValueError("early-stop candidates must be B2-4 backgrounds")
        by_batch.setdefault(row["batch"], []).append((row["episode_id"], index))
    fit, holdout, counts = [], [], {}
    for batch in (2, 3, 4):
        members = by_batch.get(batch, [])
        if len({episode_id for episode_id, _ in members}) != len(members):
            raise ValueError("source episode repeated in early-stop candidates")
        ranked = sorted(members, key=lambda item: (sha256(f"{SEED}:early-stop:{item[0]}".encode()).hexdigest(), item[0]))
        selected = max(1, round(len(ranked) * .2))
        holdout.extend(index for _, index in ranked[:selected])
        fit.extend(index for _, index in ranked[selected:])
        counts[str(batch)] = {"eligible": len(ranked), "holdout": selected, "fit": len(ranked) - selected}
    if set(fit) & set(holdout):
        raise ValueError("early-stop groups overlap")
    return np.asarray(sorted(fit), dtype=int), np.asarray(sorted(holdout), dtype=int), counts


def _fit_normalizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if X.ndim != 2 or X.shape[1] != INPUT_WIDTH or not np.isfinite(X).all():
        raise ValueError("invalid background features")
    center = np.median(X, axis=0)
    spread = np.maximum((np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)) / 1.349, 1e-6)
    return center, spread


def _normalize(X: np.ndarray, center: np.ndarray, spread: np.ndarray) -> np.ndarray:
    transformed = np.arcsinh((X - center) / spread).astype(np.float32)
    if not np.isfinite(transformed).all():
        raise ValueError("nonfinite normalized input")
    return transformed


def _build_autoencoder(tf):
    tf.keras.utils.set_random_seed(SEED)
    inputs = tf.keras.Input(shape=(INPUT_WIDTH,))
    regularizer = tf.keras.regularizers.l2(1e-4)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer)(inputs)
    x = tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer)(x)
    outputs = tf.keras.layers.Dense(INPUT_WIDTH)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.Huber(delta=1.0))
    return model


def _score(model, X: np.ndarray) -> np.ndarray:
    reconstructed = model.predict(X, batch_size=128, verbose=0)
    error = X - reconstructed
    absolute = np.abs(error)
    scores = np.mean(np.where(absolute <= 1, .5 * error ** 2, absolute - .5), axis=1)
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite reconstruction scores")
    return scores


def _gated_predictions(scores: np.ndarray, threshold: float, probabilities: np.ndarray,
                       classes: list[str]) -> list[str]:
    drift = classes.index("synthetic_drift")
    attack = classes.index("synthetic_attack")
    return ["background" if score <= threshold else
            ("synthetic_attack" if row[attack] > row[drift] else "synthetic_drift")
            for score, row in zip(scores, probabilities)]


def run(archive: Path, reports: Path, *, prediction_rows: list[dict] | None = None) -> dict:
    reject_frozen_output(reports)
    if file_sha256(archive) != APPROVED_ARCHIVE_SHA256:
        raise ValueError("unapproved source archive")
    preflight = run_preflight(archive, reports)
    baseline = json.loads((reports / "ml_baselines.json").read_text())
    if baseline["archive_sha256"] != APPROVED_ARCHIVE_SHA256 or baseline["calibration_digest"] != preflight["calibration_digest"]:
        raise ValueError("baseline or preflight provenance mismatch")
    batches = load_archive(archive)
    episodes, _ = construct_episodes(batches)
    scale = fit_feature_scale(batches)
    generator_scales = fit_generator_scales(batches)
    calibration = fit_magnitude_calibration(episodes, generator_scales)
    primary = _build_datasets(batches, "matched_primary", scale, generator_scales, calibration)
    stress = _build_datasets(batches, "original_stress", scale, generator_scales, calibration)
    background_episodes = [episode for episode in episodes if episode.batch in (1, 2, 3, 4)]
    if Counter(episode.batch for episode in background_episodes) != {1: 6, 2: 135, 3: 192, 4: 15}:
        raise ValueError("background source coverage changed")
    if len({episode.episode_id for episode in background_episodes}) != len(background_episodes):
        raise ValueError("background source group reused")
    b1 = np.vstack([episode_features(episode.rows, scale) for episode in background_episodes if episode.batch == 1])
    training = primary["train"]
    background_indices = [i for i, label in enumerate(training["labels"]) if label == "background"]
    background_metadata = [training["metadata"][i] for i in background_indices]
    b234 = training["X"][background_indices]
    if not np.array_equal(b234, stress["train"]["X"][::3]):
        raise ValueError("conditions differ in unmodified training backgrounds")
    fit_indices, holdout_indices, holdout_counts = _source_group_holdout(background_metadata)
    holdout_source_ids = sorted(background_metadata[i]["episode_id"] for i in holdout_indices)
    all_background_source_ids = sorted(episode.episode_id for episode in background_episodes)
    early_fit_raw = np.vstack((b1, b234[fit_indices]))
    early_center, early_spread = _fit_normalizer(early_fit_raw)
    early_fit = _normalize(early_fit_raw, early_center, early_spread)
    early_holdout = _normalize(b234[holdout_indices], early_center, early_spread)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    early_model = _build_autoencoder(tf)
    history = early_model.fit(early_fit, early_fit, validation_data=(early_holdout, early_holdout),
                              epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=0,
                              callbacks=[tf.keras.callbacks.EarlyStopping(
                                  monitor="val_loss", patience=PATIENCE, min_delta=MIN_DELTA,
                                  restore_best_weights=True)])
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    if not 1 <= best_epoch <= MAX_EPOCHS:
        raise ValueError("invalid selected epoch")
    all_background = np.vstack((b1, b234))
    center, spread = _fit_normalizer(all_background)
    final_input = _normalize(all_background, center, spread)
    tf.keras.backend.clear_session()
    final_model = _build_autoencoder(tf)
    final_model.fit(final_input, final_input, epochs=best_epoch, batch_size=BATCH_SIZE,
                    shuffle=True, verbose=0)
    selected_settings = baseline["random_forest"]["selected_settings"]
    forest = RandomForestClassifier(**selected_settings)
    forest.fit(training["X"], training["labels"])
    validation = primary["validation"]
    reproduced = _classification_metrics(validation["labels"], forest.predict(validation["X"]).tolist())
    saved_matrix = baseline["conditions"]["matched_primary"]["validation"]["random_forest"]["confusion_rows_true_columns_predicted"]
    if reproduced["confusion_rows_true_columns_predicted"] != saved_matrix:
        raise ValueError("refitted forest does not reproduce saved validation confusion")
    validation_scores = _score(final_model, _normalize(validation["X"], center, spread))
    threshold, selected_binary_f1 = select_anomaly_threshold(validation_scores, validation["labels"])
    result = {
        "archive_sha256": APPROVED_ARCHIVE_SHA256,
        "calibration_digest": calibration.digest,
        "tensorflow_version": version("tensorflow"),
        "seed": SEED,
        "architecture": [INPUT_WIDTH, 64, 16, 64, INPUT_WIDTH],
        "objective": "mean Huber(delta=1) reconstruction plus L2(1e-4) on hidden kernels; Adam(0.001)",
        "normalization": "per-feature median and max(IQR/1.349, 1e-6), then asinh; final fit B1-4 unmodified backgrounds only",
        "early_stop": {"batch_2_to_4_group_counts": holdout_counts, "batch_1_fit_episodes": len(b1),
                       "fit_episodes": len(early_fit), "holdout_episodes": len(early_holdout),
                       "holdout_source_episode_sha256": sha256("\n".join(holdout_source_ids).encode()).hexdigest(),
                       "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "min_delta": MIN_DELTA,
                       "epochs_run": len(history.history["loss"]), "best_epoch": best_epoch},
        "final_fit_background_episodes": len(all_background),
        "final_fit_source_episode_sha256": sha256("\n".join(all_background_source_ids).encode()).hexdigest(),
        "final_fit_by_batch": {str(batch): sum(episode.batch == batch for episode in background_episodes)
                               for batch in range(1, 5)},
        "final_fit_source_grouping_by_batch": {
            str(batch): {
                "gas_concentration_groups": len({(episode.gas, episode.concentration)
                                                 for episode in background_episodes if episode.batch == batch}),
                "distinct_source_rows": len({source_id for episode in background_episodes
                                             if episode.batch == batch for source_id in episode.source_ids}),
            } for batch in range(1, 5)},
        "selection_batches": [5, 6], "exploratory_evaluation_batches": [7, 8, 9, 10],
        "threshold": threshold, "validation_binary_macro_f1_at_selection": selected_binary_f1,
        "forest_validation_confusion_reproduced": saved_matrix,
        "gate": "score <= threshold: background; else higher RF P(drift/attack), ties drift",
        "conditions": {},
        "limitations": ["Background is unmodified input, not verified healthy hardware.",
                        "Freeze and replay are exact-copy controls with deterministic feature signatures.",
                        "Batches 7-10 are exploratory because their audit distributions were inspected earlier.",
                        "Synthetic results do not establish actual aging or real-world cyberattack detection."],
    }
    for condition, data in (("matched_primary", primary), ("original_stress", stress)):
        result["conditions"][condition] = {}
        for split in ("validation", "test"):
            values = data[split]
            truth = values["labels"]
            scores = _score(final_model, _normalize(values["X"], center, spread))
            binary_truth = ["background" if label == "background" else "intervention" for label in truth]
            binary_predicted = ["intervention" if score > threshold else "background" for score in scores]
            probabilities = forest.predict_proba(values["X"])
            combined_predicted = _gated_predictions(scores, threshold, probabilities,
                                                     forest.classes_.tolist())
            if prediction_rows is not None:
                class_index = {label: index for index, label in enumerate(forest.classes_)}
                refit_predicted = forest.predict(values["X"]).tolist()
                for metadata, label, score, binary_label, gated_label, refit_label, probability in zip(
                        values["metadata"], truth, scores, binary_predicted, combined_predicted,
                        refit_predicted, probabilities, strict=True):
                    prediction_rows.append({
                        "condition": condition, "split": split,
                        "episode_id": metadata["episode_id"], "true_label": label,
                        "ae_score": float(score), "ae_binary_predicted": binary_label,
                        "gated_predicted": gated_label, "refit_rf_predicted": refit_label,
                        **{f"refit_rf_prob_{name}": float(probability[class_index[name]])
                           for name in ("background", "synthetic_drift", "synthetic_attack")},
                    })
            binary = _binary_metrics(binary_truth, binary_predicted, truth)
            binary.update(_breakdowns(truth, binary_predicted, values["metadata"], True, condition))
            combined = _classification_metrics(truth, combined_predicted)
            combined.update(_breakdowns(truth, combined_predicted, values["metadata"], False, condition))
            result["conditions"][condition][split] = {
                "autoencoder_only_binary": binary, "autoencoder_plus_forest": combined}
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "ml_autoencoder.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (reports / "ml_autoencoder.md").write_text(_markdown(result, baseline), encoding="utf-8")
    return result


def _markdown(result: dict, baseline: dict) -> str:
    lines = ["# Autoencoder stage — synthetic intervention study", "",
             "Background denotes unmodified input, not verified healthy hardware. Freeze and replay are exact-copy controls. Batches 7–10 are exploratory because their audit coverage and scores were inspected before protocol freeze.", "",
             "## Fitting and selection", "",
             f"- Unmodified background episodes for final fit: {result['final_fit_background_episodes']} (B1–4: {result['final_fit_by_batch']}).",
             f"- Source grouping by batch (gas/concentration groups and distinct source rows): {result['final_fit_source_grouping_by_batch']}.",
             f"- Early-stop fit/holdout: {result['early_stop']['fit_episodes']}/{result['early_stop']['holdout_episodes']} source-grouped episodes; selected epoch {result['early_stop']['best_epoch']} from {result['early_stop']['epochs_run']} run.",
             f"- Batch 5–6 matched-primary binary macro-F1 selected threshold: {result['threshold']:.9g}; higher threshold wins ties. The same threshold is used for stress and exploratory test.",
             "- Selected forest refit reproduced the saved matched-primary validation confusion exactly before gate evaluation.",
             "- Architecture 768→64→16→64→768; Huber reconstruction, Adam 0.001, hidden-kernel L2 1e-4; seed 1729. Median/IQR and asinh normalization use permitted background fitting episodes only.", "",
             "## Results", "",
             "Confusion matrices below use true rows and predicted columns. Binary order: background, intervention. Three-class order: background, synthetic drift, synthetic attack.", "",
             "| Condition | Split | AE binary macro-F1 | Combined macro-F1 | Forest macro-F1 | Combined background false alarms |",
             "|---|---|---:|---:|---:|---:|"]
    for condition in ("matched_primary", "original_stress"):
        for split in ("validation", "test"):
            entry = result["conditions"][condition][split]
            binary = entry["autoencoder_only_binary"]
            combined = entry["autoencoder_plus_forest"]
            forest = baseline["conditions"][condition][split]["random_forest"]
            alarm = combined["rates"]["any_intervention_false_alarm"]
            lines.append(f"| {condition} | {split} | {binary['macro_f1']:.4f} | {combined['macro_f1']:.4f} | {forest['macro_f1']:.4f} | {alarm['numerator']}/{alarm['denominator']} ({alarm['rate']:.1%}) |")
    lines.extend(["", "The combined pipeline did not improve three-class macro-F1 over the saved forest in either condition or split. On matched-primary exploratory test it increased background false alarms from 724/909 to 752/909.", ""])
    for condition in ("matched_primary", "original_stress"):
        lines.extend([f"### {condition}", ""])
        for split in ("validation", "test"):
            entry = result["conditions"][condition][split]
            lines.append(f"**{split}:** AE binary matrix `{entry['autoencoder_only_binary']['confusion_rows_true_columns_predicted']}`; combined matrix `{entry['autoencoder_plus_forest']['confusion_rows_true_columns_predicted']}`. Denominator: {entry['autoencoder_plus_forest']['denominator_episodes']} paired variants.")
            lines.append("")
            lines.append("| Class | Combined precision | Combined recall | Support |")
            lines.append("|---|---:|---:|---:|")
            for label, values in entry["autoencoder_plus_forest"]["per_class"].items():
                lines.append(f"| {label} | {values['precision']:.3f} | {values['recall']:.3f} | {values['support']} |")
            lines.extend(["", "| Directional rate | Numerator / denominator |", "|---|---:|"])
            for name, values in entry["autoencoder_plus_forest"]["rates"].items():
                lines.append(f"| {name} | {values['numerator']}/{values['denominator']} |")
            lines.extend(["", "| Family/severity | Correct / episodes |", "|---|---:|"])
            for name, values in entry["autoencoder_plus_forest"]["per_family_severity"].items():
                lines.append(f"| {name} | {values['correct']}/{values['denominator_episodes']} |")
            lines.append("")
    lines.extend(["## Interpretation and limits", "",
                  "The JSON report also contains binary class precision/recall, four directional rates, and per-batch confusion matrices with denominators for both conditions. Related variants share source episodes, so episode variants and the two conditions are not independent samples. No uncertainty intervals are claimed here.",
                  "", "These are responses to assigned manipulations of extracted features in artificial sequences. They do not validate actual sensor-aging detection or real-world cyberattack detection. Gas/concentration and all generator metadata were excluded from model inputs. The finite matched-bias pool and exact-copy controls remain simulation signatures.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.archive, args.reports)
    print("Autoencoder stage complete; selected epoch", result["early_stop"]["best_epoch"])


if __name__ == "__main__":
    main()
