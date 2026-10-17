"""Build final figures and tables from frozen audit and model reports only.

No dataset parsing, model fitting, prediction, or third-party package is used.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from xml.sax.saxutils import escape


CONDITIONS = ("matched_primary", "original_stress")
SPLITS = ("validation", "test")
BINARY_LABELS = ("background", "intervention")
THREE_LABELS = ("background", "synthetic_drift", "synthetic_attack")
MODELS = (
    ("binary", "Isolation Forest", "ml_baselines", "anomaly_only"),
    ("binary", "Autoencoder", "ml_autoencoder", "autoencoder_only_binary"),
    ("three_class", "Random Forest", "ml_baselines", "random_forest"),
    ("three_class", "AE + forest", "ml_autoencoder", "autoencoder_plus_forest"),
)
INPUTS = (
    "batch_summary.csv", "batch_gas_summary.csv", "source_manifest.json", "ml_baselines.json",
    "ml_autoencoder.json", "ml_preflight.json", "data_audit.md",
)
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
INK = "#18232D"
MUTED = "#53616C"
GRID = "#D9E1E5"


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual: float, expected: float, message: str) -> None:
    require(abs(float(actual) - expected) <= 1e-9, message)


def load_and_validate(reports: Path) -> tuple[dict, list[dict], dict]:
    """Read frozen reports and reject inconsistent provenance or saved metrics."""
    root = reports.parent
    paths = {name: reports / name for name in INPUTS}
    paths["experimental_protocol.md"] = root / "docs/experimental_protocol.md"
    for name, path in paths.items():
        require(path.is_file(), f"missing frozen input: {name}")
    source_hashes = {name: digest(path) for name, path in paths.items()}
    with (reports / "batch_summary.csv").open(newline="") as stream:
        batches = list(csv.DictReader(stream))
    with (reports / "batch_gas_summary.csv").open(newline="") as stream:
        gas_rows = list(csv.DictReader(stream))
    data = {name: json.loads((reports / f"{name}.json").read_text())
            for name in ("source_manifest", "ml_baselines", "ml_autoencoder", "ml_preflight")}
    manifest, baseline, autoencoder, preflight = (data[name] for name in
        ("source_manifest", "ml_baselines", "ml_autoencoder", "ml_preflight"))
    archive_hash = manifest["archive_sha256"]
    calibration_hash = baseline["calibration_digest"]
    for name, item in (("baseline", baseline), ("autoencoder", autoencoder), ("preflight", preflight)):
        require(item["archive_sha256"] == archive_hash, f"{name} archive provenance mismatch")
        require(item["calibration_digest"] == calibration_hash, f"{name} calibration provenance mismatch")
    raw_archive = root / "data/raw/uci-270.zip"
    require(raw_archive.is_file() and digest(raw_archive) == archive_hash,
            "raw archive no longer matches frozen source manifest")
    source_hashes["data/raw/uci-270.zip"] = archive_hash
    for filename, expected in manifest["pdf_sha256"].items():
        pdf = root / filename
        require(pdf.is_file() and digest(pdf) == expected, f"submitted PDF hash changed: {filename}")
        source_hashes[filename] = expected
    require(len(batches) == 10 and [int(row["batch"]) for row in batches] == list(range(1, 11)),
            "batch summary must contain B1–B10 exactly once in order")
    require(len(gas_rows) == 60 and
            {(int(row["batch"]), int(row["gas_code"])) for row in gas_rows} ==
            {(batch, gas) for batch in range(1, 11) for gas in range(1, 7)},
            "batch-gas summary must contain each B1–B10/gas pair exactly once")
    require(sum(int(row["rows"]) for row in batches) == manifest["total_rows"],
            "audit row total differs from source manifest")
    for row in batches:
        batch = row["batch"]
        require(row["split"] == manifest["splits"][batch], f"batch {batch} split mismatch")
        selected = [gas for gas in gas_rows if gas["batch"] == batch]
        require(all(gas["split"] == row["split"] and
                    int(gas["rows"]) == sum(int(gas[key]) for key in
                                            ("eligible", "no_exact_reference", "sparse_reference"))
                    for gas in selected), f"batch {batch} gas coverage denominators disagree")
        for key in ("rows", "eligible", "no_exact_reference", "sparse_reference"):
            require(sum(int(gas[key]) for gas in selected) == int(row[key]),
                    f"batch {batch} gas totals disagree for {key}")
        eligible = int(row["eligible"])
        require(int(row["rows"]) == eligible + int(row["no_exact_reference"]) + int(row["sparse_reference"]),
                f"batch {batch} coverage denominators disagree")
        for quantile in ("90", "95", "99"):
            require(eligible == int(row[f"within_q{quantile}"]) + int(row[f"outside_q{quantile}"]),
                    f"batch {batch} q{quantile} departure denominator differs from eligible")
    require(manifest["reference_rows_for_leave_one_out"] == int(batches[0]["eligible"]),
            "Batch 1 reference coverage mismatch")
    require(baseline["feature_scale_fit_batches"] == [1, 2, 3, 4]
            and baseline["model_fit_batches"] == [2, 3, 4]
            and baseline["selection_batches"] == [5, 6]
            and autoencoder["selection_batches"] == [5, 6]
            and baseline["exploratory_evaluation_batches"] == [7, 8, 9, 10]
            and autoencoder["exploratory_evaluation_batches"] == [7, 8, 9, 10],
            "frozen fitting/selection/exploratory split metadata changed")
    require(preflight["model_input_metadata_fields"] == [] and preflight["model_input_width"] == 768,
            "preflight model-input contract changed")
    for condition in CONDITIONS:
        require(condition in baseline["conditions"] and condition in autoencoder["conditions"]
                and condition in preflight["conditions"], f"missing condition: {condition}")
        for split in SPLITS:
            require(split in baseline["conditions"][condition]
                    and split in autoencoder["conditions"][condition], f"missing {condition}/{split}")
            supports = None
            for task, display, source, key in MODELS:
                result = data[source]["conditions"][condition][split][key]
                validate_model_result(result, task, f"{condition}/{split}/{display}")
                actual_support = ([result["per_class"]["background"]["support"],
                                   result["per_class"]["intervention"]["support"]]
                                  if task == "binary" else
                                  [result["per_class"][label]["support"] for label in THREE_LABELS])
                three_support = ([actual_support[0], actual_support[1] // 2, actual_support[1] // 2]
                                 if task == "binary" else actual_support)
                require(actual_support[1] % 2 == 0 if task == "binary" else True,
                        f"{condition}/{split}/{display} odd intervention support")
                if supports is None:
                    supports = three_support
                require(three_support == supports, f"{condition}/{split}/{display} true-label supports disagree")
                for batch in ((5, 6) if split == "validation" else (7, 8, 9, 10)):
                    require(str(batch) in result["per_batch"], f"missing per-batch result {batch}")
                summed = [[0] * (2 if task == "binary" else 3) for _ in range(2 if task == "binary" else 3)]
                for batch_result in result["per_batch"].values():
                    validate_model_result(batch_result, task, f"{condition}/{split}/{display}/batch",
                                          nested=False)
                    for i, row in enumerate(batch_result["confusion_rows_true_columns_predicted"]):
                        for j, count in enumerate(row):
                            summed[i][j] += count
                require(summed == result["confusion_rows_true_columns_predicted"],
                        f"{condition}/{split}/{display} per-batch matrices do not sum to overall matrix")
            preflight_counts = preflight["conditions"][condition]["class_counts_by_split"][split]
            require([preflight_counts[label] for label in THREE_LABELS] == supports,
                    f"{condition}/{split} preflight class counts disagree")
            signatures = preflight["conditions"][condition]["exact_copy_signatures_by_split_family"][split]
            for family, field in (("freeze", "zero_active_variation_sensor_blocks"),
                                  ("replay", "zero_lag_four_sensor_blocks")):
                require(signatures[family][field] == signatures[family]["episodes"],
                        f"{condition}/{split}/{family} exact-copy control signature incomplete")
            for family, counts in signatures.items():
                if family != "freeze":
                    require(counts["zero_active_variation_sensor_blocks"] == 0,
                            f"unexpected freeze signature in {condition}/{split}/{family}")
                if family != "replay":
                    require(counts["zero_lag_four_sensor_blocks"] == 0,
                            f"unexpected replay signature in {condition}/{split}/{family}")
    for task, display, source, key in MODELS:
        for split in SPLITS:
            primary = _result(data, source, "matched_primary", split, key)
            stress = _result(data, source, "original_stress", split, key)
            require(primary["confusion_rows_true_columns_predicted"][0] ==
                    stress["confusion_rows_true_columns_predicted"][0],
                    f"{display}/{split}: background outcomes differ between conditions")
            for batch in primary["per_batch"]:
                require(primary["per_batch"][batch]["rates"]["any_intervention_false_alarm"] ==
                        stress["per_batch"][batch]["rates"]["any_intervention_false_alarm"],
                        f"{display}/{split}/B{batch}: background false alarms differ between conditions")
    require(autoencoder["forest_validation_confusion_reproduced"] ==
            baseline["conditions"]["matched_primary"]["validation"]["random_forest"]
            ["confusion_rows_true_columns_predicted"], "refitted forest validation matrix differs from saved result")
    provenance = {"source_sha256": source_hashes, "archive_sha256": archive_hash,
                  "calibration_digest": calibration_hash,
                  "interpretation": "Derived figures/tables only; no model predictions recalculated."}
    return data, batches, provenance


def validate_model_result(result: dict, task: str, context: str, nested: bool = True) -> None:
    """Validate saved row=true/column=predicted matrices against every stated denominator."""
    labels = BINARY_LABELS if task == "binary" else THREE_LABELS
    matrix = result["confusion_rows_true_columns_predicted"]
    width = len(labels)
    require(result["labels"] == list(labels) and len(matrix) == width and
            all(len(row) == width for row in matrix), f"{context}: task/label/matrix orientation mismatch")
    require(all(type(value) is int and value >= 0 for row in matrix for value in row),
            f"{context}: invalid confusion count")
    require(sum(map(sum, matrix)) == result["denominator_episodes"],
            f"{context}: overall confusion denominator mismatch")
    f1_values = []
    for i, label in enumerate(labels):
        support = sum(matrix[i]); predicted = sum(row[i] for row in matrix)
        tp = matrix[i][i]
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        saved = result["per_class"][label]
        require(saved["support"] == support and saved["predicted"] == predicted,
                f"{context}: {label} support/prediction orientation mismatch")
        close(saved["precision"], precision, f"{context}: {label} precision mismatch")
        close(saved["recall"], recall, f"{context}: {label} recall mismatch")
    close(result["macro_f1"], sum(f1_values) / width, f"{context}: macro-F1 mismatch")
    rates = result["rates"]
    if task == "binary":
        _check_rate(rates["any_intervention_false_alarm"], matrix[0][1], sum(matrix[0]), context)
        require("attack_false_alarm" not in rates, f"{context}: binary/three-class rates mixed")
        if nested:
            drift = rates["synthetic_drift_detection"]
            attack = rates["synthetic_attack_detection"]
            require(drift["denominator"] + attack["denominator"] == sum(matrix[1]) and
                    drift["numerator"] + attack["numerator"] == matrix[1][1],
                    f"{context}: binary drift/attack detection totals mismatch")
            _check_rate(drift, drift["numerator"], drift["denominator"], context)
            _check_rate(attack, attack["numerator"], attack["denominator"], context)
    else:
        _check_rate(rates["attack_false_alarm"], matrix[0][2], sum(matrix[0]), context)
        _check_rate(rates["any_intervention_false_alarm"], matrix[0][1] + matrix[0][2], sum(matrix[0]), context)
        _check_rate(rates["drift_mistaken_for_attack"], matrix[1][2], sum(matrix[1]), context)
        _check_rate(rates["attack_mistaken_for_background_or_drift"], matrix[2][0] + matrix[2][1],
                    sum(matrix[2]), context)
    if nested:
        families = result["per_family_severity"]
        require(families, f"{context}: missing family/severity breakdown")
        drift_total = attack_total = 0
        for name, values in families.items():
            family = name.split(":", 1)[0]
            expected_label = ("intervention" if task == "binary" else
                              "synthetic_drift" if family == "gradual_gain" else "synthetic_attack")
            total = values["denominator_episodes"]
            require(sum(values["predicted_counts"].values()) == total and
                    values["correct"] == values["predicted_counts"].get(expected_label, 0),
                    f"{context}/{name}: family predicted totals mismatch")
            close(values["recall"], values["correct"] / total,
                  f"{context}/{name}: family recall mismatch")
            if family == "gradual_gain":
                drift_total += total
            else:
                attack_total += total
        require(drift_total == (sum(matrix[1]) if task == "three_class" else
                                rates["synthetic_drift_detection"]["denominator"]),
                f"{context}: family drift denominators mismatch")
        require(attack_total == (sum(matrix[2]) if task == "three_class" else
                                 rates["synthetic_attack_detection"]["denominator"]),
                f"{context}: family attack denominators mismatch")


def _check_rate(saved: dict, numerator: int, denominator: int, context: str) -> None:
    require(saved["numerator"] == numerator and saved["denominator"] == denominator,
            f"{context}: rate numerator/denominator mismatch")
    close(saved["rate"], numerator / denominator if denominator else 0,
          f"{context}: rate value mismatch")


def _result(data: dict, source: str, condition: str, split: str, key: str) -> dict:
    return data[source]["conditions"][condition][split][key]


def derive_tables(data: dict, batches: list[dict]) -> dict[str, list[dict]]:
    tables: dict[str, list[dict]] = {name: [] for name in (
        "track_a_by_batch", "binary_models", "three_class_models",
        "per_batch_false_alarms", "attack_family_severity")}
    for source in batches:
        row = {"batch": int(source["batch"]), "split": source["split"],
               "rows": int(source["rows"]), "eligible": int(source["eligible"]),
               "no_exact_reference": int(source["no_exact_reference"]),
               "sparse_reference": int(source["sparse_reference"])}
        row["excluded"] = row["no_exact_reference"] + row["sparse_reference"]
        for key in ("no_exact_reference", "sparse_reference", "excluded"):
            row[f"{key}_denominator"] = row["rows"]
        row["eligible_rate"] = row["eligible"] / row["rows"]
        for q in (90, 95, 99):
            row[f"outside_q{q}"] = int(source[f"outside_q{q}"])
            row[f"eligible_q{q}_denominator"] = row["eligible"]
            row[f"outside_q{q}_rate"] = row[f"outside_q{q}"] / row["eligible"]
        tables["track_a_by_batch"].append(row)
    for condition in CONDITIONS:
        for split in SPLITS:
            for task, model, source, key in MODELS:
                entry = _result(data, source, condition, split, key)
                rates = entry["rates"]
                row = {"condition": condition, "split": split,
                       "evaluation_status": "exploratory" if split == "test" else "validation",
                       "model": model, "episodes": entry["denominator_episodes"],
                       "macro_f1": entry["macro_f1"]}
                for label in (BINARY_LABELS if task == "binary" else THREE_LABELS):
                    values = entry["per_class"][label]
                    row[f"{label}_support"] = values["support"]
                    row[f"{label}_precision"] = values["precision"]
                    row[f"{label}_recall"] = values["recall"]
                for rate_name, values in rates.items():
                    row[f"{rate_name}_numerator"] = values["numerator"]
                    row[f"{rate_name}_denominator"] = values["denominator"]
                    row[f"{rate_name}_rate"] = values["rate"]
                tables["binary_models" if task == "binary" else "three_class_models"].append(row)
                for batch, batch_entry in sorted(entry["per_batch"].items(), key=lambda item: int(item[0])):
                    alarm = batch_entry["rates"]["any_intervention_false_alarm"]
                    tables["per_batch_false_alarms"].append({
                        "condition": condition, "split": split,
                        "evaluation_status": row["evaluation_status"], "batch": int(batch),
                        "task": task, "model": model, "numerator": alarm["numerator"],
                        "denominator": alarm["denominator"], "rate": alarm["rate"]})
                for name, values in sorted(entry["per_family_severity"].items()):
                    family, severity = name.split(":", 1)
                    tables["attack_family_severity"].append({
                        "condition": condition, "split": split,
                        "evaluation_status": row["evaluation_status"],
                        "task": task, "model": model, "family": family,
                        "severity": severity, "exact_copy_control": family in ("freeze", "replay"),
                        "correct_or_detected": values["correct"],
                        "denominator": values["denominator_episodes"], "rate": values["recall"],
                        "predicted_counts": json.dumps(values["predicted_counts"], sort_keys=True)})
    return tables


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"empty derived table: {path.name}")
    columns = list(rows[0])
    require(all(set(row) == set(columns) for row in rows), f"inconsistent table columns: {path.name}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class Svg:
    """Small vector drawing helper with escaped text and a fixed visual style."""

    def __init__(self, width: int, height: int, title: str, subtitle: str):
        self.width, self.height = width, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
                      '<rect width="100%" height="100%" fill="#FFFFFF"/>']
        self.text(52, 48, title, size=28, weight="700")
        self.text(52, 78, subtitle, size=14, color=MUTED)

    def rect(self, x: float, y: float, width: float, height: float, fill: str,
             stroke: str = "none", radius: int = 0) -> None:
        self.parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(width, 0):.2f}" height="{max(height, 0):.2f}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>')

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str = GRID,
             width: float = 1, dash: str | None = None) -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"{extra}/>')

    def text(self, x: float, y: float, content: object, size: int = 14, color: str = INK,
             anchor: str = "start", weight: str = "400", rotate: int | None = None) -> None:
        transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
        self.parts.append(f'<text x="{x:.2f}" y="{y:.2f}" fill="{color}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}"{transform}>{escape(str(content))}</text>')

    def circle(self, x: float, y: float, radius: float, fill: str) -> None:
        self.parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{fill}"/>')

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.parts + ["</svg>", ""]))


def draw_track_a(rows: list[dict], manifest: dict, path: Path) -> None:
    svg = Svg(1850, 1040, "Track A · reference coverage and departure sensitivity",
              "Observed UCI records. Departures are from the Batch 1 reference envelope; they are not verified sensor drift.")
    svg.text(70, 125, "A · Reference matching", size=20, weight="700")
    svg.text(925, 125, "B · Outside-envelope fraction among eligible records", size=20, weight="700")
    svg.text(70, 150, "Stacked fraction of records, with eligible/total shown below", size=13, color=MUTED)
    svg.text(925, 150, "Each label gives outside/eligible; q90, q95, q99 are Batch 1 cutoffs", size=13, color=MUTED)
    top, bottom = 210, 700
    for x0 in (95, 950):
        for tick in (0, .25, .5, .75, 1):
            y = bottom - tick * (bottom - top)
            svg.line(x0, y, x0 + 755, y)
            svg.text(x0 - 10, y + 5, f"{tick:.0%}", size=12, anchor="end", color=MUTED)
    for index, row in enumerate(rows):
        x = 105 + index * 74
        fractions = (row["eligible"] / row["rows"], row["no_exact_reference"] / row["rows"],
                     row["sparse_reference"] / row["rows"])
        y = bottom
        for fraction, color in zip(fractions, (BLUE, ORANGE, "#B8C3CB")):
            height = fraction * (bottom - top)
            y -= height
            svg.rect(x, y, 46, height, color)
        svg.text(x + 23, 730, f"B{row['batch']}", anchor="middle", weight="700")
        svg.text(x + 23, 754, f"{row['eligible']}/{row['rows']}", size=10, anchor="middle")
        x2 = 956 + index * 75
        for j, (quantile, color) in enumerate(((90, PURPLE), (95, GREEN), (99, ORANGE))):
            rate = row[f"outside_q{quantile}_rate"]
            bar_x = x2 + j * 16
            height = rate * (bottom - top)
            svg.rect(bar_x, bottom - height, 13, height, color)
            svg.text(bar_x + 6, 790 + j * 25, f"{row[f'outside_q{quantile}']}/{row['eligible']}",
                     size=9, anchor="middle", rotate=-35)
        svg.text(x2 + 24, 730, f"B{row['batch']}", anchor="middle", weight="700")
    for x, color, label in ((110, BLUE, "Eligible"), (260, ORANGE, "No exact B1 pair"),
                            (485, "#B8C3CB", "Sparse B1 pair"),
                            (970, PURPLE, "q90"), (1080, GREEN, "q95"), (1190, ORANGE, "q99")):
        svg.rect(x, 875, 16, 16, color)
        svg.text(x + 23, 889, label, size=13)
    svg.line(105 + 6 * 74 - 14, 185, 105 + 6 * 74 - 14, 810, MUTED, 1.5, "5,5")
    svg.line(956 + 6 * 75 - 10, 185, 956 + 6 * 75 - 10, 810, MUTED, 1.5, "5,5")
    svg.text(70, 945, "B1: in-sample reference counts. B5–6: validation. B7–10: exploratory test (separators before B7).", size=14)
    svg.text(70, 973, "Cutoffs: " + ", ".join(f"q{int(float(q)*100)}={value:.4f}" for q, value in sorted(manifest["cutoffs"].items())),
             size=13, color=MUTED)
    svg.save(path)


def draw_confusions(data: dict, condition: str, task: str, path: Path) -> None:
    choices = [item for item in MODELS if item[0] == task]
    labels = BINARY_LABELS if task == "binary" else THREE_LABELS
    title = f"Track B · {task.replace('_', ' ')} confusion matrices · {condition.replace('_', ' ')}"
    svg = Svg(1740, 1020, title, "Rows are assigned true labels; columns are predictions. Each cell is a count, with its true-row denominator alongside.")
    cell = 86 if task == "binary" else 72
    for r, (t, display, source, key) in enumerate(choices):
        for c, split in enumerate(SPLITS):
            entry = _result(data, source, condition, split, key)
            matrix = entry["confusion_rows_true_columns_predicted"]
            left = 145 + c * 820
            top = 205 + r * 390
            svg.text(left, top - 66, f"{display} · {'validation B5–6' if split == 'validation' else 'exploratory test B7–10'}",
                     size=19, weight="700")
            svg.text(left, top - 40, f"n={entry['denominator_episodes']} episodes; macro-F1={entry['macro_f1']:.3f}",
                     size=13, color=MUTED)
            x0 = left + 215
            for j, label in enumerate(labels):
                svg.text(x0 + (j + .5) * cell, top - 10, label.replace("synthetic_", ""),
                         size=13, anchor="middle", color=MUTED)
            for i, label in enumerate(labels):
                denom = sum(matrix[i])
                svg.text(x0 - 14, top + (i + .56) * cell,
                         f"{label.replace('synthetic_', '')}  n={denom}",
                         size=13, anchor="end")
                for j, count in enumerate(matrix[i]):
                    ratio = count / denom if denom else 0
                    shade = int(246 - ratio * 138)
                    fill = f"#{shade:02x}{int(shade + (255-shade)*.20):02x}ff"
                    svg.rect(x0 + j * cell, top + i * cell, cell - 3, cell - 3, fill, radius=3)
                    svg.text(x0 + (j + .5) * cell, top + (i + .55) * cell,
                             count, size=20, weight="700", anchor="middle")
    svg.text(55, 960, "The binary task combines drift and attack as intervention; its macro-F1 is not compared with three-class macro-F1.",
             size=14, color=MUTED)
    svg.save(path)


def draw_false_alarms(data: dict, path: Path) -> None:
    svg = Svg(1850, 1140, "Track B · background false alarms by batch",
              "Bars show background episodes flagged as any intervention; labels give numerator/denominator. Both bias conditions share these backgrounds.")
    for panel, task in enumerate(("binary", "three_class")):
        top, bottom = 180 + panel * 435, 475 + panel * 435
        choices = [item for item in MODELS if item[0] == task]
        svg.text(72, top - 35, "Binary anomaly detection" if task == "binary" else "Three-class classification",
                 size=21, weight="700")
        for tick in (0, .25, .5, .75, 1):
            y = bottom - tick * (bottom - top)
            svg.line(90, y, 1780, y)
            svg.text(80, y + 5, f"{tick:.0%}", size=12, anchor="end", color=MUTED)
        for c, batch in enumerate((5, 6, 7, 8, 9, 10)):
            x0 = 130 + c * 275
            split = "validation" if batch <= 6 else "test"
            for m, (_, display, source, key) in enumerate(choices):
                result = _result(data, source, "matched_primary", split, key)
                alarm = result["per_batch"][str(batch)]["rates"]["any_intervention_false_alarm"]
                x = x0 + m * 85
                height = alarm["rate"] * (bottom - top)
                svg.rect(x, bottom - height, 60, height, BLUE if m == 0 else ORANGE)
                svg.text(x + 30, bottom - height - 10,
                         f"{alarm['numerator']}/{alarm['denominator']}", size=11, anchor="middle", weight="700")
            svg.text(x0 + 73, bottom + 29, f"B{batch}", anchor="middle", weight="700")
            svg.text(x0 + 73, bottom + 49,
                     "validation" if batch <= 6 else "exploratory", size=11,
                     anchor="middle", color=MUTED)
        svg.line(130 + 2 * 275 - 30, top - 20, 130 + 2 * 275 - 30, bottom + 55,
                 MUTED, 1.5, "5,5")
        for m, (_, display, _, _) in enumerate(choices):
            svg.rect(135 + m * 270, bottom + 75, 15, 15, BLUE if m == 0 else ORANGE)
            svg.text(158 + m * 270, bottom + 88, display, size=13)
    svg.text(72, 1095, "A false alarm here is an unmodified episode assigned any synthetic intervention; it is not a measured deployment alarm rate.",
             size=14, color=MUTED)
    svg.save(path)


def draw_family(data: dict, path: Path) -> None:
    families = ("bias", "noise", "scaling", "freeze", "replay")
    svg = Svg(1800, 1040, "Track B · synthetic attack-family classification",
              "Exploratory B7–10. Numerator/denominator is correct attack-class predictions; freeze/replay are exact-copy controls (†).")
    for panel, condition in enumerate(CONDITIONS):
        left = 120 + panel * 850
        right = left + 610
        svg.text(left, 150, condition.replace("_", " ").title(), size=21, weight="700")
        for tick in (0, .25, .5, .75, 1):
            x = left + tick * (right - left)
            svg.line(x, 205, x, 765)
            svg.text(x, 790, f"{tick:.0%}", size=12, anchor="middle", color=MUTED)
        for index, family in enumerate(families):
            y = 265 + index * 110
            svg.text(left - 15, y + 7, family + (" †" if family in ("freeze", "replay") else ""),
                     size=16, anchor="end", weight="700" if family in ("freeze", "replay") else "400")
            for m, (_, display, source, key) in enumerate(MODELS[2:]):
                breakdown = _result(data, source, condition, "test", key)["per_family_severity"]
                matching = [value for name, value in breakdown.items() if name.startswith(family + ":")]
                numerator = sum(item["correct"] for item in matching)
                denominator = sum(item["denominator_episodes"] for item in matching)
                require(denominator > 0, f"missing family {family}")
                x = left + (numerator / denominator) * (right - left)
                yy = y - 17 + m * 34
                svg.circle(x, yy, 8, BLUE if m == 0 else ORANGE)
                label_x = min(x + 14, right - 22)
                svg.text(label_x, yy + 5, f"{numerator}/{denominator}", size=12,
                         color=BLUE if m == 0 else ORANGE, weight="700")
    svg.rect(120, 860, 16, 16, BLUE)
    svg.text(143, 873, "Random Forest", size=14)
    svg.rect(300, 860, 16, 16, ORANGE)
    svg.text(323, 873, "Autoencoder + forest", size=14)
    svg.text(120, 920, "† Exact-copy controls have deterministic zero-block signatures; their apparent success does not establish general attack detection.",
             size=14, color=MUTED)
    svg.text(120, 950, "All other families edit extracted feature streams. The matched-primary bias uses frozen training-derived magnitudes; stress uses the original bias grid.",
             size=14, color=MUTED)
    svg.save(path)


def _rate_text(values: dict) -> str:
    return f"{values['numerator']}/{values['denominator']} ({values['rate']:.1%})"


def build_report(data: dict, batches: list[dict], tables: dict[str, list[dict]],
                 provenance: dict) -> str:
    manifest = data["source_manifest"]
    test_rows = [row for row in tables["track_a_by_batch"] if row["split"] == "test"]
    covered = sum(row["eligible"] for row in test_rows)
    total = sum(row["rows"] for row in test_rows)
    lines = ["# Final results · Drift-or-Attack", "",
             "These figures and tables are derived from frozen audit and model-result files. No predictions or thresholds were recomputed. **Batches 7–10 are exploratory:** their audit coverage and score distributions were inspected before the protocol was frozen.", "",
             "## Track A · observational Batch 1 reference departures", "",
             f"Strict gas/concentration matching covers **{covered}/{total} ({covered/total:.1%})** B7–10 records; {total-covered} are excluded. An outside-envelope record is a **reference departure**, not a verified natural-drift label. Batch 1 counts are in-sample reference counts and are not calibrated false-alarm estimates. Cutoffs (Batch 1 leave-one-out only): " +
             ", ".join(f"q{int(float(q)*100)}={value:.6f}" for q, value in sorted(manifest["cutoffs"].items())) + ".", "",
             "![Track A reference coverage and sensitivity](figures/track_a_coverage_sensitivity.svg)", "",
             "| Batch | Role | Eligible / all | No exact B1 pair / all | Sparse B1 pair / all | Excluded / all | Outside q90 / eligible | Outside q95 / eligible | Outside q99 / eligible |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in tables["track_a_by_batch"]:
        role = "exploratory test" if row["split"] == "test" else row["split"]
        lines.append(f"| {row['batch']} | {role} | {row['eligible']}/{row['rows']} | {row['no_exact_reference']}/{row['rows']} | {row['sparse_reference']}/{row['rows']} | {row['excluded']}/{row['rows']} | " +
                     " | ".join(f"{row[f'outside_q{q}']}/{row['eligible']}" for q in (90, 95, 99)) + " |")
    lines.extend(["", "Track A uses laboratory gas and concentration for exact reference matching. Unmatched and sparse strata receive no departure state. It supplies no cyberattack labels.", "",
                  "## Track B · controlled synthetic interventions", "",
                  "`background` means unmodified input, not verified healthy hardware. Binary detection and three-class classification are **different tasks**; their macro-F1 values are not compared directly. Primary and original-stress conditions share source episodes and are reported separately. All rates below show numerator/denominator.", ""])
    for task, heading, keys in (("binary", "Binary anomaly detection", MODELS[:2]),
                                ("three_class", "Three-class classification", MODELS[2:])):
        lines.extend([f"### {heading}", "",
                      "| Condition | Split | Model | Macro-F1 | Background false alarms | " +
                      ("Drift detected | Attack detected |" if task == "binary" else
                       "Drift→attack | Attack→background/drift |"),
                      "|---|---|---|---:|---:|---:|---:|"])
        for condition in CONDITIONS:
            for split in SPLITS:
                for _, display, source, key in keys:
                    result = _result(data, source, condition, split, key)
                    rates = result["rates"]
                    if task == "binary":
                        details = (_rate_text(rates["synthetic_drift_detection"]),
                                   _rate_text(rates["synthetic_attack_detection"]))
                    else:
                        details = (_rate_text(rates["drift_mistaken_for_attack"]),
                                   _rate_text(rates["attack_mistaken_for_background_or_drift"]))
                    lines.append(f"| {condition} | {'exploratory test' if split == 'test' else 'validation'} | {display} | "
                                 f"{result['macro_f1']:.3f} | {_rate_text(rates['any_intervention_false_alarm'])} | {details[0]} | {details[1]} |")
        lines.extend(["", "Per-class precision and recall (P/R), with true-class support:", "",
                      "| Condition | Split | Model | Class | Precision | Recall | Support |",
                      "|---|---|---|---|---:|---:|---:|"])
        for condition in CONDITIONS:
            for split in SPLITS:
                for _, display, source, key in keys:
                    result = _result(data, source, condition, split, key)
                    for label in (BINARY_LABELS if task == "binary" else THREE_LABELS):
                        values = result["per_class"][label]
                        lines.append(f"| {condition} | {'exploratory test' if split == 'test' else 'validation'} | "
                                     f"{display} | {label} | {values['precision']:.3f} | {values['recall']:.3f} | {values['support']} |")
        lines.append("")
        for condition in CONDITIONS:
            lines.append(f"![{heading}, {condition} confusion matrices](figures/{task}_confusions_{condition}.svg)")
        lines.append("")
    lines.extend(["### Batch and family diagnostics", "",
                  "![Background false alarms by batch](figures/per_batch_false_alarms.svg)", "",
                  "The plotted background episodes are identical in the matched-primary and original-stress conditions. Batch 5–6 bars are validation; Batch 7–10 bars are exploratory. Full per-batch fractions appear in `tables/per_batch_false_alarms.csv`.", "",
                  "![Attack-family classification by condition](figures/attack_family_recall.svg)", "",
                  "Freeze and replay are **exact-copy controls** with deterministic zero-block feature signatures. The severity-specific counts, and binary detector results by family, are in `tables/attack_family_severity.csv`. Its binary `correct_or_detected` column means intervention detected; its three-class value means correct assigned class. These rates must be interpreted separately.", "",
                  "### Interpretation", "",
                  "On matched-primary exploratory test episodes, the autoencoder plus forest has lower three-class macro-F1 than the forest alone, while its background false alarms rise from **724/909** to **752/909**. The autoencoder-only detector has higher binary macro-F1 than Isolation Forest but still flags **752/909** backgrounds. No improvement in three-class classification is claimed.", "",
                  "## Validity and leakage limits", "",
                  "- Batches 7–10 are exploratory because their audit coverage and score distributions were previously inspected. They were not used for model or threshold selection.",
                  "- The strict Track A reference rule excludes unmatched or sparse gas/concentration strata. Later-batch reference departures are not verified sensor aging.",
                  "- Track B episode order is seeded and artificial, without record timestamps or a real intervention onset. Detection delay in elapsed time is unsupported.",
                  "- Synthetic operations edit extracted sensor features, not physical waveforms. Neither real sensor-aging detection nor real cyberattack detection is validated.",
                  "- The matched-primary bias draws from finite training-derived pools. Exact magnitudes repeat across splits and may be learnable simulation signatures. Mean magnitude matching does not equalize temporal shape or remove residual paired gaps.",
                  "- Freeze and replay have exact-copy signatures. Their aggregate contribution must not be generalized to subtle attacks.",
                  "- Gas identity, concentration, batch, source IDs, seeds, and intervention metadata are excluded as direct model inputs, but sensor features can indirectly encode gas/concentration and batch conditions.",
                  "- Related variants share source episodes; matched-primary and stress reuse those sources. They are not independent replicates. No uncertainty intervals are claimed.",
                  "- Background means unmodified UCI input; verified healthy hardware, independently annotated aging, and real cyberattacks are unavailable.", "",
                  "## Reproducibility", "",
                  "The full derived tables are under `tables/`, vector figures under `figures/`, and input SHA-256 hashes in `source_hashes.json`. The frozen protocol is `docs/experimental_protocol.md`; the original audit and model JSON files remain authoritative for all saved counts.", ""])
    return "\n".join(lines)


def build(reports: Path, output: Path) -> list[Path]:
    data, batches, provenance = load_and_validate(reports)
    require(not output.exists() or output.is_dir(), "output path exists but is not a directory")
    require(output.resolve() == (reports / "final_results").resolve(),
            "generated output must be reports/final_results/")
    tables = derive_tables(data, batches)
    figures = output / "figures"
    table_dir = output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    for name, rows in tables.items():
        path = table_dir / f"{name}.csv"
        write_csv(path, rows)
        produced.append(path)
    manifest = data["source_manifest"]
    track_path = figures / "track_a_coverage_sensitivity.svg"
    draw_track_a(tables["track_a_by_batch"], manifest, track_path)
    produced.append(track_path)
    for task in ("binary", "three_class"):
        for condition in CONDITIONS:
            path = figures / f"{task}_confusions_{condition}.svg"
            draw_confusions(data, condition, task, path)
            produced.append(path)
    alarm_path = figures / "per_batch_false_alarms.svg"
    draw_false_alarms(data, alarm_path)
    produced.append(alarm_path)
    family_path = figures / "attack_family_recall.svg"
    draw_family(data, family_path)
    produced.append(family_path)
    report_path = output / "final_results.md"
    report_path.write_text(build_report(data, batches, tables, provenance))
    produced.append(report_path)
    provenance_path = output / "source_hashes.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    produced.append(provenance_path)
    return produced


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = build(args.reports, args.output)
    print(f"Generated {len(files)} files in {args.output}; all frozen-source checks passed.")


if __name__ == "__main__":
    main()
