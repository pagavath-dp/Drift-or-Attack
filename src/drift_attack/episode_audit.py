"""Validate and summarize synthetic episode generation without training models."""

from __future__ import annotations

import argparse
import csv
import json
from hashlib import sha256
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from struct import pack

from .audit import APPROVED_ARCHIVE_SHA256, file_sha256
from .dataset import load_archive
from .episodes import (ACTIVE_POSITIONS, ATTACK_FAMILIES, MASTER_SEED,
                       BIAS_CONDITIONS, BIAS_LEVELS, GAIN_LEVELS,
                       calibration_payload, changed_positions, construct_episodes,
                       fit_generator_scales, fit_magnitude_calibration,
                       generate_variants, standardized_change)


def inspect(archive: Path, output: Path, condition: str) -> dict:
    if condition not in BIAS_CONDITIONS:
        raise ValueError("unknown bias condition")
    if file_sha256(archive) != APPROVED_ARCHIVE_SHA256:
        raise ValueError("archive SHA-256 differs from inspected UCI archive")
    batches = load_archive(archive)
    episodes, coverage = construct_episodes(batches)
    scales = fit_generator_scales(batches)
    calibration = fit_magnitude_calibration(episodes, scales) if condition == "matched_primary" else None
    fit_rows = [row for batch in range(1, 5) for row in batches[batch]]
    training_min = [min(row.features[j] for row in fit_rows) for j in range(128)]
    training_max = [max(row.features[j] for row in fit_rows) for j in range(128)]
    counts = {split: Counter() for split in ("train", "validation", "test")}
    families = {split: Counter() for split in counts}
    family_by_batch = {batch: Counter() for batch in range(2, 11)}
    effects: dict[str, list[float]] = defaultdict(list)
    effects_by_split_severity: dict[str, dict[str, list[float]]] = {
        split: defaultdict(list) for split in counts
    }
    paired_bias_drift: dict[str, dict[str, dict[str, list[float]]]] = {
        split: defaultdict(lambda: {"bias": [], "drift": [], "absolute_gap": []}) for split in counts
    }
    induced_out_of_training_range = Counter()
    manifest_rows = []
    seen_episode_ids = set()
    generated_vector_owner: dict[str, str] = {}
    for episode in episodes:
        if episode.batch == 1:
            continue
        if episode.episode_id in seen_episode_ids:
            raise AssertionError("duplicate episode ID")
        seen_episode_ids.add(episode.episode_id)
        variants = generate_variants(episode, scales, bias_condition=condition, calibration=calibration)
        if {variant.label for variant in variants} != {"background", "synthetic_drift", "synthetic_attack"}:
            raise AssertionError("each episode needs all three labels")
        for variant in variants:
            if (variant.episode_id != episode.episode_id or variant.batch != episode.batch
                    or variant.split != episode.split or variant.source_ids != episode.source_ids):
                raise AssertionError("paired variant crossed its source episode or partition")
            if len(variant.rows) != 8 or any(len(row) != 128 for row in variant.rows):
                raise AssertionError("padding or feature-width mismatch")
            for generated_row in variant.rows:
                fingerprint = sha256(pack("!128d", *generated_row)).hexdigest()
                previous_owner = generated_vector_owner.get(fingerprint)
                if previous_owner is not None and previous_owner != episode.episode_id:
                    raise AssertionError("generated feature-vector duplicate across base episodes")
                generated_vector_owner[fingerprint] = episode.episode_id
            expected = () if variant.label == "background" else ACTIVE_POSITIONS
            if changed_positions(episode, variant) != expected:
                raise AssertionError(f"unexpected intervention duration: {episode.episode_id} {variant.family}")
            first = variant.sensor * 8
            for t, (source_row, generated_row) in enumerate(zip(episode.rows, variant.rows, strict=True)):
                for j, (before, after) in enumerate(zip(source_row, generated_row, strict=True)):
                    if before == after:
                        continue
                    if t not in ACTIVE_POSITIONS or not first <= j < first + 8:
                        raise AssertionError("intervention changed source selection or another sensor")
                    if training_min[j] <= before <= training_max[j] and not training_min[j] <= after <= training_max[j]:
                        induced_out_of_training_range[variant.family] += 1
            counts[episode.split][variant.label] += 1
            if variant.label != "background":
                magnitude = standardized_change(episode, variant, scales)
                effects[variant.family].append(magnitude)
                if condition == "matched_primary" and variant.family == "bias":
                    severity_key = f"bias:gain_level={GAIN_LEVELS[episode.rotation_index % len(GAIN_LEVELS)]}"
                else:
                    severity_key = f"{variant.family}:{variant.severity}"
                effects_by_split_severity[episode.split][severity_key].append(magnitude)
                if variant.family == "bias" and condition == "matched_primary":
                    if abs(magnitude - variant.severity) > 1e-7 * max(1.0, variant.severity):
                        raise AssertionError("matched bias standardized magnitude differs from frozen target")
        attack = variants[2]
        attack_level = episode.rotation_index % len(GAIN_LEVELS)
        if attack.family == "bias":
            pair = paired_bias_drift[episode.split][str(GAIN_LEVELS[attack_level])]
            bias_magnitude = standardized_change(episode, attack, scales)
            drift_magnitude = standardized_change(episode, variants[1], scales)
            pair["bias"].append(bias_magnitude)
            pair["drift"].append(drift_magnitude)
            pair["absolute_gap"].append(abs(bias_magnitude - drift_magnitude))
        family_severity_key = (f"bias:gain_level={GAIN_LEVELS[attack_level]}"
                               if condition == "matched_primary" and attack.family == "bias"
                               else f"{attack.family}:{attack.severity}")
        families[episode.split][family_severity_key] += 1
        family_by_batch[episode.batch][attack.family] += 1
        manifest_rows.append({
            "episode_id": episode.episode_id,
            "batch": episode.batch,
            "split": episode.split,
            "gas_code_lab_only": episode.gas,
            "concentration_lab_only": episode.concentration,
            "source_ids": ";".join(f"{b}:{line}" for b, line in episode.source_ids),
            "attack_family": attack.family,
            "attack_gain_level_index": attack_level,
            "attack_severity": "fixed" if attack.severity is None else attack.severity,
            "drift_severity": variants[1].severity,
            "sensor_generator_only": attack.sensor,
        })
    for split in counts:
        if any(counts[split][label] == 0 for label in ("background", "synthetic_drift", "synthetic_attack")):
            raise AssertionError(f"a class is absent in {split}")
        if any(not any(key.startswith(family + ":") for key in families[split]) for family in ATTACK_FAMILIES):
            raise AssertionError(f"an attack family is absent in {split}")
    diagnostics = {
        "master_seed": MASTER_SEED,
        "bias_condition": condition,
        "original_bias_grid": list(BIAS_LEVELS),
        "calibration_digest": calibration.digest if calibration else None,
        "calibration_source_episode_count": len(calibration.source_episodes) if calibration else None,
        "archive_sha256": APPROVED_ARCHIVE_SHA256,
        "episode_length": 8,
        "active_positions_zero_based": list(ACTIVE_POSITIONS),
        "scale_fit_batches": [1, 2, 3, 4],
        "scale_min": min(scales),
        "scale_median": median(scales),
        "scale_max": max(scales),
        "coverage_by_batch": {str(batch): values for batch, values in coverage.items()},
        "class_counts_by_split": {split: dict(counter) for split, counter in counts.items()},
        "attack_family_severity_by_split": {split: dict(counter) for split, counter in families.items()},
        "attack_family_by_batch": {str(batch): dict(counter) for batch, counter in family_by_batch.items()},
        "mean_standardized_change_by_family": {
            family: {"episodes": len(values), "min": min(values), "median": median(values), "max": max(values)}
            for family, values in sorted(effects.items())
        },
        "standardized_change_by_split_family_severity": {
            split: {key: {"episodes": len(values), "min": min(values),
                          "median": median(values), "max": max(values)}
                    for key, values in sorted(groups.items())}
            for split, groups in effects_by_split_severity.items()
        },
        "paired_bias_drift_by_split_gain": {
            split: {gain: {"episodes": len(group["bias"]),
                           "bias_median": median(group["bias"]),
                           "drift_median": median(group["drift"]),
                           "median_absolute_paired_gap": median(group["absolute_gap"])}
                    for gain, group in sorted(groups.items(), key=lambda item: float(item[0]))}
            for split, groups in paired_bias_drift.items()
        },
        "calibration_pool_summary": (
            {str(gain): {"episodes": len(calibration.pools[level]),
                         "min": min(calibration.pools[level]),
                         "median": median(calibration.pools[level]),
                         "max": max(calibration.pools[level])}
             for level, gain in enumerate(GAIN_LEVELS)} if calibration else None
        ),
        "induced_out_of_training_range_feature_values": dict(induced_out_of_training_range),
        "exact_source_vector_duplicates": 0,
        "source_row_reuse_across_base_episodes": 0,
        "generated_vector_duplicates_across_base_episodes": 0,
        "paired_variants_crossing_partitions": 0,
        "padding_or_duration_mismatch": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    stem = f"episode_diagnostics_{condition}"
    (output / f"{stem}.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    with (output / f"episode_manifest_{condition}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    (output / f"{stem}.md").write_text(_report(diagnostics))
    if calibration is not None:
        payload = calibration_payload(calibration)
        payload["calibration_digest"] = calibration.digest
        (output / "magnitude_calibration.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return diagnostics


def _report(data: dict) -> str:
    lines = [
        f"# Stage 1 synthetic episode diagnostics — {data['bias_condition']}",
        "",
        "These are generator checks, not classifier results or real sensor-health labels. Episodes are artificial seeded orderings within one batch, gas, and concentration. Gas/concentration are laboratory grouping metadata only.",
        "",
        f"- Archive SHA-256: `{data['archive_sha256']}`; master seed: {data['master_seed']}.",
        f"- Original approved bias grid: {data['original_bias_grid']}; calibration digest: `{data['calibration_digest']}`.",
        "- Source and generated-vector duplicates across base episodes, source-row reuse, cross-partition paired variants, and padding/duration mismatches: all zero (generation stops if any occur).",
        "- Every variant has eight positions; intervention positions are 5–7 (zero-based). Background positions are unmodified. Each intervention changes only one shared selected eight-feature sensor block.",
        f"- Generator scales use Batches 1–4 only. Min/median/max: {data['scale_min']:.6g}/{data['scale_median']:.6g}/{data['scale_max']:.6g}.",
        "",
        "| Batch | Split | Source rows | Eight-row episodes | Used rows | Excluded rows |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for batch_text, values in data["coverage_by_batch"].items():
        batch = int(batch_text)
        split = "reference" if batch == 1 else ("train" if batch <= 4 else "validation" if batch <= 6 else "test")
        lines.append(f"| {batch} | {split} | {values['source_rows']} | {values['episodes']} | {values['used_source_rows']} | {values['excluded_source_rows']} |")
    lines += [
        "",
        "Batch 1 episodes are counted for source feasibility only; Track B variants are generated for Batches 2–10.",
        "",
        "| Split | Background | Synthetic drift | Synthetic attack |",
        "|---|---:|---:|---:|",
    ]
    for split, counts in data["class_counts_by_split"].items():
        lines.append(f"| {split} | {counts['background']} | {counts['synthetic_drift']} | {counts['synthetic_attack']} |")
    lines += [
        "",
        "| Split | Bias | Scaling | Noise | Freeze | Replay |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, counts in data["attack_family_severity_by_split"].items():
        totals = {family: sum(value for key, value in counts.items() if key.startswith(family + ":"))
                  for family in ATTACK_FAMILIES}
        lines.append(f"| {split} | {totals['bias']} | {totals['scaling']} | {totals['noise']} | {totals['freeze']} | {totals['replay']} |")
    lines += [
        "",
        "## Intervention diagnostics",
        "",
        "| Family | Episodes | Median mean absolute standardized change | Induced values beyond training range / affected values |",
        "|---|---:|---:|---:|",
    ]
    for family, values in data["mean_standardized_change_by_family"].items():
        induced = data['induced_out_of_training_range_feature_values'].get(family, 0)
        affected = values['episodes'] * len(ACTIVE_POSITIONS) * 8
        lines.append(f"| {family} | {values['episodes']} | {values['median']:.3g} | {induced}/{affected} ({induced/affected:.1%}) |")
    lines += [
        "",
        "## Per-split, per-family and severity magnitudes",
        "",
        "| Split | Family and severity | Episodes | Median mean standardized change | Range |",
        "|---|---|---:|---:|---:|",
    ]
    for split, groups in data["standardized_change_by_split_family_severity"].items():
        for key, values in groups.items():
            lines.append(f"| {split} | {key} | {values['episodes']} | {values['median']:.5g} | {values['min']:.5g}–{values['max']:.5g} |")
    lines += [
        "",
        "## Paired bias-versus-drift magnitude comparison",
        "",
        "Both variants use the same base episode and gain-level index. The median absolute paired gap shows residual magnitude mismatch; distribution matching does not guarantee equal magnitude within a pair.",
        "",
        "| Split | Gain level | Bias episodes | Bias median | Paired drift median | Median absolute paired gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, groups in data["paired_bias_drift_by_split_gain"].items():
        for gain, values in groups.items():
            lines.append(f"| {split} | {gain} | {values['episodes']} | {values['bias_median']:.5g} | {values['drift_median']:.5g} | {values['median_absolute_paired_gap']:.5g} |")
    if data["calibration_pool_summary"] is not None:
        lines += [
            "",
            "## Frozen training-only magnitude pools",
            "",
            f"The calibration contains {data['calibration_source_episode_count']} B2–4 source episodes and three pools. Full source IDs, row IDs, values, scale provenance, and digest are in `magnitude_calibration.json`.",
            "",
            "| Gain level | Pool entries | Minimum | Median | Maximum |",
            "|---:|---:|---:|---:|---:|",
        ]
        for gain, values in data["calibration_pool_summary"].items():
            lines.append(f"| {gain} | {values['episodes']} | {values['min']:.5g} | {values['median']:.5g} | {values['max']:.5g} |")
    bias_median = data["mean_standardized_change_by_family"]["bias"]["median"]
    drift_median = data["mean_standardized_change_by_family"]["gradual_gain"]["median"]
    lines += [
        "",
        f"Across all splits and levels, bias median is {bias_median:.3g} and gradual-gain median is {drift_median:.3g} ({bias_median/drift_median:.2f}×). This pooled ratio mixes severity levels and is descriptive only. Bias induces {data['induced_out_of_training_range_feature_values'].get('bias', 0)} values beyond B1-4 feature ranges. No severity used model outcomes.",
        "",
        "The condition-specific JSON contains exact family/severity counts by split and family counts by batch. The condition-specific manifest records source and grouping metadata for auditing; those columns must never be passed as model features.",
        "",
        "## Unresolved scientific limitations",
        "",
        "- Source order is artificial; a ramp versus an abrupt step is an injected signature. Equal active duration removes a trivial duration cue but cannot make intent causally identifiable. A gradual attacker can mimic synthetic drift.",
        "- The primary bias uses exact values from finite frozen training pools. Their reuse may be a learnable simulation signature. Static pairing and duration checks cannot rule out every future model-input shortcut.",
        "- All interventions edit extracted feature blocks rather than raw sensor waveforms. Numerical validity does not establish physical plausibility or security realism.",
        "- Background is unmodified, not verified healthy. The generator does not infer natural drift or real cyberattack labels.",
        "- Batches 7–10 are exploratory held-out evaluation because their audit coverage and score distributions were inspected before protocol freeze. No model was trained or evaluated here.",
        "- Model-input shortcut tests, operating-threshold selection, and statistical uncertainty checks remain for later approved stages.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=BIAS_CONDITIONS, required=True)
    args = parser.parse_args()
    result = inspect(args.archive, args.output, args.condition)
    print("Generated episode diagnostics:", {split: counts["background"] for split, counts in result["class_counts_by_split"].items()})


if __name__ == "__main__":
    main()
