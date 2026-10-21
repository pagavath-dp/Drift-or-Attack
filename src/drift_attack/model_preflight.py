"""Mandatory static leakage and simulation-signature review before model fitting."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

from .audit import APPROVED_ARCHIVE_SHA256, file_sha256, reject_frozen_output
from .dataset import load_archive
from .episodes import (ACTIVE_POSITIONS, GAIN_LEVELS, calibration_payload,
                       changed_positions, construct_episodes, fit_generator_scales,
                       fit_magnitude_calibration, generate_variants,
                       standardized_change)
from .features import FEATURE_GROUPS, episode_features, fit_feature_scale


def exact_copy_signature(features) -> dict[str, int]:
    """Count sensor blocks with exact-zero variation or lag-four differences."""
    if features.shape != (len(FEATURE_GROUPS) * 128,):
        raise ValueError("unexpected episode feature width")
    return {
        "zero_active_variation_sensor_blocks": sum(
            all(features[512 + sensor * 8:512 + (sensor + 1) * 8] == 0)
            for sensor in range(16)),
        "zero_lag_four_sensor_blocks": sum(
            all(features[640 + sensor * 8:640 + (sensor + 1) * 8] == 0)
            for sensor in range(16)),
    }


def inspect(archive: Path, reports: Path) -> dict:
    reject_frozen_output(reports)
    if file_sha256(archive) != APPROVED_ARCHIVE_SHA256:
        raise ValueError("unapproved source archive")
    batches = load_archive(archive)
    episodes, coverage = construct_episodes(batches)
    generator_scales = fit_generator_scales(batches)
    feature_scale = fit_feature_scale(batches)
    calibration = fit_magnitude_calibration(episodes, generator_scales)
    recorded = json.loads((reports / "magnitude_calibration.json").read_text())
    if recorded.pop("calibration_digest") != calibration.digest or recorded != calibration_payload(calibration):
        raise ValueError("frozen calibration record differs from current B1-4 data/code")
    for condition in ("matched_primary", "original_stress"):
        prior = json.loads((reports / f"episode_diagnostics_{condition}.json").read_text())
        if prior["archive_sha256"] != APPROVED_ARCHIVE_SHA256 or prior["calibration_digest"] != (
                calibration.digest if condition == "matched_primary" else None):
            raise ValueError("condition diagnostics provenance mismatch")
        for key in ("exact_source_vector_duplicates", "source_row_reuse_across_base_episodes",
                    "generated_vector_duplicates_across_base_episodes", "paired_variants_crossing_partitions",
                    "padding_or_duration_mismatch"):
            if prior[key] != 0:
                raise ValueError(f"leakage gate failed: {key}")
    by_condition: dict[str, dict] = {}
    used_source_ids: dict[str, set] = {split: set() for split in ("train", "validation", "test")}
    for condition in ("matched_primary", "original_stress"):
        paired: dict[str, dict[str, dict[str, list[float]]]] = {
            split: defaultdict(lambda: {"attack": [], "drift": [], "gap": []})
            for split in ("train", "validation", "test")
        }
        bias_values: dict[str, list[float]] = defaultdict(list)
        feature_fingerprints: dict[str, str] = {}
        class_counts: dict[str, Counter] = defaultdict(Counter)
        exact_copy_counts: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
        for episode in episodes:
            if episode.batch == 1:
                continue
            if condition == "matched_primary":
                for source_id in episode.source_ids:
                    if source_id in used_source_ids[episode.split]:
                        raise ValueError("source row reused across base episodes")
                    used_source_ids[episode.split].add(source_id)
            variants = generate_variants(episode, generator_scales, bias_condition=condition,
                                         calibration=calibration if condition == "matched_primary" else None)
            if {variant.label for variant in variants} != {
                    "background", "synthetic_drift", "synthetic_attack"}:
                raise ValueError("incomplete paired triplet")
            for variant in variants:
                if (variant.split != episode.split or variant.batch != episode.batch or
                        variant.source_ids != episode.source_ids or variant.episode_id != episode.episode_id):
                    raise ValueError("paired variant left source partition")
                expected = () if variant.label == "background" else ACTIVE_POSITIONS
                if changed_positions(episode, variant) != expected or len(variant.rows) != 8:
                    raise ValueError("label-specific padding or duration artifact")
                features = episode_features(variant.rows, feature_scale)
                signatures = exact_copy_signature(features)
                family = "background" if variant.label == "background" else variant.family
                signature_counts = exact_copy_counts[episode.split][family]
                signature_counts["episodes"] += 1
                for name, blocks in signatures.items():
                    signature_counts[name] += int(blocks > 0)
                    signature_counts[name + "_total_blocks"] += blocks
                if family == "freeze" and signatures["zero_active_variation_sensor_blocks"] != 1:
                    raise ValueError("freeze exact-copy signature changed")
                if family == "replay" and signatures["zero_lag_four_sensor_blocks"] != 1:
                    raise ValueError("replay exact-copy signature changed")
                fingerprint = sha256(features.tobytes()).hexdigest()
                owner = feature_fingerprints.get(fingerprint)
                if owner is not None and owner != episode.episode_id:
                    raise ValueError("exact model-input collision across source episodes")
                feature_fingerprints[fingerprint] = episode.episode_id
                class_counts[episode.split][variant.label] += 1
            attack = variants[2]
            if attack.family == "bias" and condition == "matched_primary":
                level = episode.rotation_index % len(GAIN_LEVELS)
                if attack.severity not in calibration.pools[level]:
                    raise ValueError("primary bias value absent from frozen training pool")
                bias_values[episode.split].append(attack.severity)
            level = episode.rotation_index % len(GAIN_LEVELS)
            key = f"{attack.family}:level={level}:parameter={attack.severity if condition == 'original_stress' else ('frozen_pool' if attack.family == 'bias' else attack.severity)}"
            group = paired[episode.split][key]
            attack_magnitude = standardized_change(episode, attack, generator_scales)
            drift_magnitude = standardized_change(episode, variants[1], generator_scales)
            group["attack"].append(attack_magnitude)
            group["drift"].append(drift_magnitude)
            group["gap"].append(abs(attack_magnitude - drift_magnitude))
        by_condition[condition] = {
            "class_counts_by_split": {split: dict(counts) for split, counts in class_counts.items()},
            "exact_copy_signatures_by_split_family": {
                split: {family: dict(counts) for family, counts in sorted(families.items())}
                for split, families in exact_copy_counts.items()},
            "paired_magnitude_by_split_family_severity": {
                split: {key: {"denominator_episodes": len(values["attack"]),
                              "attack_median": median(values["attack"]),
                              "paired_drift_median": median(values["drift"]),
                              "median_absolute_pair_gap": median(values["gap"])}
                        for key, values in sorted(groups.items())}
                for split, groups in paired.items()
            },
        }
        if condition == "matched_primary":
            by_condition[condition]["finite_pool_signature"] = {
                split: {"bias_episodes": len(values),
                        "distinct_exact_magnitudes": len(set(values)),
                        "episodes_with_reused_magnitude_within_split": sum(
                            count for count in Counter(values).values() if count > 1)}
                for split, values in bias_values.items()
            }
            by_condition[condition]["cross_split_shared_bias_magnitudes"] = {
                f"{left}_and_{right}": len(set(bias_values[left]) & set(bias_values[right]))
                for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
            }
    if len(set().union(*used_source_ids.values())) != sum(map(len, used_source_ids.values())):
        raise ValueError("source row crossed partitions")
    result = {
        "archive_sha256": APPROVED_ARCHIVE_SHA256,
        "calibration_digest": calibration.digest,
        "scale_fit_batches": [1, 2, 3, 4],
        "magnitude_pool_batches": [2, 3, 4],
        "model_input_width": len(FEATURE_GROUPS) * 128,
        "model_input_groups": list(FEATURE_GROUPS),
        "model_input_metadata_fields": [],
        "variant_length": 8,
        "intervention_positions_zero_based": list(ACTIVE_POSITIONS),
        "coverage_by_batch": {str(batch): row for batch, row in coverage.items()},
        "conditions": by_condition,
        "limitations": [
            "Freeze and replay are exact-copy controls; their signatures do not establish general real-world attack detection.",
            "Exact finite-pool bias magnitudes repeat and may form a simulation signature.",
            "Static checks establish equal shape, source pairing and split isolation; they cannot prove the future model lacks all shortcuts.",
            "Episode order is artificial and interventions edit extracted features, not sensor waveforms.",
            "Batch 7-10 evaluation remains exploratory because audit distributions were previously inspected.",
        ],
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "ml_preflight.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.archive, args.reports)
    print("Preflight passed; calibration", result["calibration_digest"])


if __name__ == "__main__":
    main()
