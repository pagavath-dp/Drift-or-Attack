"""Deterministic artificial episodes and feature-level intervention variants.

This module creates simulation labels. It does not infer sensor health or attack
intent from UCI records, and it does not train a model.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from random import Random
from struct import pack

from .audit import APPROVED_ARCHIVE_SHA256, SPLITS
from .dataset import FEATURE_COUNT, Measurement
from .proxy import percentile


MASTER_SEED = 1729
EPISODE_LENGTH = 8
ONSET = 5
ACTIVE_POSITIONS = (5, 6, 7)
ATTACK_FAMILIES = ("bias", "scaling", "noise", "freeze", "replay")
GAIN_LEVELS = (0.05, 0.10, 0.20)
BIAS_LEVELS = (0.5, 1.0, 2.0)
NOISE_LEVELS = (0.25, 0.5, 1.0)
BIAS_CONDITIONS = ("matched_primary", "original_stress")


def _digest(*parts: object) -> str:
    content = "|".join(map(str, (MASTER_SEED, APPROVED_ARCHIVE_SHA256, *parts)))
    return sha256(content.encode("ascii")).hexdigest()


def _rng(*parts: object) -> Random:
    return Random(int(_digest(*parts)[:16], 16))


@dataclass(frozen=True)
class Episode:
    episode_id: str
    batch: int
    split: str
    gas: int  # grouping metadata only; never a model input
    concentration: float  # grouping metadata only; never a model input
    source_ids: tuple[tuple[int, int], ...]
    rows: tuple[tuple[float, ...], ...]
    rotation_index: int


@dataclass(frozen=True)
class Variant:
    episode_id: str
    batch: int
    split: str
    source_ids: tuple[tuple[int, int], ...]
    label: str
    family: str
    severity: float | None
    sensor: int  # generator metadata only; never a model input
    rows: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class MagnitudeCalibration:
    """Frozen B2-4 drift magnitudes, using feature scales fitted on B1-4."""

    scales_sha256: str
    # (episode ID, batch, source row IDs, magnitudes for the three gain levels)
    source_episodes: tuple[tuple[str, int, tuple[tuple[int, int], ...], tuple[float, ...]], ...]
    pools: tuple[tuple[float, ...], ...]
    digest: str


def _scales_sha256(scales: tuple[float, ...]) -> str:
    return sha256(pack(f"!{FEATURE_COUNT}d", *scales)).hexdigest()


def _sensor_and_sign(episode: Episode) -> tuple[int, int]:
    sensor = int(_digest("sensor", episode.episode_id)[:8], 16) % 16
    sign = 1 if int(_digest("sign", episode.episode_id)[:2], 16) % 2 == 0 else -1
    return sensor, sign


def standardized_change(source: Episode, variant: Variant, scales: tuple[float, ...]) -> float:
    """Mean absolute standardized change over 3 positions x 8 selected features."""
    if source.episode_id != variant.episode_id or len(scales) != FEATURE_COUNT:
        raise ValueError("source/variant or scale-width mismatch")
    first = variant.sensor * 8
    values = [abs(variant.rows[t][j] - source.rows[t][j]) / scales[j]
              for t in ACTIVE_POSITIONS for j in range(first, first + 8)]
    result = sum(values) / len(values)
    if not isfinite(result):
        raise ValueError("non-finite standardized change")
    return result


def calibration_payload(calibration: MagnitudeCalibration) -> dict:
    """JSON-ready provenance, including every source episode and pool value."""
    return {
        "archive_sha256": APPROVED_ARCHIVE_SHA256,
        "master_seed": MASTER_SEED,
        "feature_scale_rule": "max(IQR(B1-4)/1.349, 1e-12)",
        "scale_fit_batches": [1, 2, 3, 4],
        "magnitude_source_batches": [2, 3, 4],
        "episode_length": EPISODE_LENGTH,
        "active_positions_zero_based": list(ACTIVE_POSITIONS),
        "gain_levels": list(GAIN_LEVELS),
        "statistic": "mean(abs(modified-original)/feature_scale) over 3 active positions and 8 sensor features",
        "scales_sha256": calibration.scales_sha256,
        "source_episodes": [
            {"episode_id": episode_id, "batch": batch, "source_ids": [list(item) for item in source_ids],
             "magnitudes_by_gain_level": list(magnitudes)}
            for episode_id, batch, source_ids, magnitudes in calibration.source_episodes
        ],
        "pools_by_gain_level": [list(pool) for pool in calibration.pools],
    }


def _canonical_digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def fit_magnitude_calibration(episodes: list[Episode], scales: tuple[float, ...]) -> MagnitudeCalibration:
    """Construct all three severity pools solely from B2-4 base episodes."""
    if len(scales) != FEATURE_COUNT or any(not isfinite(x) or x <= 0 for x in scales):
        raise ValueError("128 positive finite B1-4 feature scales required")
    training = sorted((episode for episode in episodes if episode.split == "train"),
                      key=lambda episode: episode.episode_id)
    if not training or {episode.batch for episode in training} != {2, 3, 4}:
        raise ValueError("calibration requires training episodes in Batches 2-4")
    if any(episode.split != "train" or episode.batch not in (2, 3, 4) for episode in training):
        raise ValueError("calibration received a non-training episode")
    if len({episode.episode_id for episode in training}) != len(training):
        raise ValueError("duplicate calibration episode ID")
    records = []
    for episode in training:
        sensor, sign = _sensor_and_sign(episode)
        first = sensor * 8
        magnitudes = []
        for gain in GAIN_LEVELS:
            values = [abs(episode.rows[t][j] * (1 + sign * gain *
                       (t - ONSET + 1) / len(ACTIVE_POSITIONS)) - episode.rows[t][j]) / scales[j]
                      for t in ACTIVE_POSITIONS for j in range(first, first + 8)]
            magnitude = sum(values) / len(values)
            if not isfinite(magnitude) or magnitude <= 0:
                raise ValueError("calibration magnitude must be positive and finite")
            magnitudes.append(magnitude)
        records.append((episode.episode_id, episode.batch, episode.source_ids, tuple(magnitudes)))
    pools = tuple(tuple(sorted(record[3][level] for record in records))
                  for level in range(len(GAIN_LEVELS)))
    preliminary = MagnitudeCalibration(_scales_sha256(scales), tuple(records), pools, "")
    return MagnitudeCalibration(preliminary.scales_sha256, preliminary.source_episodes,
                                preliminary.pools, _canonical_digest(calibration_payload(preliminary)))


def matched_bias_magnitude(calibration: MagnitudeCalibration, episode: Episode, level: int) -> float:
    """Domain-separated stable draw; validation/test values never enter the pool."""
    if episode.split not in ("train", "validation", "test") or level not in range(len(GAIN_LEVELS)):
        raise ValueError("invalid episode split or severity level")
    if calibration.digest != _canonical_digest(calibration_payload(calibration)):
        raise ValueError("calibration digest mismatch")
    pool = calibration.pools[level]
    if not pool or any(not isfinite(value) or value <= 0 for value in pool):
        raise ValueError("invalid magnitude pool")
    index = int(_digest("matched-bias-pool", episode.episode_id, level), 16) % len(pool)
    return pool[index]


def _feature_fingerprint(features: tuple[float, ...]) -> str:
    return sha256(pack("!128d", *features)).hexdigest()


def construct_episodes(batches: dict[int, list[Measurement]]) -> tuple[list[Episode], dict[int, dict[str, int]]]:
    """Reject duplicate source vectors and form disjoint within-batch groups."""
    if set(batches) != set(range(1, 11)):
        raise ValueError("all ten original batches are required")
    seen_vectors: dict[str, tuple[int, int]] = {}
    seen_ids: set[tuple[int, int]] = set()
    episodes: list[Episode] = []
    coverage: dict[int, dict[str, int]] = {}
    for batch in range(1, 11):
        groups: dict[tuple[int, float], list[Measurement]] = defaultdict(list)
        for row in batches[batch]:
            if row.batch != batch or len(row.features) != FEATURE_COUNT:
                raise ValueError("batch or feature-width mismatch")
            source_id = (row.batch, row.line)
            if source_id in seen_ids:
                raise ValueError(f"source row reused: {source_id}")
            seen_ids.add(source_id)
            fingerprint = _feature_fingerprint(row.features)
            if fingerprint in seen_vectors:
                raise ValueError(f"exact feature-vector duplicate: {seen_vectors[fingerprint]} and {source_id}")
            seen_vectors[fingerprint] = source_id
            groups[(row.gas, row.concentration)].append(row)
        used = 0
        batch_episodes: list[Episode] = []
        for (gas, concentration), members in sorted(groups.items()):
            ordered = list(members)
            _rng("source-order", batch, gas, concentration).shuffle(ordered)
            for start in range(0, len(ordered) - EPISODE_LENGTH + 1, EPISODE_LENGTH):
                selected = ordered[start:start + EPISODE_LENGTH]
                source_ids = tuple((row.batch, row.line) for row in selected)
                episode_id = _digest("episode", *source_ids)[:24]
                batch_episodes.append(Episode(
                    episode_id, batch, SPLITS[batch], gas, concentration,
                    source_ids, tuple(row.features for row in selected), -1,
                ))
                used += EPISODE_LENGTH
        for rotation_index, episode in enumerate(sorted(batch_episodes, key=lambda x: x.episode_id)):
            episodes.append(Episode(
                episode.episode_id, episode.batch, episode.split, episode.gas,
                episode.concentration, episode.source_ids, episode.rows, rotation_index,
            ))
        coverage[batch] = {
            "source_rows": len(batches[batch]),
            "groups": len(groups),
            "groups_with_8_or_more": sum(len(members) >= EPISODE_LENGTH for members in groups.values()),
            "episodes": len(batch_episodes),
            "used_source_rows": used,
            "excluded_source_rows": len(batches[batch]) - used,
        }
    used_ids = [source_id for episode in episodes for source_id in episode.source_ids]
    if len(used_ids) != len(set(used_ids)):
        raise AssertionError("a source row appears in two base episodes")
    return episodes, coverage


def fit_generator_scales(batches: dict[int, list[Measurement]]) -> tuple[float, ...]:
    """Fit injection magnitudes to B1-4 only, without validation/test data."""
    if not set(range(1, 5)).issubset(batches):
        raise ValueError("Batches 1-4 required for generator scales")
    rows = [row for batch in range(1, 5) for row in batches[batch]]
    return tuple(max((percentile([row.features[j] for row in rows], 0.75)
                      - percentile([row.features[j] for row in rows], 0.25)) / 1.349,
                     1e-12) for j in range(FEATURE_COUNT))


def generate_variants(episode: Episode, scales: tuple[float, ...], *,
                      bias_condition: str = "original_stress",
                      calibration: MagnitudeCalibration | None = None) -> tuple[Variant, Variant, Variant]:
    if episode.split not in ("train", "validation", "test") or len(episode.rows) != EPISODE_LENGTH:
        raise ValueError("Track B requires an eight-record B2-10 episode")
    if len(scales) != FEATURE_COUNT or any(not isfinite(x) or x <= 0 for x in scales):
        raise ValueError("128 positive finite training-only scales required")
    if bias_condition not in BIAS_CONDITIONS:
        raise ValueError("unknown bias condition")
    if bias_condition == "matched_primary" and (calibration is None or
            calibration.scales_sha256 != _scales_sha256(scales)):
        raise ValueError("matched condition requires calibration for these scales")
    sensor, sign = _sensor_and_sign(episode)
    first = sensor * 8
    level = episode.rotation_index % 3
    family = ATTACK_FAMILIES[episode.rotation_index % len(ATTACK_FAMILIES)]
    drift_severity = GAIN_LEVELS[level]
    attack_severity = {
        "bias": (matched_bias_magnitude(calibration, episode, level)
                 if bias_condition == "matched_primary" and family == "bias" else BIAS_LEVELS[level]),
        "scaling": GAIN_LEVELS[level],
        "noise": NOISE_LEVELS[level],
        "freeze": None,
        "replay": None,
    }[family]
    original = episode.rows
    drift = list(original)
    attack = list(original)
    noise_rng = _rng("noise", episode.episode_id)
    for position in ACTIVE_POSITIONS:
        drift_row = list(original[position])
        attack_row = list(original[position])
        progress = (position - ONSET + 1) / len(ACTIVE_POSITIONS)
        for j in range(first, first + 8):
            drift_row[j] = original[position][j] * (1 + sign * drift_severity * progress)
            if family == "bias":
                attack_row[j] = original[position][j] + sign * attack_severity * scales[j]
            elif family == "scaling":
                attack_row[j] = original[position][j] * (1 + sign * attack_severity)
            elif family == "noise":
                attack_row[j] = original[position][j] + noise_rng.gauss(0, attack_severity * scales[j])
            elif family == "freeze":
                attack_row[j] = original[ONSET - 1][j]
            elif family == "replay":
                attack_row[j] = original[position - 4][j]
        drift[position] = tuple(drift_row)
        attack[position] = tuple(attack_row)
    for rows in (drift, attack):
        if any(not isfinite(value) for row in rows for value in row):
            raise ValueError("non-finite generated feature")
    common = (episode.episode_id, episode.batch, episode.split, episode.source_ids)
    return (
        Variant(*common, "background", "none", None, sensor, original),
        Variant(*common, "synthetic_drift", "gradual_gain", drift_severity, sensor, tuple(drift)),
        Variant(*common, "synthetic_attack", family, attack_severity, sensor, tuple(attack)),
    )


def changed_positions(source: Episode, variant: Variant) -> tuple[int, ...]:
    return tuple(t for t in range(EPISODE_LENGTH) if source.rows[t] != variant.rows[t])
