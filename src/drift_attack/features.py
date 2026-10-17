"""Numeric-only episode features. No laboratory or generator metadata enters here."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import FEATURE_COUNT, Measurement


FEATURE_GROUPS = (
    "pre_mean", "active_minus_pre_mean", "onset_step", "active_slope",
    "active_variation", "lag_four_similarity",
)
INPUT_WIDTH = FEATURE_COUNT * len(FEATURE_GROUPS)


@dataclass(frozen=True)
class FeatureScale:
    center: np.ndarray
    spread: np.ndarray


def fit_feature_scale(batches: dict[int, list[Measurement]]) -> FeatureScale:
    """Fit only on unmodified B1-4 source rows, before any model or variant."""
    if not all(batch in batches for batch in range(1, 5)):
        raise ValueError("Batches 1-4 required for feature scaling")
    rows = np.asarray([row.features for batch in range(1, 5) for row in batches[batch]], dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != FEATURE_COUNT or not np.isfinite(rows).all():
        raise ValueError("invalid B1-4 feature matrix")
    center = np.median(rows, axis=0)
    spread = np.maximum((np.percentile(rows, 75, axis=0) - np.percentile(rows, 25, axis=0)) / 1.349,
                        1e-12)
    return FeatureScale(center, spread)


def episode_features(rows: tuple[tuple[float, ...], ...], scale: FeatureScale) -> np.ndarray:
    """Six 128-channel summaries from an eight-record artificial sequence."""
    values = np.asarray(rows, dtype=np.float64)
    if values.shape != (8, FEATURE_COUNT) or not np.isfinite(values).all():
        raise ValueError("expected eight finite 128-feature rows")
    if scale.center.shape != (FEATURE_COUNT,) or scale.spread.shape != (FEATURE_COUNT,):
        raise ValueError("invalid feature scale shape")
    if not np.isfinite(scale.center).all() or not np.isfinite(scale.spread).all() or np.any(scale.spread <= 0):
        raise ValueError("invalid feature scale values")
    z = (values - scale.center) / scale.spread
    pre = z[:5].mean(axis=0)
    active = z[5:]
    groups = (
        pre,
        active.mean(axis=0) - pre,
        z[5] - z[4],
        (z[7] - z[5]) / 2,
        (np.abs(z[6] - z[5]) + np.abs(z[7] - z[6])) / 2,
        (np.abs(z[5] - z[1]) + np.abs(z[6] - z[2]) + np.abs(z[7] - z[3])) / 3,
    )
    result = np.concatenate(groups)
    if result.shape != (INPUT_WIDTH,) or not np.isfinite(result).all():
        raise ValueError("invalid constructed feature vector")
    return result
