"""Batch 1-only reference-envelope feasibility analysis.

Scores describe distance from a reference; they are not ground-truth drift labels.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import median, stdev

from .dataset import FEATURE_COUNT, Measurement


MIN_GROUP_SIZE = 5
CUTOFF_QUANTILES = (0.90, 0.95, 0.99)


def percentile(values: list[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        raise ValueError("nonempty values and probability in [0, 1] required")
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (location - lower) * (ordered[upper] - ordered[lower])


def _center(rows: list[Measurement]) -> tuple[float, ...]:
    return tuple(median(row.features[j] for row in rows) for j in range(FEATURE_COUNT))


def _distance(features: tuple[float, ...], center: tuple[float, ...],
              scales: tuple[float, ...]) -> float:
    return sum(abs(value - base) / scale for value, base, scale
               in zip(features, center, scales, strict=True)) / FEATURE_COUNT


@dataclass(frozen=True)
class ReferenceEnvelope:
    group_sizes: dict[tuple[int, float], int]
    centers: dict[tuple[int, float], tuple[float, ...]]
    scales: tuple[float, ...]
    leave_one_out_scores: tuple[float, ...]
    cutoffs: dict[float, float]

    def status(self, row: Measurement) -> str:
        size = self.group_sizes.get((row.gas, row.concentration), 0)
        if size == 0:
            return "no_exact_reference"
        if size < MIN_GROUP_SIZE:
            return "sparse_reference"
        return "eligible"

    def score(self, row: Measurement) -> float | None:
        if self.status(row) != "eligible":
            return None
        return _distance(row.features, self.centers[(row.gas, row.concentration)], self.scales)


def fit_reference(rows: list[Measurement]) -> ReferenceEnvelope:
    if not rows or any(row.batch != 1 for row in rows):
        raise ValueError("reference fitting accepts Batch 1 only")
    if any(len(row.features) != FEATURE_COUNT for row in rows):
        raise ValueError("all reference rows need 128 features")
    groups: dict[tuple[int, float], list[Measurement]] = defaultdict(list)
    for row in rows:
        groups[(row.gas, row.concentration)].append(row)
    scales = tuple(max(stdev(row.features[j] for row in rows), 1e-12)
                   for j in range(FEATURE_COUNT))
    centers = {key: _center(members) for key, members in groups.items()
               if len(members) >= MIN_GROUP_SIZE}
    loo_scores: list[float] = []
    for members in groups.values():
        if len(members) < MIN_GROUP_SIZE:
            continue
        for i, row in enumerate(members):
            other_center = _center(members[:i] + members[i+1:])
            loo_scores.append(_distance(row.features, other_center, scales))
    if not loo_scores:
        raise ValueError("no Batch 1 reference groups have sufficient replicates")
    cutoffs = {q: percentile(loo_scores, q) for q in CUTOFF_QUANTILES}
    return ReferenceEnvelope(
        group_sizes={key: len(members) for key, members in groups.items()},
        centers=centers,
        scales=scales,
        leave_one_out_scores=tuple(loo_scores),
        cutoffs=cutoffs,
    )
