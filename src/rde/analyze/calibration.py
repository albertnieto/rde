"""Calibrate representation_complexity against known synthetic families."""

from __future__ import annotations

from typing import Any

import numpy as np

from rde.analyze.query import distribution_summary


def complexity_by_group(
    rows: list[dict[str, Any]],
    *,
    metric: str = "metric.representation_complexity",
    group_by: str = "size",
) -> list[dict[str, Any]]:
    """Distribution summary of a complexity metric grouped by size or generator."""
    return distribution_summary(rows, metric, group_by=group_by)


def separation_score(
    rows: list[dict[str, Any]],
    *,
    metric: str = "metric.representation_complexity",
    label_key: str = "generator",
) -> float:
    """Between-group / within-group variance ratio (higher = better separation)."""
    groups: dict[Any, list[float]] = {}
    for row in rows:
        label = row.get(label_key)
        val = row.get(metric)
        if label is None or not isinstance(val, (int, float)) or not np.isfinite(float(val)):
            continue
        groups.setdefault(label, []).append(float(val))
    if len(groups) < 2:
        return float("nan")
    all_vals = [v for vals in groups.values() for v in vals]
    grand_mean = float(np.mean(all_vals))
    between = sum(len(vals) * (float(np.mean(vals)) - grand_mean) ** 2 for vals in groups.values())
    within = sum(sum((v - float(np.mean(vals))) ** 2 for v in vals) for vals in groups.values())
    if within <= 0:
        return float("nan")
    return float(between / within)
