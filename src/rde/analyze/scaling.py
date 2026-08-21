"""Scaling-law fits for obstruction witnesses vs N (v0.2 Track C2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ScalingFit:
    witness: str
    slope: float
    intercept: float
    r_squared: float
    exponential_growth: bool
    n_sizes: int
    empirical_scaling: str


def _median_by_size(rows: list[dict[str, Any]], column: str) -> dict[int, float]:
    buckets: dict[int, list[float]] = {}
    for row in rows:
        size = row.get("size")
        val = row.get(column)
        if size is None or val is None:
            continue
        try:
            fv = float(val)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            buckets.setdefault(int(size), []).append(fv)
    return {s: float(np.median(vs)) for s, vs in buckets.items() if vs}


def fit_log_scaling(
    rows: list[dict[str, Any]],
    column: str,
    *,
    exp_slope_threshold: float = 0.3,
) -> ScalingFit | None:
    """Fit log(median witness) ~ a*N + b across sizes."""
    medians = _median_by_size(rows, column)
    if len(medians) < 2:
        return None
    sizes = np.array(sorted(medians.keys()), dtype=float)
    vals = np.array([medians[int(s)] for s in sizes], dtype=float)
    positive = vals > 0
    if positive.sum() < 2:
        return None
    sizes = sizes[positive]
    vals = vals[positive]
    log_vals = np.log(vals)
    coeffs = np.polyfit(sizes, log_vals, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    pred = slope * sizes + intercept
    ss_res = float(np.sum((log_vals - pred) ** 2))
    ss_tot = float(np.sum((log_vals - log_vals.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    exp_flag = slope > exp_slope_threshold
    scaling_str = f"log({column}) ~ {slope:.4f}*N + {intercept:.4f} (R2={r2:.3f})"
    return ScalingFit(
        witness=column,
        slope=slope,
        intercept=intercept,
        r_squared=r2,
        exponential_growth=exp_flag,
        n_sizes=len(sizes),
        empirical_scaling=scaling_str,
    )


def fit_all_witnesses(
    rows: list[dict[str, Any]],
    witness_columns: list[str],
) -> list[ScalingFit]:
    fits: list[ScalingFit] = []
    for col in witness_columns:
        fit = fit_log_scaling(rows, col)
        if fit is not None:
            fits.append(fit)
    return fits
