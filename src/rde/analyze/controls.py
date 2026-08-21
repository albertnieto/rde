"""Null baselines and positive controls for RDE v0.3."""

from __future__ import annotations

from typing import Any

import numpy as np


def nr_baseline_r2(rows: list[dict[str, Any]], target_col: str) -> float:
    """R² of predicting target from size + family_index only."""
    ys = []
    xs_n = []
    xs_r = []
    for row in rows:
        y = row.get(target_col)
        if y is None:
            continue
        try:
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(yf):
            continue
        ys.append(yf)
        xs_n.append(float(row.get("size", 0)))
        xs_r.append(float(row.get("family_index", 0)))
    if len(ys) < 3:
        return float("nan")
    y_arr = np.array(ys)
    X = np.column_stack([np.ones(len(ys)), xs_n, xs_r])
    coef, _, _, _ = np.linalg.lstsq(X, y_arr, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y_arr - pred) ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    if ss_tot <= 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def generator_median_baseline(
    rows: list[dict[str, Any]],
    target_col: str,
) -> dict[str, float]:
    """Mean absolute error vs generator-stratified median."""
    groups: dict[str, list[float]] = {}
    for row in rows:
        gen = str(row.get("generator", "unknown"))
        y = row.get(target_col)
        if y is None:
            continue
        try:
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if np.isfinite(yf):
            groups.setdefault(gen, []).append(yf)
    medians = {g: float(np.median(v)) for g, v in groups.items() if v}
    errors: list[float] = []
    for row in rows:
        gen = str(row.get("generator", "unknown"))
        y = row.get(target_col)
        if y is None or gen not in medians:
            continue
        try:
            errors.append(abs(float(y) - medians[gen]))
        except (TypeError, ValueError):
            continue
    return {"mae": float(np.mean(errors)) if errors else float("nan"), "n_groups": float(len(medians))}


def permuted_target_null(
    rows: list[dict[str, Any]],
    target_col: str,
    *,
    seed: int = 0,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    targets = [row.get(target_col) for row in rows]
    shuffled = list(rng.permutation(targets))
    return [{**row, target_col: t} for row, t in zip(rows, shuffled)]
