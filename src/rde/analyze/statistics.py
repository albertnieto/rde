"""Cluster bootstrap, FDR, and permutation tests for RDE v0.3."""

from __future__ import annotations

from typing import Any

import numpy as np
from rde.features.numeric import safe_pearson_r


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[tuple[float, float, bool]]:
    """Return (p, q, reject) for each p-value."""
    if not p_values:
        return []
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]
    q_vals = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        q = min(prev, sorted_p[i] * m / rank)
        q_vals[i] = q
        prev = q
    out: list[tuple[float, float, bool]] = [(0.0, 0.0, False)] * m
    for i, orig_idx in enumerate(order):
        out[orig_idx] = (float(p_values[orig_idx]), float(q_vals[i]), float(q_vals[i]) <= alpha)
    return out


def cluster_bootstrap_pearson(
    rows: list[dict[str, Any]],
    x_col: str,
    y_col: str,
    *,
    n_bootstrap: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap Pearson r by resampling instance_id clusters."""
    by_inst: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        iid = str(row.get("instance_id", ""))
        xv = row.get(x_col)
        yv = row.get(y_col)
        if xv is None or yv is None:
            continue
        try:
            xf, yf = float(xv), float(yv)
        except (TypeError, ValueError):
            continue
        if np.isfinite(xf) and np.isfinite(yf):
            by_inst.setdefault(iid, []).append((xf, yf))

    if len(by_inst) < 2:
        return {"pearson_r": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    inst_ids = list(by_inst.keys())

    def _pearson_from_sample(ids: list[str]) -> float:
        xs: list[float] = []
        ys: list[float] = []
        for iid in ids:
            for x, y in by_inst[iid]:
                xs.append(x)
                ys.append(y)
        if len(xs) < 2:
            return float("nan")
        return safe_pearson_r(xs, ys)

    rng = np.random.default_rng(seed)
    obs = _pearson_from_sample(inst_ids)
    boots: list[float] = []
    for _ in range(n_bootstrap):
        sample = rng.choice(inst_ids, size=len(inst_ids), replace=True).tolist()
        r = _pearson_from_sample(sample)
        if np.isfinite(r):
            boots.append(r)
    if not boots:
        return {"pearson_r": obs, "ci_low": float("nan"), "ci_high": float("nan")}
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"pearson_r": obs, "ci_low": float(lo), "ci_high": float(hi)}


def permutation_max_statistic(
    rows: list[dict[str, Any]],
    candidate_cols: list[str],
    target_col: str,
    *,
    n_perm: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Max |r| under a global permutation of the complete target vector.

    The selection correction must cover every searched candidate in one
    universe.  Stratifying this outer null by size/generator would test a
    different, narrower hypothesis and can understate the best-of-catalog
    penalty.  Missing targets remain missing; only finite target entries are
    permuted.
    """

    def max_abs_r(data: list[dict[str, Any]]) -> float:
        best = 0.0
        ys = [float(r[target_col]) for r in data if np.isfinite(float(r.get(target_col, float("nan"))))]
        if len(ys) < 3:
            return 0.0
        for col in candidate_cols:
            xs = []
            paired = []
            for r in data:
                xv = r.get(col)
                yv = r.get(target_col)
                if xv is None or yv is None:
                    continue
                try:
                    xf, yf = float(xv), float(yv)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(xf) and np.isfinite(yf):
                    paired.append((xf, yf))
            if len(paired) < 3:
                continue
            xs_arr = np.array([p[0] for p in paired])
            ys_arr = np.array([p[1] for p in paired])
            r = abs(safe_pearson_r(xs_arr, ys_arr))
            best = max(best, r)
        return best

    obs = max_abs_r(rows)
    rng = np.random.default_rng(seed)
    finite_indices = [
        index
        for index, row in enumerate(rows)
        if np.isfinite(float(row.get(target_col, float("nan"))))
    ]
    finite_targets = [rows[index][target_col] for index in finite_indices]
    null_max: list[float] = []
    for _ in range(n_perm):
        shuffled = list(rng.permutation(finite_targets))
        permuted = [dict(row) for row in rows]
        for index, value in zip(finite_indices, shuffled):
            permuted[index][target_col] = value
        null_max.append(max_abs_r(permuted))
    # Include the observed statistic in the randomization reference set.
    # This prevents a finite permutation test from reporting p=0 and is the
    # standard conservative correction for a Monte Carlo p-value.
    exceedances = sum(value >= obs for value in null_max)
    p = (exceedances + 1) / (len(null_max) + 1)
    return {"observed_max_abs_r": obs, "permutation_p": p, "n_perm": float(n_perm)}
