"""Prospective power planning from persisted RDE campaign data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rde.analyze.statistics import cluster_bootstrap_pearson, permutation_max_statistic


@dataclass
class PowerPlan:
    instances_per_generator: int
    target_power: float
    min_effect_r: float
    estimated_variance: float
    within_instance_correlation: float
    locked: bool = False
    notes: list[str] | None = None


def estimate_target_variance(rows: list[dict[str, Any]], target_col: str) -> float:
    vals = [float(r[target_col]) for r in rows if np.isfinite(float(r.get(target_col, float("nan"))))]
    return float(np.var(vals)) if len(vals) > 1 else float("nan")


def estimate_within_instance_correlation(
    rows: list[dict[str, Any]],
    target_col: str,
) -> float:
    by_inst: dict[str, list[float]] = {}
    for row in rows:
        iid = str(row.get("instance_id", ""))
        y = row.get(target_col)
        if y is None:
            continue
        try:
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if np.isfinite(yf):
            by_inst.setdefault(iid, []).append(yf)
    if len(by_inst) < 2:
        return float("nan")
    icc_samples: list[float] = []
    for vals in by_inst.values():
        if len(vals) >= 2:
            icc_samples.append(float(np.std(vals) / max(np.mean(vals), 1e-12)))
    return float(np.median(icc_samples)) if icc_samples else float("nan")


def simulate_power(
    rows: list[dict[str, Any]],
    *,
    target_col: str = "metric.representation_complexity",
    candidate_cols: list[str] | None = None,
    min_effect_r: float = 0.35,
    target_power: float = 0.90,
    max_instances_cap: int = 500,
) -> PowerPlan:
    """Select smallest balanced instances-per-generator count reaching target power."""
    cols = candidate_cols or [c for c in rows[0].keys() if c.startswith("matrix.")][:5] if rows else []
    var = estimate_target_variance(rows, target_col)
    icc = estimate_within_instance_correlation(rows, target_col)

    best_n = max_instances_cap
    notes: list[str] = []
    for n in (50, 100, 150, 200, 250, 300, 400, 500):
        if n > max_instances_cap:
            break
        sample = rows[: min(len(rows), n * 5)]
        if len(sample) < 10:
            continue
        boot = cluster_bootstrap_pearson(sample, cols[0] if cols else "size", target_col, n_bootstrap=100)
        ci_width = boot["ci_high"] - boot["ci_low"] if np.isfinite(boot["ci_high"]) else float("inf")
        perm = permutation_max_statistic(sample, cols, target_col, n_perm=50) if cols else {"permutation_p": 1.0}
        if ci_width < 2 * (1 - min_effect_r) and perm.get("permutation_p", 1.0) < 0.10:
            best_n = n
            notes.append(f"n={n} achieves proxy power at r>={min_effect_r}")
            break
    else:
        notes.append(f"required n may exceed cap {max_instances_cap}; revise scientific question")

    return PowerPlan(
        instances_per_generator=best_n,
        target_power=target_power,
        min_effect_r=min_effect_r,
        estimated_variance=var,
        within_instance_correlation=icc,
        locked=False,
        notes=notes,
    )


def power_plan_to_dict(plan: PowerPlan) -> dict[str, Any]:
    return {
        "instances_per_generator": plan.instances_per_generator,
        "target_power": plan.target_power,
        "min_effect_r": plan.min_effect_r,
        "estimated_variance": plan.estimated_variance,
        "within_instance_correlation": plan.within_instance_correlation,
        "locked": plan.locked,
        "notes": plan.notes or [],
    }
