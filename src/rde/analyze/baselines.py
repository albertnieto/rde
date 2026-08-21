"""Baseline comparator for scoped lower bounds (v0.2 Track C3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rde.analyze.scaling import ScalingFit


@dataclass
class BaselineMatch:
    baseline_id: str
    witness: str
    matched: bool
    reason: str


def _theorem005_slice_rank_match(fit: ScalingFit) -> BaselineMatch:
    """THEOREM-005: balanced slice rank ~ 2^(N/2) => log slope ~ 0.693/2 ≈ 0.35."""
    matched = fit.exponential_growth and fit.slope >= 0.25
    reason = (
        f"slope={fit.slope:.3f} suggests ~2^(N/2) growth"
        if matched
        else f"slope={fit.slope:.3f} below THEOREM-005 threshold"
    )
    return BaselineMatch("THEOREM-005", fit.witness, matched, reason)


def _theorem020_temporal_match(rows: list[dict[str, Any]], witness: str) -> BaselineMatch:
    """THEOREM-020: temporal spectrum collapses to success count — low recurrence order."""
    col = witness
    vals = [float(r[col]) for r in rows if col in r and np.isfinite(float(r[col]))]
    if not vals:
        return BaselineMatch("THEOREM-020", witness, False, "no data")
    med = float(np.median(vals))
    matched = med <= 4.0 and "recurrence" in witness
    reason = f"median={med:.2f}; low order consistent with success-count collapse"
    return BaselineMatch("THEOREM-020", witness, matched, reason)


def _exp014_reference() -> BaselineMatch:
    return BaselineMatch(
        "EXP-014",
        "trajectory_u_acum",
        False,
        "reference only — scoped trajectory u_acum feasibility negatives",
    )


def compare_baselines(
    fits: list[ScalingFit],
    rows: list[dict[str, Any]],
) -> list[BaselineMatch]:
    """Tag witnesses against known scoped negatives."""
    matches: list[BaselineMatch] = []
    for fit in fits:
        w = fit.witness.lower()
        if "slice_rank" in w or "balanced_slice" in w or fit.witness.endswith("slice_rank_end"):
            matches.append(_theorem005_slice_rank_match(fit))
        if "recurrence" in w:
            matches.append(_theorem020_temporal_match(rows, fit.witness))
    matches.append(_exp014_reference())
    return matches
