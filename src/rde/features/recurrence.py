"""Recurrence-order proxies with v0.3 identification gates."""

from __future__ import annotations

import numpy as np

# Main feature indices for trajectory science (unchanged)
MAIN_FEATURE_INDICES = (1, 2, 4, 8, 16)

# Minimum contiguous diagnostic grid points to attempt order identification
MIN_IDENTIFICATION_SLICES = 12
MIN_WITHHELD_SLICES = 3


def _krylov_rank(vectors: list[np.ndarray], tol: float = 1e-10) -> int:
    if not vectors:
        return 0
    mat = np.column_stack([np.asarray(v, dtype=float).ravel() for v in vectors])
    if mat.size == 0:
        return 0
    _, s, _ = np.linalg.svd(mat, full_matrices=False)
    s = s[s > tol * max(float(s[0]), 1.0)]
    return int(s.size)


def recurrence_identification_eligible(
    n_measured_slices: int,
    *,
    min_slices: int = MIN_IDENTIFICATION_SLICES,
) -> bool:
    """Five main slices cannot certify recurrence order."""
    return n_measured_slices >= min_slices


def estimate_recurrence_order(
    indices: list[int],
    slices: list[np.ndarray],
    *,
    max_order: int = 8,
    withheld_slices: list[np.ndarray] | None = None,
) -> dict[str, float]:
    if len(slices) < 2:
        return {"recurrence.identification_eligible": 0.0}

    order = sorted(range(len(indices)), key=lambda i: indices[i])
    sorted_slices = [slices[i] for i in order]
    n_slices = len(sorted_slices)
    eligible = recurrence_identification_eligible(n_slices)

    full_rank = _krylov_rank(sorted_slices)
    best_order = len(sorted_slices)
    for k in range(1, min(max_order, len(sorted_slices)) + 1):
        if _krylov_rank(sorted_slices[:k]) >= full_rank:
            best_order = k
            break
    orbit_rank = _krylov_rank(sorted_slices)

    out: dict[str, float] = {
        "recurrence.krylov_rank_full": float(full_rank),
        "recurrence.krylov_orbit_dimension": float(orbit_rank),
        "recurrence.estimated_order": float(best_order),
        "recurrence.n_slices": float(n_slices),
        "recurrence.identification_eligible": 1.0 if eligible else 0.0,
        "recurrence.promotion_eligible": 1.0 if eligible and best_order <= 32 else 0.0,
    }

    if withheld_slices and eligible and len(withheld_slices) >= MIN_WITHHELD_SLICES:
        train_rank = _krylov_rank(sorted_slices)
        test_rank = _krylov_rank(sorted_slices + withheld_slices)
        out["recurrence.withheld_rank_delta"] = float(test_rank - train_rank)
        out["recurrence.withheld_stable"] = 1.0 if test_rank == train_rank else 0.0
    else:
        out["recurrence.withheld_rank_delta"] = float("nan")
        out["recurrence.withheld_stable"] = 0.0

    return out
