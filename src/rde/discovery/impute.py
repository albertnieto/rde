"""Column imputation helpers shared by latent / autoencoder paths."""

from __future__ import annotations

import numpy as np


def impute_matrix(X: np.ndarray) -> np.ndarray:
    """Replace NaN with column means (0 when column is all-NaN)."""
    out = np.asarray(X, dtype=float).copy()
    finite = np.isfinite(out)
    counts = finite.sum(axis=0)
    totals = np.sum(np.where(finite, out, 0.0), axis=0)
    means = np.divide(
        totals,
        counts,
        out=np.zeros_like(totals, dtype=float),
        where=counts > 0,
    )
    out[~finite] = np.broadcast_to(means, out.shape)[~finite]
    return out


def impute_column(col: np.ndarray) -> np.ndarray:
    """Replace NaN in a 1-D column."""
    out = np.asarray(col, dtype=float).copy()
    mask = np.isfinite(out)
    if not mask.all():
        fill = float(np.nanmean(out[mask])) if mask.any() else 0.0
        out[~mask] = fill
    return out
