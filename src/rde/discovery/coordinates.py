"""Coordinate-system search — linear projections (Phase 6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rde.discovery.latent import pca_latent
from rde.features.numeric import safe_pearson_r


@dataclass
class CoordinateResult:
    method: str
    n_components: int
    explained_variance: list[float]
    update_simplicity: float
    target_correlation: float


def search_linear_coordinates(
    feature_matrix: np.ndarray,
    *,
    target: np.ndarray,
    update_proxy: np.ndarray | None = None,
    max_components: int = 16,
) -> CoordinateResult:
    """Find low-dim linear coords maximizing target correlation (single SVD)."""
    max_k = min(max_components, feature_matrix.shape[1], feature_matrix.shape[0])
    if max_k < 1:
        return CoordinateResult(
            method="pca_linear",
            n_components=0,
            explained_variance=[],
            update_simplicity=float("nan"),
            target_correlation=float("nan"),
        )
    res = pca_latent(feature_matrix, n_components=max_k, target=None)
    t = target.copy()
    mask = np.isfinite(t)
    if not mask.any():
        t[~mask] = 0.0
    else:
        t[~mask] = float(np.nanmean(t[mask]))

    best_corr = -1.0
    best_k = 1
    best_var: list[float] = []
    codes = res.latent_codes
    for k in range(1, max_k + 1):
        corrs = []
        for i in range(k):
            c = codes[:, i]
            if np.std(c) > 0 and np.std(t) > 0:
                corrs.append(abs(safe_pearson_r(c, t)))
        c = max(corrs) if corrs else 0.0
        if c > best_corr:
            best_corr = c
            best_k = k
            best_var = res.explained_variance_ratio[:k]

    update_sim = float("nan")
    if update_proxy is not None and np.isfinite(update_proxy).any() and codes.size:
        from rde.discovery.latent import ridge_predict_columns

        k_use = min(best_k, codes.shape[1])
        _, r2 = ridge_predict_columns(codes[:, :k_use], update_proxy)
        update_sim = float(r2)
    return CoordinateResult(
        method="pca_linear",
        n_components=best_k,
        explained_variance=best_var,
        update_simplicity=update_sim,
        target_correlation=best_corr,
    )
