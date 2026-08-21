"""2D manifold embedding of (instance, r) feature rows (Phase 4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rde.analyze.feature_table import FeatureTable
from rde.discovery.latent import pca_latent
from rde.features.numeric import safe_pearson_r


@dataclass
class ManifoldResult:
    n_components: int
    explained_variance: list[float]
    target_correlation: float
    embedding: list[list[float]]


def embed_trajectory_manifold(
    feature_matrix: np.ndarray,
    *,
    target: np.ndarray,
    n_components: int = 2,
) -> ManifoldResult:
    """PCA embedding of feature rows; report max |corr| with target."""
    k = min(n_components, feature_matrix.shape[0], feature_matrix.shape[1])
    if k < 1:
        return ManifoldResult(0, [], float("nan"), [])
    res = pca_latent(feature_matrix, n_components=k, target=None)
    codes = res.latent_codes
    t = target.copy()
    mask = np.isfinite(t)
    if not mask.any():
        return ManifoldResult(k, res.explained_variance_ratio[:k], float("nan"), codes.tolist())

    best = -1.0
    for j in range(codes.shape[1]):
        col = codes[:, j]
        m = mask & np.isfinite(col)
        if m.sum() < 3:
            continue
        xc = col[m]
        yc = t[m]
        if float(np.std(xc)) == 0.0:
            continue
        r = safe_pearson_r(xc, yc)
        if np.isfinite(r):
            best = max(best, abs(r))
    return ManifoldResult(
        n_components=k,
        explained_variance=res.explained_variance_ratio[:k],
        target_correlation=best if best >= 0 else float("nan"),
        embedding=codes.tolist(),
    )


def manifold_from_rows(
    rows: list[dict[str, Any]],
    target: str,
    *,
    feature_columns: list[str] | None = None,
    n_components: int = 2,
) -> ManifoldResult:
    if not rows:
        return ManifoldResult(0, [], float("nan"), [])
    if feature_columns is None:
        feature_columns = [
            c
            for c in sorted(rows[0])
            if isinstance(rows[0].get(c), (int, float))
            and c not in {target, "size", "seed", "family_index"}
            and not str(c).startswith("metric.")
        ][:24]
    table = FeatureTable.from_rows(rows, columns=[*feature_columns, target])
    X = table.data[:, : len(feature_columns)]
    y = table.column(target)
    col_ok = np.all(np.isfinite(X), axis=0)
    if not col_ok.any():
        return ManifoldResult(0, [], float("nan"), [])
    X = X[:, col_ok]
    return embed_trajectory_manifold(X, target=y, n_components=n_components)
