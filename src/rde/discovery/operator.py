"""Linear update-step predictor (Phase 6 — operator learning proxy)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rde.analyze.feature_table import FeatureTable
from rde.discovery.latent import ridge_predict_columns


@dataclass
class OperatorResult:
    method: str
    train_r_squared: float
    n_rows: int
    feature_columns: list[str]
    update_key: str = "update.step_ratio_mean"


def learn_update_predictor(
    rows: list[dict[str, Any]],
    feature_columns: list[str],
    *,
    update_key: str = "update.step_ratio_mean",
) -> OperatorResult:
    """Ridge model: descriptors → update.step_ratio_mean (low-rank update proxy)."""
    if not rows or not feature_columns:
        return OperatorResult("ridge_update", float("nan"), 0, [])

    use_cols = [c for c in feature_columns if c != update_key][:24]
    table = FeatureTable.from_rows(rows, columns=[*use_cols, update_key])
    X = table.data[:, : len(use_cols)]
    y = table.column(update_key)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if mask.sum() < max(4, X.shape[1] + 1):
        return OperatorResult("ridge_update", float("nan"), int(mask.sum()), use_cols, update_key)

    _, r2 = ridge_predict_columns(X[mask], y[mask])
    return OperatorResult("ridge_update", r2, int(mask.sum()), use_cols, update_key)
