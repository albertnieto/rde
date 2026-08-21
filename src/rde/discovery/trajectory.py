"""Trajectory predictor: (features, r) → target (Phase 4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rde.analyze.feature_table import FeatureTable
from rde.discovery.latent import ridge_predict_columns


@dataclass
class TrajectoryPredictorResult:
    train_r_squared: float
    extrapolation_r_squared: float
    n_train: int
    n_test: int
    feature_columns: list[str]


def predict_trajectory_target(
    rows: list[dict[str, Any]],
    target: str,
    *,
    feature_columns: list[str] | None = None,
    train_sizes: set[int] | None = None,
) -> TrajectoryPredictorResult:
    """Ridge model on [family_index, descriptors] predicting target; optional size holdout."""
    if not rows:
        return TrajectoryPredictorResult(float("nan"), float("nan"), 0, 0, [])

    cols = feature_columns or []
    if not cols:
        cols = [
            c
            for c in sorted({k for row in rows for k in row if isinstance(row.get(k), (int, float))})
            if c not in {target, "size", "seed", "run_id", "instance_id"}
            and not str(c).startswith("metric.")
        ][:12]

    use_cols = ["family_index", *cols]
    all_sizes = sorted({int(row["size"]) for row in rows if row.get("size") is not None})
    if train_sizes is None and len(all_sizes) >= 2:
        split = max(1, len(all_sizes) // 2)
        train_sizes = set(all_sizes[:split])
    elif train_sizes is None:
        train_sizes = set(all_sizes)

    train_rows = [r for r in rows if r.get("size") in train_sizes]
    test_rows = [r for r in rows if r.get("size") not in train_sizes]

    train_table = FeatureTable.from_rows(train_rows, columns=[*use_cols, target])
    X_train = train_table.data[:, : len(use_cols)]
    y_train = train_table.column(target)
    _, train_r2 = ridge_predict_columns(X_train, y_train)

    extrap_r2 = float("nan")
    if test_rows:
        test_table = FeatureTable.from_rows(test_rows, columns=[*use_cols, target])
        X_test = test_table.data[:, : len(use_cols)]
        y_test = test_table.column(target)
        X_aug = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        reg = 1e-3 * np.eye(X_aug.shape[1])
        beta = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y_train)
        pred = np.hstack([np.ones((X_test.shape[0], 1)), X_test]) @ beta
        ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
        if ss_tot > 0:
            extrap_r2 = float(1.0 - np.sum((y_test - pred) ** 2) / ss_tot)

    return TrajectoryPredictorResult(
        train_r_squared=train_r2,
        extrapolation_r_squared=extrap_r2,
        n_train=len(train_rows),
        n_test=len(test_rows),
        feature_columns=use_cols,
    )
