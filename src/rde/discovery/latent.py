"""Latent variable discovery — PCA, training loops, checkpointing (Phase 4)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rde.analyze.feature_table import FeatureTable
from rde.discovery.checkpoint import load_latent_checkpoint, save_latent_checkpoint
from rde.discovery.datasets import build_feature_dataset, load_feature_matrix
from rde.discovery.impute import impute_matrix
from rde.features.numeric import safe_pearson_r


@dataclass
class LatentResult:
    latent_dim: int
    explained_variance_ratio: list[float]
    latent_codes: np.ndarray
    target_correlations: dict[str, float]
    reconstruction_error: float
    checkpoint_name: str = "pca"
    components: np.ndarray | None = None
    singular_values: np.ndarray | None = None


def _pairwise_pearson(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pearson correlations for every column pair with pairwise masking."""
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    pair_mask = (
        np.isfinite(left_arr)[:, :, None]
        & np.isfinite(right_arr)[:, None, :]
        & (np.abs(left_arr)[:, :, None] < 1e100)
        & (np.abs(right_arr)[:, None, :] < 1e100)
    )
    counts = np.sum(pair_mask, axis=0)
    left_values = np.where(pair_mask, left_arr[:, :, None], 0.0)
    right_values = np.where(pair_mask, right_arr[:, None, :], 0.0)
    left_sum = np.sum(left_values, axis=0)
    right_sum = np.sum(right_values, axis=0)
    left_sq_sum = np.sum(left_values**2, axis=0)
    right_sq_sum = np.sum(right_values**2, axis=0)
    cross_sum = np.sum(left_values * right_values, axis=0)
    safe_counts = np.maximum(counts, 1)
    left_var = np.maximum(
        left_sq_sum - left_sum**2 / safe_counts,
        0.0,
    )
    right_var = np.maximum(
        right_sq_sum - right_sum**2 / safe_counts,
        0.0,
    )
    covariance = cross_sum - left_sum * right_sum / safe_counts
    denominator = np.sqrt(left_var * right_var)
    correlations = np.full(denominator.shape, np.nan, dtype=float)
    valid = (counts >= 3) & (denominator > 0.0)
    correlations[valid] = covariance[valid] / denominator[valid]
    return correlations, counts


def ridge_predict_columns(
    X: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float = 1e-3,
) -> tuple[np.ndarray, float]:
    """Ridge regression predicting target from feature columns; returns (coef, train_r2)."""
    target_arr = np.asarray(target, dtype=float)
    target_mask = np.isfinite(target_arr)
    if int(target_mask.sum()) < 3:
        return np.full(X.shape[1] + 1, np.nan), float("nan")
    target_values = target_arr[target_mask]
    if float(np.std(target_values)) <= 0.0:
        return np.full(X.shape[1] + 1, np.nan), float("nan")
    X_clean = impute_matrix(X)
    t = target_arr[target_mask]
    X_aug = np.hstack([np.ones((X_clean.shape[0], 1)), X_clean])
    X_fit = X_aug[target_mask]
    if X_fit.shape[0] < X_fit.shape[1]:
        # Centering makes the intercept unregularized while the dual solve
        # keeps the linear system bounded by the number of samples.
        x_features = X_fit[:, 1:]
        x_mean = x_features.mean(axis=0)
        t_mean = float(t.mean())
        x_centered = x_features - x_mean
        dual = np.linalg.solve(
            x_centered @ x_centered.T + alpha * np.eye(X_fit.shape[0]),
            t - t_mean,
        )
        feature_coef = x_centered.T @ dual
        coef = np.concatenate([[t_mean - x_mean @ feature_coef], feature_coef])
        pred = X_fit @ coef
    else:
        reg = alpha * np.eye(X_aug.shape[1])
        reg[0, 0] = 0.0
        coef = np.linalg.solve(X_fit.T @ X_fit + reg, X_fit.T @ t)
        pred = X_fit @ coef
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = float(1.0 - np.sum((t - pred) ** 2) / ss_tot)
    return coef, r2


def ridge_predict_multi(
    X: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Ridge-regress multiple target columns with one normal-equation solve."""
    target_arr = np.asarray(targets, dtype=float)
    if target_arr.ndim == 1:
        target_arr = target_arr.reshape(-1, 1)
    if target_arr.ndim != 2 or target_arr.shape[0] != np.asarray(X).shape[0]:
        raise ValueError("targets must be a 1D/2D array with one row per X sample")

    X_clean = impute_matrix(X)
    X_aug = np.hstack([np.ones((X_clean.shape[0], 1)), X_clean])
    valid_rows = np.all(np.isfinite(target_arr), axis=1)
    if int(valid_rows.sum()) < 3:
        return (
            np.full((X_aug.shape[1], target_arr.shape[1]), np.nan),
            np.full(target_arr.shape[1], np.nan),
        )

    X_fit = X_aug[valid_rows]
    target_fit = target_arr[valid_rows]
    if X_fit.shape[0] < X_fit.shape[1]:
        x_features = X_fit[:, 1:]
        x_mean = x_features.mean(axis=0)
        target_mean = target_fit.mean(axis=0)
        x_centered = x_features - x_mean
        dual = np.linalg.solve(
            x_centered @ x_centered.T + alpha * np.eye(X_fit.shape[0]),
            target_fit - target_mean,
        )
        feature_coef = x_centered.T @ dual
        coef = np.vstack([target_mean - x_mean @ feature_coef, feature_coef])
        pred = X_fit @ coef
    else:
        reg = alpha * np.eye(X_aug.shape[1])
        reg[0, 0] = 0.0
        coef = np.linalg.solve(X_fit.T @ X_fit + reg, X_fit.T @ target_fit)
        pred = X_fit @ coef
    centered = target_fit - target_fit.mean(axis=0, keepdims=True)
    ss_tot = np.sum(centered**2, axis=0)
    ss_res = np.sum((target_fit - pred) ** 2, axis=0)
    r2 = np.full(target_arr.shape[1], np.nan, dtype=float)
    valid_targets = ss_tot > 0.0
    r2[valid_targets] = 1.0 - ss_res[valid_targets] / ss_tot[valid_targets]
    return coef, r2


def pca_latent(
    X: np.ndarray,
    *,
    n_components: int,
    target: np.ndarray | None = None,
) -> LatentResult:
    """PCA via SVD — no sklearn dependency."""
    X_clean = impute_matrix(X)
    X_centered = X_clean - X_clean.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(X_centered, full_matrices=False)
    k = min(n_components, vt.shape[0])
    components = vt[:k]
    codes = X_centered @ components.T
    var = (s[:k] ** 2) / max(float((s**2).sum()), 1e-12)
    recon = codes @ components + X_clean.mean(axis=0, keepdims=True)
    mse = float(np.mean((X_clean - recon) ** 2))
    corrs: dict[str, float] = {}
    if target is not None:
        target_arr = np.asarray(target, dtype=float)
        target_mask = np.isfinite(target_arr)
        t = target_arr[target_mask]
        for i in range(k):
            c = codes[:, i]
            if t.size >= 3 and np.std(c[target_mask]) > 0 and np.std(t) > 0:
                corrs[f"latent_{i}"] = safe_pearson_r(c[target_mask], t)
    return LatentResult(
        latent_dim=k,
        explained_variance_ratio=[float(v) for v in var],
        latent_codes=codes,
        target_correlations=corrs,
        reconstruction_error=mse,
        components=components,
        singular_values=s[:k],
    )


def discover_latent_from_run(
    run_id: str,
    store_root: str | Path,
    *,
    target_column: str = "metric.representation_complexity",
    n_components: int = 8,
    exclude_prefixes: tuple[str, ...] = ("metric.", "run_id"),
    checkpoint: bool = True,
) -> LatentResult:
    X, cols, rows = load_feature_matrix(run_id, store_root, target=target_column)
    use_cols = [
        c
        for c in cols
        if c != target_column and not any(c.startswith(p) for p in exclude_prefixes)
    ]
    idx = [cols.index(c) for c in use_cols]
    target = np.array([row.get(target_column, float("nan")) for row in rows], dtype=float)
    X_clean = impute_matrix(X[:, idx])
    result = pca_latent(X_clean, n_components=n_components, target=target)
    if checkpoint and result.latent_codes.size:
        k = result.latent_dim
        save_latent_checkpoint(
            store_root,
            run_id,
            result.checkpoint_name,
            arrays={
                "components": result.components,
                "singular_values": result.singular_values,
            },
            metadata={
                "method": "pca",
                "target_column": target_column,
                "feature_columns": use_cols,
                "latent_dim": k,
                "reconstruction_error": result.reconstruction_error,
            },
        )
    return result


def correlate_latent_descriptors(
    latent_codes: np.ndarray,
    latent_names: list[str],
    rows: list[dict[str, Any]],
    *,
    target: str,
    max_descriptors: int = 24,
    min_abs_r: float = 0.2,
) -> list[dict[str, Any]]:
    """Correlate each latent dimension with descriptor columns and the target."""
    from rde.analyze.query import numeric_columns

    if latent_codes.size == 0 or not rows:
        return []

    skip = set(latent_names) | {target, "size", "seed", "run_id", "instance_id", "family_index"}
    desc_cols = [
        c for c in numeric_columns(rows, exclude=skip) if not c.startswith("metric.")
    ][:max_descriptors]

    table = FeatureTable.from_rows(
        rows,
        columns=[*desc_cols, target],
    )
    target_arr = table.column(target)
    descriptor_matrix = table.data[:, : len(desc_cols)]
    codes = np.asarray(latent_codes[:, : len(latent_names)], dtype=float)
    correlations, counts = _pairwise_pearson(codes, descriptor_matrix)
    out: list[dict[str, Any]] = []

    for j, name in enumerate(latent_names[: codes.shape[1]]):
        for col_index, col in enumerate(desc_cols):
            r = correlations[j, col_index]
            if not np.isfinite(r) or abs(r) < min_abs_r:
                continue
            out.append(
                {
                    "latent_column": name,
                    "descriptor_column": col,
                    "pearson_r": float(r),
                    "n": int(counts[j, col_index]),
                }
            )
    target_correlations, target_counts = _pairwise_pearson(
        codes,
        target_arr[:, None],
    )
    for j, name in enumerate(latent_names[: codes.shape[1]]):
        rt = target_correlations[j, 0]
        if np.isfinite(rt) and abs(rt) >= min_abs_r:
            out.append(
                {
                    "latent_column": name,
                    "descriptor_column": target,
                    "pearson_r": float(rt),
                    "n": int(target_counts[j, 0]),
                }
                )

    out.sort(key=lambda d: -abs(float(d["pearson_r"])))
    return out[:40]


def load_pca_from_checkpoint(
    run_id: str,
    store_root: str | Path,
    X: np.ndarray,
    *,
    target: np.ndarray | None = None,
) -> LatentResult | None:
    """Reconstruct PCA latent codes from a saved checkpoint."""
    try:
        arrays, meta = load_latent_checkpoint(store_root, run_id, "pca")
    except FileNotFoundError:
        return None
    components = arrays["components"]
    feat_cols = meta.get("feature_columns")
    if feat_cols and X.shape[1] != len(feat_cols):
        return None
    X_clean = impute_matrix(X)
    codes = (X_clean - X_clean.mean(axis=0, keepdims=True)) @ components.T
    k = components.shape[0]
    sv = arrays.get("singular_values", np.ones(k))
    var = (sv**2) / max(float((sv**2).sum()), 1e-12)
    recon = codes @ components + X_clean.mean(axis=0, keepdims=True)
    mse = float(np.mean((X_clean - recon) ** 2))
    corrs: dict[str, float] = {}
    if target is not None:
        target_arr = np.asarray(target, dtype=float)
        target_mask = np.isfinite(target_arr)
        t = target_arr[target_mask]
        for i in range(k):
            c = codes[:, i]
            if t.size >= 3 and np.std(c[target_mask]) > 0 and np.std(t) > 0:
                corrs[f"latent_{i}"] = safe_pearson_r(c[target_mask], t)
    return LatentResult(
        latent_dim=k,
        explained_variance_ratio=[float(v) for v in var],
        latent_codes=codes,
        target_correlations=corrs,
        reconstruction_error=mse,
        checkpoint_name="pca",
        components=components,
        singular_values=sv,
    )
