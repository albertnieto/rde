"""Descriptors measuring how slice complexity evolves with family index r."""

from __future__ import annotations

import math

import numpy as np


def _slope(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 2:
        return float("nan")
    coeffs = np.polyfit(xs.astype(float), ys.astype(float), deg=1)
    return float(coeffs[0])


def _log_slope(indices: list[int], values: list[float]) -> float:
    xs = np.log(np.maximum(np.asarray(indices, dtype=float), 1.0))
    ys = np.asarray(values, dtype=float)
    mask = np.isfinite(ys)
    if mask.sum() < 2:
        return float("nan")
    return _slope(xs[mask], ys[mask])


def _batch_histogram_entropy(stacked: np.ndarray, bins: int = 32) -> np.ndarray:
    """Compute per-row histogram entropy without a histogram Python loop."""
    finite = np.isfinite(stacked)
    count = np.sum(finite, axis=1)
    safe_values = np.where(finite, stacked, 0.0)
    mins = np.min(
        np.where(finite, safe_values, np.inf),
        axis=1,
    )
    maxes = np.max(
        np.where(finite, safe_values, -np.inf),
        axis=1,
    )
    mins = np.where(count > 0, mins, 0.0)
    maxes = np.where(count > 0, maxes, 0.0)
    span = maxes - mins
    safe_span = np.where(span > 0.0, span, 1.0)
    normalized = (safe_values - mins[:, None]) / safe_span[:, None]
    bin_indices = np.clip(
        np.floor(normalized * bins).astype(np.intp),
        0,
        bins - 1,
    )

    histogram = np.zeros((stacked.shape[0], bins), dtype=np.intp)
    row_indices = np.broadcast_to(
        np.arange(stacked.shape[0])[:, None],
        stacked.shape,
    )
    np.add.at(
        histogram,
        (row_indices[finite], bin_indices[finite]),
        1,
    )
    totals = histogram.sum(axis=1)
    probabilities = np.divide(
        histogram,
        totals[:, None],
        out=np.zeros_like(histogram, dtype=float),
        where=totals[:, None] > 0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(
            np.where(
                probabilities > 0.0,
                probabilities * np.log(probabilities),
                0.0,
            ),
            axis=1,
        )
    return entropy


def _batch_effective_rank(stacked: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Effective rank of each row after one batched absolute sort."""
    sorted_abs = np.sort(np.abs(stacked), axis=1)[:, ::-1]
    energy = np.where(sorted_abs > eps, sorted_abs**2, 0.0)
    numerator = np.sum(energy, axis=1)
    denominator = np.sum(energy**2, axis=1)
    return np.divide(
        numerator**2,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > eps,
    )


def _batch_slice_rank(stacked: np.ndarray) -> np.ndarray:
    """Balanced bipartition rank for a batch of equal-sized vectors."""
    size = stacked.shape[1]
    if size == 0 or (size & (size - 1)) != 0:
        return np.full(stacked.shape[0], np.nan, dtype=float)
    dim = int(round(math.log2(size)))
    half = dim // 2
    if half == 0:
        return np.ones(stacked.shape[0], dtype=float)
    left_dim = 2**half
    right_dim = 2 ** (dim - half)
    singular_values = np.linalg.svd(
        stacked.reshape(stacked.shape[0], left_dim, right_dim),
        compute_uv=False,
    )
    return np.sum(singular_values > 1e-10, axis=1).astype(float)


def compute_r_dynamics(
    indices: list[int],
    slices: list[np.ndarray],
) -> dict[str, float]:
    """Summarize trajectory dynamics across materialized family indices."""
    if not indices or not slices or len(indices) != len(slices):
        return {}

    order = sorted(range(len(indices)), key=lambda i: indices[i])
    rs = np.asarray([indices[i] for i in order], dtype=int)
    arrays = [
        np.asarray(slices[i], dtype=float).ravel()
        for i in order
    ]
    size_groups: dict[int, list[int]] = {}
    for i, array in enumerate(arrays):
        size_groups.setdefault(array.size, []).append(i)

    ent_arr = np.full(len(arrays), np.nan, dtype=float)
    rank_arr = np.full(len(arrays), np.nan, dtype=float)
    sr_arr = np.full(len(arrays), np.nan, dtype=float)
    for group_indices in size_groups.values():
        stacked = np.stack([arrays[i] for i in group_indices])
        group = np.asarray(group_indices, dtype=int)
        ent_arr[group] = _batch_histogram_entropy(stacked)
        rank_arr[group] = _batch_effective_rank(stacked)
        sr_arr[group] = _batch_slice_rank(stacked)

    out: dict[str, float] = {
        "dynamics.n_points": float(len(rs)),
        "dynamics.entropy_start": float(ent_arr[0]),
        "dynamics.entropy_end": float(ent_arr[-1]),
        "dynamics.entropy_delta": float(ent_arr[-1] - ent_arr[0]),
        "dynamics.entropy_slope_log_r": _log_slope(rs, ent_arr),
        "dynamics.effective_rank_start": float(rank_arr[0]),
        "dynamics.effective_rank_end": float(rank_arr[-1]),
        "dynamics.effective_rank_delta": float(rank_arr[-1] - rank_arr[0]),
        "dynamics.effective_rank_slope_r": _slope(np.array(rs, dtype=float), rank_arr),
        "dynamics.slice_rank_start": float(sr_arr[0]),
        "dynamics.slice_rank_end": float(sr_arr[-1]),
        "dynamics.slice_rank_delta": float(sr_arr[-1] - sr_arr[0]),
        "dynamics.slice_rank_slope_r": _slope(np.array(rs, dtype=float), sr_arr),
    }

    if len(sr_arr) >= 2:
        growth = sr_arr[-1] / max(sr_arr[0], 1.0)
        out["dynamics.slice_rank_growth_ratio"] = float(growth)

    # Plateau detection: first index where slice rank stops increasing
    plateau_candidates = np.flatnonzero(
        np.isfinite(sr_arr) & (sr_arr >= sr_arr[-1] - 1e-9)
    )
    plateau_at = rs[plateau_candidates[0]] if plateau_candidates.size else rs[-1]
    out["dynamics.slice_rank_plateau_r"] = float(plateau_at)

    return out
