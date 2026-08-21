"""Landscape descriptors for 1D cost/energy vectors over binary assignments."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from rde.core.instance import InstanceRecord
from rde.core.protocols import SimpleFamilySlice
from rde.features.numeric import safe_kurtosis, safe_pearson_r, safe_skew
from rde.features.stats import shannon_entropy_histogram


@lru_cache(maxsize=8)
def _hamming_weights(length: int) -> np.ndarray:
    """Return reusable Hamming weights for a power-of-two assignment space."""
    if length < 1 or (length & (length - 1)) != 0:
        raise ValueError("length must be a positive power of two")
    n_bits = length.bit_length() - 1
    nums = np.arange(length, dtype=np.uint64)
    bits = np.arange(n_bits, dtype=np.uint64)
    return ((nums[:, None] >> bits[None, :]) & 1).sum(axis=1, dtype=float)


def hamming_energy_correlation(values: np.ndarray) -> float:
    """Pearson r between Hamming weight and cost (requires len = 2^N)."""
    arr = np.asarray(values, dtype=float).ravel()
    n = arr.size
    if n < 4 or (n & (n - 1)) != 0:
        return float("nan")
    return safe_pearson_r(_hamming_weights(n), arr, min_count=4)


def count_local_minima(values: np.ndarray) -> int:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size <= 2:
        return 1 if arr.size else 0
    left = np.concatenate(([arr[0]], arr[:-1]))
    right = np.concatenate((arr[1:], [arr[-1]]))
    return int(np.sum((arr <= left) & (arr <= right)))


def level_set_masses(values: np.ndarray, *, levels: tuple[float, ...] = (0.1, 0.25, 0.5)) -> dict[str, float]:
    """Fraction of mass below (min + level * span) for each level in (0,1]."""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return {}
    gmin = float(np.min(arr))
    span = float(np.max(arr) - gmin)
    if span <= 0:
        return {f"level_{int(100 * lv)}pct": 1.0 for lv in levels}
    out: dict[str, float] = {}
    for lv in levels:
        threshold = gmin + lv * span
        out[f"level_{int(100 * lv)}pct"] = float(np.mean(arr <= threshold))
    return out


def descriptor_for_vector(values: np.ndarray, name: str = "array") -> dict[str, float]:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return {}
    p = f"landscape.{name}"
    out: dict[str, float] = {
        f"{p}.count": float(arr.size),
        f"{p}.mean": float(np.mean(arr)),
        f"{p}.std": float(np.std(arr)),
        f"{p}.range": float(np.max(arr) - np.min(arr)),
        f"{p}.entropy": shannon_entropy_histogram(arr),
        f"{p}.local_minima": float(count_local_minima(arr)),
        f"{p}.unique_values": float(len(np.unique(np.round(arr, 12)))),
        f"{p}.multimodality_proxy": float(count_local_minima(arr) / max(np.sqrt(arr.size), 1.0)),
        f"{p}.hamming_energy_r": hamming_energy_correlation(arr),
    }
    for q, label in ((10, "p10"), (50, "p50"), (90, "p90")):
        out[f"{p}.quantile_{label}"] = float(np.percentile(arr, q))
    if arr.size >= 3:
        diffs = np.diff(np.sort(arr))
        out[f"{p}.ruggedness"] = float(np.std(diffs) / max(np.mean(np.abs(diffs)), 1e-12))
        out[f"{p}.skewness"] = safe_skew(arr)
        out[f"{p}.kurtosis"] = safe_kurtosis(arr)
        sorted_arr = np.sort(arr)
        grad = np.diff(sorted_arr)
        out[f"{p}.mean_abs_gradient"] = float(np.mean(np.abs(grad)))
        out[f"{p}.autocorr_lag1"] = _autocorr(arr, lag=1)
        out[f"{p}.autocorr_lag2"] = _autocorr(arr, lag=2)
    # Basin proxy: fraction within 10% of global minimum.
    gmin = float(np.min(arr))
    span = float(np.max(arr) - gmin)
    if span > 0:
        out[f"{p}.basin_mass_10pct"] = float(np.mean(arr <= gmin + 0.1 * span))
        out[f"{p}.basin_mass_25pct"] = float(np.mean(arr <= gmin + 0.25 * span))
        out[f"{p}.basin_mass_50pct"] = float(np.mean(arr <= gmin + 0.5 * span))
        for key, val in level_set_masses(arr).items():
            out[f"{p}.{key}"] = val
    return out


def _autocorr(values: np.ndarray, *, lag: int) -> float:
    arr = np.asarray(values, dtype=float).ravel()
    if lag < 1 or arr.size <= lag + 1:
        # Need at least 2 paired samples after the lag split.
        return float("nan")
    return safe_pearson_r(arr[:-lag], arr[lag:], min_count=2)


def descriptor(
    _instance: InstanceRecord,
    _slice_: SimpleFamilySlice | None,
    array: np.ndarray | None,
) -> dict[str, float]:
    if array is None:
        return {}
    arr = np.asarray(array, dtype=float)
    if arr.ndim == 1:
        return descriptor_for_vector(arr)
    return {}
