"""Statistical descriptors for array payloads."""

from __future__ import annotations

import numpy as np

from rde.core.instance import InstanceRecord
from rde.core.protocols import SimpleFamilySlice
from rde.features.numeric import safe_kurtosis, safe_skew


def shannon_entropy_histogram(values: np.ndarray, bins: int = 32) -> float:
    """Entropy of a fixed-bin histogram (base e)."""
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    counts, _ = np.histogram(arr, bins=bins, density=False)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log(probs)))


def descriptor(
    _instance: InstanceRecord,
    _slice_: SimpleFamilySlice | None,
    array: np.ndarray | None,
) -> dict[str, float]:
    if array is None:
        return {}
    arr = np.asarray(array, dtype=float).ravel()
    if arr.size == 0:
        return {"stats.count": 0.0}

    out: dict[str, float] = {
        "stats.count": float(arr.size),
        "stats.mean": float(np.mean(arr)),
        "stats.std": float(np.std(arr)),
        "stats.variance": float(np.var(arr)),
        "stats.min": float(np.min(arr)),
        "stats.max": float(np.max(arr)),
        "stats.median": float(np.median(arr)),
        "stats.range_ratio": float((np.max(arr) - np.min(arr)) / max(abs(np.mean(arr)), 1e-12)),
        "stats.zero_fraction": float(np.mean(arr == 0.0)),
        "stats.shannon_entropy": shannon_entropy_histogram(arr),
    }

    if arr.size >= 3:
        out["stats.skewness"] = safe_skew(arr)
        out["stats.kurtosis"] = safe_kurtosis(arr)
        q1, q3 = np.quantile(arr, [0.25, 0.75])
        out["stats.iqr"] = float(q3 - q1)
        out["stats.cv"] = float(np.std(arr) / max(abs(np.mean(arr)), 1e-12))

    abs_arr = np.abs(arr)
    total = float(np.sum(abs_arr))
    if total > 0:
        probs = abs_arr / total
        probs = probs[probs > 0]
        out["stats.participation_ratio"] = float(1.0 / np.sum(probs**2))

    return out
