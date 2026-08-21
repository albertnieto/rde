"""Numerically safe scalar helpers for feature extractors."""

from __future__ import annotations

import numpy as np


def safe_pearson_r(x: np.ndarray, y: np.ndarray, *, min_count: int = 2) -> float:
    """Pearson r, or NaN when undefined (no NumPy RuntimeWarning).

    Correlation is undefined when either series is non-finite, shorter than
    ``min_count``, has extreme magnitude, or has zero sample standard deviation.
    """
    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    if xv.size != yv.size or xv.size < min_count:
        return float("nan")
    mask = np.isfinite(xv) & np.isfinite(yv)
    if int(mask.sum()) < min_count:
        return float("nan")
    xs = xv[mask]
    ys = yv[mask]
    if not np.all(np.abs(xs) < 1e100) or not np.all(np.abs(ys) < 1e100):
        return float("nan")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        xstd = float(np.std(xs))
        ystd = float(np.std(ys))
        if not np.isfinite(xstd) or not np.isfinite(ystd) or xstd == 0.0 or ystd == 0.0:
            return float("nan")
        r = float(np.corrcoef(xs, ys)[0, 1])
    return r if np.isfinite(r) else float("nan")


def safe_skew(values: np.ndarray) -> float:
    """Sample skewness, or NaN for constant / too-short inputs."""
    from scipy import stats as sp_stats

    arr = np.asarray(values, dtype=float).ravel()
    if arr.size < 3 or float(np.std(arr)) == 0.0:
        return float("nan")
    return float(sp_stats.skew(arr))


def safe_kurtosis(values: np.ndarray) -> float:
    """Sample excess kurtosis, or NaN for constant / too-short inputs."""
    from scipy import stats as sp_stats

    arr = np.asarray(values, dtype=float).ravel()
    if arr.size < 4 or float(np.std(arr)) == 0.0:
        return float("nan")
    return float(sp_stats.kurtosis(arr))
