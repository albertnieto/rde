"""Balanced bipartition slice rank (shared by dynamics and domain metrics)."""

from __future__ import annotations

import math

import numpy as np


def balanced_slice_rank(values: np.ndarray) -> float:
    """Rank of reshaped vector at balanced bipartition (power-of-two length)."""
    arr = np.asarray(values, dtype=float).ravel()
    n = arr.size
    if n == 0 or (n & (n - 1)) != 0:
        return float("nan")
    dim = int(round(math.log2(n)))
    half = dim // 2
    if half == 0:
        return 1.0
    left_dim = 2**half
    right_dim = 2 ** (dim - half)
    mat = arr.reshape(left_dim, right_dim)
    return float(np.linalg.matrix_rank(mat, tol=1e-10))


def log_slice_rank(values: np.ndarray) -> float:
    rank = balanced_slice_rank(values)
    n = int(np.asarray(values).size)
    if math.isnan(rank) or n <= 1:
        return float("nan")
    return float(math.log2(max(rank, 1.0)) / math.log2(n))
