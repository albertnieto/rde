"""Spectral descriptors for vectors and matrices."""

from __future__ import annotations

import numpy as np

from rde.core.instance import InstanceRecord
from rde.core.protocols import SimpleFamilySlice


def _as_1d(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=float).ravel()
    if arr.size == 0:
        return arr
    return arr


def effective_rank(singular_values: np.ndarray, eps: float = 1e-12) -> float:
    """Participation-ratio-style effective rank from singular values."""
    s = singular_values[singular_values > eps]
    if s.size == 0:
        return 0.0
    s2 = s**2
    denom = np.sum(s2)
    if denom <= eps:
        return 0.0
    return float((np.sum(s2) ** 2) / np.sum(s2**2))


def _gini(values: np.ndarray) -> float:
    arr = np.abs(np.asarray(values, dtype=float).ravel())
    if arr.size == 0:
        return 0.0
    sorted_arr = np.sort(arr)
    cum = np.cumsum(sorted_arr)
    if cum[-1] <= 0:
        return 0.0
    n = sorted_arr.size
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def _hoyer_sparsity(values: np.ndarray) -> float:
    arr = np.abs(np.asarray(values, dtype=float).ravel())
    l1 = float(np.sum(arr))
    l2 = float(np.linalg.norm(arr))
    n = arr.size
    if l1 <= 0 or n == 0:
        return 0.0
    return float((np.sqrt(n) - l1 / l2) / max(np.sqrt(n) - 1.0, 1e-12))


def descriptor(
    _instance: InstanceRecord,
    _slice_: SimpleFamilySlice | None,
    array: np.ndarray | None,
) -> dict[str, float]:
    if array is None:
        return {}
    arr = np.asarray(array, dtype=float)
    out: dict[str, float] = {}

    if arr.ndim == 1:
        vec = _as_1d(arr)
        norm = float(np.linalg.norm(vec))
        out["spectral.l2_norm"] = norm
        if norm > 0:
            probs = (vec**2) / np.sum(vec**2)
            probs = probs[probs > 0]
            out["spectral.singular_value_entropy"] = float(-np.sum(probs * np.log(probs)))
        s = np.sort(np.abs(vec))[::-1]
        out["spectral.effective_rank"] = effective_rank(s)
        if vec.size >= 2:
            out["spectral.spectral_gap"] = float(s[0] - s[1]) if s.size >= 2 else 0.0
        out["spectral.gini"] = _gini(vec)
        out["spectral.hoyer_sparsity"] = _hoyer_sparsity(vec)
        out["spectral.crest_factor"] = float(np.max(np.abs(vec)) / max(norm, 1e-15))
    elif arr.ndim == 2:
        s = np.linalg.svd(arr, compute_uv=False)
        out["spectral.max_singular_value"] = float(s[0]) if s.size else 0.0
        out["spectral.min_singular_value"] = float(s[-1]) if s.size else 0.0
        out["spectral.condition_number"] = (
            float(s[0] / s[-1]) if s.size and s[-1] > 1e-15 else float("inf")
        )
        out["spectral.effective_rank"] = effective_rank(s)
        if s.size >= 2:
            out["spectral.spectral_gap"] = float(s[0] - s[1])
    else:
        return descriptor(_instance, _slice_, arr.ravel())

    return out
