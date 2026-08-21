"""Walsh-Hadamard / Fourier descriptors on boolean or ±1 arrays."""

from __future__ import annotations

import numpy as np

from rde.core.instance import InstanceRecord
from rde.core.protocols import SimpleFamilySlice


def walsh_hadamard(values: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard transform on a length-2^k vector.

    Each stage operates on a reshaped view of all butterfly pairs at once.
    The stage loop is only logarithmic in the vector length; individual
    butterflies never cross the Python/NumPy boundary.
    """
    arr = np.asarray(values, dtype=float).ravel().copy()
    n = arr.size
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError("Walsh-Hadamard requires length 2^k")
    h = 1
    while h < n:
        pairs = arr.reshape(-1, 2, h)
        top = pairs[:, 0, :].copy()
        bot = pairs[:, 1, :].copy()
        pairs[:, 0, :] = top + bot
        pairs[:, 1, :] = top - bot
        h *= 2
    arr /= np.sqrt(n)
    return arr


def walsh_hadamard_batch(matrix: np.ndarray) -> np.ndarray:
    """FWHT along the last axis for each row of a 2-D array."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 1:
        return walsh_hadamard(arr)
    if arr.ndim != 2:
        raise ValueError("walsh_hadamard_batch expects a 1-D or 2-D array")
    n = arr.shape[-1]
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError("Walsh-Hadamard requires last axis length 2^k")
    out = arr.copy()
    batch_size = out.shape[0]
    h = 1
    while h < n:
        pairs = out.reshape(batch_size, -1, 2, h)
        top = pairs[:, :, 0, :].copy()
        bot = pairs[:, :, 1, :].copy()
        pairs[:, :, 0, :] = top + bot
        pairs[:, :, 1, :] = top - bot
        h *= 2
    out /= np.sqrt(n)
    return out


def _to_pm1(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if np.all((arr >= 0) & (arr <= 1)):
        return 2.0 * arr - 1.0
    return arr


def descriptor(
    _instance: InstanceRecord,
    _slice_: SimpleFamilySlice | None,
    array: np.ndarray | None,
) -> dict[str, float]:
    if array is None:
        return {}
    arr = np.asarray(array, dtype=float).ravel()
    n = arr.size
    if n == 0 or (n & (n - 1)) != 0:
        return {"fourier.valid_length": 0.0}

    pm1 = _to_pm1(arr)
    coeffs = walsh_hadamard(pm1)
    abs_c = np.abs(coeffs)
    total = float(np.sum(abs_c))
    if total <= 0:
        return {"fourier.valid_length": 1.0, "fourier.sparsity": 1.0}

    threshold = 1e-8 * abs_c.max()
    sparsity = float(np.count_nonzero(abs_c <= threshold) / n)
    sorted_abs = np.sort(abs_c)[::-1]
    top_k = max(1, n // 8)
    mass_top = float(np.sum(sorted_abs[:top_k]) / total)

    return {
        "fourier.valid_length": 1.0,
        "fourier.sparsity": sparsity,
        "fourier.top_mass_fraction": mass_top,
        "fourier.max_coeff": float(abs_c.max()),
        "fourier.l1_norm": total,
        "fourier.tail_loglog_slope": _tail_loglog_slope(sorted_abs),
    }


def _tail_loglog_slope(sorted_abs: np.ndarray) -> float:
    ys = sorted_abs[sorted_abs > 0][:32]
    if ys.size < 4:
        return float("nan")
    xs = np.arange(1, ys.size + 1, dtype=float)
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def descriptor_for_vector(values: np.ndarray, name: str = "array") -> dict[str, float]:
    """Walsh descriptors with prefixed keys."""
    base = descriptor(None, None, values)  # type: ignore[arg-type]
    if not base:
        return {}
    out: dict[str, float] = {}
    for key, val in base.items():
        suffix = key.split(".", 1)[-1] if key.startswith("fourier.") else key
        out[f"fourier.{name}.{suffix}"] = val
    return out
