"""Walsh descriptors on log-amplitude vectors."""

from __future__ import annotations

import math

import numpy as np

from rde.features.fourier import walsh_hadamard


def descriptor_for_log_amplitudes(values: np.ndarray, name: str = "w_amp") -> dict[str, float]:
    """Walsh spectrum of log(w_amp), padded/truncated to power-of-two length."""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return {}
    positive = np.maximum(arr, 1e-300)
    logv = np.log(positive)
    n = 1 << int(math.ceil(math.log2(max(arr.size, 1))))
    if n > 4096:
        # Subsample log amplitudes to cap Walsh cost on large N.
        idx = np.linspace(0, arr.size - 1, num=min(4096, n), dtype=int)
        logv = logv[idx]
        n = 1 << int(math.ceil(math.log2(max(logv.size, 1))))
    padded = np.zeros(n, dtype=float)
    padded[: min(logv.size, n)] = logv[:n]
    coeffs = walsh_hadamard(padded)
    abs_c = np.abs(coeffs)
    total = float(np.sum(abs_c))
    p = f"walsh_log.{name}"
    if total <= 0:
        return {f"{p}.valid_length": 1.0, f"{p}.sparsity": 1.0}
    sorted_abs = np.sort(abs_c)[::-1]
    top_k = max(1, n // 8)
    mass_top = float(np.sum(sorted_abs[:top_k]) / total)
    # Heavy-tail proxy: log-log slope of top coefficients.
    tail_slope = float("nan")
    if sorted_abs.size >= 8:
        ys = sorted_abs[sorted_abs > 0][:32]
        if ys.size >= 4:
            xs = np.arange(1, ys.size + 1, dtype=float)
            tail_slope = float(np.polyfit(np.log(xs), np.log(ys), 1)[0])
    return {
        f"{p}.valid_length": 1.0,
        f"{p}.sparsity": float(np.count_nonzero(abs_c <= 1e-8 * abs_c.max()) / n),
        f"{p}.top_mass_fraction": mass_top,
        f"{p}.max_coeff": float(abs_c.max()),
        f"{p}.l1_norm": total,
        f"{p}.tail_loglog_slope": tail_slope,
        f"{p}.entropy": _coeff_entropy(abs_c),
    }


def _coeff_entropy(abs_c: np.ndarray) -> float:
    total = float(abs_c.sum())
    if total <= 0:
        return 0.0
    p = abs_c[abs_c > 0] / total
    return float(-np.sum(p * np.log(p)))
