"""Overlap descriptors between family slices (QueryIntent.OVERLAP)."""

from __future__ import annotations

import numpy as np


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    if x.size != y.size:
        return float("nan")
    return float(abs(np.vdot(x, y)))


def compute_overlap_descriptors(
    indices: list[int],
    slices: list[np.ndarray],
) -> dict[str, float]:
    if len(indices) < 2 or len(slices) < 2:
        return {}
    order = sorted(range(len(indices)), key=lambda i: indices[i])
    sorted_slices = [slices[i] for i in order]
    sorted_rs = [indices[i] for i in order]
    mat = np.stack([np.asarray(s, dtype=float).ravel() for s in sorted_slices], axis=0)
    gram = np.abs(mat @ mat.T)
    n = gram.shape[0]
    tri_i, tri_j = np.triu_indices(n, k=1)
    pairs = gram[tri_i, tri_j]
    diag = np.arange(n - 1)
    consecutive = gram[diag, diag + 1]
    return {
        "overlap.n_pairs": float(len(pairs)),
        "overlap.mean": float(np.mean(pairs)),
        "overlap.min": float(np.min(pairs)),
        "overlap.max": float(np.max(pairs)),
        "overlap.consecutive_mean": float(np.mean(consecutive)) if consecutive.size else float("nan"),
        "overlap.consecutive_min": float(np.min(consecutive)) if consecutive.size else float("nan"),
        "overlap.first_r": float(sorted_rs[0]),
        "overlap.last_r": float(sorted_rs[-1]),
    }
