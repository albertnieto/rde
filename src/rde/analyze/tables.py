"""Shared helpers for numeric feature-table columns."""

from __future__ import annotations

from typing import Any

import numpy as np


def numeric_columns(rows: list[dict[str, Any]], *, exclude: set[str] | None = None) -> list[str]:
    skip = exclude or set()
    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
    out: list[str] = []
    for key in sorted(keys):
        if key in skip:
            continue
        vals = [row.get(key) for row in rows]
        if all(isinstance(v, (int, float)) or v is None for v in vals):
            out.append(key)
    return out


def group_indices_by_size(rows: list[dict[str, Any]]) -> dict[Any, np.ndarray]:
    """Bucket row indices by ``size``.

    Callers that check cross-size stability for many candidates against the
    same ``rows`` (e.g. one ranking pass over hundreds of templates or
    thousands of GP candidates) should compute this once and reuse it,
    instead of rescanning ``rows`` per candidate.
    """
    groups: dict[Any, list[int]] = {}
    for i, row in enumerate(rows):
        size = row.get("size")
        if size is not None:
            groups.setdefault(size, []).append(i)
    return {size: np.array(idxs, dtype=int) for size, idxs in groups.items()}


def default_train_test_split(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray] | None:
    """Default extrapolation split: train on the smaller sizes, test on the larger ones.

    Returns ``None`` when fewer than two distinct sizes are present. Compute
    once per ranking pass and reuse across candidates via
    ``extrapolation_r_squared(..., split=...)``.
    """
    all_sizes = sorted({int(row["size"]) for row in rows if row.get("size") is not None})
    if len(all_sizes) < 2:
        return None
    n_split = max(1, len(all_sizes) // 2)
    train_sizes = set(all_sizes[:n_split])
    test_sizes = set(all_sizes[n_split:])
    train_idx = np.array([i for i, row in enumerate(rows) if row.get("size") in train_sizes], dtype=int)
    test_idx = np.array([i for i, row in enumerate(rows) if row.get("size") in test_sizes], dtype=int)
    return train_idx, test_idx
