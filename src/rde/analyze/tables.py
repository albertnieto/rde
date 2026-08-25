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


def contract_excluded_columns(rows: list[dict[str, Any]], domain_id: str | None) -> set[str]:
    """Columns present in `rows` that `domain_id`'s `DomainContract` explicitly excludes.

    Shared by every Mode 1 candidate-variable-selection site (`rde.expression
    .generators.metric_variable_columns`, `rde.descriptor_gen.rank
    .descriptor_variable_columns`, `rde.discovery.symbolic`'s equation/GP
    feature selection) so a raw `OUTCOME`/`TARGET_DERIVED` scalar a domain
    re-exposes for its own bookkeeping (e.g. `hsp_functions`'s
    `structure_strength`, present so `metric.structure_strength` can be
    computed from it) can never be silently selected as a predictor
    candidate — found via a real tautological `R^2=1.0` "discovery"
    (`structure_strength` predicting `metric.structure_strength`) that an
    existing per-conjecture leak-audit check did not catch, because none of
    these call sites consulted the contract at all before this fix.

    Fail-*open*: `None` domain_id or a domain with no registered contract
    returns an empty set (unchanged candidate pool) rather than excluding
    everything; only columns the contract actively marks
    `predictor_eligible=False` are excluded.
    """
    if domain_id is None:
        return set()
    from rde.core.feature_contract import catalog_for_domain

    try:
        catalog = catalog_for_domain(domain_id)
    except KeyError:
        return set()
    return {c for c in numeric_columns(rows, exclude=set()) if catalog.is_explicitly_excluded(c)}


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
