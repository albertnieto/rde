"""Shared test helpers for RDE."""

from __future__ import annotations

from rde.core.registry import Registry
from rde.features import register_builtin_descriptors, register_builtin_metrics
from rde.testing import SyntheticPolyDomain


def toy_registry() -> Registry:
    reg = Registry()
    reg.register_domain(SyntheticPolyDomain())
    register_builtin_descriptors(reg)
    register_builtin_metrics(reg)
    return reg


def analyze_rows(n: int = 20) -> list[dict]:
    """Rows with instance grouping for analyze/power-plan tests."""
    rows: list[dict] = []
    for i in range(n):
        rows.append(
            {
                "instance_id": f"inst_{i // 5}",
                "size": 4,
                "family_index": 1,
                "generator": "gen_a" if i % 2 == 0 else "gen_b",
                "matrix.trace": float(i),
                "metric.log_slice_rank": float(i * 0.1),
            }
        )
    return rows


def obstruction_rows(n_sizes: list[int]) -> list[dict]:
    """Rows with exponential witness columns for obstruction / promote-lb tests."""
    rows: list[dict] = []
    for n in n_sizes:
        for i in range(5):
            rows.append(
                {
                    "size": n,
                    "seed": i,
                    "metric.log_slice_rank": float(n),
                    "dynamics.slice_rank_end": float(2 ** (n / 2)),
                    "recurrence.estimated_order": float(min(4, n)),
                    "spectral.effective_rank": float(2 ** (n / 3)),
                }
            )
    return rows
