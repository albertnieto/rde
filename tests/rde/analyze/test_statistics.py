"""Tests for v0.3 statistics and controls."""

from __future__ import annotations

import numpy as np

from rde.analyze.controls import nr_baseline_r2, permuted_target_null
from rde.analyze.statistics import (
    benjamini_hochberg,
    cluster_bootstrap_pearson,
    permutation_max_statistic,
)


def _rows(n: int = 20):
    out = []
    for i in range(n):
        out.append(
            {
                "instance_id": f"inst_{i // 5}",
                "size": 4,
                "family_index": 1,
                "matrix.trace": float(i),
                "metric.log_slice_rank": float(i * 0.1),
            }
        )
    return out


def test_cluster_bootstrap_finite():
    boot = cluster_bootstrap_pearson(_rows(), "matrix.trace", "metric.log_slice_rank", n_bootstrap=50)
    assert np.isfinite(boot["pearson_r"])


def test_fdr_rejects_some():
    results = benjamini_hochberg([0.001, 0.04, 0.20, 0.50], alpha=0.05)
    assert results[0][2] is True
    assert results[-1][2] is False


def test_permuted_null_shuffles():
    rows = _rows(10)
    shuffled = permuted_target_null(rows, "metric.log_slice_rank", seed=0)
    orig = [r["metric.log_slice_rank"] for r in rows]
    new = [r["metric.log_slice_rank"] for r in shuffled]
    assert sorted(orig) == sorted(new)
    assert orig != new


def test_nr_baseline_r2():
    rows = _rows()
    r2 = nr_baseline_r2(rows, "metric.log_slice_rank")
    assert np.isfinite(r2)


def test_selection_null_permutates_complete_target_universe(monkeypatch):
    rows = [
        {"size": 2, "generator": "a", "x": 0.0, "target": 0.0},
        {"size": 2, "generator": "a", "x": 1.0, "target": 1.0},
        {"size": 3, "generator": "b", "x": 2.0, "target": 2.0},
        {"size": 3, "generator": "b", "x": 3.0, "target": 3.0},
    ]

    class GlobalRng:
        def permutation(self, values):
            values = np.asarray(values)
            assert len(values) == len(rows)
            return values[::-1]

    monkeypatch.setattr(np.random, "default_rng", lambda _seed: GlobalRng())
    result = permutation_max_statistic(rows, ["x"], "target", n_perm=1)
    assert result["n_perm"] == 1.0
