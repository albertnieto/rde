"""Tests for Phase 2 expansion, Phase 2b catalog, and Phase 3 metric generators."""

from __future__ import annotations

import numpy as np

from rde.descriptor_gen.enumerate import catalog_size, default_pipeline_templates
from rde.expression.generators import enumerate_metric_candidates, metric_variable_columns
from rde.features.graph import descriptor_for_matrix
from rde.features.landscape import descriptor_for_vector
from rde.features.matrix import descriptor_for_matrix as matrix_desc
from rde.features.spectral import descriptor
from rde.features.stats import descriptor as stats_descriptor
from rde.core.instance import InstanceRecord


def _inst() -> InstanceRecord:
    return InstanceRecord(domain_id="t", size=4, seed=0, params={})


def test_spectral_gini_and_hoyer():
    vec = np.array([3.0, 1.0, 0.5, 0.1])
    out = descriptor(_inst(), None, vec)
    assert "spectral.gini" in out
    assert "spectral.hoyer_sparsity" in out
    assert 0.0 <= out["spectral.gini"] <= 1.0


def test_matrix_expanded_keys():
    q = np.array([[1.0, 0.5], [0.5, 2.0]])
    out = matrix_desc(q)
    assert "matrix.array.sparsity" in out
    assert "matrix.array.diagonal_dominance" in out
    assert "matrix.array.off_diagonal_fro_ratio" in out


def test_graph_weighted_stats():
    mat = np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    out = descriptor_for_matrix(mat)
    assert "graph.array.weighted_edge_sum" in out
    assert out["graph.array.weighted_edge_sum"] > 0


def test_landscape_autocorr():
    v = np.sin(np.linspace(0, 4 * np.pi, 32))
    out = descriptor_for_vector(v)
    assert "landscape.array.autocorr_lag1" in out
    assert np.isfinite(out["landscape.array.autocorr_lag1"])


def test_stats_extended_keys():
    arr = np.linspace(-1.0, 1.0, 20)
    out = stats_descriptor(_inst(), None, arr)
    assert "stats.median" in out
    assert "stats.iqr" in out
    assert "stats.cv" in out


def test_descriptor_gen_catalog_size():
    n = catalog_size()
    assert n >= 120
    assert len(default_pipeline_templates()) == n


def test_metric_generators_include_ratios():
    vars_ = ["a", "b", "c"]
    cands = list(
        enumerate_metric_candidates(
            vars_,
            max_depth=1,
            max_candidates=50,
            include_unary_chains=False,
            include_products=False,
        )
    )
    assert len(cands) >= 6
    kinds = {c.kind for c in cands}
    assert "div" in kinds
    cands2 = list(
        enumerate_metric_candidates(
            vars_,
            max_depth=1,
            max_candidates=20,
            include_ratios=False,
            include_unary_chains=False,
        )
    )
    assert any(c.kind == "mul" for c in cands2)


def test_metric_variable_columns_excludes_metrics():
    rows = [
        {"spectral.effective_rank": 1.0, "metric.foo": 2.0, "target": 3.0},
        {"spectral.effective_rank": 2.0, "metric.foo": 4.0, "target": 5.0},
    ]
    cols = metric_variable_columns(rows, "target", max_vars=8)
    assert "metric.foo" not in cols
    assert "spectral.effective_rank" in cols
