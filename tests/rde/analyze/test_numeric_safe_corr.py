"""Degenerate correlation / moment inputs must not emit RuntimeWarnings."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from rde.core.instance import InstanceRecord
from rde.features.graph import descriptor_for_matrix
from rde.features.landscape import descriptor_for_vector, hamming_energy_correlation
from rde.features.numeric import safe_kurtosis, safe_pearson_r, safe_skew
from rde.features.stats import descriptor as stats_descriptor


def _warn_errors():
    return warnings.catch_warnings()


def test_safe_pearson_undefined_cases():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert np.isnan(safe_pearson_r(np.ones(4), np.arange(4.0)))
        assert np.isnan(safe_pearson_r(np.arange(4.0), np.ones(4)))
        assert np.isnan(safe_pearson_r(np.array([1.0]), np.array([2.0])))
        assert np.isnan(safe_pearson_r(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0])))


def test_landscape_degenerate_no_runtime_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = descriptor_for_vector(np.array([1.0, 2.0, 3.0]), name="costs")
        assert np.isnan(out["landscape.costs.autocorr_lag2"])
        assert np.isfinite(out["landscape.costs.autocorr_lag1"])
        out2 = descriptor_for_vector(np.ones(8), name="costs")
        assert np.isnan(out2["landscape.costs.autocorr_lag1"])
        assert np.isnan(out2["landscape.costs.skewness"])
        assert np.isnan(hamming_energy_correlation(np.ones(8)))


def test_graph_assortativity_regular_no_runtime_warning():
    # Complete graph K3: constant degree → undefined assortativity.
    q = np.ones((3, 3)) - np.eye(3)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = descriptor_for_matrix(q, name="Q")
    assert np.isnan(out["graph.Q.assortativity"])

    # Two disjoint edges: node-degree variance > 0 but endpoint-degree var = 0.
    q2 = np.zeros((4, 4))
    q2[0, 1] = q2[1, 0] = 1.0
    q2[2, 3] = q2[3, 2] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out2 = descriptor_for_matrix(q2, name="Q")
    assert np.isnan(out2["graph.Q.assortativity"])


def test_stats_constant_no_runtime_warning():
    inst = InstanceRecord(domain_id="d", size=2, seed=0, params={})
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = stats_descriptor(inst, None, np.ones(10))
    assert np.isnan(out["stats.skewness"])
    assert np.isnan(out["stats.kurtosis"])
    assert np.isnan(safe_skew(np.ones(5)))
    assert np.isnan(safe_kurtosis(np.ones(5)))


def test_safe_pearson_finite_case():
    x = np.arange(8.0)
    y = 2.0 * x + 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        r = safe_pearson_r(x, y)
    assert r == pytest.approx(1.0)
