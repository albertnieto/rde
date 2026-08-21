"""Unit tests for spectral descriptors."""

from __future__ import annotations

import numpy as np

from rde.features.spectral import descriptor, effective_rank
from rde.core.instance import InstanceRecord


def _inst() -> InstanceRecord:
    return InstanceRecord(domain_id="test", size=4, seed=0, params={})


def test_effective_rank_single_component():
    assert effective_rank(np.array([1.0, 0.0, 0.0])) == 1.0


def test_effective_rank_uniform():
    s = np.ones(4)
    assert np.isclose(effective_rank(s), 4.0)


def test_vector_descriptor_norm_and_rank():
    vec = np.array([3.0, 4.0])
    out = descriptor(_inst(), None, vec)
    assert np.isclose(out["spectral.l2_norm"], 5.0)
    assert out["spectral.effective_rank"] >= 1.0
    assert out["spectral.spectral_gap"] == 1.0


def test_matrix_descriptor_singular_values():
    mat = np.diag([2.0, 1.0, 0.5])
    out = descriptor(_inst(), None, mat)
    assert np.isclose(out["spectral.max_singular_value"], 2.0)
    assert np.isclose(out["spectral.min_singular_value"], 0.5)
    assert out["spectral.effective_rank"] >= 1.0


def test_empty_array_returns_empty():
    assert descriptor(_inst(), None, None) == {}
