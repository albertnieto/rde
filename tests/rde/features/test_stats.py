"""Unit tests for statistical descriptors."""

from __future__ import annotations

import numpy as np

from rde.features.stats import descriptor, shannon_entropy_histogram
from rde.core.instance import InstanceRecord


def _inst() -> InstanceRecord:
    return InstanceRecord(domain_id="test", size=4, seed=0, params={})


def test_shannon_entropy_constant_is_zero():
    assert shannon_entropy_histogram(np.ones(10)) == 0.0


def test_descriptor_basic_moments():
    arr = np.array([0.0, 1.0, 2.0, 3.0])
    out = descriptor(_inst(), None, arr)
    assert out["stats.count"] == 4.0
    assert np.isclose(out["stats.mean"], 1.5)
    assert np.isclose(out["stats.min"], 0.0)
    assert np.isclose(out["stats.max"], 3.0)
    assert "stats.participation_ratio" in out


def test_empty_array_count_zero():
    out = descriptor(_inst(), None, np.array([]))
    assert out["stats.count"] == 0.0
