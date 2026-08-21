"""Tests for SyntheticPoly toy domain and registry."""

from __future__ import annotations

import pytest

from rde.core.instance import InstanceRecord
from rde.core.registry import Registry
from rde.testing import SyntheticPolyDomain


def test_synthetic_poly_generate_and_materialize():
    domain = SyntheticPolyDomain()
    instances = domain.generate(n=3, size=8, seed=42)
    assert len(instances) == 3
    for inst in instances:
        assert inst.domain_id == "synthetic_poly"
        assert inst.size == 8
        assert "a" in inst.params
        slice_ = domain.materialize(inst, index=2)
        assert slice_.index == 2
        assert slice_.values.shape == (8,)
        assert slice_.kind == "polynomial_trajectory"


def test_instance_id_stable():
    a = InstanceRecord(domain_id="d", size=4, seed=1, params={"x": 1})
    b = InstanceRecord(domain_id="d", size=4, seed=1, params={"x": 1})
    assert a.instance_id == b.instance_id
    assert len(a.instance_id) == 16


def test_registry_domain_roundtrip():
    reg = Registry()
    domain = SyntheticPolyDomain()
    reg.register_domain(domain)
    assert reg.get_domain("synthetic_poly") is domain
    assert reg.list_domains() == ["synthetic_poly"]

    with pytest.raises(ValueError, match="already registered"):
        reg.register_domain(domain)
