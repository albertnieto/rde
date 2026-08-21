"""Registry edge-case tests."""

from __future__ import annotations

import pytest

from rde.features import register_builtin_descriptors
from rde.core.protocols import CallableDescriptor, QueryIntent
from rde.core.registry import Registry
from rde.testing import SyntheticPolyDomain


def test_unknown_domain_raises():
    reg = Registry()
    with pytest.raises(KeyError, match="Unknown domain"):
        reg.get_domain("missing")


def test_duplicate_domain_raises():
    reg = Registry()
    domain = SyntheticPolyDomain()
    reg.register_domain(domain)
    with pytest.raises(ValueError, match="already registered"):
        reg.register_domain(domain)


def test_duplicate_descriptor_raises():
    reg = Registry()
    register_builtin_descriptors(reg)
    with pytest.raises(ValueError, match="already registered"):
        reg.register_descriptor(CallableDescriptor("spectral", lambda *a: {}))


def test_unknown_metric_raises():
    reg = Registry()
    with pytest.raises(KeyError, match="Unknown metric"):
        reg.get_metric("missing")


def test_register_metric_fn():
    reg = Registry()
    reg.register_metric_fn("dummy", QueryIntent.RANK, lambda i, s, d: 1.0)
    assert reg.get_metric("dummy").score(None, None, {}) == 1.0  # type: ignore[arg-type]
