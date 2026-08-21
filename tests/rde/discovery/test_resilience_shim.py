"""Tests for discovery-layer resilience re-exports."""

from __future__ import annotations


def test_discovery_resilience_matches_runtime():
    from rde.discovery import resilience as discovery_resilience
    from rde.runtime import resilience as runtime_resilience

    assert discovery_resilience.soft_call is runtime_resilience.soft_call
    assert discovery_resilience.soft_map is runtime_resilience.soft_map
    assert discovery_resilience.SOFT_FAIL_EXCEPTIONS is runtime_resilience.SOFT_FAIL_EXCEPTIONS
