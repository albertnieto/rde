"""Observe-layer cache contract: expensive domains implement prepare_instance."""

from __future__ import annotations

from rde.core.plugins import build_registry

# Domains whose materialize() and primitive_features() share an expensive
# primitive. Methodology §9 / ARCHITECTURE.md: the worker calls
# prepare_instance once.
_EXPENSIVE_OBSERVE_DOMAINS = (
    "hsp_functions",
    "tsp_cost_landscape",
    "tsp_landscape_stats",
    "tsp_clustered",
)


def test_expensive_observe_domains_implement_prepare_instance():
    for domain_id in _EXPENSIVE_OBSERVE_DOMAINS:
        domain = build_registry(domain_id).get_domain(domain_id)
        assert hasattr(domain, "prepare_instance"), domain_id
        assert callable(domain.prepare_instance)
