"""Central registry for domains, descriptors, and metrics."""

from __future__ import annotations

from rde.core.protocols import (
    CallableDescriptor,
    CallableMetric,
    Descriptor,
    DescriptorFn,
    Domain,
    Metric,
    MetricFn,
    QueryIntent,
)


class Registry:
    """In-memory plugin registry."""

    def __init__(self) -> None:
        self._domains: dict[str, Domain] = {}
        self._descriptors: dict[str, Descriptor] = {}
        self._metrics: dict[str, Metric] = {}

    def register_domain(self, domain: Domain) -> None:
        domain_id = domain.domain_id
        if domain_id in self._domains:
            raise ValueError(f"Domain already registered: {domain_id!r}")
        self._domains[domain_id] = domain

    def get_domain(self, domain_id: str) -> Domain:
        if domain_id not in self._domains:
            raise KeyError(f"Unknown domain: {domain_id!r}")
        return self._domains[domain_id]

    def list_domains(self) -> list[str]:
        return sorted(self._domains)

    def register_descriptor(self, descriptor: Descriptor) -> None:
        if descriptor.name in self._descriptors:
            raise ValueError(f"Descriptor already registered: {descriptor.name!r}")
        self._descriptors[descriptor.name] = descriptor

    def register_descriptor_fn(self, name: str, fn: DescriptorFn) -> None:
        self.register_descriptor(CallableDescriptor(name, fn))

    def get_descriptor(self, name: str) -> Descriptor:
        if name not in self._descriptors:
            raise KeyError(f"Unknown descriptor: {name!r}")
        return self._descriptors[name]

    def list_descriptors(self) -> list[str]:
        return sorted(self._descriptors)

    def register_metric(self, metric: Metric) -> None:
        if metric.name in self._metrics:
            raise ValueError(f"Metric already registered: {metric.name!r}")
        self._metrics[metric.name] = metric

    def register_metric_fn(self, name: str, intent: QueryIntent, fn: MetricFn) -> None:
        self.register_metric(CallableMetric(name, intent, fn))

    def get_metric(self, name: str) -> Metric:
        if name not in self._metrics:
            raise KeyError(f"Unknown metric: {name!r}")
        return self._metrics[name]

    def list_metrics(self) -> list[str]:
        return sorted(self._metrics)


_GLOBAL_REGISTRY = Registry()


def get_registry() -> Registry:
    return _GLOBAL_REGISTRY
