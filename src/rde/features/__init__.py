"""Generic array feature extractors (domain-agnostic)."""

from rde.features import compress, dynamics, fourier, graph, landscape, matrix, spectral, stats
from rde.runtime.metrics import register_composite_metrics
from rde.core.protocols import CallableDescriptor
from rde.core.registry import Registry
from rde.descriptor_gen.pipeline import generated_descriptor

__all__ = [
    "dynamics",
    "fourier",
    "graph",
    "landscape",
    "matrix",
    "register_builtin_descriptors",
    "register_builtin_metrics",
    "register_generated_descriptors",
    "spectral",
    "stats",
]


def register_builtin_descriptors(registry: Registry) -> None:
    """Register built-in domain-agnostic descriptors."""
    registry.register_descriptor(CallableDescriptor("spectral", spectral.descriptor))
    registry.register_descriptor(CallableDescriptor("stats", stats.descriptor))
    registry.register_descriptor(CallableDescriptor("compress", compress.descriptor))
    registry.register_descriptor(CallableDescriptor("fourier", fourier.descriptor))
    registry.register_descriptor(CallableDescriptor("matrix", matrix.descriptor))
    registry.register_descriptor(CallableDescriptor("graph", graph.descriptor))
    registry.register_descriptor(CallableDescriptor("landscape", landscape.descriptor))
    registry.register_descriptor(CallableDescriptor("gen", generated_descriptor))


def register_generated_descriptors(registry: Registry) -> None:
    """Alias for registering parameterized descriptor generators only."""
    if "gen" not in registry.list_descriptors():
        registry.register_descriptor(CallableDescriptor("gen", generated_descriptor))


def register_builtin_metrics(registry: Registry) -> None:
    """Register composite representation metrics."""
    register_composite_metrics(registry)
