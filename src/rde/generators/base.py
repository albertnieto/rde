"""Instance generator protocol and registry (Phase 1)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rde.core.instance import InstanceRecord


@runtime_checkable
class InstanceGenerator(Protocol):
    """Produces InstanceRecord batches for a fixed domain."""

    @property
    def generator_id(self) -> str:
        ...

    @property
    def domain_id(self) -> str:
        ...

    def generate(self, n: int, size: int, seed: int) -> list[InstanceRecord]:
        ...


_GENERATORS: dict[str, InstanceGenerator] = {}


def register_generator(generator: InstanceGenerator) -> None:
    _GENERATORS[generator.generator_id] = generator


def get_generator(generator_id: str) -> InstanceGenerator:
    if generator_id not in _GENERATORS:
        raise KeyError(f"Unknown generator: {generator_id!r}")
    return _GENERATORS[generator_id]


def list_generators(domain_id: str | None = None) -> list[str]:
    if domain_id is None:
        return sorted(_GENERATORS)
    return sorted(gid for gid, g in _GENERATORS.items() if g.domain_id == domain_id)


def register_builtin_generators() -> None:
    from rde.generators.synthetic import SyntheticPolyGenerators

    for gen in SyntheticPolyGenerators.all():
        register_generator(gen)
