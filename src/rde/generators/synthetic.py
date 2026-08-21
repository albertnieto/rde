"""Structured generators for the synthetic_poly test domain."""

from __future__ import annotations

import numpy as np

from rde.core.instance import InstanceRecord


class _SyntheticPolyGenerator:
    def __init__(self, generator_id: str, *, a_range: tuple[float, float], b_range: tuple[float, float]) -> None:
        self._generator_id = generator_id
        self._a_range = a_range
        self._b_range = b_range

    @property
    def generator_id(self) -> str:
        return self._generator_id

    @property
    def domain_id(self) -> str:
        return "synthetic_poly"

    def generate(self, n: int, size: int, seed: int) -> list[InstanceRecord]:
        rng = np.random.default_rng(seed)
        instances: list[InstanceRecord] = []
        for i in range(n):
            a = float(rng.uniform(*self._a_range))
            b = float(rng.uniform(*self._b_range))
            instances.append(
                InstanceRecord(
                    domain_id=self.domain_id,
                    size=size,
                    seed=seed + i,
                    params={"a": a, "b": b, "generator": self.generator_id},
                )
            )
        return instances


class SyntheticPolyGenerators:
    @staticmethod
    def all() -> list[_SyntheticPolyGenerator]:
        return [
            _SyntheticPolyGenerator("poly_wide_coeff", a_range=(0.1, 3.0), b_range=(-2.0, 2.0)),
            _SyntheticPolyGenerator("poly_narrow_coeff", a_range=(0.9, 1.1), b_range=(-0.1, 0.1)),
            _SyntheticPolyGenerator("poly_positive_slope", a_range=(1.0, 2.5), b_range=(0.0, 1.0)),
        ]
