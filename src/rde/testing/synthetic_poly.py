"""Test-only synthetic polynomial domain (not for production science)."""

from __future__ import annotations

import numpy as np

from rde.core.instance import InstanceRecord
from rde.core.protocols import SimpleFamilySlice


class SyntheticPolyDomain:
    """Family: psi_r(x) = (a*x + b)^r on a discrete grid x in [0, 1]."""

    domain_id = "synthetic_poly"

    def generate(self, n: int, size: int, seed: int) -> list[InstanceRecord]:
        rng = np.random.default_rng(seed)
        instances: list[InstanceRecord] = []
        for i in range(n):
            a = float(rng.uniform(0.5, 2.0))
            b = float(rng.uniform(-1.0, 1.0))
            inst_seed = seed + i
            instances.append(
                InstanceRecord(
                    domain_id=self.domain_id,
                    size=size,
                    seed=inst_seed,
                    params={"a": a, "b": b},
                )
            )
        return instances

    def materialize(self, instance: InstanceRecord, index: int) -> SimpleFamilySlice:
        a = float(instance.params["a"])
        b = float(instance.params["b"])
        n = instance.size
        x = np.linspace(0.0, 1.0, n, dtype=float)
        values = (a * x + b) ** index
        return SimpleFamilySlice(values=values, index=index, kind="polynomial_trajectory")

    def materialize_indices(
        self,
        instance: InstanceRecord,
        indices: list[int],
        *,
        cache: dict | None = None,
    ) -> dict[int, SimpleFamilySlice]:
        return {idx: self.materialize(instance, idx) for idx in indices}

    def primitive_features(
        self,
        instance: InstanceRecord,
        *,
        cache: dict | None = None,
    ) -> dict[str, float | np.ndarray]:
        return {
            "a": float(instance.params["a"]),
            "b": float(instance.params["b"]),
        }
