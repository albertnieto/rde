"""Block-separable minimization — a `SynthesisDomain` example (not for
production science; see `rde.testing.synthetic_poly` for the analogous
forward-discovery example domain).

This is the literal toy example from the reverse-engineering brief: an
instance is a block-diagonal cost function

    C(x) = C_1(x_1) + C_2(x_2) + ... + C_k(x_k)

(``Q = block_diag(Q_1, ..., Q_k)`` in QUBO language, kept domain-agnostic
here — RDE core never imports external domain adapters). Each block has a small fixed bit-width
so it is a genuine, independently-checkable optimization problem, not an
answer smuggled into the instance. Solving the whole instance by brute force
costs ``Theta(2^(block_width * n_blocks))`` — exponential in the number of
blocks — while decomposing along block boundaries and combining by summing
independently-found optima is exact and ``Theta(n_blocks)``. It exists to
give `rde.synthesis` an end-to-end, verifiable demonstration that the search
actually rediscovers "decompose, solve independently, merge" rather than
assuming it.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

from rde.core.instance import InstanceRecord
from rde.core.protocols import SimpleFamilySlice, SynthesisSolution

BLOCK_WIDTH = 3  # bits per block; brute-forcing one block costs 2**BLOCK_WIDTH (a constant)


def _tables(instance: Any) -> list[list[float]]:
    if isinstance(instance, InstanceRecord):
        return instance.params["tables"]
    return instance["tables"]


class BlockSeparableDomain:
    """`SynthesisDomain` (and minimal `Domain`) over block-diagonal cost tables."""

    domain_id = "block_separable"

    # ---- Domain protocol (generation / registry compatibility) ----

    def generate(self, n: int, size: int, seed: int) -> list[InstanceRecord]:
        rng = np.random.default_rng(seed)
        instances: list[InstanceRecord] = []
        for i in range(n):
            tables = rng.uniform(-1.0, 1.0, size=(size, 2**BLOCK_WIDTH)).round(6).tolist()
            instances.append(
                InstanceRecord(
                    domain_id=self.domain_id,
                    size=size,
                    seed=seed + i,
                    params={"tables": tables},
                )
            )
        return instances

    def materialize(self, instance: InstanceRecord, index: int) -> SimpleFamilySlice:
        tables = _tables(instance)
        block_minima = np.array([min(t) for t in tables], dtype=float)
        return SimpleFamilySlice(values=block_minima, index=index, kind="block_minima")

    def primitive_features(self, instance: InstanceRecord) -> dict[str, float | np.ndarray]:
        tables = _tables(instance)
        return {"n_blocks": float(len(tables)), "block_width": float(BLOCK_WIDTH)}

    # ---- SynthesisDomain protocol (Mode 2) ----

    def base_case_cost_exponent(self) -> float:
        """log2 of the per-block brute-force branching factor — the rate at
        which `brute_force`'s joint enumeration grows with n_blocks."""
        return float(BLOCK_WIDTH)

    def size_of(self, instance: Any) -> int:
        return len(_tables(instance))

    def brute_force(self, instance: Any) -> SynthesisSolution:
        """Exact optimum via full joint enumeration — the reference oracle.

        Deliberately does not exploit separability even though the domain
        happens to be separable: it enumerates the whole joint assignment
        space, so it is a legitimate, un-gamed ground truth for
        `verify_skeleton` to check candidate skeletons against.
        """
        tables = _tables(instance)
        if not tables:
            return SynthesisSolution(assignment=(), cost=0.0)
        if len(tables) == 1:
            table = tables[0]
            best_idx = min(range(len(table)), key=lambda j: table[j])
            return SynthesisSolution(assignment=(best_idx,), cost=float(table[best_idx]))
        best_assignment: tuple[int, ...] | None = None
        best_cost = math.inf
        for combo in itertools.product(*(range(len(t)) for t in tables)):
            cost = sum(t[j] for t, j in zip(tables, combo))
            if cost < best_cost:
                best_cost = cost
                best_assignment = combo
        assert best_assignment is not None
        return SynthesisSolution(assignment=best_assignment, cost=best_cost)

    def decompose_flat(self, instance: Any) -> list[Any] | None:
        tables = _tables(instance)
        if len(tables) < 2:
            return None
        return [{"tables": [t]} for t in tables]

    def decompose_divide(self, instance: Any, branches: int) -> list[Any] | None:
        """Contiguous, order-preserving split into `branches` near-equal groups.

        Must stay order-preserving (not e.g. round-robin `idx % branches`):
        `combine` reconstructs the full assignment by concatenating
        sub-solutions in group order, and that only reproduces the original
        block order — which `cost()` then zips against `tables` — if every
        level of recursive splitting partitions the sequence into contiguous
        slices. A non-contiguous split (round-robin, interleaved) scrambles
        that correspondence across uneven recursive splits and silently
        mismatches assignments to blocks.
        """
        tables = _tables(instance)
        n = len(tables)
        if branches < 2 or n < branches:
            return None
        base_size, remainder = divmod(n, branches)
        groups: list[list[list[float]]] = []
        start = 0
        for i in range(branches):
            size = base_size + (1 if i < remainder else 0)
            groups.append(tables[start : start + size])
            start += size
        return [{"tables": g} for g in groups]

    def combine(self, instance: Any, sub_solutions: list[SynthesisSolution]) -> SynthesisSolution:
        assignment: tuple[int, ...] = tuple(
            j for sol in sub_solutions for j in (sol.assignment if isinstance(sol.assignment, tuple) else (sol.assignment,))
        )
        return SynthesisSolution(assignment=assignment, cost=sum(sol.cost for sol in sub_solutions))

    def cost(self, instance: Any, solution: SynthesisSolution) -> float:
        """Independently re-evaluate `solution.assignment` against `instance`
        — never trusts `solution.cost`, so this is a real correctness check."""
        tables = _tables(instance)
        assignment = solution.assignment
        if len(assignment) != len(tables):
            raise ValueError(
                f"assignment length {len(assignment)} does not match {len(tables)} blocks"
            )
        return float(sum(t[j] for t, j in zip(tables, assignment)))
