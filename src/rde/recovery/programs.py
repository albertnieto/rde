"""Enumerated collision-algebra programs for Mode 2 HSP recovery search.

Each program is a (bag, reducer, post) triple on colliding query pairs.
None of these names a group family. The experiment searches this catalog;
finding ``xor_mode_high_half`` on Heisenberg is a lead about this bit
embedding, not an automatic new quantum algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np

from rde.core.protocols import QueryTape
from rde.recovery.extractors import (
    AdditiveGcdExtractor,
    AdditiveSumExtractor,
    XorCollisionExtractor,
    _mode_or_none,
    _pair_values,
)

BAGS = ("xor", "diff", "sum")
REDUCERS = ("mode", "gcd")
POSTS = ("id", "low_half", "high_half", "low_3")


def _postprocess(value: int | None, post: str, n_bits: int) -> int | None:
    if value is None:
        return None
    m = max(1, int(n_bits) // 2)
    if post == "id":
        return int(value)
    if post == "low_half":
        return int(value) & ((1 << m) - 1)
    if post == "high_half":
        return int(value) >> m
    if post == "low_3":
        return int(value) & 7
    raise ValueError(post)


def _gcd_all(values: list[int]) -> int | None:
    if not values:
        return None
    g = 0
    for d in values:
        g = gcd(g, d)
    return int(g) if g else None


def _reduce(values: list[int], reducer: str) -> int | None:
    if reducer == "mode":
        return _mode_or_none(values)
    if reducer == "gcd":
        return _gcd_all(values)
    raise ValueError(reducer)


@dataclass(frozen=True)
class CollisionProgram:
    bag: str
    reducer: str
    post: str

    @property
    def protocol_id(self) -> str:
        return f"{self.bag}_{self.reducer}_{self.post}"

    def extract(self, tape: QueryTape) -> int | None:
        values = _pair_values(tape, self.bag)
        return _postprocess(_reduce(values, self.reducer), self.post, tape.n_bits)


@dataclass(frozen=True)
class UniqueMaskedProgram:
    bag: str
    mask_bits: int

    @property
    def protocol_id(self) -> str:
        return f"{self.bag}_unique_low_{self.mask_bits}"

    def extract(self, tape: QueryTape) -> tuple[int, ...] | None:
        mask = (1 << int(self.mask_bits)) - 1
        values = [v & mask for v in _pair_values(tape, self.bag)]
        uniq = tuple(sorted(set(values)))
        return uniq or None


@dataclass(frozen=True)
class Gf2SpanProgram:
    @property
    def protocol_id(self) -> str:
        return "xor_gf2_basis"

    def extract(self, tape: QueryTape) -> tuple[int, ...] | None:
        values = _pair_values(tape, "xor")
        if not values:
            return None
        n_bits = int(tape.n_bits)
        bits = np.array([[(v >> i) & 1 for i in range(n_bits)] for v in values], dtype=np.uint8)
        basis = _gf2_basis_ints(bits)
        return tuple(basis) or None


def _gf2_basis_ints(mat: np.ndarray) -> list[int]:
    work = np.asarray(mat, dtype=np.uint8).copy()
    if work.size == 0:
        return []
    rows, cols = work.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for r in range(rank, rows):
            if work[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        work[[rank, pivot]] = work[[pivot, rank]]
        for r in range(rows):
            if r != rank and work[r, col]:
                work[r] ^= work[rank]
        rank += 1
        if rank == rows:
            break
    out: list[int] = []
    for r in range(rank):
        val = 0
        for c in range(cols):
            if work[r, c]:
                val |= 1 << c
        if val:
            out.append(int(val))
    return out


def enumerate_recovery_programs() -> tuple[object, ...]:
    """Finite catalog of recovery programs. Poly(n) per program; no 2^n loops."""
    programs: list[object] = []
    for bag in BAGS:
        for reducer in REDUCERS:
            for post in POSTS:
                programs.append(CollisionProgram(bag, reducer, post))
        programs.append(UniqueMaskedProgram(bag, 3))
    programs.append(Gf2SpanProgram())
    programs.extend(default_pipeline_extractors())
    return tuple(programs)


def default_pipeline_extractors() -> tuple[object, ...]:
    return (XorCollisionExtractor(), AdditiveSumExtractor(), AdditiveGcdExtractor())


PIPELINE_PROTOCOL_BY_FAMILY = {
    "simon": "xor_collision_mode",
    "shor_cyclic": "additive_gcd",
    "dihedral_kuperberg": "additive_sum_mode",
}

TEXTBOOK_PROTOCOL_IDS = frozenset(PIPELINE_PROTOCOL_BY_FAMILY.values())

DISCOVERY_FAMILIES = ("heisenberg_noncentral", "quaternion_coset")
PIPELINE_FAMILIES = ("simon", "shor_cyclic", "dihedral_kuperberg")
CONTROL_FAMILY = "generic_random_control"
ALL_CAMPAIGN_FAMILIES = PIPELINE_FAMILIES + DISCOVERY_FAMILIES + (CONTROL_FAMILY,)
