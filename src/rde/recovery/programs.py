"""Enumerated collision-algebra programs for Mode 2 HSP recovery search.

Most programs here are a (bag, reducer, post) triple on colliding query
pairs; `GroupClosureProgram` reads collision-group membership directly
instead, and `PairCombine` composes two programs into one paired-output
program (see `rde.recovery.search_space` for the depth-2 search that builds
and scores those pairings). None of these names a group family -- what
family a given program or pair happens to solve is discovered by
`campaign.py`'s discovery/confirmatory split, not designed in by picking a
family and hand-fitting a class to it. Finding ``xor_mode_high_half`` on
Heisenberg is a lead about this bit embedding, not an automatic new quantum
algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any

import numpy as np

from rde.core.protocols import QueryTape
from rde.recovery.extractors import (
    AdditiveGcdExtractor,
    AdditiveSumExtractor,
    XorCollisionExtractor,
    _mode_or_none,
    _mode_or_none_confident,
    _pair_values,
)
from rde.recovery.tape import collision_groups

BAGS = ("xor", "diff", "sum", "ratio")
REDUCERS = ("mode", "gcd")
POSTS = ("id", "low_half", "high_half", "low_3")

# Fixed, not swept -- one preregistered margin, not a hyperparameter the
# search gets to tune per family.
MODE_CONFIDENCE_RATIO = 3.0


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


@dataclass(frozen=True)
class ConfidentCollisionProgram:
    """Like `CollisionProgram`, but the mode step abstains below a confidence margin.

    Same bag x post grammar as `CollisionProgram`; the only new degree of
    freedom is refusing to answer on a weak plurality rather than always
    guessing. A general hardening of the existing grammar, not written with
    any one family in mind -- verified during design to turn a family where
    plain pooled `mode` guesses wrong most of the time into one where it
    never guesses wrong, at the cost of abstaining when the query budget
    does not yet give a clear plurality.
    """

    bag: str
    post: str
    min_ratio: float = MODE_CONFIDENCE_RATIO

    @property
    def protocol_id(self) -> str:
        return f"{self.bag}_mode_confident_{self.post}"

    def extract(self, tape: QueryTape) -> int | None:
        values = _pair_values(tape, self.bag)
        value = _mode_or_none_confident(values, self.min_ratio)
        return _postprocess(value, self.post, tape.n_bits)


_CLOSURE_IDENTITY = {"xor": 0, "mult": 1}


def _closed_under(values: frozenset[int], op: str, modulus: int) -> bool:
    for a in values:
        for b in values:
            c = (a ^ b) if op == "xor" else (a * b) % modulus
            if c not in values:
                return False
    return True


def _relative_group_values(group: list[int], op: str, mask: int, modulus: int) -> frozenset[int]:
    """Every member's relationship to the group's own base, including the identity.

    Relative to `group[0]` rather than each member's raw value -- a hidden
    subgroup H shows up as *some* coset g*H (xor: g^H) of members sharing a
    label, and g itself need not be H's identity element. Reading raw values
    only recognizes H when the sampled group happens to include the literal
    identity; reading values relative to the base always exposes it (base
    relative to itself), independent of which coset representative was
    sampled.
    """
    base = group[0]
    identity = _CLOSURE_IDENTITY[op]
    if op == "xor":
        base_m = base & mask
        return frozenset({identity} | {(other & mask) ^ base_m for other in group[1:]})
    # op == "mult": undefined when base shares a factor with modulus.
    try:
        inv = pow(base, -1, modulus)
    except ValueError:
        return frozenset()
    return frozenset({identity} | {(other * inv) % modulus for other in group[1:]})


@dataclass(frozen=True)
class GroupClosureProgram:
    """Smallest observed collision-group value-set, closed under `op`, containing its identity.

    Reads collision-*group membership* directly (every raw value sharing one
    label), not a pairwise combination of two members -- the shape a hidden
    *subgroup* (as opposed to a hidden *shift*) leaves in a query tape: a
    whole coset collapsing onto one output label. Family-agnostic: applies
    to any family whose planted object happens to collapse to a small
    op-closed set at this bit width, not written for any specific family.
    `op="xor"` reads a hidden subgroup of the additive/XOR group (mask_bits
    is the free parameter, same small range `UniqueMaskedProgram` already
    enumerates); `op="mult"` reads a hidden subgroup of the *multiplicative*
    group mod `tape.modulus` instead -- mask_bits is unused there since the
    raw domain values already are the group's own coordinates, not bits
    embedded in a larger register.
    """

    mask_bits: int
    op: str = "xor"

    @property
    def protocol_id(self) -> str:
        return f"{self.op}_closure_mask{self.mask_bits}"

    def extract(self, tape: QueryTape) -> tuple[int, ...] | None:
        mask = (1 << int(self.mask_bits)) - 1
        modulus = int(tape.modulus)
        candidates: list[frozenset[int]] = []
        for group in collision_groups(tape):
            if len(group) < 2:
                continue
            values = _relative_group_values(group, self.op, mask, modulus)
            if len(values) > 1 and _closed_under(values, self.op, modulus):
                candidates.append(values)
        if not candidates:
            return None
        # Largest observed closed set, not smallest: a subgroup H can nest
        # smaller subgroups (e.g. order-4 H containing an order-2 {identity,
        # h} pair), so an under-sampled coset can look "closed" by
        # coincidence at a smaller size. A larger set staying closed under
        # every pairwise combination is exponentially less likely by chance,
        # so it is the stronger evidence.
        best = max(candidates, key=len)
        return tuple(sorted(best))


@dataclass(frozen=True)
class PairCombine:
    """Depth-2 composition: run two independent depth-1 pipelines, report their paired result.

    General composition, not "xor+sum for one specific family" by
    construction -- any two `RecoveryProtocol`-conforming objects can be
    paired. Which pair (if any) actually recovers a two-mechanism family's
    two independent secrets is for `search_space.search_recovery_chains` to
    find via the discovery/confirmatory split, not hardcoded here.
    """

    protocol_a: Any
    protocol_b: Any

    @property
    def protocol_id(self) -> str:
        return f"pair[{self.protocol_a.protocol_id}|{self.protocol_b.protocol_id}]"

    def extract(self, tape: QueryTape) -> tuple[Any, Any]:
        return (self.protocol_a.extract(tape), self.protocol_b.extract(tape))


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

DISCOVERY_FAMILIES = (
    "heisenberg_noncentral",
    "quaternion_coset",
    "abelian_dihedral_blend",
    "multiplicative_fold",
)
PIPELINE_FAMILIES = ("simon", "shor_cyclic", "dihedral_kuperberg")
CONTROL_FAMILY = "generic_random_control"
ALL_CAMPAIGN_FAMILIES = PIPELINE_FAMILIES + DISCOVERY_FAMILIES + (CONTROL_FAMILY,)
