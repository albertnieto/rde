"""Representation search: rank the grammar's primitives for one batch of data.

Phase 3 of the representation-discovery roadmap. This is exhaustive
evaluation, not beam/evolutionary search, by design: `grammar.py` ships six
primitives, all sharing one object type, so evaluating every primitive
once is already optimal — a heuristic search algorithm over six items would
be complexity for its own sake, not a capability. If the grammar grows past
what exhaustive evaluation can cover cheaply, add beam/evolutionary search
here then; do not add it speculatively now.

Multi-hop composition (`R1 -> R2 -> R3`) is not offered on top of *this*
flat grammar (every primitive here shares the same `object_type`/carrier
domain, so chaining two of them collapses to using the second directly —
see `layered.py`'s module docstring for the exact argument and where it
stops applying). It becomes meaningful the moment a primitive's carrier
differs from its input, which is what `layered.py` (depth-2) and
`program_search.py` (depth-`K`) build on top of this module for —
`rank_representations`'s `chain_max_depth` parameter opts into ranking
those composed chains alongside the flat grammar, not a parallel ranking
path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from rde.representation.array_backend import ArraySearchBackend, get_array_backend
from rde.representation.certificate import Certificate, certify_roundtrip
from rde.representation.cost import computational_cost
from rde.representation.grammar import build_primitive_representations
from rde.representation.representation import Representation


@dataclass(frozen=True)
class SearchCandidate:
    """One grammar primitive, evaluated on one batch.

    `conversion_cost` (see `cost.py`) is a separate axis from `complexity`:
    complexity is "how simple does the encoded value look", conversion cost
    is "how expensive was it to get there" — a representation can be simple
    to look at and expensive to compute, or vice versa. Ranking on
    `complexity` alone (as `rank_representations` does) does not hide that
    tradeoff; `conversion_cost` stays available as its own field for
    `pareto.objectives_from_candidates`.
    """

    representation_id: str
    representation: Representation
    certificate: Certificate
    complexity: float
    conversion_cost: float


def rank_representations(
    values: Any,
    *,
    n: int,
    backend: ArraySearchBackend | None = None,
    object_type: str | None = None,
    tolerance: float = 1e-6,
    primitive_subset: Sequence[str] | None = None,
    chain_max_depth: int | None = None,
) -> list[SearchCandidate]:
    """Certify and score every grammar primitive against `values`, best first.

    Ordering: verified roundtrips before refuted ones, then ascending
    description complexity (`Representation.complexity`, lower = simpler).

    `primitive_subset` restricts which grammar primitives are even
    considered (see `grammar.build_primitive_representations`) — used by
    `holdout.py` to genuinely withhold primitives, not merely filter
    results after the fact.

    `chain_max_depth`, when given, ranks `program_search.enumerate_chains`'s
    output (depth-1..`chain_max_depth` typed chains, which already include
    every flat grammar primitive as its own depth-1 chain) instead of the
    flat grammar alone — this is the "wire chain search into `search.py`'s
    ranking" the theory doc's roadmap flagged as missing (see
    `docs/representation-synthesis-theory.md` §10). Left `None` (default),
    behavior is unchanged from before this parameter existed: only
    `grammar.py`'s single-stage primitives are ranked, no `program_search`
    import even happens.
    """
    backend = backend or get_array_backend()
    if chain_max_depth is None:
        grammar = build_primitive_representations(
            n, object_type=object_type, backend=backend, primitive_subset=primitive_subset
        )
    else:
        from rde.representation.program_search import enumerate_chains

        grammar = enumerate_chains(
            n,
            max_depth=chain_max_depth,
            object_type=object_type,
            backend=backend,
            primitive_subset=primitive_subset,
        )

    candidates: list[SearchCandidate] = []
    for representation_id, representation in grammar.items():
        certificate = certify_roundtrip(representation, values, tolerance=tolerance)
        encoded = representation.encode(values)
        complexity = (
            representation.complexity(encoded)
            if representation.complexity is not None
            else float("inf")
        )
        candidates.append(
            SearchCandidate(
                representation_id=representation_id,
                representation=representation,
                certificate=certificate,
                complexity=complexity,
                conversion_cost=computational_cost(representation_id, n),
            )
        )

    candidates.sort(key=lambda c: (c.certificate.status != "verified", c.complexity))
    return candidates


def best_representation(
    values: Any,
    *,
    n: int,
    backend: ArraySearchBackend | None = None,
    object_type: str | None = None,
    tolerance: float = 1e-6,
    primitive_subset: Sequence[str] | None = None,
    chain_max_depth: int | None = None,
) -> SearchCandidate:
    """Convenience wrapper: the top-ranked candidate from `rank_representations`."""
    ranked = rank_representations(
        values,
        n=n,
        backend=backend,
        object_type=object_type,
        tolerance=tolerance,
        primitive_subset=primitive_subset,
        chain_max_depth=chain_max_depth,
    )
    if not ranked:
        raise ValueError("grammar produced no candidates for this N")
    return ranked[0]
