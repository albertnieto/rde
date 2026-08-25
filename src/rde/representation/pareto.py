"""Multi-objective ranking over representation candidates (Phase 5).

Every objective here is minimized (description complexity, conversion cost,
roundtrip error, ...). Dominance is the whole `(K, K, M)` pairwise
comparison computed in one broadcast — never a Python double loop over
candidate pairs, which is what naive Pareto-frontier code usually does.

This intentionally does not implement Q(R) as a single learned/fitted
scalar score (section 10 of the original proposal sketches one) — a linear
scalarization of un-normalized, differently-scaled objectives would silently
encode arbitrary weight choices as if they were discovered, which is the
overclaim this project's methodology forbids. `weighted_score` is offered as
an explicit, caller-supplied scalarization for when one is actually wanted;
`pareto_rank` (the frontier) is the default because it makes no such
weighting choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from rde.representation.search import SearchCandidate


def dominance_matrix(objectives: np.ndarray) -> np.ndarray:
    """`dominance[i, j]` is True iff row `i` Pareto-dominates row `j`.

    `i` dominates `j` iff `objectives[i] <= objectives[j]` in every column
    and strictly less in at least one. Computed as a single `(K, K, M)`
    broadcast comparison, not a loop over the `K**2` candidate pairs.
    """
    obj = np.asarray(objectives, dtype=float)
    if obj.ndim != 2:
        raise ValueError(f"objectives must be 2-D (K, M), got shape {obj.shape}")
    le = obj[:, None, :] <= obj[None, :, :]
    lt = obj[:, None, :] < obj[None, :, :]
    dominates = np.all(le, axis=2) & np.any(lt, axis=2)
    np.fill_diagonal(dominates, False)
    return dominates


@dataclass(frozen=True)
class ParetoResult:
    """`frontier_mask[i]` is True iff no other row dominates row `i`."""

    objectives: np.ndarray
    dominance: np.ndarray
    frontier_mask: np.ndarray

    @property
    def frontier_indices(self) -> np.ndarray:
        return np.flatnonzero(self.frontier_mask)


def pareto_rank(objectives: np.ndarray) -> ParetoResult:
    """Dominance matrix and non-dominated (frontier) mask for `objectives`."""
    obj = np.asarray(objectives, dtype=float)
    dominance = dominance_matrix(obj)
    dominated = np.any(dominance, axis=0)
    return ParetoResult(objectives=obj, dominance=dominance, frontier_mask=~dominated)


OBJECTIVE_NAMES: tuple[str, str, str] = ("complexity", "roundtrip_error", "conversion_cost")


def objectives_from_candidates(
    candidates: Sequence[SearchCandidate],
) -> tuple[np.ndarray, list[str]]:
    """`(complexity, roundtrip_error, conversion_cost)` objectives for `rank_representations` output.

    All three minimized, in `OBJECTIVE_NAMES` order. `conversion_cost`
    (`cost.computational_cost`) closes the gap where description complexity
    and "how expensive was it to compute" used to be tracked in unrelated
    places — a representation that looks simple but cost `O(n^2)` to reach
    no longer silently outranks one that cost `O(n log n)`, unless the
    frontier says the simplicity was worth it.

    Extracting three scalar fields from each of the `K` candidate objects is
    a small (grammar-sized, not data-sized) Python comprehension — the
    objectives themselves were already computed by fully vectorized batch
    calls in `search.rank_representations`.
    """
    if not candidates:
        return np.empty((0, 3), dtype=float), []
    objectives = np.array(
        [[c.complexity, c.certificate.error, c.conversion_cost] for c in candidates], dtype=float
    )
    ids = [c.representation_id for c in candidates]
    return objectives, ids


def canonical_representation(candidates: Sequence[SearchCandidate]) -> SearchCandidate | None:
    """Pick one representative from the Pareto frontier via one explicit, disclosed rule.

    Gap closure: `search.py` and the representation-synthesis theory doc
    both name canonicalization as unimplemented. It turns out not to need
    new machinery — `pareto_rank` already computes the frontier; this adds
    exactly one fixed, named selection rule over it, not a caller-tunable
    scalarization (that would repeat the overclaim `weighted_score`'s
    docstring already refuses to make the default).

    Rule, applied only to candidates on the Pareto frontier (never to a
    dominated one, however it breaks ties): lowest `complexity`; ties
    broken by lowest `conversion_cost`; further ties by lowest
    `certificate.error`; final ties by `representation_id` (alphabetical,
    for a fully deterministic result). Returns `None` for empty `candidates`.
    """
    if not candidates:
        return None
    objectives, ids = objectives_from_candidates(candidates)
    frontier_indices = pareto_rank(objectives).frontier_indices
    by_id = {c.representation_id: c for c in candidates}

    def _tie_break_key(index: int) -> tuple[float, float, float, str]:
        candidate = by_id[ids[index]]
        return (
            candidate.complexity,
            candidate.conversion_cost,
            candidate.certificate.error,
            candidate.representation_id,
        )

    best_index = min(frontier_indices, key=_tie_break_key)
    return by_id[ids[best_index]]


def weighted_score(objectives: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    """Explicit linear scalarization `-sum(weights * objectives)` (higher is better).

    Caller-supplied weights only — see module docstring for why this is not
    the default ranking.
    """
    obj = np.asarray(objectives, dtype=float)
    w = np.asarray(weights, dtype=float)
    if obj.shape[1] != w.shape[0]:
        raise ValueError(f"objectives has {obj.shape[1]} columns but {w.shape[0]} weights given")
    return -(obj * w[None, :]).sum(axis=1)
