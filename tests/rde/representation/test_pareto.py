"""Tests for vectorized Pareto dominance/frontier ranking (Phase 5)."""

from __future__ import annotations

import numpy as np
import pytest

from rde.representation import (
    canonical_representation,
    dominance_matrix,
    objectives_from_candidates,
    pareto_rank,
    weighted_score,
)
from rde.representation.certificate import Certificate
from rde.representation.search import SearchCandidate


def test_dominance_matrix_basic_relationships():
    # row0=[1,1] dominates row1=[2,2] and row2=[1,2]; row2 dominates row1.
    objectives = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 2.0]])
    dominance = dominance_matrix(objectives)
    assert dominance[0, 1]
    assert dominance[0, 2]
    assert dominance[2, 1]
    assert not dominance[1, 0]
    assert not dominance[1, 2]
    assert not dominance[2, 0]


def test_dominance_matrix_diagonal_is_false():
    objectives = np.array([[1.0, 2.0], [3.0, 4.0]])
    dominance = dominance_matrix(objectives)
    assert not dominance[0, 0]
    assert not dominance[1, 1]


def test_dominance_matrix_rejects_non_2d_input():
    with pytest.raises(ValueError):
        dominance_matrix(np.array([1.0, 2.0, 3.0]))


def test_pareto_rank_frontier_excludes_dominated_points():
    objectives = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 2.0], [0.5, 3.0]])
    result = pareto_rank(objectives)
    assert set(result.frontier_indices.tolist()) == {0, 3}


def test_pareto_rank_frontier_includes_all_when_mutually_non_dominated():
    # A diagonal frontier: each point strictly better in exactly one objective.
    objectives = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
    result = pareto_rank(objectives)
    assert set(result.frontier_indices.tolist()) == {0, 1, 2, 3}


def test_pareto_rank_frontier_excludes_identical_dominated_duplicate():
    # Two identical points do not dominate each other (must be strictly less
    # in at least one objective), so both stay on the frontier.
    objectives = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    result = pareto_rank(objectives)
    assert set(result.frontier_indices.tolist()) == {0, 1}


def _fake_candidate(
    representation_id: str, complexity: float, error: float, conversion_cost: float = 0.0
) -> SearchCandidate:
    cert = Certificate(
        representation_id=representation_id,
        object_type="numeric_batch_4",
        claim="exact_roundtrip",
        status="verified",
        error=error,
        tolerance=1e-9,
    )
    return SearchCandidate(
        representation_id=representation_id,
        representation=None,
        certificate=cert,
        complexity=complexity,
        conversion_cost=conversion_cost,
    )


def test_objectives_from_candidates_shape_and_order():
    candidates = [
        _fake_candidate("a", complexity=3.0, error=1e-10, conversion_cost=8.0),
        _fake_candidate("b", complexity=1.0, error=1e-12, conversion_cost=64.0),
    ]
    objectives, ids = objectives_from_candidates(candidates)
    assert objectives.shape == (2, 3)
    assert ids == ["a", "b"]
    assert objectives[0, 0] == 3.0
    assert objectives[1, 0] == 1.0
    assert objectives[0, 2] == 8.0
    assert objectives[1, 2] == 64.0


def test_objectives_from_candidates_empty_list():
    objectives, ids = objectives_from_candidates([])
    assert objectives.shape == (0, 3)
    assert ids == []


def test_objectives_from_candidates_cost_can_flip_pareto_preference():
    # "b" looks simpler (lower complexity) but cost far more to compute;
    # "a" is both simpler-to-compute and only slightly more complex — "b"
    # must not dominate "a" once conversion_cost is part of the objectives.
    candidates = [
        _fake_candidate("a", complexity=2.0, error=1e-10, conversion_cost=4.0),
        _fake_candidate("b", complexity=1.0, error=1e-10, conversion_cost=400.0),
    ]
    objectives, ids = objectives_from_candidates(candidates)
    result = pareto_rank(objectives)
    assert set(ids[i] for i in result.frontier_indices) == {"a", "b"}


def test_weighted_score_prefers_lower_objectives():
    objectives = np.array([[1.0, 1.0], [2.0, 2.0]])
    scores = weighted_score(objectives, weights=[1.0, 1.0])
    assert scores[0] > scores[1]


def test_weighted_score_rejects_mismatched_weight_length():
    objectives = np.array([[1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(ValueError):
        weighted_score(objectives, weights=[1.0, 1.0, 1.0])


def test_canonical_representation_returns_none_for_empty_candidates():
    assert canonical_representation([]) is None


def test_canonical_representation_picks_lowest_complexity_on_the_frontier():
    candidates = [
        _fake_candidate("a", complexity=3.0, error=1e-10, conversion_cost=1.0),
        _fake_candidate("b", complexity=1.0, error=1e-10, conversion_cost=1.0),
    ]
    result = canonical_representation(candidates)
    assert result.representation_id == "b"


def test_canonical_representation_never_picks_a_dominated_candidate():
    # "a" is dominated by "b" in every objective; canonicalization must
    # never surface a dominated candidate even if some tie-break field
    # would otherwise favor it.
    candidates = [
        _fake_candidate("a", complexity=5.0, error=1e-8, conversion_cost=5.0),
        _fake_candidate("b", complexity=1.0, error=1e-10, conversion_cost=1.0),
    ]
    result = canonical_representation(candidates)
    assert result.representation_id == "b"


def test_canonical_representation_breaks_ties_by_conversion_cost_then_error():
    # Equal complexity -> lower conversion_cost wins.
    tie_on_complexity = [
        _fake_candidate("expensive", complexity=1.0, error=1e-10, conversion_cost=100.0),
        _fake_candidate("cheap", complexity=1.0, error=1e-10, conversion_cost=1.0),
    ]
    assert canonical_representation(tie_on_complexity).representation_id == "cheap"

    # Equal complexity and conversion_cost -> lower roundtrip error wins.
    tie_on_cost_too = [
        _fake_candidate("noisier", complexity=1.0, error=1e-6, conversion_cost=1.0),
        _fake_candidate("cleaner", complexity=1.0, error=1e-10, conversion_cost=1.0),
    ]
    assert canonical_representation(tie_on_cost_too).representation_id == "cleaner"


def test_canonical_representation_is_deterministic_on_full_ties():
    # Fully tied candidates must still resolve deterministically (by id).
    candidates = [
        _fake_candidate("z_candidate", complexity=1.0, error=1e-10, conversion_cost=1.0),
        _fake_candidate("a_candidate", complexity=1.0, error=1e-10, conversion_cost=1.0),
    ]
    result = canonical_representation(candidates)
    assert result.representation_id == "a_candidate"


def test_canonical_representation_matches_real_grammar_ranking():
    from rde.representation import rank_representations
    from rde.representation.array_backend import NumpySearchBackend

    backend = NumpySearchBackend()
    n = 6
    nodes = np.arange(n, dtype=float)
    batch = np.stack([2.0 + 0.5 * nodes, -1.0 + 1.5 * nodes])
    ranked = rank_representations(batch, n=n, backend=backend)
    canonical = canonical_representation(ranked)
    assert canonical is not None
    assert canonical.representation_id == "polynomial_vandermonde"
