"""Tests for exhaustive grammar ranking (Phase 3 search)."""

from __future__ import annotations

import numpy as np

from rde.representation import best_representation, rank_representations
from rde.representation.array_backend import NumpySearchBackend
from rde.representation.grammar import primitive_names

BACKEND = NumpySearchBackend()


def test_rank_representations_covers_every_grammar_primitive():
    rng = np.random.default_rng(0)
    batch = rng.normal(size=(4, 7))
    ranked = rank_representations(batch, n=7, backend=BACKEND)
    assert {c.representation_id for c in ranked} == set(primitive_names()) - {"matrix_reshape"}


def test_rank_representations_respects_primitive_subset():
    rng = np.random.default_rng(0)
    batch = rng.normal(size=(4, 7))
    ranked = rank_representations(
        batch, n=7, backend=BACKEND, primitive_subset=["identity", "difference"]
    )
    assert {c.representation_id for c in ranked} == {"identity", "difference"}


def test_best_representation_respects_primitive_subset():
    rng = np.random.default_rng(0)
    batch = rng.normal(size=(4, 7))
    best = best_representation(batch, n=7, backend=BACKEND, primitive_subset=["difference"])
    assert best.representation_id == "difference"


def test_rank_representations_orders_verified_before_refuted():
    rng = np.random.default_rng(0)
    batch = rng.normal(size=(4, 12))
    # tolerance tight enough that some ill-conditioned primitives may refute
    ranked = rank_representations(batch, n=12, backend=BACKEND, tolerance=1e-300)
    statuses = [c.certificate.status == "verified" for c in ranked]
    # once a False appears, no True should follow
    assert statuses == sorted(statuses, reverse=True)


def test_rank_representations_orders_by_ascending_complexity_within_verified_group():
    rng = np.random.default_rng(0)
    batch = rng.normal(size=(4, 8))
    ranked = rank_representations(batch, n=8, backend=BACKEND)
    verified = [c for c in ranked if c.certificate.status == "verified"]
    complexities = [c.complexity for c in verified]
    assert complexities == sorted(complexities)


def test_periodic_signal_favors_dft_over_identity():
    n = 8
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rng = np.random.default_rng(0)
    batch = np.stack([np.sin(3 * t + phase) for phase in rng.normal(size=6)])
    best = best_representation(batch, n=n, backend=BACKEND)
    assert best.representation_id in {"dft", "dft_full"}
    ranked = {c.representation_id: c for c in rank_representations(batch, n=n, backend=BACKEND)}
    assert ranked["dft"].complexity < ranked["identity"].complexity


def test_low_degree_polynomial_signal_favors_polynomial_vandermonde():
    # Exactly (not approximately) low-degree: the Vandermonde matrix at
    # n=6 has cond(V) ~ 5.8e3 (see grammar.py docstring), so any injected
    # per-sample noise gets amplified by ~cond(V) once inverted into
    # coefficient space — a real numerical property of this primitive, not
    # a test artifact. Adding noise here would defeat the sparsity signal
    # this test checks for, not merely make the test "more realistic".
    n = 6
    nodes = np.arange(n, dtype=float)
    slopes = np.array([0.5, -0.3, 1.2, 0.0, 2.0, -1.0])
    intercepts = np.array([2.0, 1.0, -4.0, 0.5, 0.0, 3.0])
    batch = intercepts[:, None] + slopes[:, None] * nodes[None, :]
    best = best_representation(batch, n=n, backend=BACKEND)
    assert best.representation_id == "polynomial_vandermonde"


def test_rank_representations_chain_max_depth_none_matches_flat_grammar_default():
    # Backward-compat: chain_max_depth defaults to None, which must produce
    # exactly the pre-existing flat-grammar behavior -- no program_search
    # import, no composed ids in the result.
    rng = np.random.default_rng(0)
    batch = rng.normal(size=(4, 16))
    flat = rank_representations(batch, n=16, backend=BACKEND)
    explicit_none = rank_representations(batch, n=16, backend=BACKEND, chain_max_depth=None)
    assert {c.representation_id for c in flat} == {c.representation_id for c in explicit_none}
    assert all("+" not in c.representation_id for c in flat)


def test_rank_representations_chain_max_depth_includes_composed_chains():
    rng = np.random.default_rng(1)
    n = 16
    seg1 = np.linspace(0.0, 5.0, 9)
    seg2 = np.linspace(5.2, 20.0, 7)
    x = np.concatenate([seg1, seg2])
    batch = np.stack([x, x.copy()])
    ranked = rank_representations(batch, n=n, backend=BACKEND, chain_max_depth=3)
    ids = {c.representation_id for c in ranked}
    assert "identity" in ids  # depth-1 chains still include the flat grammar
    assert "sorted_permutation+sorted_then_difference+sorted_then_difference" in ids


def test_rank_representations_chain_max_depth_top_candidate_never_worse_than_flat_grammar_best():
    # Depth-3 chains genuinely widen what rank_representations can find:
    # its top candidate with chain_max_depth=3 must never be worse than the
    # flat-grammar (depth-1-only) top candidate over the same data, since
    # every flat-grammar primitive is itself included as a depth-1 chain.
    n = 12
    rng = np.random.default_rng(2)
    seg1 = np.linspace(0.0, 5.0, 7)
    seg2 = np.linspace(5.2, 20.0, 5)
    sorted_vals = np.concatenate([seg1, seg2])
    x = sorted_vals.copy()
    rng.shuffle(x)
    batch = np.stack([x, x.copy()])
    flat_best = best_representation(batch, n=n, backend=BACKEND)
    chain_best = best_representation(batch, n=n, backend=BACKEND, chain_max_depth=3)
    assert chain_best.complexity <= flat_best.complexity

    # And the specific depth-3-only win test_program_search.py locks in for
    # this exact scenario is reachable through rank_representations too, not
    # just through program_search.enumerate_chains directly.
    ranked = {
        c.representation_id: c
        for c in rank_representations(batch, n=n, backend=BACKEND, chain_max_depth=3)
    }
    depth3_id = "sorted_permutation+sorted_then_difference+sorted_then_difference"
    depth2_id = "sorted_permutation+sorted_then_difference"
    assert ranked[depth3_id].complexity < ranked[depth2_id].complexity


def test_rank_representations_chain_max_depth_conversion_cost_is_additive_over_stages():
    from rde.representation.cost import computational_cost

    n = 12
    seg1 = np.linspace(0.0, 5.0, 7)
    seg2 = np.linspace(5.2, 20.0, 5)
    x = np.concatenate([seg1, seg2])
    batch = np.stack([x, x.copy()])
    ranked = {
        c.representation_id: c
        for c in rank_representations(batch, n=n, backend=BACKEND, chain_max_depth=3)
    }
    chain_id = "sorted_permutation+sorted_then_difference+sorted_then_difference"
    assert ranked[chain_id].conversion_cost == computational_cost(chain_id, n)
    assert ranked[chain_id].conversion_cost == computational_cost("sorted_permutation", n) + 2 * computational_cost(
        "sorted_then_difference", n
    )


def test_best_representation_matches_rank_representations_head():
    rng = np.random.default_rng(2)
    batch = rng.normal(size=(3, 6))
    ranked = rank_representations(batch, n=6, backend=BACKEND)
    best = best_representation(batch, n=6, backend=BACKEND)
    assert best.representation_id == ranked[0].representation_id
    assert best.complexity == ranked[0].complexity
