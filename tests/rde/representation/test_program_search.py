"""Tests for exhaustive typed-chain program search (`program_search.py`)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from rde.representation.array_backend import NumpySearchBackend
from rde.representation.program_search import (
    ChainSearchResult,
    atomic_registry,
    enumerate_chains,
    search_chains,
)


@pytest.fixture
def backend() -> NumpySearchBackend:
    return NumpySearchBackend()


def test_atomic_registry_includes_grammar_and_layered_stage2_primitives(backend):
    registry = atomic_registry(8, backend=backend)
    # grammar.py primitives
    assert "identity" in registry
    assert "difference" in registry
    assert "dft_full" in registry
    # layered.py stage-2 primitives, present standalone (uncomposed)
    assert "sort_by_magnitude" in registry
    assert "sorted_then_difference" in registry


def test_enumerate_chains_depth_1_matches_atomic_real_vector_starts(backend):
    chains = enumerate_chains(8, max_depth=1, backend=backend)
    registry = atomic_registry(8, backend=backend)
    expected_ids = {rid for rid, rep in registry.items() if rep.input_carrier_kind == "real_vector"}
    assert set(chains.keys()) == expected_ids


def test_enumerate_chains_respects_max_depth(backend):
    chains = enumerate_chains(8, max_depth=2, backend=backend)
    for representation_id in chains:
        assert representation_id.count("+") + 1 <= 2


def test_enumerate_chains_allows_self_chaining_sorted_then_difference(backend):
    # sorted_then_difference's carrier_kind == its own input_carrier_kind
    # ("sorted_pair"), so it is a legitimate depth-3 continuation of itself.
    chains = enumerate_chains(12, max_depth=3, backend=backend)
    assert "sorted_permutation+sorted_then_difference+sorted_then_difference" in chains


def test_enumerate_chains_never_composes_incompatible_carrier_kinds(backend):
    # row_dft's carrier_kind ("complex_matrix") has no registered stage-2
    # continuation; it must never appear as a non-terminal chain stage.
    chains = enumerate_chains(9, max_depth=3, backend=backend)
    for representation_id in chains:
        stages = representation_id.split("+")
        assert "row_dft" not in stages[:-1]


def test_enumerate_chains_allows_matrix_reshape_to_continue_into_row_dft(backend):
    # matrix_reshape's carrier_kind ("matrix") now has a registered stage-2
    # continuation (row_dft) -- unlike the still-terminal carrier kinds
    # checked above, matrix_reshape may legitimately appear as a
    # non-terminal chain stage.
    chains = enumerate_chains(16, max_depth=2, backend=backend)
    assert "matrix_reshape+row_dft" in chains


def test_enumerate_chains_allows_self_chaining_sorted_complex_then_difference(backend):
    # sorted_complex_then_difference's carrier_kind == its own
    # input_carrier_kind ("sorted_complex_pair"), same self-chaining
    # property sorted_then_difference has for "sorted_pair".
    chains = enumerate_chains(16, max_depth=4, backend=backend)
    assert "dft_full+sort_by_magnitude+sorted_complex_then_difference" in chains
    assert (
        "dft_full+sort_by_magnitude+sorted_complex_then_difference+sorted_complex_then_difference"
        in chains
    )


def test_enumerate_chains_produces_no_complex_cast_warnings(backend):
    # Regression test: dft_full's decode (ifft) always returns complex
    # dtype; a chain stage decoding *through* it (e.g. difference's cumsum)
    # must not silently warn-and-truncate.
    chains = enumerate_chains(16, max_depth=4, backend=backend)
    batch = np.stack([np.linspace(0.0, 1.0, 16), np.linspace(-2.0, 3.0, 16)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for representation in chains.values():
            representation.decode(representation.encode(batch))


def test_depth_3_self_chained_difference_beats_depth_2_on_piecewise_linear_sorted_data(backend):
    # Verified numerically before being written here: second-order
    # differencing of already-sorted, piecewise-linear data compresses
    # further than first-order differencing alone.
    n = 12
    chains = enumerate_chains(n, max_depth=3, backend=backend)
    seg1 = np.linspace(0.0, 5.0, 7)
    seg2 = np.linspace(5.2, 20.0, 5)
    sorted_vals = np.concatenate([seg1, seg2])
    rng = np.random.default_rng(2)
    x = sorted_vals.copy()
    rng.shuffle(x)
    batch = np.stack([x, x.copy()])

    depth2 = chains["sorted_permutation+sorted_then_difference"]
    depth3 = chains["sorted_permutation+sorted_then_difference+sorted_then_difference"]

    depth2_complexity = depth2.complexity(depth2.encode(batch))
    depth3_complexity = depth3.complexity(depth3.encode(batch))
    assert depth3_complexity < depth2_complexity

    np.testing.assert_allclose(depth3.decode(depth3.encode(batch)), batch, atol=1e-9)


def test_depth_3_self_chained_difference_beats_identity_when_permutation_is_cheap(backend):
    # Verified numerically before being written here: once permutation
    # storage is charged by actual structure (`permutation_complexity`)
    # rather than a flat ~n regardless of content, a sort permutation that
    # is already near-identity (data collected in roughly sorted order)
    # costs close to nothing, letting the depth-3 self-chained difference
    # beat the permutation-free `identity` baseline outright -- not just
    # its own depth-2 base (already covered by the test above).
    n = 12
    chains = enumerate_chains(n, max_depth=3, backend=backend)
    seg1 = np.linspace(0.0, 5.0, 7)
    seg2 = np.linspace(5.2, 20.0, 5)
    x = np.concatenate([seg1, seg2])  # already ascending -> sort permutation ~ identity
    batch = np.stack([x, x.copy()])

    identity = chains["identity"]
    depth3 = chains["sorted_permutation+sorted_then_difference+sorted_then_difference"]
    identity_complexity = identity.complexity(identity.encode(batch))
    depth3_complexity = depth3.complexity(depth3.encode(batch))
    assert depth3_complexity < identity_complexity


def test_sorted_complex_then_difference_beats_depth_2_on_conjugate_paired_spectrum(backend):
    # Verified numerically before being written here: zero-phase cosine
    # components make each DFT conjugate-pair bin X[k]/X[n-k] purely real
    # and numerically identical (X[n-k] = conj(X[k]) = X[k] when X[k] has
    # no imaginary part) -- sort_by_magnitude places those equal-valued
    # bins adjacent, so differencing them is exactly zero. A random-phase
    # variant of the same frequencies does not produce this win (the
    # conjugate pairs are no longer numerically equal), confirming this is
    # a real structural property of the data, not an artifact of the metric.
    n = 16
    rng = np.random.default_rng(6)
    t = np.arange(n)
    freqs = [1, 2, 3, 4, 5]
    batch = np.stack(
        [sum(3.0 * np.cos(2 * np.pi * f * t / n) for f in freqs) for _ in range(4)]
    )

    chains = enumerate_chains(n, max_depth=4, backend=backend)
    depth2 = chains["dft_full+sort_by_magnitude"]
    depth3 = chains["dft_full+sort_by_magnitude+sorted_complex_then_difference"]

    depth2_complexity = depth2.complexity(depth2.encode(batch))
    depth3_complexity = depth3.complexity(depth3.encode(batch))
    assert depth3_complexity < depth2_complexity

    reconstructed = depth3.decode(depth3.encode(batch))
    np.testing.assert_allclose(np.real(reconstructed), batch, atol=1e-9)


def _piecewise_linear_batch(rng: np.random.Generator, n: int, count: int) -> np.ndarray:
    rows = []
    for _ in range(count):
        knot = rng.integers(2, n - 2)
        left = np.linspace(0.0, rng.uniform(1.0, 5.0), knot)
        right = np.linspace(left[-1], left[-1] + rng.uniform(1.0, 5.0), n - knot)
        rows.append(np.concatenate([left, right]))
    return np.stack(rows)


def test_search_chains_returns_results_sorted_by_holdout_complexity(backend):
    rng = np.random.default_rng(0)
    n = 10
    train = _piecewise_linear_batch(rng, n, 6)
    holdout = _piecewise_linear_batch(rng, n, 6)

    results = search_chains(train, holdout, n=n, max_depth=3, backend=backend)
    assert results
    assert all(isinstance(r, ChainSearchResult) for r in results)
    complexities = [r.holdout_complexity for r in results]
    assert complexities == sorted(complexities)


def test_search_chains_drops_chains_that_fail_to_verify_on_either_batch(backend):
    rng = np.random.default_rng(0)
    n = 10
    train = _piecewise_linear_batch(rng, n, 6)
    holdout = _piecewise_linear_batch(rng, n, 6)

    results = search_chains(train, holdout, n=n, max_depth=3, backend=backend)
    for r in results:
        assert r.train_certificate.status == "verified"
        assert r.holdout_certificate.status == "verified"


def test_search_chains_generalization_ratio_is_holdout_over_train(backend):
    rng = np.random.default_rng(0)
    n = 10
    train = _piecewise_linear_batch(rng, n, 6)
    holdout = _piecewise_linear_batch(rng, n, 6)

    results = search_chains(train, holdout, n=n, max_depth=3, backend=backend)
    for r in results:
        if r.train_complexity > 0:
            assert r.generalization_ratio == pytest.approx(r.holdout_complexity / r.train_complexity)
        else:
            expected = 1.0 if r.holdout_complexity == 0 else float("inf")
            assert r.generalization_ratio == expected


def test_search_chains_requires_holdout_batch_argument(backend):
    # holdout_batch is positional/required, not an optional keyword with a
    # silent default of "reuse train" — that would defeat the check this
    # function exists for.
    import inspect

    sig = inspect.signature(search_chains)
    assert "holdout_batch" in sig.parameters
    assert sig.parameters["holdout_batch"].default is inspect.Parameter.empty


def test_search_chains_ranks_by_holdout_not_train_complexity(backend):
    # A chain that compresses well on train but not on an independent
    # holdout batch must not be ranked ahead of one that generalizes,
    # purely because search_chains sorts by holdout_complexity.
    rng = np.random.default_rng(3)
    n = 10
    train = _piecewise_linear_batch(rng, n, 6)
    holdout = _piecewise_linear_batch(rng, n, 6)
    results = search_chains(train, holdout, n=n, max_depth=3, backend=backend)
    if len(results) >= 2:
        for a, b in zip(results, results[1:]):
            assert a.holdout_complexity <= b.holdout_complexity
