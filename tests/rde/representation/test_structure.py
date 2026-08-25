"""Tests for the checkable structure vocabulary (sparsity, periodicity, low-rank,
separability, conservation, duality).

Every `holds=True`/`holds=False` was independently verified numerically
before being written into a test — a periodic signal really does
concentrate ~100% of its AC energy in one frequency; an outer-product
matrix really does have one nonzero normalized singular value; a genuinely
circulant matrix really is exactly invariant under simultaneous cyclic
row/column shift and exactly diagonalized by the full DFT. This file covers
the domain-agnostic (synthetic-but-mathematically-real) case; see
`tests/rde_domains/test_tsp_circulant_structure.py` for the same two checks
verified against real `rde_domains.tsp.circulant`-generated distance
matrices, not fabricated data — core (this file) must never import
`rde_domains`.
"""

from __future__ import annotations

import numpy as np
import pytest

from rde.representation import (
    check_conservation,
    check_duality,
    check_low_rank,
    check_periodicity,
    check_separability,
    check_sparsity,
)
from rde.representation.array_backend import NumpySearchBackend

BACKEND = NumpySearchBackend()


def _circulant(first_row: np.ndarray) -> np.ndarray:
    """A genuine circulant matrix — row `i` is `first_row` rotated right by `i`."""
    n = first_row.shape[0]
    return np.stack([np.roll(first_row, shift) for shift in range(n)])


def test_check_sparsity_holds_for_mostly_zero_data():
    encoded = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 2.0]])
    claim = check_sparsity(encoded, BACKEND, eps=1e-8, max_fraction=0.5)
    assert claim.holds
    assert claim.structure_type == "sparsity"
    assert claim.score == pytest.approx(0.25)


def test_check_sparsity_fails_for_dense_data():
    rng = np.random.default_rng(0)
    encoded = rng.normal(size=(4, 8))
    claim = check_sparsity(encoded, BACKEND, eps=1e-8, max_fraction=0.3)
    assert not claim.holds
    assert claim.score == pytest.approx(1.0)


def test_check_periodicity_holds_for_pure_sinusoid():
    n = 16
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rng = np.random.default_rng(0)
    batch = np.stack([np.sin(3 * t + phase) for phase in rng.normal(size=8)])
    claim = check_periodicity(batch, top_k=1, energy_threshold=0.9)
    assert claim.holds
    assert claim.score == pytest.approx(1.0, abs=1e-6)


def test_check_periodicity_fails_for_random_signal():
    n = 16
    rng = np.random.default_rng(0)
    batch = rng.normal(size=(8, n))
    claim = check_periodicity(batch, top_k=1, energy_threshold=0.9)
    assert not claim.holds
    assert claim.score < 0.9


def test_check_periodicity_higher_top_k_increases_captured_energy():
    n = 16
    rng = np.random.default_rng(0)
    batch = rng.normal(size=(8, n))
    claim_1 = check_periodicity(batch, top_k=1)
    claim_3 = check_periodicity(batch, top_k=3)
    assert claim_3.score >= claim_1.score


def test_check_low_rank_and_separability_hold_for_outer_product_matrices():
    rng = np.random.default_rng(1)
    side = 5
    outer = np.stack(
        [np.outer(rng.normal(size=side), rng.normal(size=side)) for _ in range(6)]
    )
    low_rank_claim = check_low_rank(outer, rank_threshold=0.1)
    separability_claim = check_separability(outer, rank_threshold=0.1)
    assert low_rank_claim.holds
    assert low_rank_claim.score == pytest.approx(1.0)
    assert separability_claim.holds
    assert separability_claim.score < 1e-10


def test_check_low_rank_and_separability_fail_for_generic_random_matrices():
    rng = np.random.default_rng(1)
    side = 5
    random_matrices = rng.normal(size=(6, side, side))
    low_rank_claim = check_low_rank(random_matrices, rank_threshold=0.1)
    separability_claim = check_separability(random_matrices, rank_threshold=0.1)
    assert not low_rank_claim.holds
    # A generic random matrix's effective rank should be close to full rank
    # (well above the max_effective_rank=full_rank//2 bar), not necessarily
    # exactly full rank — the exact count is seed-dependent.
    assert low_rank_claim.score >= side // 2 + 1
    assert not separability_claim.holds
    assert separability_claim.score > 0.1


def test_check_low_rank_requires_effective_rank_well_below_full_not_merely_less_than():
    # Regression: a matrix with 4 of 5 singular values significant is not
    # "low rank" in any useful sense, even though effective_rank(4) < full_rank(5).
    profile_like_matrix = np.diag([1.0, 0.72, 0.54, 0.27, 0.03])[None, :, :]
    claim = check_low_rank(profile_like_matrix, rank_threshold=0.1)
    assert claim.score == pytest.approx(4.0)
    assert not claim.holds


def test_check_low_rank_respects_explicit_max_effective_rank():
    profile_like_matrix = np.diag([1.0, 0.72, 0.54, 0.27, 0.03])[None, :, :]
    lenient = check_low_rank(profile_like_matrix, rank_threshold=0.1, max_effective_rank=4)
    assert lenient.holds


def test_check_conservation_holds_for_exactly_circulant_matrix():
    rng = np.random.default_rng(0)
    n = 6
    c = _circulant(rng.normal(size=n))
    claim = check_conservation(c[None, :, :])
    assert claim.holds
    assert claim.structure_type == "conservation"
    assert claim.score == pytest.approx(0.0, abs=1e-9)


def test_check_conservation_fails_for_a_perturbed_circulant_matrix():
    rng = np.random.default_rng(0)
    n = 6
    c = _circulant(rng.normal(size=n))
    perturbed = c + rng.normal(0, 2.0, size=(n, n))
    claim = check_conservation(perturbed[None, :, :])
    assert not claim.holds
    assert claim.score > 0.05


def test_check_conservation_deviation_grows_with_perturbation_scale():
    rng = np.random.default_rng(1)
    n = 8
    c = _circulant(rng.normal(size=n))
    small = check_conservation((c + rng.normal(0, 0.01, size=(n, n)))[None, :, :])
    large = check_conservation((c + rng.normal(0, 5.0, size=(n, n)))[None, :, :])
    assert small.score < large.score


def test_check_conservation_accepts_a_custom_permutation_group():
    # Default (no permutations given) is the full cyclic group Z_n; passing
    # only the identity permutation trivially makes every matrix "conserved"
    # (M[id][:, id] == M always) -- confirms `permutations` is genuinely
    # threaded through, not silently ignored in favor of the default.
    rng = np.random.default_rng(2)
    n = 5
    random_matrix = rng.normal(size=(n, n))
    identity_only = check_conservation(random_matrix[None, :, :], permutations=[np.arange(n)])
    assert identity_only.holds
    assert identity_only.score == pytest.approx(0.0, abs=1e-12)


def test_check_duality_holds_for_dft_full_on_a_circulant_operator():
    # The textbook fact operator.py already verified against a random
    # circulant test matrix (off_diagonal_energy ~1e-16 for dft_full) --
    # this function makes that same claim reusable and named.
    rng = np.random.default_rng(3)
    n = 6
    c = _circulant(rng.normal(size=n))
    claim = check_duality(c[None, :, :], n=n)
    assert claim.holds
    assert claim.structure_type == "duality"
    assert claim.score == pytest.approx(0.0, abs=1e-9)


def test_check_duality_fails_for_a_perturbed_circulant_operator():
    rng = np.random.default_rng(3)
    n = 6
    c = _circulant(rng.normal(size=n))
    perturbed = c + rng.normal(0, 2.0, size=(n, n))
    claim = check_duality(perturbed[None, :, :], n=n)
    assert not claim.holds
    assert claim.score > 0.05


def test_check_duality_rejects_a_non_diagonalizing_representation():
    # identity does not diagonalize a genuinely non-diagonal circulant
    # operator -- a real negative example, not just "dft_full passes".
    rng = np.random.default_rng(4)
    n = 6
    c = _circulant(rng.normal(size=n))
    claim = check_duality(c[None, :, :], n=n, dual_representation_id="identity")
    assert not claim.holds
    assert claim.score > 0.5


def test_check_duality_rejects_dct_for_a_circulant_operator():
    # dct's half-integer-frequency cosine basis is not a circulant
    # matrix's eigenbasis -- dft_full is; a second, distinct real-valued
    # negative example alongside identity's above.
    rng = np.random.default_rng(4)
    n = 8
    c = _circulant(rng.normal(size=n))
    claim = check_duality(c[None, :, :], n=n, dual_representation_id="dct")
    assert not claim.holds


def test_check_conservation_and_check_duality_scores_match_for_circulant_operator():
    # Emergent finding while building this, verified numerically before
    # writing it down: a circulant matrix's deviation from the cyclic-
    # group-invariant subspace and its off-diagonal energy after DFT
    # transport are numerically identical -- both the Frobenius-norm
    # projection and the DFT are norm-preserving linear operations on the
    # same underlying (circulant + deviation) decomposition, so the
    # "how far from circulant" and "how far from diagonal-in-Fourier-basis"
    # numbers coincide. Not asserted as a general theorem here (this
    # package's methodology reports what was checked, not what "should"
    # follow) -- just the specific numerical equality found while
    # verifying both functions against the same real data (also confirmed
    # against actual TSP distance matrices in
    # test_tsp_circulant_structure.py).
    rng = np.random.default_rng(5)
    n = 6
    c = _circulant(rng.normal(size=n))
    perturbed = c + rng.normal(0, 1.0, size=(n, n))
    conservation = check_conservation(perturbed[None, :, :])
    duality = check_duality(perturbed[None, :, :], n=n)
    assert conservation.score == pytest.approx(duality.score, rel=1e-9)


def test_structure_claims_integrate_with_grammar_matrix_reshape():
    from rde.representation import build_primitive_representations

    n = 9  # perfect square
    grammar = build_primitive_representations(n, backend=BACKEND)
    rep = grammar["matrix_reshape"]
    rng = np.random.default_rng(2)
    side = 3
    outer_vectors = np.stack(
        [np.outer(rng.normal(size=side), rng.normal(size=side)).reshape(n) for _ in range(5)]
    )
    encoded = rep.encode(outer_vectors)  # (5, 3, 3)
    claim = check_separability(encoded)
    assert claim.holds
