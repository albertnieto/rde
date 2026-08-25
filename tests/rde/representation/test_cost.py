"""Tests for the asymptotic conversion-cost model."""

from __future__ import annotations

import pytest

from rde.representation import computational_cost
from rde.representation.grammar import primitive_names


@pytest.mark.parametrize("representation_id", primitive_names())
def test_computational_cost_is_defined_for_every_grammar_primitive(representation_id):
    cost = computational_cost(representation_id, 8)
    assert cost >= 0.0


def test_identity_and_matrix_reshape_are_free():
    assert computational_cost("identity", 16) == 0.0
    assert computational_cost("matrix_reshape", 16) == 0.0


def test_polynomial_vandermonde_scales_quadratically():
    small = computational_cost("polynomial_vandermonde", 4)
    large = computational_cost("polynomial_vandermonde", 8)
    # doubling n should ~quadruple a quadratic cost
    assert large / small == pytest.approx(4.0, rel=0.05)


def test_dft_scales_like_n_log_n_not_quadratically():
    small = computational_cost("dft", 8)
    large = computational_cost("dft", 64)
    # n log2(n): 8*3=24 -> 64*6=384, ratio 16, far below the n^2 ratio of 64
    assert large / small == pytest.approx(16.0, rel=0.05)
    assert large / small < 64.0


def test_dft_full_costs_more_than_compact_dft():
    n = 16
    assert computational_cost("dft_full", n) > computational_cost("dft", n)


def test_dct_scales_quadratically_like_polynomial_vandermonde():
    # dct is implemented as a literal dense-matrix matmul (see cost.py's
    # docstring), the same algorithm shape as polynomial_vandermonde --
    # not the O(n log n) fast-DCT algorithm.
    small = computational_cost("dct", 4)
    large = computational_cost("dct", 8)
    assert large / small == pytest.approx(4.0, rel=0.05)
    assert computational_cost("dct", 16) == pytest.approx(computational_cost("polynomial_vandermonde", 16))


def test_polynomial_eventually_more_expensive_than_dft_at_large_n():
    # Quadratic beats n log n asymptotically; find where the crossover holds.
    n = 4096
    assert computational_cost("polynomial_vandermonde", n) > computational_cost("dft", n)


def test_computational_cost_rejects_unknown_representation_id():
    with pytest.raises(KeyError):
        computational_cost("not_a_real_primitive", 8)


def test_computational_cost_rejects_non_positive_n():
    with pytest.raises(ValueError):
        computational_cost("identity", 0)
    with pytest.raises(ValueError):
        computational_cost("identity", -3)


def test_computational_cost_handles_n_equals_one():
    # log2(1) == 0 would zero out every cost that depends on log2(n);
    # the n<=1 branch guards against that degenerate case.
    assert computational_cost("dft", 1) >= 0.0
    assert computational_cost("sorted_permutation", 1) >= 0.0
