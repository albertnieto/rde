"""Tests for formal (SymPy, exact-rational) verification."""

from __future__ import annotations

import numpy as np
import pytest

from rde.representation import discover_parity_claim, prove_vandermonde_inverse
from rde.representation.grammar import _vandermonde_matrix


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_prove_vandermonde_inverse_proves_for_small_n(n):
    cert = prove_vandermonde_inverse(n)
    assert cert.status == "proved"
    assert cert.representation_id == "polynomial_vandermonde"


def test_prove_vandermonde_inverse_matches_grammar_numeric_matrix():
    """The symbolic proof's matrix must be the same matrix grammar.py uses at runtime."""
    import sympy as sp

    n = 5
    numeric_matrix = _vandermonde_matrix(n)
    nodes = [sp.Integer(i) for i in range(n)]
    symbolic_matrix = sp.Matrix(n, n, lambda i, j: nodes[i] ** j)
    symbolic_as_float = np.array(symbolic_matrix.evalf().tolist(), dtype=float)
    assert np.allclose(symbolic_as_float, numeric_matrix)


def test_discover_parity_claim_proves_even_polynomial():
    # p(x) = 1 + 3x^2 -> even coefficients only, p(x) == p(-x)
    coeffs = np.array([1.0, 0.0, 3.0, 0.0])
    cert = discover_parity_claim(coeffs)
    assert cert.status == "proved"
    assert cert.claim == "p(x) == p(-x)"


def test_discover_parity_claim_disproves_odd_component():
    coeffs = np.array([1.0, 2.0, 3.0, 0.0])
    cert = discover_parity_claim(coeffs)
    assert cert.status == "disproved"
    assert "odd" in cert.detail


def test_discover_parity_claim_screens_near_negligible_odd_coefficients_numerically():
    # Odd coefficients below eps should still pass the numeric screen and be
    # proved as if they were exactly zero (rounded to rational).
    coeffs = np.array([1.0, 1e-10, 3.0, -1e-10])
    cert = discover_parity_claim(coeffs, eps=1e-8)
    assert cert.status == "proved"


def test_discover_parity_claim_respects_stricter_eps():
    coeffs = np.array([1.0, 1e-6, 3.0, 0.0])
    loose = discover_parity_claim(coeffs, eps=1e-5)
    strict = discover_parity_claim(coeffs, eps=1e-8)
    assert loose.status == "proved"
    assert strict.status == "disproved"


def test_formal_certificate_to_payload_round_trips_fields():
    cert = prove_vandermonde_inverse(3)
    payload = cert.to_payload()
    assert payload["representation_id"] == "polynomial_vandermonde"
    assert payload["status"] == "proved"
    assert "n=3" in payload["detail"]
