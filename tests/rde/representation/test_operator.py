"""Tests for operator transport across representations (Phase 6).

The headline result: the full complex DFT (`dft_full`) diagonalizes any
circulant operator (circulant matrices share the DFT eigenbasis — a
textbook fact, verified numerically here, not asserted).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import circulant

from rde.representation import (
    linear_probe_matrices,
    off_diagonal_energy,
    probe_encode_matrix,
    rank_by_diagonalization,
    transport_operator,
)
from rde.representation.array_backend import NumpySearchBackend
from rde.representation.grammar import build_primitive_representations

BACKEND = NumpySearchBackend()


def test_probe_encode_matrix_matches_direct_encode_for_non_square_compact_dft():
    # dft (rfft) is non-square: linear_probe_matrices rejects it (decode-side
    # probing is unsound), but encode-only probing must still be sound.
    n = 8
    grammar = build_primitive_representations(n, backend=BACKEND)
    rep = grammar["dft"]
    encode_matrix = probe_encode_matrix(rep, n)
    assert encode_matrix.shape == (n // 2 + 1, n)
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    assert np.allclose(encode_matrix @ x, rep.encode(x[None, :])[0])


def test_probe_encode_matrix_rejects_nonlinear_representation():
    n = 5
    grammar = build_primitive_representations(n, backend=BACKEND)
    with pytest.raises(ValueError):
        probe_encode_matrix(grammar["sorted_permutation"], n)


def test_linear_probe_matrices_rejects_non_square_compact_dft():
    n = 8
    grammar = build_primitive_representations(n, backend=BACKEND)
    with pytest.raises(ValueError):
        linear_probe_matrices(grammar["dft"], n)


def test_rank_by_diagonalization_excludes_compact_dft_but_keeps_dft_full():
    # dft's decode-side probing is unsound (see linear_probe_matrices'
    # docstring); rank_by_diagonalization must not silently include it.
    n = 8
    rng = np.random.default_rng(7)
    U = circulant(rng.normal(size=n)).T
    ranked = rank_by_diagonalization(U, n=n, backend=BACKEND)
    ids = {c.representation_id for c in ranked}
    assert "dft" not in ids
    assert "dft_full" in ids


def test_linear_probe_matrices_identity_representation_gives_identity_matrices():
    n = 5
    grammar = build_primitive_representations(n, backend=BACKEND)
    encode_matrix, decode_matrix = linear_probe_matrices(grammar["identity"], n)
    assert np.allclose(encode_matrix, np.eye(n))
    assert np.allclose(decode_matrix, np.eye(n))


def test_linear_probe_matrices_matches_direct_encode_decode():
    n = 6
    grammar = build_primitive_representations(n, backend=BACKEND)
    rep = grammar["difference"]
    encode_matrix, decode_matrix = linear_probe_matrices(rep, n)
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    assert np.allclose(encode_matrix @ x, rep.encode(x[None, :])[0])
    assert np.allclose(decode_matrix @ (encode_matrix @ x), rep.decode(rep.encode(x[None, :]))[0])


def test_linear_probe_matrices_rejects_nonlinear_representation():
    n = 5
    grammar = build_primitive_representations(n, backend=BACKEND)
    with pytest.raises(ValueError):
        linear_probe_matrices(grammar["sorted_permutation"], n)


def test_linear_probe_matrices_rejects_non_flat_carrier():
    n = 9
    grammar = build_primitive_representations(n, backend=BACKEND)
    with pytest.raises(ValueError):
        linear_probe_matrices(grammar["matrix_reshape"], n)


def test_transport_operator_identity_matrices_is_a_no_op():
    n = 4
    rng = np.random.default_rng(0)
    U = rng.normal(size=(n, n))
    transported = transport_operator(U, np.eye(n), np.eye(n))
    assert np.allclose(transported, U)


def test_transport_operator_supports_batched_operators():
    n = 4
    rng = np.random.default_rng(0)
    U_batch = rng.normal(size=(3, n, n))
    encode_matrix = rng.normal(size=(n, n))
    decode_matrix = np.linalg.inv(encode_matrix)
    transported = transport_operator(U_batch, encode_matrix, decode_matrix)
    assert transported.shape == (3, n, n)
    for i in range(3):
        assert np.allclose(transported[i], encode_matrix @ U_batch[i] @ decode_matrix)


def test_off_diagonal_energy_zero_for_diagonal_matrix():
    matrix = np.diag([1.0, 2.0, 3.0])
    assert off_diagonal_energy(matrix) == pytest.approx(0.0)


def test_off_diagonal_energy_positive_for_dense_matrix():
    matrix = np.array([[1.0, 1.0], [1.0, 1.0]])
    energy = off_diagonal_energy(matrix)
    assert energy > 0.5


def test_off_diagonal_energy_supports_batched_input():
    matrices = np.stack([np.diag([1.0, 2.0]), np.array([[0.0, 1.0], [1.0, 0.0]])])
    energies = off_diagonal_energy(matrices)
    assert energies.shape == (2,)
    assert energies[0] == pytest.approx(0.0)
    assert energies[1] == pytest.approx(1.0)


def test_off_diagonal_energy_handles_zero_matrix_without_division_error():
    matrix = np.zeros((3, 3))
    assert off_diagonal_energy(matrix) == pytest.approx(0.0)


def test_dft_full_diagonalizes_circulant_operator():
    n = 8
    rng = np.random.default_rng(7)
    first_row = rng.normal(size=n)
    U = circulant(first_row).T
    grammar = build_primitive_representations(n, backend=BACKEND)
    encode_matrix, decode_matrix = linear_probe_matrices(grammar["dft_full"], n)
    transported = transport_operator(U, encode_matrix, decode_matrix)
    assert off_diagonal_energy(transported) < 1e-8


def test_identity_representation_does_not_diagonalize_circulant_operator():
    n = 8
    rng = np.random.default_rng(7)
    first_row = rng.normal(size=n)
    U = circulant(first_row).T
    grammar = build_primitive_representations(n, backend=BACKEND)
    encode_matrix, decode_matrix = linear_probe_matrices(grammar["identity"], n)
    transported = transport_operator(U, encode_matrix, decode_matrix)
    assert off_diagonal_energy(transported) > 0.9


def test_rank_by_diagonalization_puts_dft_variants_first_for_circulant_operator():
    n = 8
    rng = np.random.default_rng(7)
    first_row = rng.normal(size=n)
    U = circulant(first_row).T
    ranked = rank_by_diagonalization(U, n=n, backend=BACKEND)
    assert ranked[0].representation_id in {"dft", "dft_full"}
    assert ranked[0].off_diagonal_energy < 1e-6
    assert ranked[-1].off_diagonal_energy > ranked[0].off_diagonal_energy


def test_rank_by_diagonalization_excludes_nonlinear_and_non_flat_primitives():
    n = 9
    rng = np.random.default_rng(0)
    U = rng.normal(size=(n, n))
    ranked = rank_by_diagonalization(U, n=n, backend=BACKEND)
    ids = {c.representation_id for c in ranked}
    assert "sorted_permutation" not in ids
    assert "matrix_reshape" not in ids
