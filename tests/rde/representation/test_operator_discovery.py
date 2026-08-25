"""Tests for genuine operator discovery from samples (not transport of a known operator).

`discover_linear_operator` never receives the true operator directly in
these tests — only paired `(X, Y)` samples — mirroring how it must be used
for real discovery.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import circulant

from rde.representation import (
    OperatorRecovery,
    discover_and_rank_diagonalization,
    discover_linear_operator,
)
from rde.representation.array_backend import NumpySearchBackend

BACKEND = NumpySearchBackend()


def test_discover_linear_operator_recovers_exact_operator_from_n_samples():
    n = 8
    rng = np.random.default_rng(5)
    U_true = circulant(rng.normal(size=n)).T
    X = rng.normal(size=(n, n))
    Y = X @ U_true.T

    recovery = discover_linear_operator(X, Y)
    assert isinstance(recovery, OperatorRecovery)
    assert recovery.n == n
    assert recovery.n_samples == n
    assert np.max(np.abs(recovery.operator - U_true)) < 1e-8
    assert recovery.residual < 1e-10


def test_discover_linear_operator_recovers_approximately_under_noise():
    n = 6
    rng = np.random.default_rng(9)
    U_true = rng.normal(size=(n, n))
    X = rng.normal(size=(3 * n, n))
    Y = X @ U_true.T + rng.normal(scale=1e-6, size=(3 * n, n))

    recovery = discover_linear_operator(X, Y)
    assert np.max(np.abs(recovery.operator - U_true)) < 1e-4
    assert recovery.residual < 1e-4


def test_discover_linear_operator_rejects_underdetermined_system():
    n = 8
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n - 1, n))  # one sample short of identifiable
    Y = rng.normal(size=(n - 1, n))
    with pytest.raises(ValueError):
        discover_linear_operator(X, Y)


def test_discover_linear_operator_rejects_mismatched_shapes():
    X = np.zeros((5, 4))
    Y = np.zeros((5, 3))
    with pytest.raises(ValueError):
        discover_linear_operator(X, Y)


def test_discover_linear_operator_rejects_non_2d_input():
    with pytest.raises(ValueError):
        discover_linear_operator(np.zeros(4), np.zeros(4))


def test_discover_and_rank_diagonalization_finds_dft_full_without_seeing_true_operator():
    n = 8
    rng = np.random.default_rng(11)
    U_true = circulant(rng.normal(size=n)).T
    X = rng.normal(size=(n, n))
    Y = X @ U_true.T  # the only thing discover_and_rank_diagonalization receives

    recovery, ranking = discover_and_rank_diagonalization(X, Y, backend=BACKEND)
    assert recovery.residual < 1e-8
    assert ranking[0].representation_id == "dft_full"
    assert ranking[0].off_diagonal_energy < 1e-6
    # A non-circulant recovered operator's transport should look far less diagonal.
    assert ranking[-1].off_diagonal_energy > ranking[0].off_diagonal_energy


def test_discover_and_rank_diagonalization_on_non_circulant_operator_does_not_favor_dft():
    n = 6
    rng = np.random.default_rng(13)
    # A generic dense random operator has no reason to be diagonalized by
    # any of the grammar's primitives — dft_full should not automatically win.
    U_true = rng.normal(size=(n, n))
    X = rng.normal(size=(n, n))
    Y = X @ U_true.T

    recovery, ranking = discover_and_rank_diagonalization(X, Y, backend=BACKEND)
    assert recovery.residual < 1e-8
    # No primitive should achieve near-perfect diagonalization of a generic operator.
    assert ranking[0].off_diagonal_energy > 0.5
