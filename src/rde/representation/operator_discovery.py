"""Operator discovery from samples — recovering `U`, not transporting a known one.

Gap closure: `operator.py`'s `transport_operator`/`rank_by_diagonalization`
assume `U` is already a known `(n, n)` matrix — that is operator
*transport*. This module closes the harder gap: given only paired samples
`(X, Y)` with `Y[i] ~= U @ X[i]` for an unknown linear `U`, recover `U`
itself via least squares (`discover_linear_operator`), then hand the
*recovered* (not the true) operator to `rank_by_diagonalization` to find
which representation compresses/diagonalizes it
(`discover_and_rank_diagonalization`). The discovery pipeline never
receives `U` directly, only the samples — verified before this module was
written: recovering a random circulant `U` (`n=8`) from exactly `n`
noiseless samples reproduces it to `~1e-15`, and from `3n` samples with
`1e-6` sample noise reproduces it to `~1e-6` (see
`tests/rde/representation/test_operator_discovery.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rde.representation.array_backend import ArraySearchBackend
from rde.representation.operator import DiagonalizationCandidate, rank_by_diagonalization


@dataclass(frozen=True)
class OperatorRecovery:
    """A linear operator recovered from paired samples, with its own fit evidence."""

    operator: np.ndarray
    residual: float
    rank: int
    n_samples: int
    n: int


def discover_linear_operator(X: np.ndarray, Y: np.ndarray) -> OperatorRecovery:
    """Recover the `(n, n)` linear operator `U` such that `Y[i] ~= U @ X[i]`, from samples.

    Least squares (`np.linalg.lstsq`, solved as `X @ U.T ~= Y`): exact when
    `X` has at least `n` independent rows and the data is noiseless; a
    best-fit approximation otherwise. `residual` is the relative Frobenius
    reconstruction error `||X @ U.T - Y|| / ||Y||` — the fit's own honest
    evidence, not a claim that `U` is exactly recovered.

    Raises `ValueError` with fewer than `n` samples: an `(n, n)` operator
    has `n^2` free parameters and each sample pair contributes only `n`
    equations, so fewer than `n` samples leave `U` underdetermined
    (infinitely many operators fit the data) — this is a real
    identifiability limit, not an implementation choice to relax.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have the same shape; got {X.shape} and {Y.shape}")
    if X.ndim != 2:
        raise ValueError(f"X, Y must be 2-D (B, n); got shape {X.shape}")
    b, n = X.shape
    if b < n:
        raise ValueError(
            f"discover_linear_operator needs at least n={n} independent samples to "
            f"uniquely determine an (n, n) operator ({n * n} free parameters, n equations "
            f"per sample); got only {b}"
        )

    u_transpose, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
    reconstructed = X @ u_transpose
    denominator = float(np.linalg.norm(Y))
    numerator = float(np.linalg.norm(reconstructed - Y))
    residual = numerator / denominator if denominator > 0 else numerator

    return OperatorRecovery(
        operator=u_transpose.T,
        residual=residual,
        rank=int(rank),
        n_samples=b,
        n=n,
    )


def discover_and_rank_diagonalization(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    backend: ArraySearchBackend | None = None,
    object_type: str | None = None,
) -> tuple[OperatorRecovery, list[DiagonalizationCandidate]]:
    """Recover `U` from samples, then rank the grammar by how well it diagonalizes `U`.

    Composes `discover_linear_operator` with `operator.rank_by_diagonalization`
    on the *recovered* operator — the ranking is only as trustworthy as
    `OperatorRecovery.residual` says the recovery itself was; a caller
    should check `residual` before trusting the ranking.
    """
    recovery = discover_linear_operator(X, Y)
    ranking = rank_by_diagonalization(
        recovery.operator, n=recovery.n, backend=backend, object_type=object_type
    )
    return recovery, ranking
