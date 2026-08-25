"""Formal (SymPy, exact-rational) verification — distinct from numeric certification.

Phase 4 of the representation-discovery roadmap. `certificate.py`'s
`Certificate.status in {"verified", "refuted"}` is numerical evidence: a
floating-point roundtrip within tolerance. This module never reuses that
vocabulary — `FormalCertificate.status in {"proved", "disproved"}` — because
conflating "observed numerically" with "proved" is exactly the overclaim
this project's methodology forbids (see `docs/ARCHITECTURE.md` /
`docs/methodology.md`).

Symbolic construction here (building a SymPy polynomial term by term,
inverting a small exact-rational matrix) is inherently per-term work that
NumPy/MLX cannot vectorize — there is no batched-array equivalent of
"construct a symbolic expression." That is unrelated to, and does not
excuse, looping over batch/data dimensions elsewhere in this package; those
paths (`array_backend.py`, `grammar.py`) stay fully vectorized. Anything
that *can* be checked numerically first (e.g. whether coefficients look
negligible) is vectorized with NumPy before any symbolic work happens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp


@dataclass(frozen=True)
class FormalCertificate:
    """A machine-checked exact-rational proof (or disproof) of one claim."""

    representation_id: str
    claim: str
    status: str
    detail: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "representation_id": self.representation_id,
            "claim": self.claim,
            "status": self.status,
            "detail": self.detail,
        }


def prove_vandermonde_inverse(n: int) -> FormalCertificate:
    """Prove `V @ V_inv == I == V_inv @ V` exactly, for the degree-`(n-1)`

    Vandermonde matrix over integer nodes `0..n-1` (increasing powers, same
    convention as `grammar._vandermonde_matrix` / `np.vander(..., increasing=True)`).
    Built and inverted independently in SymPy's exact rational arithmetic —
    this does not reuse or trust the NumPy/LAPACK inverse `grammar.py` uses
    at runtime; it is an independent proof that primitive is a true bijection.
    """
    nodes = [sp.Integer(i) for i in range(n)]
    vandermonde = sp.Matrix(n, n, lambda i, j: nodes[i] ** j)
    inverse = vandermonde.inv()
    right_residual = sp.expand(vandermonde * inverse - sp.eye(n))
    left_residual = sp.expand(inverse * vandermonde - sp.eye(n))
    proved = right_residual.is_zero_matrix and left_residual.is_zero_matrix
    return FormalCertificate(
        representation_id="polynomial_vandermonde",
        claim="V @ V_inv == I == V_inv @ V",
        status="proved" if proved else "disproved",
        detail=f"n={n}, exact rational arithmetic, residual is_zero_matrix={proved}",
    )


def discover_parity_claim(coeffs: np.ndarray, *, eps: float = 1e-8) -> FormalCertificate:
    """Check `p(x) == p(-x)` for one row of `polynomial_vandermonde` coefficients.

    Coefficient `k` is the degree-`k` term (increasing powers). First screens
    odd-degree coefficients against `eps` with a single vectorized NumPy
    comparison; only if that numeric screen passes does it round those
    coefficients to rationals and prove the identity symbolically by exact
    polynomial expansion. A claim that fails the numeric screen is reported
    "disproved" without ever reaching SymPy — cheap numeric evidence gates
    expensive symbolic work, not the other way around.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    odd_coeffs = coeffs[1::2]
    if odd_coeffs.size and not bool(np.all(np.abs(odd_coeffs) < eps)):
        return FormalCertificate(
            representation_id="polynomial_vandermonde",
            claim="p(x) == p(-x)",
            status="disproved",
            detail="odd-degree coefficients are not numerically negligible",
        )

    x = sp.symbols("x")
    rational_coeffs = [sp.nsimplify(float(c), rational=True, tolerance=eps) for c in coeffs]
    polynomial = sum(c * x**k for k, c in enumerate(rational_coeffs))
    difference = sp.expand(polynomial - polynomial.subs(x, -x))
    proved = difference == 0
    return FormalCertificate(
        representation_id="polynomial_vandermonde",
        claim="p(x) == p(-x)",
        status="proved" if proved else "disproved",
        detail=f"expand(p(x) - p(-x)) = {difference}",
    )
