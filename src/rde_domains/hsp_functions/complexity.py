"""Closed-form quantum/classical query-complexity reference table.

Interpretive reference only -- deliberately **not** wired into the RDE
regression pipeline: a naive classical collision-search baseline is
confounded by the very noise/break parameters this domain varies, so it
cannot honestly serve as a per-row RDE target. These numbers exist to state
why the three held-out calibration families are the right validation set:
they are exactly the three named points in the literature with a *known*
poly(log|X|)-or-subexponential quantum query algorithm (Simon's algorithm,
Shor-style period finding, Kuperberg's dihedral hidden subgroup algorithm),
contrasted with a generic classical Theta(sqrt(|X|/|K|))
collision/birthday-bound lower bound for finding a size-|K| hidden
subgroup element by black-box queries alone.

The dihedral constant below is illustrative / order-of-magnitude only,
matching the literature's own O(...) form (verified against Kuperberg's
Theorem 1.1: time and query complexity 2^O(sqrt(log N))) -- not a
numerically precise claim. Do not cite the specific constant used here as
an established result; cite Kuperberg's theorem for the asymptotic form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryComplexityReference:
    family: str
    log2_x_size: int
    log2_k_true: int  # log2 |K_true|
    q_quantum_queries: float
    q_classical_queries: float
    quantum_basis: str
    classical_basis: str

    @property
    def log2_gap(self) -> float:
        return math.log2(max(1.0, self.q_classical_queries)) - math.log2(max(1.0, self.q_quantum_queries))


def calibration_reference(family: str, n_bits: int) -> QueryComplexityReference:
    """Closed-form Q_quantum / Q_classical for one of the three held-out families.

    `n_bits` = log2|X|; every calibration family in this domain is
    constructed with |K_true| = 2 (see `functions.make_instance`), so the
    generic birthday-bound classical baseline is Theta(sqrt(|X|/2)) for
    all three -- the group's representation theory, not |K|, is what
    varies.
    """
    x_size = 1 << n_bits
    q_classical = math.sqrt(x_size / 2.0)  # generic collision/birthday-bound scaling

    if family == "simon":
        # Kitaev 1995 / Mosca-Ekert 1999: abelian HSP is solved via Fourier
        # sampling in O(log|G|) queries; Simon's own analysis gives the
        # standard tight-enough asymptotic Theta(n) (coupon-collector-style:
        # n-1 linearly independent samples over GF(2)^n needed whp). A small
        # constant buffer is added for the repetition needed to guarantee
        # independence whp -- illustrative, not a razor-precise count.
        q_quantum = float(n_bits + 4)
        return QueryComplexityReference(
            family, n_bits, 1, q_quantum, q_classical,
            quantum_basis="Simon/abelian HSP, Theta(n)",
            classical_basis="generic birthday-bound collision search, Theta(sqrt(|X|/|K|))",
        )
    if family == "shor_cyclic":
        q_quantum = float(n_bits + 4)
        return QueryComplexityReference(
            family, n_bits, 1, q_quantum, q_classical,
            quantum_basis="cyclic/abelian HSP, O(log N)",
            classical_basis="generic birthday-bound collision search, Theta(sqrt(|X|/|K|))",
        )
    if family == "dihedral_kuperberg":
        # Kuperberg's Theorem 1.1: 2^O(sqrt(log N)) -- illustrative constant c=1.
        q_quantum = float(2.0 ** math.sqrt(max(1.0, n_bits)))
        return QueryComplexityReference(
            family, n_bits, 1, q_quantum, q_classical,
            quantum_basis="Kuperberg dihedral HSP, 2^O(sqrt(log N)), illustrative constant",
            classical_basis="generic birthday-bound collision search, Theta(sqrt(|X|/|K|))",
        )
    raise ValueError(f"no closed-form calibration reference for family {family!r}")
