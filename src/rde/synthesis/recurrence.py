"""Recurrence-complexity solver (Mode 2).

Before a candidate algorithm skeleton is ever executed against a domain, its
*shape* already implies an asymptotic cost via a standard recurrence. This
module answers that question symbolically and cheaply, so a search over
skeletons can reject exponential shapes (e.g. ``T(n) = 2*T(n-1) + poly(n)``)
without paying for a single domain call, and only spend domain calls
verifying shapes that are already known to meet the target
(``T(n) = 2*T(n/2) + poly(n)``).

Three recurrence shapes are supported, covering the common divide-and-conquer
and peeling patterns:

- ``base``      — no recursion; solve directly at declared cost.
- ``divide``    — ``T(n) = a * T(n / b) + Theta(n^d)`` (master theorem).
- ``subtract``  — ``T(n) = a * T(n - c) + Theta(n^d)`` (linear/exponential
                  peeling recursion).
- ``flat``      — one-shot split into ``n`` independent O(1)-size leaves plus
                  an ``Theta(n^d)`` combine step (not a recursion at all, but
                  expressed here for a uniform interface).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

CostKind = Literal["poly", "exp"]


@dataclass(frozen=True)
class CostClass:
    """A closed-form asymptotic cost: either polynomial or exponential in n."""

    kind: CostKind
    degree: float | None = None
    log_factor: bool = False
    exp_base: float | None = None
    exp_rate: float = 1.0

    @property
    def is_polynomial(self) -> bool:
        return self.kind == "poly"

    def pretty(self) -> str:
        if self.kind == "exp":
            base = self.exp_base if self.exp_base is not None else 2.0
            rate = f"{self.exp_rate:g}*" if self.exp_rate != 1.0 else ""
            return f"Theta({base:g}^({rate}n))"
        degree = self.degree if self.degree is not None else 0.0
        deg_str = "1" if abs(degree - 1.0) < 1e-9 else f"{degree:g}"
        if abs(degree) < 1e-9 and not self.log_factor:
            return "Theta(1)"
        body = f"n^{deg_str}" if abs(degree) > 1e-9 else "1"
        if self.log_factor:
            body = f"{body} * log(n)" if body != "1" else "log(n)"
        return f"Theta({body})"


@dataclass(frozen=True)
class Recurrence:
    """One candidate skeleton's recurrence, in one of the four shapes above."""

    shape: Literal["base", "divide", "subtract", "flat"]
    # base
    base_cost: CostClass | None = None
    # divide / subtract: number of recursive branches
    branches: int = 1
    # divide: subproblem size = ceil(n / divisor)
    divisor: float = 2.0
    # subtract: subproblem size = n - shrink
    shrink: int = 1
    # divide / subtract / flat: combine-step cost degree, Theta(n^combine_degree)
    combine_degree: float = 1.0

    def describe(self) -> str:
        if self.shape == "base":
            return "solve directly (no decomposition)"
        if self.shape == "divide":
            return f"decompose into {self.branches} branch(es) of size n/{self.divisor:g}, combine at degree {self.combine_degree:g}"
        if self.shape == "subtract":
            return f"decompose into {self.branches} branch(es) of size n-{self.shrink}, combine at degree {self.combine_degree:g}"
        return f"flat-decompose into n independent O(1) leaves, combine at degree {self.combine_degree:g}"


_EPS = 1e-9


def solve_recurrence(rec: Recurrence) -> CostClass:
    """Closed-form asymptotic solution for one recurrence shape.

    ``divide`` uses the standard three-case master theorem (constant a, b);
    the regularity condition for case 3 is assumed, matching the textbook
    statement — this is a search-time proxy, not a full derivation.
    ``subtract`` distinguishes the two qualitatively different regimes: one
    branch per level stays polynomial, more than one branch is exponential in
    n (this is exactly the ``T(n)=2T(n-1)+poly(n)`` example that motivated
    this module — it must come out exponential, and does).
    """
    if rec.shape == "base":
        if rec.base_cost is None:
            raise ValueError("base recurrence requires base_cost")
        return rec.base_cost

    if rec.shape == "flat":
        # n branches of O(1) leaf cost + Theta(n^d) combine: touching n
        # leaf results is itself at least linear, so the total is
        # Theta(n^max(1, d)).
        return CostClass(kind="poly", degree=max(1.0, rec.combine_degree))

    a = float(rec.branches)
    d = float(rec.combine_degree)

    if rec.shape == "divide":
        b = float(rec.divisor)
        if b <= 1.0:
            raise ValueError("divide recurrence requires divisor > 1")
        b_pow_d = b**d
        if a < b_pow_d - _EPS:
            return CostClass(kind="poly", degree=d)
        if abs(a - b_pow_d) <= _EPS:
            return CostClass(kind="poly", degree=d, log_factor=True)
        log_b_a = math.log(a) / math.log(b)
        return CostClass(kind="poly", degree=log_b_a)

    # shape == "subtract"
    c = max(1, rec.shrink)
    if a <= 1.0 + _EPS:
        # T(n) = T(n - c) + Theta(n^d)  =>  Theta(n^(d+1)) (Riemann-sum style)
        return CostClass(kind="poly", degree=d + 1.0)
    # T(n) = a*T(n - c) + Theta(n^d), a > 1  =>  Theta(a^(n/c)), dominates poly(n)
    return CostClass(kind="exp", exp_base=a, exp_rate=1.0 / c)


def meets_target(solution: CostClass, target_degree: float | None) -> bool:
    """Whether `solution` satisfies a declared polynomial resource budget.

    `target_degree=None` accepts any polynomial (the informal "poly(N)"
    budget from the reverse-engineering brief); a numeric value additionally
    caps the polynomial degree.
    """
    if not solution.is_polynomial:
        return False
    if target_degree is None:
        return True
    return (solution.degree or 0.0) <= target_degree + _EPS
