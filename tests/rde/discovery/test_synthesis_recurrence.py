"""Recurrence-complexity solver (Mode 2, ALGO-057).

Validates `rde.synthesis.recurrence` against textbook closed forms so the
synthesis search's symbolic pruning stage is trustworthy before anything
touches a domain. Two cases matter most for this project: `divide` a=2,b=2,d=1
(merge sort, polynomial) and `subtract` a=2,c=1,d=0 (the exact
`T(n)=2*T(n-1)+poly(n)` example from the reverse-engineering brief this
module was written to formalize, which must come out exponential).
"""

from __future__ import annotations

import math

from rde.synthesis.recurrence import Recurrence, meets_target, solve_recurrence


def test_base_case_passes_through():
    from rde.synthesis.recurrence import CostClass

    rec = Recurrence(shape="base", base_cost=CostClass(kind="exp", exp_base=2.0, exp_rate=1.0))
    solved = solve_recurrence(rec)
    assert solved.kind == "exp"
    assert not solved.is_polynomial


def test_divide_merge_sort_is_n_log_n():
    # T(n) = 2*T(n/2) + Theta(n)
    rec = Recurrence(shape="divide", branches=2, divisor=2.0, combine_degree=1.0)
    solved = solve_recurrence(rec)
    assert solved.is_polynomial
    assert math.isclose(solved.degree, 1.0)
    assert solved.log_factor


def test_divide_binary_search_is_log_n():
    # T(n) = 1*T(n/2) + Theta(1)
    rec = Recurrence(shape="divide", branches=1, divisor=2.0, combine_degree=0.0)
    solved = solve_recurrence(rec)
    assert solved.is_polynomial
    assert math.isclose(solved.degree, 0.0)
    assert solved.log_factor


def test_divide_naive_matrix_multiply_is_n_cubed():
    # T(n) = 8*T(n/2) + Theta(n^2)
    rec = Recurrence(shape="divide", branches=8, divisor=2.0, combine_degree=2.0)
    solved = solve_recurrence(rec)
    assert solved.is_polynomial
    assert math.isclose(solved.degree, 3.0, abs_tol=1e-9)
    assert not solved.log_factor


def test_divide_karatsuba_is_n_to_log2_3():
    # T(n) = 3*T(n/2) + Theta(n)
    rec = Recurrence(shape="divide", branches=3, divisor=2.0, combine_degree=1.0)
    solved = solve_recurrence(rec)
    assert solved.is_polynomial
    assert math.isclose(solved.degree, math.log2(3), abs_tol=1e-9)


def test_subtract_with_one_branch_is_polynomial():
    # T(n) = T(n-1) + Theta(n) -> Theta(n^2), e.g. selection-sort-style peeling
    rec = Recurrence(shape="subtract", branches=1, shrink=1, combine_degree=1.0)
    solved = solve_recurrence(rec)
    assert solved.is_polynomial
    assert math.isclose(solved.degree, 2.0, abs_tol=1e-9)


def test_subtract_with_two_branches_is_exponential():
    # T(n) = 2*T(n-1) + poly(n) -- exactly the brief's cautionary example.
    rec = Recurrence(shape="subtract", branches=2, shrink=1, combine_degree=0.0)
    solved = solve_recurrence(rec)
    assert not solved.is_polynomial
    assert solved.kind == "exp"
    assert math.isclose(solved.exp_base, 2.0)


def test_flat_decomposition_is_at_least_linear():
    rec = Recurrence(shape="flat", combine_degree=0.0)
    solved = solve_recurrence(rec)
    assert solved.is_polynomial
    assert math.isclose(solved.degree, 1.0)

    rec2 = Recurrence(shape="flat", combine_degree=2.0)
    solved2 = solve_recurrence(rec2)
    assert math.isclose(solved2.degree, 2.0)


def test_meets_target_respects_degree_cap():
    poly_cubic = solve_recurrence(Recurrence(shape="divide", branches=8, divisor=2.0, combine_degree=2.0))
    exp_case = solve_recurrence(Recurrence(shape="subtract", branches=2, shrink=1, combine_degree=0.0))

    assert meets_target(poly_cubic, target_degree=None)
    assert meets_target(poly_cubic, target_degree=3.0)
    assert not meets_target(poly_cubic, target_degree=2.0)
    assert not meets_target(exp_case, target_degree=None)
