"""Enumerate candidate metric expressions from descriptor variables (Phase 3)."""

from __future__ import annotations

import itertools
from typing import Iterator

from rde.expression.ast import Expr
from rde.expression.ops import BINARY_OPS, UNARY_OPS


def unary_candidates(var: str, *, ops: tuple[str, ...] | None = None) -> list[Expr]:
    v = Expr.var(var)
    chosen = ops or UNARY_OPS
    out: list[Expr] = [v]
    for op in chosen:
        if op == "neg":
            out.append(Expr("neg", left=v))
        else:
            out.append(Expr(op, left=v))
    return out


def binary_candidates(a: Expr, b: Expr, *, ops: tuple[str, ...] | None = None) -> list[Expr]:
    chosen = ops or BINARY_OPS
    out: list[Expr] = []
    for op in chosen:
        if op == "pow":
            out.append(Expr("pow", left=a, right=Expr.const(2.0)))
        else:
            out.append(Expr(op, left=a, right=b))
    return out


def enumerate_expressions(
    variables: list[str],
    *,
    max_depth: int = 2,
    max_candidates: int = 10_000,
    include_self_pairs: bool = True,
) -> Iterator[Expr]:
    """Breadth-first enumeration of small expression trees over variable names."""
    if max_depth < 1 or not variables:
        return
    pool: list[Expr] = []
    for var in variables:
        pool.extend(unary_candidates(var))
    seen: set[Expr] = set()

    def emit(expr: Expr) -> Iterator[Expr]:
        if expr not in seen and len(seen) < max_candidates:
            seen.add(expr)
            yield expr

    for expr in pool:
        yield from emit(expr)

    depth = 1
    while depth < max_depth and len(seen) < max_candidates:
        next_pool: list[Expr] = []
        pair_iter = itertools.combinations_with_replacement(pool, 2) if include_self_pairs else itertools.combinations(pool, 2)
        for a, b in pair_iter:
            if len(seen) >= max_candidates:
                return
            for expr in binary_candidates(a, b):
                if len(seen) >= max_candidates:
                    return
                for emitted in emit(expr):
                    next_pool.append(emitted)
                    yield emitted
                    if len(seen) >= max_candidates:
                        return
        pool = pool + next_pool
        depth += 1
