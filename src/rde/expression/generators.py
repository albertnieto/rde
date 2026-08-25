"""Metric generator enumeration (Phase 3)."""

from __future__ import annotations

import itertools
from typing import Iterator

from rde.analyze.tables import contract_excluded_columns, numeric_columns
from rde.expression.ast import Expr
from rde.expression.enumerate import enumerate_expressions, unary_candidates


def metric_variable_columns(
    rows: list[dict],
    target: str,
    *,
    max_vars: int = 24,
    exclude_prefixes: tuple[str, ...] = ("metric.", "gen.", "derived."),
    domain_id: str | None = None,
) -> list[str]:
    """Pick descriptor columns for metric DSL search, preferring |r| with target.

    `domain_id`, when given, additionally blocks any column the domain's
    `DomainContract` explicitly marks `predictor_eligible=False` (raw
    `OUTCOME`/`TARGET_DERIVED` scalars, e.g. `structure_strength`) — fail
    *open* for columns the contract doesn't catalog at all, so domains
    without a full catalog keep their current candidate set; only columns
    the contract actively flags are removed. Without `domain_id`, behavior
    is unchanged from before this parameter existed.

    This exists because `exclude_prefixes` alone does not catch a raw,
    un-prefixed `OUTCOME` scalar re-exposed on the row for bookkeeping
    (e.g. `hsp_functions`'s `structure_strength`, present so
    `metric.structure_strength` can be computed from it) — that column
    correlates with the target by construction (`r=1.0`), which is exactly
    the leak this filter exists to catch and previously did not.
    """
    from rde.analyze.query import correlate_with_target

    skip = {target, "size", "seed", "family_index"} | contract_excluded_columns(rows, domain_id)
    hits = correlate_with_target(rows, target, min_abs_r=0.0, exclude=skip)
    ordered: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        col = hit["column"]
        if any(col.startswith(p) for p in exclude_prefixes):
            continue
        if col not in seen:
            seen.add(col)
            ordered.append(col)
    if len(ordered) < max_vars:
        for col in numeric_columns(rows, exclude=skip):
            if any(col.startswith(p) for p in exclude_prefixes):
                continue
            if col not in seen:
                seen.add(col)
                ordered.append(col)
    return ordered[:max_vars]


def enumerate_ratio_metrics(
    variables: list[str],
    *,
    max_pairs: int | None = None,
) -> Iterator[Expr]:
    """Pairwise ratio metrics div(a,b) and div(b,a)."""
    count = 0
    for a, b in itertools.combinations(variables, 2):
        va, vb = Expr.var(a), Expr.var(b)
        yield Expr("div", left=va, right=vb)
        yield Expr("div", left=vb, right=va)
        count += 2
        if max_pairs is not None and count >= max_pairs:
            return


def enumerate_product_metrics(
    variables: list[str],
    *,
    max_pairs: int | None = None,
) -> Iterator[Expr]:
    """Pairwise product metrics mul(a,b)."""
    count = 0
    for a, b in itertools.combinations_with_replacement(variables, 2):
        yield Expr("mul", left=Expr.var(a), right=Expr.var(b))
        count += 1
        if max_pairs is not None and count >= max_pairs:
            return


def enumerate_unary_metric_chains(
    variables: list[str],
    *,
    ops: tuple[str, ...] = ("log", "sqrt", "abs"),
) -> Iterator[Expr]:
    """Unary transforms and simple chains log(sqrt(x)), abs(log(x))."""
    for var in variables:
        base = unary_candidates(var, ops=ops)
        for expr in base:
            yield expr
        v = Expr.var(var)
        for op in ops:
            inner = Expr(op, left=v) if op != "neg" else Expr("neg", left=v)
            for op2 in ("abs", "sqrt"):
                if op2 == op:
                    continue
                yield Expr(op2, left=inner)


def enumerate_metric_candidates(
    variables: list[str],
    *,
    max_depth: int = 3,
    max_candidates: int = 50_000,
    include_ratios: bool = True,
    include_products: bool = True,
    include_unary_chains: bool = True,
) -> Iterator[Expr]:
    """Merge structured metric generators with depth-limited expression trees."""
    seen: set[Expr] = set()
    emitted = 0

    def emit(expr: Expr) -> Iterator[Expr]:
        nonlocal emitted
        if expr in seen:
            return
        seen.add(expr)
        emitted += 1
        yield expr

    streams: list[Iterator[Expr]] = []
    if include_ratios:
        streams.append(enumerate_ratio_metrics(variables))
    if include_products:
        streams.append(enumerate_product_metrics(variables))
    if include_unary_chains:
        streams.append(enumerate_unary_metric_chains(variables))

    for stream in streams:
        for expr in stream:
            yield from emit(expr)
            if emitted >= max_candidates:
                return

    remaining = max(0, max_candidates - emitted)
    if remaining <= 0:
        return
    for expr in enumerate_expressions(
        variables,
        max_depth=max_depth,
        max_candidates=remaining,
    ):
        yield from emit(expr)
        if emitted >= max_candidates:
            return
