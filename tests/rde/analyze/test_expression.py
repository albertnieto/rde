"""Tests for analysis helpers and Phase 3 expression/ranker skeleton."""

from __future__ import annotations

import numpy as np

from rde.analyze.query import correlate_with_target, distribution_summary, outlier_rows
from rde.expression import Expr, enumerate_expressions, evaluate_expr, r_squared
from rde.analyze.ranker import ConjectureRanker


def test_correlate_with_target_finds_linear_relation():
    rows = [{"x": float(i), "y": 2.0 * i + 1.0} for i in range(20)]
    hits = correlate_with_target(rows, "y", min_abs_r=0.9)
    assert hits
    assert hits[0]["column"] == "x"
    assert hits[0]["r_squared"] > 0.99


def test_distribution_summary_by_size():
    rows = [
        {"size": 4, "metric.log_slice_rank": 0.5},
        {"size": 4, "metric.log_slice_rank": 0.7},
        {"size": 8, "metric.log_slice_rank": 0.9},
    ]
    summary = distribution_summary(rows, "metric.log_slice_rank", group_by="size")
    assert len(summary) == 2
    assert summary[0]["size"] == 4
    assert summary[0]["count"] == 2


def test_outlier_rows():
    rows = [{"instance_id": str(i), "v": float(i)} for i in range(10)]
    rows.append({"instance_id": "out", "v": 1000.0})
    found = outlier_rows(rows, "v", z_threshold=2.0)
    assert len(found) == 1
    assert found[0]["instance_id"] == "out"


def test_expression_evaluate_and_rank():
    n = 50
    x = np.linspace(1.0, 2.0, n)
    rows = [{"a": float(v), "b": float(v * v), "target": float(3 * v)} for v in x]
    env = {"a": x, "b": x * x}
    expr = Expr("mul", left=Expr.const(3.0), right=Expr.var("a"))
    got = evaluate_expr(expr, env)
    assert np.allclose(got, 3 * x)
    assert r_squared(got, 3 * x) > 0.999

    ranker = ConjectureRanker(target_column="target", min_abs_r=0.9)
    cands = list(enumerate_expressions(["a", "b"], max_depth=2, max_candidates=50))
    top = ranker.rank_expressions(rows, cands)
    assert top
    assert top[0].r_squared > 0.99


def test_expression_enumerator_stops_inside_pair_loop_at_cap():
    candidates = list(
        enumerate_expressions(["a"], max_depth=4, max_candidates=10)
    )
    assert len(candidates) == 10
    assert len(set(candidates)) == 10
