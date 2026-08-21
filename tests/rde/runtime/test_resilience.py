"""Soft-fail / skip-one resilience: one bad item must not abort RDE stages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rde.discovery.checkpoint import (
    load_stage_progress,
    population_fingerprint,
    save_stage_progress,
    stage_completed,
)
from rde.discovery.rediscovery import assess_rediscovery
from rde.expression.ast import Expr
from rde.expression.batch import evaluate_chunk, score_expression
from rde.runtime.resilience import soft_call, soft_map


def test_soft_call_returns_fallback_and_records_error():
    errors: list[dict] = []
    out = soft_call("boom", lambda: (_ for _ in ()).throw(RecursionError("deep")), fallback=42, errors=errors)
    assert out == 42
    assert errors and errors[0]["stage"] == "boom"
    assert "RecursionError" in errors[0]["error"]


def test_soft_map_skips_bad_items():
    def fn(x: int) -> int:
        if x == 3:
            raise ValueError("bad")
        return x * 2

    out = soft_map("items", [1, 2, 3, 4], fn, skip_value=None)
    assert out == [2, 4, None, 8]


def test_score_expression_skips_recursion_error(monkeypatch):
    import rde.expression.batch as batch

    def boom(*_a, **_k):
        raise RecursionError("too deep")

    monkeypatch.setattr(batch, "_eval_values", boom)
    env = {"x": np.arange(8, dtype=float)}
    target = np.arange(8, dtype=float)
    assert score_expression(Expr.var("x"), env, target, backend="numpy") is None


def test_ranker_safe_chunk_degrades_to_per_expression(monkeypatch):
    import rde.analyze.ranker as ranker_mod
    from rde.analyze.ranker import ConjectureRanker

    ranker = ConjectureRanker(target_column="y", min_abs_r=0.0)
    env = {"x": np.linspace(0, 1, 16)}
    target = env["x"] * 2
    real_eval = evaluate_chunk

    def selective(*args, **kwargs):
        # Multi-expr chunk fails; single-expr retries succeed.
        if len(args[0]) > 1:
            raise RuntimeError("chunk failed")
        return real_eval(*args, **kwargs)

    monkeypatch.setattr(ranker_mod, "evaluate_chunk", selective)
    hits = ranker._safe_evaluate_chunk(
        [Expr.var("x"), Expr("add", left=Expr.var("x"), right=Expr.const(1.0))],
        env,
        target,
        backend="numpy",
        mlx_env=None,
        torch_env=None,
        target_mlx=None,
        force_gpu=False,
    )
    assert hits  # at least one candidate survives


def test_assess_rediscovery_soft_fails(monkeypatch):
    import rde.discovery.rediscovery as red

    monkeypatch.setattr(
        red,
        "_assess_rediscovery_inner",
        lambda *_a, **_k: (_ for _ in ()).throw(RecursionError("seed boom")),
    )
    rows = [{"x": float(i), "target": float(i)} for i in range(40)]
    assessment = assess_rediscovery(
        rows,
        "target",
        programs=[{"expression": "x", "fitness": 0.9, "pearson_r": 0.9}],
        variables=["x"],
        n_splits=3,
    )
    assert assessment.g5_met is False
    assert assessment.criteria.get("soft_failed") is True


def test_stage_progress_roundtrip(tmp_path: Path):
    save_stage_progress(
        tmp_path,
        "run_a",
        completed=["rankers", "phase4"],
        errors=[{"stage": "phase6.rediscovery", "error": "RecursionError"}],
        partial={"metric_candidates": [{"expression": "x", "r_squared": 0.9}]},
    )
    progress = load_stage_progress(tmp_path, "run_a")
    assert stage_completed(progress, "rankers")
    assert stage_completed(progress, "phase4")
    assert not stage_completed(progress, "phase6")
    assert progress["partial"]["metric_candidates"][0]["expression"] == "x"
    assert progress["errors"][0]["stage"] == "phase6.rediscovery"


def _pop(n: int, target: str = "metric.t", extra_column: bool = False) -> list[dict]:
    rows = [{"size": 6, target: float(i)} for i in range(n)]
    if extra_column:
        for row in rows:
            row["operator.coin_theta"] = 1.0
    return rows


def test_stage_progress_is_discarded_when_the_population_changes(tmp_path: Path):
    """A changed population must not resume another population's stage output.

    The run_id stays the same across a re-run, so keying the checkpoint on it
    alone restores ranker results computed on the previous rows and reports them
    against the new ones — silently, and with no verdict-level symptom.
    """
    original = _pop(20)
    save_stage_progress(
        tmp_path,
        "run_b",
        completed=["rankers"],
        errors=[],
        partial={"metric_candidates": [{"expression": "stale", "r_squared": 0.9}]},
        fingerprint=population_fingerprint(original, "metric.t"),
    )

    same = load_stage_progress(
        tmp_path, "run_b", fingerprint=population_fingerprint(original, "metric.t")
    )
    assert stage_completed(same, "rankers")

    for changed in (
        _pop(40),                        # more instances
        _pop(20, extra_column=True),     # a wider predictor grid
        _pop(20, target="metric.other"), # a different target
    ):
        target = "metric.other" if "metric.other" in changed[0] else "metric.t"
        progress = load_stage_progress(
            tmp_path, "run_b", fingerprint=population_fingerprint(changed, target)
        )
        assert not stage_completed(progress, "rankers")
        assert progress["partial"] == {}


def test_fingerprintless_checkpoint_is_not_resumed(tmp_path: Path):
    """Checkpoints written before fingerprints existed cannot vouch for a population."""
    save_stage_progress(
        tmp_path,
        "run_c",
        completed=["rankers"],
        errors=[],
        partial={"metric_candidates": [{"expression": "stale"}]},
    )
    progress = load_stage_progress(
        tmp_path, "run_c", fingerprint=population_fingerprint(_pop(20), "metric.t")
    )
    assert not stage_completed(progress, "rankers")
    # Without a fingerprint to check against, resume stays available as before.
    assert stage_completed(load_stage_progress(tmp_path, "run_c"), "rankers")
