"""Tests for expression ranker progress callbacks and MLX env reuse."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rde.analyze.ranker import ConjectureRanker
from rde.expression import Expr
from rde.expression.batch import evaluate_chunk, prepare_device_envs
from rde.expression import enumerate_expressions
from rde.expression import batch as batch_mod


def _synthetic_rows(n: int) -> list[dict[str, float]]:
    return [{"a": float(i), "b": float(i * i), "target": float(3 * i)} for i in range(n)]


def test_ranker_on_progress_reports_evaluated_counts():
    rows = _synthetic_rows(64)
    progress: list[tuple[int, int, str]] = []

    def on_progress(done: int, total: int, detail: str) -> None:
        progress.append((done, total, detail))

    ranker = ConjectureRanker(target_column="target", min_abs_r=0.1, max_results=5)
    cands = enumerate_expressions(["a", "b"], max_depth=2, max_candidates=500)
    ranker.rank_expressions_streaming(
        rows,
        cands,
        max_evaluated=500,
        chunk_size=64,
        on_progress=on_progress,
        eval_backend="numpy",
    )
    assert progress
    assert progress[-1][0] <= progress[-1][1]
    assert "done" in progress[-1][2]


def test_ranker_prepares_device_env_once_with_prebuilt_chunk_envs(monkeypatch):
    calls: list[int] = []
    real_prepare = batch_mod.prepare_device_envs

    def spy(*args, **kwargs):
        calls.append(1)
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr("rde.analyze.ranker.prepare_device_envs", spy)

    rows = _synthetic_rows(600)
    ranker = ConjectureRanker(target_column="target", min_abs_r=0.1, max_results=5)
    cands = enumerate_expressions(["a", "b"], max_depth=2, max_candidates=1500)
    hits = ranker.rank_expressions_streaming(
        rows,
        cands,
        max_evaluated=1500,
        chunk_size=512,
        eval_backend="mlx",
        require_gpu=False,
    )
    assert hits
    assert len(calls) == 1
    assert getattr(ranker, "last_eval_backend", "numpy") == "mlx"


def test_ranker_prunes_against_perfect_metadata_floor(monkeypatch):
    rows = _synthetic_rows(64)
    ranker = ConjectureRanker(target_column="target", min_abs_r=0.1, max_results=1)
    original = ConjectureRanker._make_conjecture
    calls = 0

    def perfect_metadata(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        conjecture = original(self, *args, **kwargs)
        conjecture.metadata["extrapolation_r_squared"] = 1.0
        conjecture.metadata["cross_n_sign_stability"] = 1.0
        return conjecture

    monkeypatch.setattr(ConjectureRanker, "_make_conjecture", perfect_metadata)
    candidates = [
        Expr.var("a"),
        Expr("mul", left=Expr.var("a"), right=Expr.const(0.5)),
    ]
    hits = ranker.rank_expressions_streaming(
        rows,
        iter(candidates),
        variables=["a"],
        chunk_size=1,
        eval_backend="numpy",
    )
    assert hits
    assert calls == 1


def test_ranker_checkpoint_resume_binds_candidate_universe(tmp_path):
    rows = _synthetic_rows(64)
    checkpoint = tmp_path / "metrics.checkpoint.json"
    ranker = ConjectureRanker(target_column="target", min_abs_r=0.1, max_results=5)
    candidates = list(enumerate_expressions(["a", "b"], max_depth=2, max_candidates=100))

    first = ranker.rank_expressions_streaming(
        rows,
        iter(candidates),
        variables=["a", "b"],
        max_evaluated=100,
        chunk_size=8,
        checkpoint_path=checkpoint,
        resume=False,
        eval_backend="numpy",
    )
    resumed = ranker.rank_expressions_streaming(
        rows,
        iter(candidates),
        variables=["a", "b"],
        max_evaluated=100,
        chunk_size=8,
        checkpoint_path=checkpoint,
        resume=True,
        eval_backend="numpy",
    )
    assert first
    assert resumed
    assert json.loads(checkpoint.read_text())["completed"] is True


def test_ranker_checkpoint_resumes_bounded_heap_without_reselection(tmp_path: Path):
    rows = _synthetic_rows(64)
    candidates = list(enumerate_expressions(["a", "b"], max_depth=2, max_candidates=60))
    checkpoint = tmp_path / "metric-search.json"

    def interrupted():
        for index, expr in enumerate(candidates):
            if index == 20:
                raise RuntimeError("synthetic interruption")
            yield expr

    partial_ranker = ConjectureRanker(target_column="target", min_abs_r=0.1, max_results=5)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        partial_ranker.rank_expressions_streaming(
            rows,
            interrupted(),
            variables=["a", "b"],
            max_evaluated=60,
            chunk_size=4,
            checkpoint_path=checkpoint,
            checkpoint_every=4,
            eval_backend="numpy",
        )
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["n_evaluated"] == 20
    assert saved["completed"] is False
    resumed = ConjectureRanker(target_column="target", min_abs_r=0.1, max_results=5)
    resumed_hits = resumed.rank_expressions_streaming(
        rows,
        iter(candidates),
        variables=["a", "b"],
        max_evaluated=60,
        chunk_size=4,
        checkpoint_path=checkpoint,
        checkpoint_every=4,
        resume=True,
        eval_backend="numpy",
    )
    fresh = ConjectureRanker(target_column="target", min_abs_r=0.1, max_results=5)
    fresh_hits = fresh.rank_expressions_streaming(
        rows,
        iter(candidates),
        variables=["a", "b"],
        max_evaluated=60,
        chunk_size=4,
        eval_backend="numpy",
    )
    assert [hit.expression for hit in resumed_hits] == [hit.expression for hit in fresh_hits]
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["completed"] is True


def test_mlx_hit_values_are_not_evaluated_twice(monkeypatch):
    from rde.expression import mlx_eval

    if not mlx_eval.mlx_runtime_available():
        pytest.skip("MLX/Metal unavailable")

    env = {"a": np.linspace(1.0, 2.0, 600)}
    target = 3.0 * env["a"]
    backend, mlx_env, torch_env, target_mlx, _ = prepare_device_envs(
        env,
        "mlx",
        len(target),
        target,
        force_gpu=True,
        require_gpu=True,
        requested_backend="mlx",
    )
    calls = 0
    original = mlx_eval.evaluate_expr_mlx

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(mlx_eval, "evaluate_expr_mlx", counted)
    hits = evaluate_chunk(
        [Expr.var("a")],
        env,
        target,
        min_abs_r=0.1,
        backend=backend,
        mlx_env=mlx_env,
        torch_env=torch_env,
        target_mlx=target_mlx,
        force_gpu=True,
    )
    assert hits
    assert calls == 1


def test_ranker_honors_explicit_empty_variable_list():
    """An explicit [] must not fall back to all numeric columns when env is prebuilt."""
    rows = [
        {
            "family_index": 0,
            "size": 8,
            "metric.y": 0.5,
        }
    ]
    ranker = ConjectureRanker(target_column="metric.y", min_abs_r=0.25, max_results=5)
    ranker.rank_expressions_streaming(
        rows,
        iter([]),
        variables=[],
        env={},
        target=np.array([0.5]),
        max_evaluated=0,
        compute_metadata=False,
        eval_backend="numpy",
    )
