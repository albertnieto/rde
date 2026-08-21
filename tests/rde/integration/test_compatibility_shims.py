"""Tests for deprecated RDE module shims and torch empty-env parity."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from rde.expression.ast import Expr
from rde.expression.evaluate import evaluate_expr


def test_legacy_dynamics_shim_exports():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from rde.discovery import dynamics as legacy

    from rde.discovery import state_dynamics as modern

    assert legacy.DynamicsStepResult is modern.DynamicsStepResult
    assert legacy.predict_state_dynamics is modern.predict_state_dynamics


def test_legacy_profiling_shim_exports():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from rde.runtime import profiling as legacy

    from rde.runtime.config import StageProfiler

    assert legacy.StageProfiler is StageProfiler


def test_resume_index_shim(tmp_path):
    from rde.io.resume_index import ResumeIndex

    # Construction is the deprecation boundary (not import). Assert it warns,
    # then verify the shim still behaves for leftover callers.
    with pytest.warns(DeprecationWarning, match="deprecated"):
        idx = ResumeIndex(tmp_path / "resume.db")
    idx.mark_slice_done("inst-a", 3)
    assert ("inst-a", 3) in idx.completed_slices()
    idx.close()


def test_torch_empty_env_constant_parity():
    torch = pytest.importorskip("torch")
    from rde.expression.torch_eval import evaluate_expr_torch, numpy_env_to_torch

    expr = Expr.const(3.5)
    env: dict[str, np.ndarray] = {}
    torch_env = numpy_env_to_torch(env)
    values = evaluate_expr_torch(expr, torch_env)
    expected = evaluate_expr(expr, env)
    assert np.allclose(values.detach().cpu().numpy(), expected)
