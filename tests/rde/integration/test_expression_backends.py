"""Cross-backend expression evaluation parity tests."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from rde.expression.ast import Expr
from rde.expression.dispatch import compile_expr
from rde.expression.evaluate import evaluate_expr
from rde.expression.mlx_eval import evaluate_to_numpy as mlx_eval_to_numpy
from rde.expression.torch_eval import evaluate_to_numpy as torch_eval_to_numpy


def _env(n: int = 32) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
    }


@pytest.mark.parametrize(
    "expr",
    [
        Expr("add", left=Expr.var("a"), right=Expr.var("b")),
        Expr("mul", left=Expr.var("a"), right=Expr.const(2.0)),
        Expr("sqrt", left=Expr("abs", left=Expr.var("a"))),
        Expr("log", left=Expr("add", left=Expr("abs", left=Expr.var("a")), right=Expr.const(1.0))),
    ],
)
def test_numpy_mlx_torch_agree(expr: Expr):
    env = _env()
    ref = evaluate_expr(expr, env)
    try:
        mlx_out = mlx_eval_to_numpy(expr, env)
        assert np.allclose(ref, mlx_out, rtol=1e-4, atol=1e-4, equal_nan=True)
    except ImportError:
        pytest.skip("mlx not installed")
    torch_out = torch_eval_to_numpy(expr, env, backend="torch_cpu")
    assert np.allclose(ref, torch_out, rtol=1e-4, atol=1e-4, equal_nan=True)


def test_numpy_mul_overflow_returns_inf_without_runtime_warning():
    """Clipped exp outputs can still overflow on multiply; keep IEEE inf quietly."""
    env = {"a": np.array([700.0, 1.0]), "b": np.array([700.0, 2.0])}
    expr = Expr(
        "mul",
        left=Expr("exp", left=Expr.var("a")),
        right=Expr("exp", left=Expr.var("b")),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        out = evaluate_expr(expr, env)
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)
    assert np.isinf(out[0])
    assert np.isfinite(out[1])


def test_expression_compiles_to_postorder_tape():
    expr = Expr(
        "mul",
        left=Expr("add", left=Expr.var("a"), right=Expr.const(2.0)),
        right=Expr("neg", left=Expr.var("b")),
    )
    tape = compile_expr(expr)
    assert [opcode for opcode, _payload in tape.instructions] == [
        "var",
        "const",
        "add",
        "var",
        "neg",
        "mul",
    ]


def test_deep_expression_compiles_without_recursion_error():
    """GP/DreamCoder can emit left-deep trees that used to blow lru_cache hashing."""
    expr = Expr.var("a")
    for _ in range(2_000):
        expr = Expr("add", left=expr, right=Expr.const(1.0))
    tape = compile_expr(expr)
    assert len(tape.instructions) == 4001
    assert expr.depth() == 2001
    assert expr.size() == 4001
    rendered = str(expr)
    assert rendered.startswith("((")
    env = {"a": np.array([1.0, 2.0])}
    out = evaluate_expr(expr, env)
    assert out.shape == (2,)
    assert np.allclose(out, np.array([2001.0, 2002.0]))
