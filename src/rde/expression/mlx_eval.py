"""MLX-accelerated expression evaluation (Apple Silicon GPU — primary RDE backend)."""

from __future__ import annotations

from typing import Any

import numpy as np

from rde.expression.ast import Expr
from rde.expression.dispatch import CompiledExpr, compile_expr


def mlx_available() -> bool:
    try:
        import mlx.core as mx  # noqa: F401

        return True
    except ImportError:
        return False


def mlx_runtime_available() -> bool:
    """True when MLX is installed *and* Metal/GPU is accessible."""
    if not mlx_available():
        return False
    try:
        mx = _mx()
        probe = mx.array([1.0], dtype=mx.float32)
        mx.eval(probe)
        return True
    except RuntimeError:
        return False


def _mx():
    import mlx.core as mx

    return mx


def numpy_env_to_mlx(env: dict[str, np.ndarray]) -> dict[str, Any]:
    mx = _mx()
    return {k: mx.array(np.asarray(v, dtype=np.float32)) for k, v in env.items()}


class _MlxExprBackend:
    def __init__(self, mx) -> None:
        self._mx = mx

    def env_len(self, env: dict[str, Any]) -> int:
        return len(next(iter(env.values()))) if env else 1

    def const(self, value: float, n: int) -> Any:
        return self._mx.full((n,), float(value), dtype=self._mx.float32)

    def var(self, key: str, env: dict[str, Any]) -> Any:
        if key not in env:
            raise KeyError(f"Unknown variable: {key!r}")
        return env[key]

    def neg(self, x: Any) -> Any:
        return -x

    def abs(self, x: Any) -> Any:
        return self._mx.abs(x)

    def sqrt(self, x: Any) -> Any:
        return self._mx.sqrt(self._mx.maximum(x, 0.0))

    def log(self, x: Any) -> Any:
        return self._mx.log(self._mx.maximum(x, 1e-30))

    def exp(self, x: Any) -> Any:
        return self._mx.exp(self._mx.minimum(self._mx.maximum(x, -700.0), 700.0))

    def add(self, a: Any, b: Any) -> Any:
        return a + b

    def sub(self, a: Any, b: Any) -> Any:
        return a - b

    def mul(self, a: Any, b: Any) -> Any:
        return a * b

    def div(self, num: Any, denom: Any) -> Any:
        safe = self._mx.where(
            self._mx.abs(denom) < 1e-12,
            self._mx.full(denom.shape, float("nan")),
            denom,
        )
        return num / safe

    def pow(self, base: Any, expn: Any) -> Any:
        return self._mx.power(base, expn)


def evaluate_expr_mlx(expr: Expr, env: dict[str, Any]) -> Any:
    """Evaluate expression tree with MLX arrays in env."""
    from rde.expression.dispatch import evaluate_expr_with_backend

    return evaluate_expr_with_backend(expr, env, _MlxExprBackend(_mx()))


def mlx_to_numpy(array: Any) -> np.ndarray:
    mx = _mx()
    mx.eval(array)
    return np.asarray(array, dtype=float)


def mlx_to_numpy_batch(arrays: list[Any]) -> np.ndarray:
    """Materialize same-shaped MLX arrays with one device synchronization."""
    if not arrays:
        return np.empty((0, 0), dtype=float)
    mx = _mx()
    stacked = mx.stack(arrays, axis=0)
    mx.eval(stacked)
    return np.asarray(stacked, dtype=float)


def mlx_target_to_mlx(target: np.ndarray) -> Any:
    mx = _mx()
    return mx.array(np.asarray(target, dtype=np.float32))


def _mlx_finite_mask(*arrays: Any) -> Any:
    mx = _mx()
    mask = mx.isfinite(arrays[0])
    for arr in arrays[1:]:
        mask = mask & mx.isfinite(arr)
    return mask


def mlx_pearson_r(values: Any, target: Any) -> float:
    """Device-side Pearson r; uses one device synchronization."""
    mx = _mx()
    mask = _mlx_finite_mask(values, target)
    n = mx.sum(mask.astype(mx.float32))
    x = mx.where(mask, values, mx.array(0.0, dtype=mx.float32))
    y = mx.where(mask, target, mx.array(0.0, dtype=mx.float32))
    n_safe = mx.maximum(n, mx.array(1.0, dtype=mx.float32))
    x_mean = mx.sum(x) / n_safe
    y_mean = mx.sum(y) / n_safe
    xc = mx.where(mask, values - x_mean, mx.array(0.0, dtype=mx.float32))
    yc = mx.where(mask, target - y_mean, mx.array(0.0, dtype=mx.float32))
    num = mx.sum(xc * yc)
    den = mx.sqrt(mx.sum(xc * xc) * mx.sum(yc * yc))
    r = num / mx.maximum(den, mx.array(1e-12, dtype=mx.float32))
    mx.eval(n, r)
    if float(np.asarray(n)) < 3:
        return float("nan")
    return float(np.asarray(r))


def mlx_regression_r_squared(values: Any, target: Any) -> float:
    """Device-side OLS R²; uses one device synchronization."""
    mx = _mx()
    mask = _mlx_finite_mask(values, target)
    n = mx.sum(mask.astype(mx.float32))
    x = mx.where(mask, values, mx.array(0.0, dtype=mx.float32))
    y = mx.where(mask, target, mx.array(0.0, dtype=mx.float32))
    n_safe = mx.maximum(n, mx.array(1.0, dtype=mx.float32))
    x_mean = mx.sum(x) / n_safe
    y_mean = mx.sum(y) / n_safe
    xc = mx.where(mask, values - x_mean, mx.array(0.0, dtype=mx.float32))
    yc = mx.where(mask, target - y_mean, mx.array(0.0, dtype=mx.float32))
    var_x = mx.sum(xc * xc)
    slope = mx.sum(xc * yc) / mx.maximum(var_x, mx.array(1e-12, dtype=mx.float32))
    intercept = y_mean - slope * x_mean
    pred = intercept + slope * values
    ss_res = mx.sum(mx.where(mask, (target - pred) ** 2, mx.array(0.0, dtype=mx.float32)))
    ss_tot = mx.sum(mx.where(mask, (target - y_mean) ** 2, mx.array(0.0, dtype=mx.float32)))
    r2 = mx.array(1.0, dtype=mx.float32) - ss_res / mx.maximum(
        ss_tot, mx.array(1e-12, dtype=mx.float32)
    )
    mx.eval(n, var_x, ss_tot, r2)
    if float(np.asarray(n)) < 3:
        return float("nan")
    if float(np.asarray(var_x)) <= 0.0 or float(np.asarray(ss_tot)) <= 0.0:
        return float("nan")
    return float(np.asarray(r2))


def score_expression_mlx(
    expr: Expr,
    mlx_env: dict[str, Any],
    target_mlx: Any,
    *,
    need_values: bool = False,
) -> tuple[float, float, np.ndarray | None] | None:
    """Return (r_squared, pearson_r, values_or_none) with minimal host sync."""
    try:
        values = evaluate_expr_mlx(expr, mlx_env)
    except (
        KeyError,
        ValueError,
        FloatingPointError,
        OverflowError,
        RuntimeError,
        RecursionError,
        MemoryError,
        TypeError,
    ):
        return None
    mx = _mx()
    mask = _mlx_finite_mask(values, target_mlx)
    any_finite = mx.any(mask)
    n = mx.sum(mask.astype(mx.float32))
    x = mx.where(mask, values, mx.array(0.0, dtype=mx.float32))
    y = mx.where(mask, target_mlx, mx.array(0.0, dtype=mx.float32))
    n_safe = mx.maximum(n, mx.array(1.0, dtype=mx.float32))
    x_mean = mx.sum(x) / n_safe
    y_mean = mx.sum(y) / n_safe
    xc = mx.where(mask, values - x_mean, mx.array(0.0, dtype=mx.float32))
    yc = mx.where(mask, target_mlx - y_mean, mx.array(0.0, dtype=mx.float32))
    var_x = mx.sum(xc * xc)
    covariance = mx.sum(xc * yc)
    denominator = mx.sqrt(var_x * mx.sum(yc * yc))
    pr_expr = covariance / mx.maximum(denominator, mx.array(1e-12, dtype=mx.float32))
    slope = covariance / mx.maximum(var_x, mx.array(1e-12, dtype=mx.float32))
    intercept = y_mean - slope * x_mean
    pred = intercept + slope * values
    ss_res = mx.sum(
        mx.where(mask, (target_mlx - pred) ** 2, mx.array(0.0, dtype=mx.float32))
    )
    ss_tot = mx.sum(
        mx.where(mask, (target_mlx - y_mean) ** 2, mx.array(0.0, dtype=mx.float32))
    )
    r2_expr = mx.array(1.0, dtype=mx.float32) - ss_res / mx.maximum(
        ss_tot, mx.array(1e-12, dtype=mx.float32)
    )
    # Keep the whole scalar decision on one device synchronization.  The
    # optional values pull below is intentionally separate because only
    # correlation hits need host-side arrays.
    mx.eval(any_finite, n, var_x, ss_tot, r2_expr, pr_expr)
    if not bool(np.asarray(any_finite)):
        return None
    count = float(np.asarray(n))
    var_x_value = float(np.asarray(var_x))
    ss_tot_value = float(np.asarray(ss_tot))
    r2 = float(np.asarray(r2_expr))
    pr = float(np.asarray(pr_expr))
    if count < 3 or var_x_value <= 0.0 or ss_tot_value <= 0.0:
        return None
    if not np.isfinite(r2) or not np.isfinite(pr):
        return None
    if need_values:
        return r2, pr, mlx_to_numpy(values)
    return r2, pr, None


def score_population_mlx(
    exprs: list[Expr],
    mlx_env: dict[str, Any],
    target_mlx: Any,
) -> list[tuple[float, float, Any] | None]:
    """Score many expressions with grouped, genuinely batched MLX graphs."""
    mx = _mx()
    results: list[tuple[float, float, Any] | None] = [None] * len(exprs)
    if len(exprs) == 1:
        # Preserve the lightweight single-expression path (and its one
        # evaluation hook) for callers that already provide a one-item chunk.
        try:
            values = evaluate_expr_mlx(exprs[0], mlx_env)
        except (
            KeyError,
            ValueError,
            FloatingPointError,
            OverflowError,
            RuntimeError,
            RecursionError,
            MemoryError,
            TypeError,
        ):
            return [None]
        any_finite, n, var_x, ss_tot, pr, r2 = _population_statistics_mlx(
            values[None, :], target_mlx
        )
        mx.eval(any_finite, n, var_x, ss_tot, pr, r2)
        if (
            bool(np.asarray(any_finite)[0])
            and float(np.asarray(n)[0]) >= 3
            and float(np.asarray(var_x)[0]) > 0.0
            and float(np.asarray(ss_tot)[0]) > 0.0
        ):
            pr_value = float(np.asarray(pr)[0])
            r2_value = float(np.asarray(r2)[0])
            if np.isfinite(pr_value) and np.isfinite(r2_value):
                return [(r2_value, pr_value, values)]
        return [None]

    # Compile one-by-one: a single pathological AST must not abort a chunk of
    # hundreds/thousands of candidates in a multi-million expression search.
    compiled: list[CompiledExpr | None] = []
    for expr in exprs:
        try:
            compiled.append(compile_expr(expr))
        except (
            RecursionError,
            MemoryError,
            ValueError,
            OverflowError,
            TypeError,
            RuntimeError,
        ):
            compiled.append(None)

    groups: dict[tuple[str, ...], list[int]] = {}
    for i, tape in enumerate(compiled):
        if tape is None:
            continue
        shape = tuple(opcode for opcode, _payload in tape.instructions)
        groups.setdefault(shape, []).append(i)

    pending: list[Any] = []
    group_meta: list[tuple[list[int], Any, Any, Any, Any, Any, Any, Any]] = []

    for indices in groups.values():
        tapes = [compiled[i] for i in indices]
        assert all(t is not None for t in tapes)
        try:
            values_batch = _evaluate_compiled_population_mlx(tapes, mlx_env)  # type: ignore[arg-type]
        except (
            KeyError,
            ValueError,
            FloatingPointError,
            OverflowError,
            RuntimeError,
            RecursionError,
            MemoryError,
            TypeError,
        ):
            # Group graph failed — fall back to scoring each member alone so one
            # bad shape-mate cannot zero the whole equal-shape batch.
            for i in indices:
                try:
                    scored = score_expression_mlx(exprs[i], mlx_env, target_mlx, need_values=True)
                except (
                    KeyError,
                    ValueError,
                    FloatingPointError,
                    OverflowError,
                    RuntimeError,
                    RecursionError,
                    MemoryError,
                    TypeError,
                ):
                    continue
                if scored is not None:
                    results[i] = scored
            continue
        any_finite, n, var_x, ss_tot, pr, r2 = _population_statistics_mlx(
            values_batch, target_mlx
        )
        pending.extend([any_finite, n, var_x, ss_tot, pr, r2])
        group_meta.append((indices, values_batch, any_finite, n, var_x, ss_tot, pr, r2))

    if not pending:
        return results

    try:
        mx.eval(*pending)
    except (RuntimeError, ValueError, FloatingPointError, OverflowError, MemoryError):
        # Batched eval failed — degrade to per-expression scoring for pending groups.
        for indices, *_rest in group_meta:
            for i in indices:
                if results[i] is not None:
                    continue
                try:
                    scored = score_expression_mlx(exprs[i], mlx_env, target_mlx, need_values=True)
                except (
                    KeyError,
                    ValueError,
                    FloatingPointError,
                    OverflowError,
                    RuntimeError,
                    RecursionError,
                    MemoryError,
                    TypeError,
                ):
                    continue
                if scored is not None:
                    results[i] = scored
        return results

    for indices, values_batch, any_finite, n, var_x, ss_tot, pr, r2 in group_meta:
        finite_arr = np.asarray(any_finite)
        n_arr = np.asarray(n)
        var_arr = np.asarray(var_x)
        total_arr = np.asarray(ss_tot)
        pr_arr = np.asarray(pr)
        r2_arr = np.asarray(r2)
        for row, i in enumerate(indices):
            if (
                bool(finite_arr[row])
                and n_arr[row] >= 3
                and var_arr[row] > 0.0
                and total_arr[row] > 0.0
                and np.isfinite(pr_arr[row])
                and np.isfinite(r2_arr[row])
            ):
                results[i] = (
                    float(r2_arr[row]),
                    float(pr_arr[row]),
                    values_batch[row],
                )
    return results


def _evaluate_compiled_population_mlx(
    tapes: list[CompiledExpr],
    mlx_env: dict[str, Any],
) -> Any:
    """Interpret equal-shape tapes over a leading expression batch axis."""
    if not tapes:
        raise ValueError("Cannot evaluate an empty tape population")
    mx = _mx()
    n_rows = len(next(iter(mlx_env.values()))) if mlx_env else 1
    shape = tuple(opcode for opcode, _payload in tapes[0].instructions)
    if any(
        tuple(opcode for opcode, _payload in tape.instructions) != shape
        for tape in tapes
    ):
        raise ValueError("Population tapes must have the same instruction shape")
    stack: list[Any] = []
    for step, opcode in enumerate(shape):
        payloads = [tape.instructions[step][1] for tape in tapes]
        if opcode == "const":
            values = mx.array(np.asarray(payloads, dtype=np.float32))[:, None]
            stack.append(mx.broadcast_to(values, (len(tapes), n_rows)))
        elif opcode == "var":
            values = [mlx_env[str(payload)] for payload in payloads]
            stack.append(mx.stack(values, axis=0))
        elif opcode in {"neg", "abs", "sqrt", "log", "exp"}:
            if not stack:
                raise ValueError(f"Unary expression {opcode!r} has no operand")
            value = stack.pop()
            if opcode == "neg":
                stack.append(-value)
            elif opcode == "abs":
                stack.append(mx.abs(value))
            elif opcode == "sqrt":
                stack.append(mx.sqrt(mx.maximum(value, 0.0)))
            elif opcode == "log":
                stack.append(mx.log(mx.maximum(value, 1e-30)))
            else:
                stack.append(mx.exp(mx.minimum(mx.maximum(value, -700.0), 700.0)))
        elif opcode in {"add", "sub", "mul", "div", "pow"}:
            if len(stack) < 2:
                raise ValueError(f"Binary expression {opcode!r} has missing operands")
            right = stack.pop()
            left = stack.pop()
            if opcode == "add":
                stack.append(left + right)
            elif opcode == "sub":
                stack.append(left - right)
            elif opcode == "mul":
                stack.append(left * right)
            elif opcode == "div":
                safe = mx.where(
                    mx.abs(right) < 1e-12,
                    mx.full(right.shape, float("nan")),
                    right,
                )
                stack.append(left / safe)
            else:
                stack.append(mx.power(left, right))
        else:
            raise ValueError(f"Unknown expression kind: {opcode!r}")
    if len(stack) != 1:
        raise ValueError("Compiled population did not produce one value")
    return stack[0]


def _population_statistics_mlx(values: Any, target: Any) -> tuple[Any, ...]:
    """Build vectorized score reductions for a [expression, row] array."""
    mx = _mx()
    mask = _mlx_finite_mask(values, target)
    n = mx.sum(mask.astype(mx.float32), axis=1)
    x = mx.where(mask, values, mx.array(0.0, dtype=mx.float32))
    y = mx.where(mask, target, mx.array(0.0, dtype=mx.float32))
    n_safe = mx.maximum(n, mx.array(1.0, dtype=mx.float32))[:, None]
    x_mean = mx.sum(x, axis=1, keepdims=True) / n_safe
    y_mean = mx.sum(y, axis=1, keepdims=True) / n_safe
    xc = mx.where(mask, values - x_mean, mx.array(0.0, dtype=mx.float32))
    yc = mx.where(mask, target - y_mean, mx.array(0.0, dtype=mx.float32))
    var_x = mx.sum(xc * xc, axis=1)
    covariance = mx.sum(xc * yc, axis=1)
    denominator = mx.sqrt(var_x * mx.sum(yc * yc, axis=1))
    pr = covariance / mx.maximum(denominator, mx.array(1e-12, dtype=mx.float32))
    slope = covariance / mx.maximum(var_x, mx.array(1e-12, dtype=mx.float32))
    intercept = y_mean[:, 0] - slope * x_mean[:, 0]
    pred = intercept[:, None] + slope[:, None] * values
    ss_res = mx.sum(
        mx.where(mask, (target - pred) ** 2, mx.array(0.0, dtype=mx.float32)),
        axis=1,
    )
    ss_tot = mx.sum(
        mx.where(mask, (target - y_mean) ** 2, mx.array(0.0, dtype=mx.float32)),
        axis=1,
    )
    r2 = mx.array(1.0, dtype=mx.float32) - ss_res / mx.maximum(
        ss_tot, mx.array(1e-12, dtype=mx.float32)
    )
    any_finite = mx.any(mask, axis=1)
    return any_finite, n, var_x, ss_tot, pr, r2


def evaluate_to_numpy(expr: Expr, env: dict[str, np.ndarray]) -> np.ndarray:
    mlx_env = numpy_env_to_mlx(env)
    return mlx_to_numpy(evaluate_expr_mlx(expr, mlx_env))
