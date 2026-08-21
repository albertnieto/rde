"""Chunked / streaming evaluation of expression candidates (Phase 3 scale-up)."""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from rde.backends.resolve import (
    ExprBackendName,
    GpuBackendError,
    default_expr_backend,
    expr_backend_choices,
    is_gpu_expr_backend,
    resolve_expr_backend,
)
from rde.expression.ast import Expr
from rde.expression.evaluate import evaluate_expr, pearson_r, regression_r_squared

# Re-export for callers that imported EvalBackend from here.
EvalBackend = ExprBackendName

# GPU expression eval pays off only above this row count (host↔device sync dominates below).
GPU_EXPR_MIN_ROWS = 512

ProgressCallback = Callable[[int, int, str], None]

_LOG = logging.getLogger(__name__)


def build_env(rows: list[dict[str, Any]], variables: list[str]) -> dict[str, np.ndarray]:
    """Extract a row-oriented feature table into reusable numeric columns."""
    from rde.analyze.feature_table import FeatureTable

    return FeatureTable.from_rows(rows, columns=variables).env(variables)


def build_env_from_columns(
    columns: dict[str, np.ndarray],
    variables: list[str],
) -> dict[str, np.ndarray]:
    """Build env from pre-extracted column arrays."""
    return {var: np.asarray(columns[var], dtype=float) for var in variables if var in columns}


def normalize_expr_backend(name: str | None) -> ExprBackendName:
    return resolve_expr_backend(name)


def effective_expr_backend(
    backend: ExprBackendName,
    n_rows: int,
    *,
    force_gpu: bool = False,
) -> ExprBackendName:
    """Downgrade GPU backends to NumPy when the batch is too small to amortize sync."""
    if force_gpu or backend == "numpy" or n_rows >= GPU_EXPR_MIN_ROWS:
        return backend
    return "numpy"


def prepare_device_envs(
    env: dict[str, np.ndarray],
    backend: ExprBackendName,
    n_rows: int,
    target: np.ndarray | None = None,
    *,
    force_gpu: bool = False,
    require_gpu: bool = False,
    requested_backend: ExprBackendName | None = None,
) -> tuple[
    ExprBackendName,
    dict[str, Any] | None,
    dict[str, Any] | None,
    Any | None,
    str | None,
]:
    """Build reusable MLX/torch env dicts once per fitness generation."""
    requested = requested_backend or backend
    gpu_locked = force_gpu or require_gpu or is_gpu_expr_backend(requested)
    fallback_reason: str | None = None
    if not gpu_locked and requested != "numpy" and n_rows < GPU_EXPR_MIN_ROWS:
        backend = "numpy"
        fallback_reason = f"n_rows<{GPU_EXPR_MIN_ROWS}"
    else:
        backend = effective_expr_backend(backend, n_rows, force_gpu=gpu_locked)
        if backend != requested and fallback_reason is None and not gpu_locked:
            fallback_reason = f"n_rows<{GPU_EXPR_MIN_ROWS}"
    mlx_env: dict[str, Any] | None = None
    torch_env: dict[str, Any] | None = None
    target_mlx: Any | None = None
    if backend == "mlx":
        from rde.expression.mlx_eval import mlx_target_to_mlx, numpy_env_to_mlx

        try:
            mlx_env = numpy_env_to_mlx(env)
            if target is not None:
                target_mlx = mlx_target_to_mlx(target)
        except RuntimeError as exc:
            if gpu_locked and "metal" in str(exc).lower():
                raise GpuBackendError(
                    f"MLX expression eval failed on Metal (requested={requested!r}, n_rows={n_rows}): {exc}"
                ) from exc
            if "metal" in str(exc).lower():
                backend = "numpy"
                fallback_reason = f"metal_error:{exc}"
            else:
                raise
    elif backend in {"torch", "torch_mps"}:
        from rde.expression.torch_eval import numpy_env_to_torch, torch_device_for_backend

        torch_env = numpy_env_to_torch(
            env,
            device=torch_device_for_backend(backend, strict=gpu_locked),
        )
    if gpu_locked and backend != requested:
        raise GpuBackendError(
            f"GPU expression backend required but effective={backend!r} "
            f"(requested={requested!r}, reason={fallback_reason!r}, n_rows={n_rows})"
        )
    if backend != requested and fallback_reason is None:
        fallback_reason = "unknown"
    if backend != requested and fallback_reason is not None:
        _LOG.warning(
            "expr eval backend fallback: requested=%s effective=%s reason=%s n_rows=%d",
            requested,
            backend,
            fallback_reason,
            n_rows,
        )
    return backend, mlx_env, torch_env, target_mlx, fallback_reason


def _eval_values(expr: Expr, env: dict[str, np.ndarray], *, backend: ExprBackendName) -> np.ndarray:
    if backend == "numpy":
        return evaluate_expr(expr, env)
    if backend == "mlx":
        from rde.expression.mlx_eval import evaluate_to_numpy as mlx_eval_to_numpy

        return mlx_eval_to_numpy(expr, env)
    from rde.expression.torch_eval import evaluate_to_numpy as torch_eval_to_numpy

    return torch_eval_to_numpy(expr, env, backend=backend)


def score_expression(
    expr: Expr,
    env: dict[str, np.ndarray],
    target: np.ndarray,
    *,
    backend: ExprBackendName = "numpy",
    mlx_env: dict[str, Any] | None = None,
    torch_env: dict[str, Any] | None = None,
    target_mlx: Any | None = None,
    need_values: bool = False,
) -> tuple[float, float, np.ndarray] | None:
    """Return (r_squared, pearson_r, values) or None if evaluation fails."""
    try:
        if backend == "mlx" and mlx_env is not None and target_mlx is not None:
            from rde.expression.mlx_eval import score_expression_mlx

            scored = score_expression_mlx(
                expr,
                mlx_env,
                target_mlx,
                need_values=need_values,
            )
            if scored is None:
                return None
            r2, pr, values = scored
            if need_values and values is None:
                from rde.expression.mlx_eval import evaluate_expr_mlx, mlx_to_numpy

                values = mlx_to_numpy(evaluate_expr_mlx(expr, mlx_env))
            elif not need_values:
                values = np.array([], dtype=float)
            return r2, pr, values
        if backend in {"torch", "torch_mps"} and torch_env is not None:
            from rde.backends.torch_mps_backend import _as_numpy
            from rde.expression.torch_eval import evaluate_expr_torch

            values = _as_numpy(evaluate_expr_torch(expr, torch_env))
        else:
            values = _eval_values(expr, env, backend=backend)
    except (
        KeyError,
        ValueError,
        FloatingPointError,
        OverflowError,
        ImportError,
        RuntimeError,
        RecursionError,
        MemoryError,
        TypeError,
        ArithmeticError,
    ):
        return None
    if not np.isfinite(values).any():
        return None
    r2 = regression_r_squared(values, target)
    pr = pearson_r(values, target)
    if not np.isfinite(r2) or not np.isfinite(pr):
        return None
    return r2, pr, values


def score_population(
    candidates: list[Expr],
    env: dict[str, np.ndarray],
    target: np.ndarray,
    *,
    backend: ExprBackendName = "numpy",
    mlx_env: dict[str, Any] | None = None,
    torch_env: dict[str, Any] | None = None,
    target_mlx: Any | None = None,
) -> list[tuple[float, float, Any | None] | None]:
    """Batch-score a population; MLX path uses one mx.eval per generation."""
    if backend == "mlx" and mlx_env is not None and target_mlx is not None:
        from rde.expression.mlx_eval import score_population_mlx

        return score_population_mlx(candidates, mlx_env, target_mlx)
    out: list[tuple[float, float, Any | None] | None] = []
    for expr in candidates:
        scored = score_expression(
            expr,
            env,
            target,
            backend=backend,
            mlx_env=mlx_env,
            torch_env=torch_env,
            target_mlx=target_mlx,
            need_values=False,
        )
        if scored is None:
            out.append(None)
        else:
            r2, pr, _ = scored
            out.append((r2, pr, None))
    return out


def evaluate_chunk(
    candidates: list[Expr],
    env: dict[str, np.ndarray],
    target: np.ndarray,
    *,
    min_abs_r: float = 0.0,
    backend: ExprBackendName = "numpy",
    mlx_env: dict[str, Any] | None = None,
    torch_env: dict[str, Any] | None = None,
    target_mlx: Any | None = None,
    force_gpu: bool = False,
) -> list[tuple[Expr, float, float, np.ndarray]]:
    """Evaluate a chunk of candidates; returns (expr, r2, pearson_r, values) tuples."""
    hits: list[tuple[Expr, float, float, np.ndarray]] = []
    if mlx_env is None and torch_env is None and target_mlx is None:
        backend, mlx_env, torch_env, target_mlx, _ = prepare_device_envs(
            env,
            backend,
            len(target),
            target,
            force_gpu=force_gpu,
            requested_backend=backend,
        )
    if backend == "mlx" and mlx_env is not None and target_mlx is not None:
        # Score the whole chunk in one MLX graph/eval.  Pull expression values
        # back only for candidates that survive the correlation threshold;
        # evaluating every candidate through score_expression_mlx creates a
        # Metal synchronization storm on million-expression searches.
        from rde.expression.mlx_eval import mlx_to_numpy_batch
        from rde.runtime.resilience import SOFT_FAIL_EXCEPTIONS

        try:
            scores = score_population(
                candidates,
                env,
                target,
                backend=backend,
                mlx_env=mlx_env,
                target_mlx=target_mlx,
            )
        except SOFT_FAIL_EXCEPTIONS:
            # Chunk-level failure: degrade to per-expression scoring so one
            # bad candidate cannot discard the rest of the chunk.
            scores = [
                score_expression(
                    expr,
                    env,
                    target,
                    backend=backend,
                    mlx_env=mlx_env,
                    torch_env=torch_env,
                    target_mlx=target_mlx,
                    need_values=True,
                )
                for expr in candidates
            ]
        survivors: list[tuple[Expr, float, float, Any]] = []
        for expr, scored in zip(candidates, scores):
            if scored is None:
                continue
            r2, pr, values_mlx = scored
            if not np.isfinite(r2) or not np.isfinite(pr) or abs(pr) < min_abs_r:
                continue
            if values_mlx is None:
                continue
            survivors.append((expr, r2, pr, values_mlx))
        if not survivors:
            return hits
        host_ready = [item for item in survivors if isinstance(item[3], np.ndarray)]
        device_pending = [item for item in survivors if not isinstance(item[3], np.ndarray)]
        for expr, r2, pr, values in host_ready:
            if np.isfinite(values).any():
                hits.append((expr, r2, pr, values))
        if not device_pending:
            return hits
        try:
            value_matrix = mlx_to_numpy_batch([item[3] for item in device_pending])
        except SOFT_FAIL_EXCEPTIONS:
            # Host transfer failed as a batch — pull survivors one at a time.
            from rde.expression.mlx_eval import mlx_to_numpy

            for expr, r2, pr, values_mlx in device_pending:
                try:
                    values = mlx_to_numpy(values_mlx)
                except SOFT_FAIL_EXCEPTIONS:
                    continue
                if np.isfinite(values).any():
                    hits.append((expr, r2, pr, values))
            return hits
        for (expr, r2, pr, _values_mlx), values in zip(device_pending, value_matrix):
            if np.isfinite(values).any():
                hits.append((expr, r2, pr, values))
        return hits
    for expr in candidates:
        scored = score_expression(
            expr,
            env,
            target,
            backend=backend,
            mlx_env=mlx_env,
            torch_env=torch_env,
            target_mlx=target_mlx,
            need_values=True,
        )
        if scored is None:
            continue
        r2, pr, values = scored
        if abs(pr) >= min_abs_r:
            hits.append((expr, r2, pr, values))
    return hits
