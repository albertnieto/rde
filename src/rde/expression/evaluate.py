"""Evaluate expression trees on descriptor columns."""

from __future__ import annotations

import math

import numpy as np

from rde.expression.ast import Expr
from rde.expression.dispatch import evaluate_expr_with_backend


class _NumpyExprBackend:
    def env_len(self, env: dict[str, np.ndarray]) -> int:
        return len(next(iter(env.values()))) if env else 1

    def const(self, value: float, n: int) -> np.ndarray:
        return np.full(n, value)

    def var(self, key: str, env: dict[str, np.ndarray]) -> np.ndarray:
        if key not in env:
            raise KeyError(f"Unknown variable: {key!r}")
        return np.asarray(env[key], dtype=float)

    def neg(self, x: np.ndarray) -> np.ndarray:
        return -x

    def abs(self, x: np.ndarray) -> np.ndarray:
        return np.abs(x)

    def sqrt(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(x, 0.0))

    def log(self, x: np.ndarray) -> np.ndarray:
        return np.log(np.maximum(x, 1e-300))

    def exp(self, x: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(x, -700, 700))

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    def sub(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - b

    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Expression search can multiply two already-huge finite values
        # (e.g. clipped exp outputs ~1e304); keep IEEE inf without warning,
        # matching pow()'s overflow policy.
        with np.errstate(over="ignore", invalid="ignore"):
            return np.multiply(a, b)

    def div(self, num: np.ndarray, denom: np.ndarray) -> np.ndarray:
        return num / np.where(np.abs(denom) < 1e-12, np.nan, denom)

    def pow(self, base: np.ndarray, expn: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", over="ignore"):
            return np.power(base, expn)


_NUMPY_BACKEND = _NumpyExprBackend()


def evaluate_expr(expr: Expr, env: dict[str, np.ndarray]) -> np.ndarray:
    """Vectorized evaluation; env maps variable names to equal-length arrays."""
    return evaluate_expr_with_backend(expr, env, _NUMPY_BACKEND)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    from rde.features.numeric import safe_pearson_r

    return safe_pearson_r(x, y, min_count=3)


def regression_r_squared(x: np.ndarray, y: np.ndarray) -> float:
    """Ordinary least-squares R² (not squared Pearson correlation)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    xv = np.asarray(x[mask], dtype=float)
    yv = np.asarray(y[mask], dtype=float)
    # Expression search can emit huge magnitudes; reject before std/lstsq warns.
    if not np.all(np.abs(xv) < 1e100) or not np.all(np.abs(yv) < 1e100):
        return float("nan")
    with np.errstate(over="ignore", invalid="ignore"):
        xstd = float(np.std(xv))
    if not np.isfinite(xstd) or xstd == 0.0:
        return float("nan")
    design = np.column_stack([xv, np.ones_like(xv)])
    try:
        coef, _, rank, _ = np.linalg.lstsq(design, yv, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan")
    if int(rank) < 2:
        return float("nan")
    pred = design @ coef
    if not np.all(np.isfinite(pred)):
        return float("nan")
    ss_res = float(np.sum((yv - pred) ** 2))
    ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
    if not np.isfinite(ss_res) or not np.isfinite(ss_tot) or ss_tot <= 0.0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def r_squared(x: np.ndarray, y: np.ndarray) -> float:
    r = pearson_r(x, y)
    return float("nan") if math.isnan(r) else float(r * r)


def predictive_r_squared(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Held-out R²: compare fixed predictions to actual targets (no refit)."""
    mask = np.isfinite(predicted) & np.isfinite(actual)
    if int(mask.sum()) < 3:
        return float("nan")
    pred = np.asarray(predicted[mask], dtype=float)
    actual_vals = np.asarray(actual[mask], dtype=float)
    if not np.all(np.abs(pred) < 1e100) or not np.all(np.abs(actual_vals) < 1e100):
        return float("nan")
    ss_res = float(np.sum((actual_vals - pred) ** 2))
    ss_tot = float(np.sum((actual_vals - np.mean(actual_vals)) ** 2))
    if not np.isfinite(ss_res) or not np.isfinite(ss_tot) or ss_tot <= 0.0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)
