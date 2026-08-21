"""Expression DSL for auto-invented metrics."""

from rde.backends.resolve import default_expr_backend, expr_backend_choices
from rde.expression.ast import Expr
from rde.expression.batch import (
    EvalBackend,
    build_env,
    evaluate_chunk,
    normalize_expr_backend,
    score_expression,
)
from rde.expression.enumerate import enumerate_expressions
from rde.expression.evaluate import evaluate_expr, pearson_r, predictive_r_squared, regression_r_squared, r_squared

__all__ = [
    "Expr",
    "EvalBackend",
    "build_env",
    "default_expr_backend",
    "enumerate_expressions",
    "evaluate_chunk",
    "evaluate_expr",
    "expr_backend_choices",
    "normalize_expr_backend",
    "pearson_r",
    "predictive_r_squared",
    "regression_r_squared",
    "r_squared",
    "score_expression",
]
