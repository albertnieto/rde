"""Cross-N generative rule discovery.

Existing cross-N helpers (`analyze.query.cross_n_sign_stability`,
`analyze.ranker.extrapolation_r_squared`) check whether a *fixed* candidate
formula stays correlated / sign-stable / extrapolates as N grows. This
module asks a different question: for a candidate ``target ~ slope*F +
intercept`` fit independently at each size, how does ``slope`` itself
change as a function of N? The answer is found by recursively handing the
resulting ``(N, slope)`` pairs to RDE's existing symbolic-regression engine
(`discovery.symbolic.discover_equation`) — no new equation search is
implemented here, only the per-size refit and the recursive call.

Domain-agnostic by construction: operates on ``rows``/``target``/candidate
expression strings only, the same contract every other RDE analysis
function uses. Nothing here assumes a particular domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rde.analyze.feature_table import FeatureTable
from rde.analyze.tables import group_indices_by_size, numeric_columns
from rde.discovery.symbolic import discover_equation
from rde.expression.batch import (
    normalize_expr_backend,
    prepare_device_envs,
    score_population,
)
from rde.expression.evaluate import evaluate_expr, regression_r_squared
from rde.expression.parse import parse_expr

# "latent_interpretation" candidates (discovery/promote.py) are display
# strings like "latent.pca_0 ≈ 2*x + 1" referencing latent.* columns that
# never exist in `rows` (latent codes live only in Phase 4's in-memory
# arrays) — never parseable, so skip them before ever trying.
_EVALUABLE_SOURCES = {"metric_dsl", "symbolic", "gp"}


@dataclass
class PerSizeFit:
    size: int
    n_rows: int
    slope: float
    intercept: float
    r_squared: float


@dataclass
class CrossNRuleResult:
    candidate_expression: str
    candidate_source: str
    status: str  # "ok" | "insufficient_sizes" | "eval_failed"
    per_size_fits: list[PerSizeFit] = field(default_factory=list)
    slope_rule_equation: str | None = None
    slope_rule_r_squared: float | None = None
    slope_rule_method: str | None = None
    stable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_expression": self.candidate_expression,
            "candidate_source": self.candidate_source,
            "status": self.status,
            "per_size_fits": [
                {
                    "size": f.size,
                    "n_rows": f.n_rows,
                    "slope": f.slope,
                    "intercept": f.intercept,
                    "r_squared": f.r_squared,
                }
                for f in self.per_size_fits
            ],
            "slope_rule": (
                {
                    "equation": self.slope_rule_equation,
                    "r_squared": self.slope_rule_r_squared,
                    "method": self.slope_rule_method,
                }
                if self.slope_rule_equation is not None
                else None
            ),
            "stable": self.stable,
        }


def _fit_per_size(
    values: np.ndarray,
    target_arr: np.ndarray,
    groups: dict[Any, np.ndarray],
    *,
    min_rows_per_size: int,
) -> list[PerSizeFit]:
    fits: list[PerSizeFit] = []
    for size, idxs in sorted(groups.items(), key=lambda kv: kv[0]):
        if len(idxs) < min_rows_per_size:
            continue
        xv = values[idxs]
        yv = target_arr[idxs]
        mask = np.isfinite(xv) & np.isfinite(yv)
        if int(mask.sum()) < min_rows_per_size:
            continue
        xv, yv = xv[mask], yv[mask]
        if float(np.std(xv)) == 0.0:
            continue
        slope, intercept = np.polyfit(xv, yv, 1)
        r2 = regression_r_squared(xv, yv)
        if not np.isfinite(r2):
            continue
        fits.append(
            PerSizeFit(
                size=int(size),
                n_rows=int(mask.sum()),
                slope=float(slope),
                intercept=float(intercept),
                r_squared=float(r2),
            )
        )
    return fits


def fit_cross_n_generative_rule(
    rows: list[dict[str, Any]],
    target: str,
    candidates: list[dict[str, Any]],
    *,
    min_sizes: int = 3,
    min_rows_per_size: int = 4,
    min_per_size_r_squared: float = 0.5,
    max_candidates: int = 5,
    use_pysr: bool = False,
    use_operon: bool = False,
) -> list[CrossNRuleResult]:
    """For each of the top candidates, fit ``target ~ slope*F + intercept``
    per size, then symbolic-regress the ``(N, slope)`` pairs to discover how
    the coefficient scales with N.

    ``candidates`` is expected to be an already-ranked list of promoted
    conjecture records (``discovery.promote``'s output shape: dicts with
    ``source``/``expression`` keys) — the first ``max_candidates`` entries
    are used as-is, on the assumption the caller already sorted them best
    first.
    """
    if not rows or not candidates:
        return []

    variables = numeric_columns(rows)
    table = FeatureTable.from_rows(rows, columns=[*variables, target])
    env = table.env(variables)
    target_arr = table.column(target)
    groups = group_indices_by_size(rows)

    results: list[CrossNRuleResult] = []
    considered = [c for c in candidates if c.get("source") in _EVALUABLE_SOURCES][:max_candidates]
    parsed: list[tuple[dict[str, Any], str, str, Any]] = []
    for cand in considered:
        expr_str = str(cand.get("expression", ""))
        source = str(cand.get("source", ""))
        try:
            tree = parse_expr(expr_str)
        except (ValueError, KeyError):
            results.append(
                CrossNRuleResult(candidate_expression=expr_str, candidate_source=source, status="eval_failed")
            )
            continue
        parsed.append((cand, expr_str, source, tree))

    if not parsed:
        return results

    requested_backend = normalize_expr_backend(None)
    backend, mlx_env, torch_env, target_mlx, _ = prepare_device_envs(
        env,
        requested_backend,
        len(rows),
        target_arr,
        requested_backend=requested_backend,
    )
    values_by_index: list[np.ndarray | None] = [None] * len(parsed)
    if backend == "mlx" and mlx_env is not None and target_mlx is not None:
        from rde.expression.mlx_eval import mlx_to_numpy_batch

        scores = score_population(
            [tree for _cand, _expr, _source, tree in parsed],
            env,
            target_arr,
            backend=backend,
            mlx_env=mlx_env,
            torch_env=torch_env,
            target_mlx=target_mlx,
        )
        valid = [
            (index, scored[2])
            for index, scored in enumerate(scores)
            if scored is not None and scored[2] is not None
        ]
        if valid:
            value_matrix = mlx_to_numpy_batch([values for _index, values in valid])
            for (index, _values), values in zip(valid, value_matrix):
                values_by_index[index] = values
    else:
        for index, (_cand, _expr, _source, tree) in enumerate(parsed):
            try:
                values_by_index[index] = np.asarray(evaluate_expr(tree, env), dtype=float)
            except (ValueError, KeyError):
                values_by_index[index] = None

    for index, (_cand, expr_str, source, _tree) in enumerate(parsed):
        values = values_by_index[index]
        if values is None:
            results.append(
                CrossNRuleResult(
                    candidate_expression=expr_str,
                    candidate_source=source,
                    status="eval_failed",
                )
            )
            continue

        per_size_fits = _fit_per_size(values, target_arr, groups, min_rows_per_size=min_rows_per_size)
        if len(per_size_fits) < min_sizes:
            results.append(
                CrossNRuleResult(
                    candidate_expression=expr_str,
                    candidate_source=source,
                    status="insufficient_sizes",
                    per_size_fits=per_size_fits,
                )
            )
            continue

        meta_rows = [{"N": float(f.size), "coef": f.slope} for f in per_size_fits]
        meta_result = discover_equation(
            meta_rows,
            "coef",
            features=["N"],
            use_pysr=use_pysr,
            use_operon=use_operon,
            use_physics_templates=True,
            use_native_gp=True,
        )
        stable = all(f.r_squared >= min_per_size_r_squared for f in per_size_fits)
        results.append(
            CrossNRuleResult(
                candidate_expression=expr_str,
                candidate_source=source,
                status="ok",
                per_size_fits=per_size_fits,
                slope_rule_equation=meta_result.equation,
                slope_rule_r_squared=meta_result.r_squared,
                slope_rule_method=meta_result.method,
                stable=stable,
            )
        )
    return results
