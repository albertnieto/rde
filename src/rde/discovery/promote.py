"""Promote top-ranked conjectures from Phases 3–6 into a single handoff list."""

from __future__ import annotations

from typing import Any

import numpy as np

from rde.discovery.symbolic import SymbolicResult


def _rank_key(record: dict[str, Any]) -> tuple[float, float, float, float, float]:
    """Larger = better (matches ConjectureRanker ordering)."""
    extrap = float(record.get("extrapolation_r_squared", float("nan")))
    extrap_key = extrap if np.isfinite(extrap) else -1.0
    cross_n = float(record.get("cross_n_sign_stability", float("nan")))
    cross_key = cross_n if np.isfinite(cross_n) else -1.0
    r2 = float(record.get("r_squared", 0.0))
    mdl = float(record.get("mdl", 0.0))
    pr = abs(float(record.get("pearson_r", 0.0)))
    return (extrap_key, cross_key, r2, -mdl, pr)


def _symbolic_mdl(sym: SymbolicResult) -> float:
    n_terms = len(sym.coefficients) if sym.coefficients else max(1, sym.equation.count("+") + 1)
    method_penalty = {
        "polynomial_fallback": 0.0,
        "physics_templates": 0.5,
        "native_gp": 1.0,
        "pysr": 1.5,
        "operon": 1.5,
        "ai_feynman": 0.5,
    }.get(sym.method, 1.5)
    return float(n_terms) + method_penalty + 0.01 * len(sym.equation)


def symbolic_to_conjecture(sym: SymbolicResult, target: str) -> dict[str, Any]:
    pr = float(sym.metadata.get("pearson_r", np.sqrt(max(sym.r_squared, 0.0))))
    if not np.isfinite(pr):
        pr = float(np.sqrt(max(sym.r_squared, 0.0)))
    return {
        "source": "symbolic",
        "expression": sym.equation,
        "target": target,
        "method": sym.method,
        "r_squared": sym.r_squared,
        "pearson_r": pr,
        "mdl": _symbolic_mdl(sym),
        "level_hint": 1 if sym.r_squared >= 0.75 else (0 if sym.r_squared >= 0.5 else 0),
        **{k: v for k, v in sym.metadata.items() if k not in {"pearson_r"}},
    }


def latent_interp_to_conjecture(interp: dict[str, Any]) -> dict[str, Any]:
    latent_col = str(interp.get("latent_column", "latent"))
    equation = str(interp.get("equation", ""))
    r2 = float(interp.get("r_squared", 0.0))
    corr = float(interp.get("main_target_correlation", 0.0))
    return {
        "source": "latent_interpretation",
        "expression": f"{latent_col} ≈ {equation}",
        "target": str(interp.get("main_target", "")),
        "latent_column": latent_col,
        "latent_source": interp.get("latent_source"),
        "method": interp.get("method"),
        "r_squared": r2,
        "pearson_r": corr,
        "mdl": float(len(equation) * 0.05 + 3.0),
        "level_hint": 1 if r2 >= 0.75 else 0,
        "main_target_correlation": corr,
        **{
            k: v
            for k, v in interp.items()
            if k
            not in {
                "latent_column",
                "equation",
                "main_target",
                "main_target_correlation",
                "method",
                "r_squared",
            }
        },
    }


def program_to_conjecture(prog: dict[str, Any], target: str) -> dict[str, Any]:
    fitness = float(prog.get("fitness", 0.0))
    pr = float(prog.get("pearson_r", 0.0))
    expr = str(prog.get("expression", ""))
    return {
        "source": "gp",
        "expression": expr,
        "target": target,
        "r_squared": fitness,
        "pearson_r": pr,
        "mdl": float(len(expr) * 0.05 + 2.0),
        "level_hint": 1 if fitness >= 0.75 else 0,
        "generation": prog.get("generation"),
    }


def promote_top_conjectures(
    *,
    target: str,
    metric_candidates: list[dict[str, Any]],
    symbolic: dict[str, Any] | None = None,
    latent_interpretations: list[dict[str, Any]] | None = None,
    programs: list[dict[str, Any]] | None = None,
    max_results: int = 10,
    min_r_squared: float = 0.5,
    leak_audit: bool = True,
    domain_id: str | None = None,
    rejected: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Merge Phase 3–6 hits; return top candidates and optionally record rejects."""
    pool: list[dict[str, Any]] = []

    for rec in metric_candidates:
        candidate = {"source": "metric_dsl", **rec}
        if float(rec.get("r_squared", 0.0)) >= min_r_squared:
            pool.append(candidate)
        elif rejected is not None:
            rejected.append(
                {
                    **candidate,
                    "promotion_rejected": True,
                    "rejection_reason": "below_min_r_squared",
                    "min_r_squared": min_r_squared,
                }
            )

    sym = symbolic or {}
    if sym.get("equation") and float(sym.get("r_squared", 0.0)) >= min_r_squared:
        sym_result = SymbolicResult(
            method=str(sym.get("method", "polynomial_fallback")),
            equation=str(sym.get("equation", "")),
            r_squared=float(sym.get("r_squared", 0.0)),
            coefficients=sym.get("coefficients", {}),
            metadata={k: v for k, v in sym.items() if k not in {"method", "equation", "r_squared", "coefficients", "latent_interpretation", "latent_interpretations"}},
        )
        pool.append(symbolic_to_conjecture(sym_result, target))
    elif sym.get("equation") and rejected is not None:
        rejected.append(
            {
                "source": "symbolic",
                **sym,
                "promotion_rejected": True,
                "rejection_reason": "below_min_r_squared",
                "min_r_squared": min_r_squared,
            }
        )

    for interp in latent_interpretations or []:
        if interp and float(interp.get("r_squared", 0.0)) >= min_r_squared:
            pool.append(latent_interp_to_conjecture(interp))
        elif interp and rejected is not None:
            rejected.append(
                {
                    "source": "latent_interpretation",
                    **interp,
                    "promotion_rejected": True,
                    "rejection_reason": "below_min_r_squared",
                    "min_r_squared": min_r_squared,
                }
            )

    latent_single = (sym or {}).get("latent_interpretation")
    if latent_single and float(latent_single.get("r_squared", 0.0)) >= min_r_squared:
        if not any(
            c.get("latent_column") == latent_single.get("latent_column")
            for c in pool
            if c.get("source") == "latent_interpretation"
        ):
            pool.append(latent_interp_to_conjecture(latent_single))
    elif latent_single and rejected is not None:
        rejected.append(
            {
                "source": "latent_interpretation",
                **latent_single,
                "promotion_rejected": True,
                "rejection_reason": "below_min_r_squared",
                "min_r_squared": min_r_squared,
            }
        )

    for prog in programs or []:
        if float(prog.get("fitness", 0.0)) >= min_r_squared:
            pool.append(program_to_conjecture(prog, target))
        elif rejected is not None:
            rejected.append(
                {
                    "source": "gp",
                    **prog,
                    "promotion_rejected": True,
                    "rejection_reason": "below_min_r_squared",
                    "min_r_squared": min_r_squared,
                }
            )

    pool.sort(key=_rank_key, reverse=True)
    seen: set[str] = set()
    promoted: list[dict[str, Any]] = []
    for rec in pool:
        key = f"{rec.get('source')}::{rec.get('expression')}"
        if key in seen:
            continue
        seen.add(key)
        promoted.append(rec)
        if len(promoted) >= max_results:
            break
    if leak_audit and promoted:
        from rde.analyze.leak_audit import audit_conjectures

        if domain_id is None:
            raise ValueError("domain_id is required when leak_audit=True")
        promoted, _summary = audit_conjectures(
            promoted,
            target=target,
            domain_id=domain_id,
        )
        promoted = promoted[:max_results]
    return promoted
