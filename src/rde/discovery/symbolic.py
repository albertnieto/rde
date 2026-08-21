"""Symbolic equation search (Phase 5) — PySR / Operon optional, polynomial fallback."""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from rde.analyze.query import correlate_with_target, numeric_columns
from rde.analyze.ranker import extrapolation_r_squared
from rde.expression.evaluate import regression_r_squared
from rde.features.numeric import safe_pearson_r
from rde.runtime.heartbeat import ProgressCallback, StageHeartbeat

_LOG = logging.getLogger(__name__)


@dataclass
class LatentSource:
    """One Phase-4 latent coordinate system to interpret in closed form."""

    codes: np.ndarray
    columns: list[str]
    source: str = "latent"


@dataclass
class SymbolicDiscoveryResult:
    """Target fit plus Phase 4→5 latent interpretations."""

    target_fit: SymbolicResult
    latent_interpretations: list[dict[str, Any]] = field(default_factory=list)
    backends: dict[str, Any] = field(default_factory=dict)


def _ensure_operon_dyld() -> None:
    """Repair pyoperon wheels linked against CI @rpath on macOS."""
    if sys.platform != "darwin":
        return
    import os
    import subprocess
    from pathlib import Path

    spec = importlib.util.find_spec("pyoperon")
    if spec is None or not spec.origin:
        return
    pkg_dir = Path(spec.origin).resolve().parent
    so_candidates = sorted(pkg_dir.glob("pyoperon*.so"))
    if not so_candidates:
        return
    so_path = so_candidates[0]
    marker = pkg_dir / ".operon_dylib_repaired"
    if marker.exists():
        return

    replacements: dict[str, tuple[str, ...]] = {
        "@rpath/libz.1.dylib": (
            "/opt/homebrew/opt/zlib/lib/libz.1.dylib",
            "/usr/local/opt/zlib/lib/libz.1.dylib",
        ),
        "@rpath/libzstd.1.dylib": (
            "/opt/homebrew/lib/libzstd.1.dylib",
            "/usr/local/lib/libzstd.1.dylib",
        ),
        "@rpath/libc++.1.dylib": ("/usr/lib/libc++.1.dylib",),
    }
    try:
        linked = subprocess.check_output(["otool", "-L", str(so_path)], text=True)
    except (OSError, subprocess.CalledProcessError):
        return

    for old, candidates in replacements.items():
        if old not in linked:
            continue
        for cand in candidates:
            if not Path(cand).exists():
                continue
            try:
                subprocess.run(
                    ["install_name_tool", "-change", old, cand, str(so_path)],
                    check=True,
                    capture_output=True,
                )
                break
            except (OSError, subprocess.CalledProcessError):
                continue

    try:
        subprocess.run(
            ["codesign", "--force", "--sign", "-", str(so_path)],
            check=True,
            capture_output=True,
        )
        marker.write_text("repaired\n", encoding="utf-8")
    except (OSError, subprocess.CalledProcessError):
        pass

    import ctypes.util

    libz = ctypes.util.find_library("z")
    if libz:
        lib_dir = os.path.dirname(libz)
        fallback = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        parts = [p for p in fallback.split(":") if p]
        if lib_dir not in parts:
            os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
                f"{lib_dir}:{fallback}" if fallback else lib_dir
            )


def _backend_probe(module: str, *, import_probe: bool = True) -> tuple[bool, str | None]:
    if importlib.util.find_spec(module) is None:
        return False, "not installed (pip install rde[symbolic])"
    if not import_probe:
        return True, None
    try:
        if module == "pyoperon":
            _ensure_operon_dyld()
        __import__(module)
    except Exception as exc:  # noqa: BLE001 — report load failures to caller
        hint = None
        if module == "pyoperon" and "libz" in str(exc).lower():
            hint = (
                "pyoperon wheel missing bundled dylibs on this platform; "
                "native_gp and physics_templates remain available"
            )
        return False, hint or str(exc)
    return True, None


def symbolic_backends() -> dict[str, Any]:
    """Report which symbolic-regression backends are usable."""
    pysr_ok, pysr_err = _backend_probe("pysr", import_probe=False)
    operon_ok, operon_err = _backend_probe("pyoperon")
    ai_ok, ai_err = _backend_probe("aifeynman", import_probe=False)
    return {
        "polynomial_fallback": True,
        "physics_templates": True,
        "native_gp": True,
        "pysr": pysr_ok,
        "operon": operon_ok,
        "ai_feynman": ai_ok,
        "errors": {
            k: v
            for k, v in (
                ("pysr", pysr_err),
                ("operon", operon_err),
                ("ai_feynman", ai_err),
            )
            if v is not None
        },
        "install_hint": "pip install -e '.[rde-symbolic]'",
    }


def _record_engine_skip(
    skipped: list[dict[str, str]],
    engine: str,
    reason: str,
    *,
    on_stage: Callable[[str], None] | None = None,
) -> None:
    _LOG.warning("symbolic engine skipped: %s — %s", engine, reason)
    if on_stage is not None:
        on_stage(f"{engine} unavailable: {reason}")
    skipped.append({"engine": engine, "reason": reason})


@dataclass
class SymbolicResult:
    method: str
    equation: str
    r_squared: float
    coefficients: dict[str, float]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SymbolicEngineOutcome:
    """Result and explicit reason when an optional engine cannot fit."""

    result: SymbolicResult | None
    reason: str | None = None


def _engine_outcome(
    result: SymbolicResult | None,
    reason: str | None,
    *,
    return_outcome: bool,
) -> SymbolicResult | SymbolicEngineOutcome | None:
    if return_outcome:
        return SymbolicEngineOutcome(result=result, reason=reason)
    return result


def _finite_rows_reason(
    mask: np.ndarray,
    features: list[str],
    engine: str,
) -> str | None:
    n_rows = int(mask.sum())
    if n_rows >= 10:
        return None
    reason = f"insufficient_finite_rows:n={n_rows};features={len(features)}"
    _LOG.warning("%s skipped: %s", engine, reason)
    return reason


def _run_symbolic_engine_job(
    job: tuple[str, list[dict[str, Any]], str, list[str], dict[str, Any]],
) -> tuple[str, SymbolicEngineOutcome]:
    """Process-pool entry point for independent CPU/GPU symbolic engines."""
    engine, rows, target, features, kwargs = job
    kwargs = {**kwargs, "_return_outcome": True}
    if engine == "ai_feynman":
        outcome = _ai_feynman_fit(rows, target, features, **kwargs)
    elif engine == "pysr":
        outcome = _pysr_fit(rows, target, features, **kwargs)
    elif engine == "operon":
        outcome = _operon_fit(rows, target, features, **kwargs)
    elif engine == "native_gp":
        outcome = _native_gp_fit(rows, target, features, **kwargs)
    else:
        raise ValueError(f"Unknown symbolic engine: {engine!r}")
    if not isinstance(outcome, SymbolicEngineOutcome):
        outcome = SymbolicEngineOutcome(
            result=outcome,
            reason=None if outcome is not None else "fit returned no result",
        )
    return engine, outcome


def _symbolic_parallel_enabled(
    *,
    gp_generations: int,
    gp_population: int,
    pysr_iterations: int,
    pysr_populations: int,
    operon_generations: int,
    operon_population: int,
) -> bool:
    """Enable process overlap automatically only for genuinely large budgets."""
    setting = os.environ.get("RDE_SYMBOLIC_PARALLEL", "auto").lower()
    if setting in {"0", "false", "no", "off"}:
        return False
    if setting in {"1", "true", "yes", "on"}:
        return True
    budget = max(
        gp_generations * gp_population,
        pysr_iterations * pysr_populations,
        operon_generations * operon_population,
    )
    return budget >= 100_000


def _feature_matrix(
    rows: list[dict[str, Any]],
    target: str,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from rde.analyze.feature_table import FeatureTable

    table_columns = list(features)
    if target not in table_columns:
        table_columns.append(target)
    table = FeatureTable.from_rows(rows, columns=table_columns)
    X = table.data[:, : len(features)]
    y = table.column(target)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X, y, mask


def _pearson_r(y: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(pred)
    if mask.sum() < 3:
        return float("nan")
    yv, pv = y[mask], pred[mask]
    if np.std(yv) <= 0 or np.std(pv) <= 0:
        return float("nan")
    return safe_pearson_r(yv, pv)


def _symbolic_mdl(method: str, equation: str, n_coefficients: int = 0) -> float:
    n_terms = n_coefficients if n_coefficients else max(1, equation.count("+") + 1)
    penalty = {
        "polynomial_fallback": 0.0,
        "physics_templates": 0.5,
        "native_gp": 1.0,
        "pysr": 1.5,
        "operon": 1.5,
        "ai_feynman": 0.5,
    }.get(method, 1.5)
    return float(n_terms) + penalty + 0.01 * len(equation)


def _finalize_symbolic_result(
    result: SymbolicResult,
    rows: list[dict[str, Any]],
    y: np.ndarray,
    pred: np.ndarray,
) -> SymbolicResult:
    pr = _pearson_r(y, pred)
    if np.isfinite(pr):
        result.metadata["pearson_r"] = pr
    result.metadata["mdl"] = _symbolic_mdl(
        result.method,
        result.equation,
        len(result.coefficients),
    )
    return _attach_extrapolation(result, rows, y, pred)


def _polynomial_fit(
    rows: list[dict[str, Any]],
    target: str,
    features: list[str],
    degree: int = 2,
) -> SymbolicResult:
    X, y, mask = _feature_matrix(rows, target, features)
    if mask.sum() < X.shape[1] + 2:
        return SymbolicResult("polynomial_fallback", "insufficient_data", 0.0, {}, {})
    X_m = X[mask]
    y_m = y[mask]
    parts = [np.ones((X_m.shape[0], 1)), X_m]
    names = ["1"] + features
    if degree >= 2 and X_m.shape[1] >= 2:
        for i in range(X_m.shape[1]):
            for j in range(i, X_m.shape[1]):
                parts.append((X_m[:, i] * X_m[:, j]).reshape(-1, 1))
                names.append(f"{features[i]}*{features[j]}")
    design = np.hstack(parts)
    coef, _, _, _ = np.linalg.lstsq(design, y_m, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((y_m - pred) ** 2))
    ss_tot = float(np.sum((y_m - y_m.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    terms = [f"{c:.4g}*{n}" for c, n in zip(coef, names) if abs(c) > 1e-10]
    eq = " + ".join(terms) if terms else "0"
    full_pred = np.full_like(y, np.nan)
    full_pred[mask] = pred
    metadata: dict[str, Any] = {"degree": degree, "n_features": len(features)}
    result = SymbolicResult(
        "polynomial_fallback",
        eq,
        float(r2),
        {n: float(c) for c, n in zip(coef, names)},
        metadata,
    )
    return _finalize_symbolic_result(result, rows, y, full_pred)


_PYSR_NAME_RE = re.compile(r"[^A-Za-z0-9_]+")


def _pysr_sanitize_variable_names(
    features: list[str],
) -> tuple[list[str], dict[str, str]]:
    """Map RDE dotted feature names to PySR-safe identifiers (alnum + underscore)."""
    safe_names: list[str] = []
    safe_to_original: dict[str, str] = {}
    used: set[str] = set()
    for index, feature in enumerate(features):
        base = _PYSR_NAME_RE.sub("_", feature)
        base = re.sub(r"_+", "_", base).strip("_")
        if not base or base[0].isdigit():
            base = f"v_{base}" if base else "v"
        candidate = base
        suffix = 0
        while candidate in used:
            suffix += 1
            candidate = f"{base}_{suffix}"
        used.add(candidate)
        safe_names.append(candidate)
        safe_to_original[candidate] = feature
    return safe_names, safe_to_original


def _pysr_restore_equation(equation: str, safe_to_original: dict[str, str]) -> str:
    """Rewrite PySR's sanitized identifiers back to canonical RDE feature names."""
    restored = equation
    for safe in sorted(safe_to_original, key=len, reverse=True):
        original = safe_to_original[safe]
        restored = re.sub(rf"\b{re.escape(safe)}\b", original, restored)
    return restored


def _attach_extrapolation(
    result: SymbolicResult,
    rows: list[dict[str, Any]],
    y: np.ndarray,
    pred: np.ndarray,
) -> SymbolicResult:
    if len({row.get("size") for row in rows if row.get("size") is not None}) >= 2:
        result.metadata["extrapolation_r_squared"] = extrapolation_r_squared(rows, pred, y)
    return result


def _pysr_fit(
    rows: list[dict[str, Any]],
    target: str,
    features: list[str],
    *,
    niterations: int = 40,
    maxsize: int = 20,
    populations: int = 15,
    _return_outcome: bool = False,
) -> SymbolicResult | SymbolicEngineOutcome | None:
    try:
        from pysr import PySRRegressor
    except Exception as exc:  # noqa: BLE001 — optional backend import
        _LOG.warning("pysr import failed: %s", exc)
        return _engine_outcome(
            None,
            f"import_failed:{type(exc).__name__}: {str(exc)[:160]}",
            return_outcome=_return_outcome,
        )
    X, y, mask = _feature_matrix(rows, target, features)
    reason = _finite_rows_reason(mask, features, "pysr")
    if reason is not None:
        return _engine_outcome(None, reason, return_outcome=_return_outcome)
    safe_features, safe_to_original = _pysr_sanitize_variable_names(features)
    try:
        model = PySRRegressor(
            niterations=niterations,
            populations=populations,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp", "log", "sqrt", "sin", "cos"],
            maxsize=maxsize,
            progress=False,
            verbosity=0,
        )
        model.fit(X[mask], y[mask], variable_names=safe_features)
        best = model.get_best()
        eq = _pysr_restore_equation(str(best["equation"]), safe_to_original)
        pred = np.full_like(y, np.nan)
        pred[mask] = model.predict(X[mask])
        r2 = regression_r_squared(pred[mask], y[mask])
        metadata: dict[str, Any] = {
            "sympy_format": eq,
            "pysr_loss": float(best.get("loss", float("nan"))),
            "pysr_variable_names": safe_features,
            "pysr_feature_columns": features,
        }
        result = SymbolicResult("pysr", eq, float(r2), {}, metadata)
        return _engine_outcome(
            _finalize_symbolic_result(result, rows, y, pred),
            None,
            return_outcome=_return_outcome,
        )
    except Exception as exc:
        _LOG.warning("pysr fit failed: %s", exc)
        return _engine_outcome(
            None,
            f"fit_exception:{type(exc).__name__}: {str(exc)[:160]}",
            return_outcome=_return_outcome,
        )


def _operon_fit(
    rows: list[dict[str, Any]],
    target: str,
    features: list[str],
    *,
    generations: int = 50,
    population_size: int = 300,
    seed: int = 0,
    _return_outcome: bool = False,
) -> SymbolicResult | SymbolicEngineOutcome | None:
    try:
        from pyoperon.sklearn import SymbolicRegressor
    except Exception as exc:  # noqa: BLE001 — optional backend import
        _LOG.warning("pyoperon import failed: %s", exc)
        return _engine_outcome(
            None,
            f"import_failed:{type(exc).__name__}: {str(exc)[:160]}",
            return_outcome=_return_outcome,
        )
    X, y, mask = _feature_matrix(rows, target, features)
    reason = _finite_rows_reason(mask, features, "operon")
    if reason is not None:
        return _engine_outcome(None, reason, return_outcome=_return_outcome)
    try:
        reg = SymbolicRegressor(
            allowed_symbols="add,sub,mul,div,constant,variable,sin,cos,exp,log,sqrt",
            generations=generations,
            population_size=population_size,
            pool_size=population_size,
            max_evaluations=max(5000, generations * population_size),
            crossover_probability=1.0,
            mutation_probability=0.25,
            random_state=seed,
            reinserter="keep-best",
        )
        reg.fit(X[mask], y[mask])
        pred = np.full_like(y, np.nan)
        pred[mask] = reg.predict(X[mask])
        r2 = regression_r_squared(pred[mask], y[mask])
        eq = "unknown"
        if getattr(reg, "pareto_front_", None):
            best = max(reg.pareto_front_, key=lambda s: float(s["objective_values"][0]))
            eq = reg.get_model_string(best["tree"], 10)
        metadata: dict[str, Any] = {"generations": generations, "population_size": population_size}
        result = SymbolicResult("operon", eq, float(r2), {}, metadata)
        return _engine_outcome(
            _finalize_symbolic_result(result, rows, y, pred),
            None,
            return_outcome=_return_outcome,
        )
    except Exception as exc:
        _LOG.warning("operon fit failed: %s", exc)
        return _engine_outcome(
            None,
            f"fit_exception:{type(exc).__name__}: {str(exc)[:160]}",
            return_outcome=_return_outcome,
        )


def _native_gp_fit(
    rows: list[dict[str, Any]],
    target: str,
    features: list[str],
    *,
    generations: int = 20,
    population: int = 48,
    seed: int = 0,
    eval_backend: str | None = None,
    require_gpu: bool = False,
    on_progress: ProgressCallback | None = None,
    _return_outcome: bool = False,
) -> SymbolicResult | SymbolicEngineOutcome | None:
    """Pure-Python GP fallback when pyoperon is unavailable."""
    from rde.discovery.programs import evolve_programs

    if len(features) < 1:
        return _engine_outcome(
            None,
            "no_features",
            return_outcome=_return_outcome,
        )
    _, y, mask = _feature_matrix(rows, target, features)
    reason = _finite_rows_reason(mask, features, "native_gp")
    if reason is not None:
        return _engine_outcome(None, reason, return_outcome=_return_outcome)
    try:
        progs = evolve_programs(
            rows,
            target,
            features,
            generations=generations,
            population=population,
            seed=seed,
            top_k=1,
            eval_backend=eval_backend,
            require_gpu=require_gpu,
            on_progress=on_progress,
        )
        if not progs:
            return _engine_outcome(
                None,
                "empty_program_population",
                return_outcome=_return_outcome,
            )
        best = progs[0]
        r2 = float(best.fitness)
        metadata: dict[str, Any] = {
            "generations": generations,
            "population_size": population,
            "pearson_r": best.pearson_r,
            "mdl": _symbolic_mdl("native_gp", best.expression, 0),
        }
        return _engine_outcome(
            SymbolicResult("native_gp", best.expression, r2, {}, metadata),
            None,
            return_outcome=_return_outcome,
        )
    except Exception as exc:
        _LOG.warning("native_gp fit failed: %s", exc)
        return _engine_outcome(
            None,
            f"fit_exception:{type(exc).__name__}: {str(exc)[:160]}",
            return_outcome=_return_outcome,
        )


def _safe_unary(col: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    out = fn(col)
    return np.where(np.isfinite(out), out, np.nan)


def _try_design(
    design: np.ndarray,
    names: list[str],
    label: str,
    y_m: np.ndarray,
) -> tuple[float, np.ndarray, list[str], str] | None:
    if design.shape[0] < design.shape[1] + 1:
        return None
    row_ok = np.all(np.isfinite(design), axis=1) & np.isfinite(y_m)
    if int(row_ok.sum()) < design.shape[1] + 1:
        return None
    d = design[row_ok]
    y = y_m[row_ok]
    try:
        coef, _, _, _ = np.linalg.lstsq(d, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    pred = d @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    terms = [f"{c:.4g}*{n}" for c, n in zip(coef, names) if abs(c) > 1e-10]
    eq = label if not terms else f"{label}: " + " + ".join(terms)
    # Align to y_m so callers can scatter with the outer feature mask.
    pred_full = np.full(y_m.shape[0], np.nan, dtype=float)
    pred_full[row_ok] = pred
    return r2, pred_full, names, eq


def _physics_template_fit(
    rows: list[dict[str, Any]],
    target: str,
    features: list[str],
) -> SymbolicResult | None:
    """AI-Feynman-style template search over common physics-inspired closed forms."""
    X, y, mask = _feature_matrix(rows, target, features)
    if mask.sum() < X.shape[1] + 2:
        return None
    X_m = X[mask]
    y_m = y[mask]
    eps = 1e-8

    best: tuple[float, np.ndarray, list[str], str, dict[str, float]] | None = None

    for j, feat in enumerate(features):
        col = X_m[:, j]
        inv = np.full_like(col, np.nan, dtype=float)
        inv_ok = np.abs(col) > eps
        inv[inv_ok] = 1.0 / col[inv_ok]
        transforms: list[tuple[str, np.ndarray]] = [
            (feat, col),
            (f"{feat}^2", col ** 2),
            (f"sqrt({feat})", _safe_unary(np.abs(col), np.sqrt)),
            (f"log({feat})", _safe_unary(np.abs(col) + eps, np.log)),
            (f"1/({feat})", inv),
        ]

        for name, transformed in transforms:
            hit = _try_design(transformed.reshape(-1, 1), [name], name, y_m)
            if hit is None:
                continue
            r2, pred, names, eq = hit
            if best is None or r2 > best[0]:
                best = (r2, pred, names, eq, {names[0]: float(pred[0]) if len(pred) else 0.0})

    if len(features) >= 2:
        for i in range(min(len(features), 4)):
            for j in range(i + 1, min(len(features), 4)):
                xi, xj = X_m[:, i], X_m[:, j]
                fi, fj = features[i], features[j]
                ratio = np.full_like(xi, np.nan, dtype=float)
                ratio_ok = np.abs(xj) > eps
                ratio[ratio_ok] = xi[ratio_ok] / xj[ratio_ok]
                pair_forms: list[tuple[str, np.ndarray]] = [
                    (f"{fi}+{fj}", xi + xj),
                    (f"{fi}-{fj}", xi - xj),
                    (f"{fi}*{fj}", xi * xj),
                    (f"{fi}/{fj}", ratio),
                    (f"sqrt({fi}^2+{fj}^2)", np.sqrt(xi ** 2 + xj ** 2 + eps)),
                ]
                for label, combo in pair_forms:
                    hit = _try_design(combo.reshape(-1, 1), [label], label, y_m)
                    if hit is None:
                        continue
                    r2, pred, names, eq = hit
                    if best is None or r2 > best[0]:
                        best = (r2, pred, names, eq, {})

        hit = _try_design(
            np.column_stack([X_m[:, k] for k in range(min(len(features), 4))]),
            features[: min(len(features), 4)],
            "linear_combo",
            y_m,
        )
        if hit is not None:
            r2, pred, names, eq = hit
            if best is None or r2 > best[0]:
                best = (r2, pred, names, eq, {})

    if best is None:
        return None
    r2, pred_m, _names, eq, coefs = best
    full_pred = np.full_like(y, np.nan)
    full_pred[mask] = pred_m
    metadata: dict[str, Any] = {"n_templates": len(features) * 5 + max(0, len(features) - 1) * 5}
    result = SymbolicResult("physics_templates", eq, float(r2), coefs, metadata)
    return _finalize_symbolic_result(result, rows, y, full_pred)


def discover_equation(
    rows: list[dict[str, Any]],
    target: str,
    *,
    top_features: int = 6,
    features: list[str] | None = None,
    use_pysr: bool = True,
    use_operon: bool = True,
    use_physics_templates: bool = True,
    use_native_gp: bool = True,
    use_ai_feynman: bool = False,
    extrapolation: bool = True,
    symbolic_min_r_squared: float = 0.5,
    gp_generations: int = 20,
    gp_population: int = 48,
    pysr_iterations: int = 40,
    pysr_maxsize: int = 20,
    pysr_populations: int = 15,
    operon_generations: int = 50,
    operon_population: int = 300,
    eval_backend: str | None = None,
    require_gpu: bool = False,
    on_stage: Callable[[str], None] | None = None,
    on_progress: ProgressCallback | None = None,
) -> SymbolicResult:
    def _step(name: str) -> None:
        if on_stage is not None:
            on_stage(name)

    if features is not None:
        feat_list = [f for f in features if f != target]
    else:
        hits = correlate_with_target(rows, target, min_abs_r=0.1)
        feat_list = [h["column"] for h in hits[:top_features]]
        if not feat_list:
            feat_list = numeric_columns(rows, exclude={target})[:top_features]

    backends = symbolic_backends()
    skipped_engines: list[dict[str, str]] = []
    _step("polynomial fit")
    candidates: list[SymbolicResult] = [_polynomial_fit(rows, target, feat_list)]
    if use_physics_templates:
        _step("physics templates")
        physics = _physics_template_fit(rows, target, feat_list)
        if physics is not None:
            candidates.append(physics)
    jobs: list[tuple[str, list[dict[str, Any]], str, list[str], dict[str, Any]]] = []

    if use_ai_feynman:
        if not backends["ai_feynman"]:
            _record_engine_skip(
                skipped_engines,
                "ai_feynman",
                backends.get("errors", {}).get("ai_feynman", "not installed"),
                on_stage=on_stage,
            )
        else:
            jobs.append(("ai_feynman", rows, target, feat_list, {}))
    if use_pysr:
        if not backends["pysr"]:
            _record_engine_skip(
                skipped_engines,
                "pysr",
                backends.get("errors", {}).get("pysr", "backend probe failed"),
                on_stage=on_stage,
            )
        else:
            jobs.append(
                (
                    "pysr",
                    rows,
                    target,
                    feat_list,
                    {
                        "niterations": pysr_iterations,
                        "maxsize": pysr_maxsize,
                        "populations": pysr_populations,
                    },
                )
            )
    if use_operon:
        if not backends["operon"]:
            _record_engine_skip(
                skipped_engines,
                "operon",
                backends.get("errors", {}).get("operon", "backend probe failed"),
                on_stage=on_stage,
            )
        else:
            jobs.append(
                (
                    "operon",
                    rows,
                    target,
                    feat_list,
                    {
                        "generations": operon_generations,
                        "population_size": operon_population,
                    },
                )
            )
    if use_native_gp:
        jobs.append(
            (
                "native_gp",
                rows,
                target,
                feat_list,
                {
                    "generations": gp_generations,
                    "population": gp_population,
                    "eval_backend": eval_backend,
                    "require_gpu": require_gpu,
                },
            )
        )

    engine_results: dict[str, SymbolicEngineOutcome] = {}
    parallel = _symbolic_parallel_enabled(
        gp_generations=gp_generations,
        gp_population=gp_population,
        pysr_iterations=pysr_iterations,
        pysr_populations=pysr_populations,
        operon_generations=operon_generations,
        operon_population=operon_population,
    )
    if parallel and len(jobs) > 1:
        for engine, *_ in jobs:
            _step(f"{engine} (parallel)")
        parallel_heartbeat = (
            StageHeartbeat(on_progress, label="symbolic engines (parallel)")
            if on_stage is None
            else None
        )
        if parallel_heartbeat is not None:
            parallel_heartbeat.start()
        try:
            with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
                for engine, outcome in executor.map(_run_symbolic_engine_job, jobs):
                    engine_results[engine] = outcome
        except Exception as exc:  # noqa: BLE001 — retain sequential fallback
            _LOG.warning("parallel symbolic engines failed; retrying sequentially: %s", exc)
            engine_results.clear()
        finally:
            if parallel_heartbeat is not None:
                parallel_heartbeat.stop(detail="complete")

    if not engine_results:
        for engine, engine_rows, engine_target, engine_features, kwargs in jobs:
            _step(engine)
            engine_kwargs = dict(kwargs)
            if engine == "native_gp":
                engine_kwargs["on_progress"] = on_progress
            engine_heartbeat = (
                StageHeartbeat(on_progress, label=f"symbolic {engine}")
                if on_stage is None
                else None
            )
            if engine_heartbeat is not None:
                engine_heartbeat.start()
            try:
                _, outcome = _run_symbolic_engine_job(
                    (engine, engine_rows, engine_target, engine_features, engine_kwargs)
                )
            finally:
                if engine_heartbeat is not None:
                    engine_heartbeat.stop(detail="complete")
            engine_results[engine] = outcome

    for engine, *_ in jobs:
        outcome = engine_results.get(engine)
        result = outcome.result if outcome is not None else None
        if result is not None:
            candidates.append(result)
        else:
            _record_engine_skip(
                skipped_engines,
                engine,
                (
                    outcome.reason
                    if outcome is not None and outcome.reason is not None
                    else "fit returned no result"
                ),
                on_stage=on_stage,
            )

    best = max(candidates, key=lambda c: (c.r_squared, -c.metadata.get("mdl", len(c.equation))))
    if best.r_squared < symbolic_min_r_squared and candidates[0].r_squared >= best.r_squared:
        best = candidates[0]

    if not extrapolation:
        best.metadata.pop("extrapolation_r_squared", None)
    best.metadata["candidates_tried"] = [c.method for c in candidates]
    best.metadata["pareto_front"] = [
        {"method": c.method, "r_squared": c.r_squared, "equation": c.equation} for c in candidates
    ]
    best.metadata["feature_columns"] = feat_list
    if skipped_engines:
        best.metadata["skipped_engines"] = skipped_engines
    return best


def _ai_feynman_fit(
    rows: list[dict[str, Any]],
    target: str,
    features: list[str],
    _return_outcome: bool = False,
) -> SymbolicResult | SymbolicEngineOutcome | None:
    """Real AI Feynman package when installed; None when unavailable or failed."""
    X, y, mask = _feature_matrix(rows, target, features)
    if len(features) < 1:
        return _engine_outcome(None, "no_features", return_outcome=_return_outcome)
    reason = _finite_rows_reason(mask, features, "ai_feynman")
    if reason is not None:
        return _engine_outcome(None, reason, return_outcome=_return_outcome)
    try:
        import aifeynman  # type: ignore[import-untyped]
    except Exception as exc:  # noqa: BLE001 — optional backend import
        _LOG.warning("aifeynman import failed: %s", exc)
        return _engine_outcome(
            None,
            f"import_failed:{type(exc).__name__}: {str(exc)[:160]}",
            return_outcome=_return_outcome,
        )
    try:
        import pandas as pd
        import tempfile
        from pathlib import Path

        df = pd.DataFrame(X[mask], columns=features)
        df["target"] = y[mask]
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "data.csv"
            df.to_csv(data_path, index=False, header=False)
            aifeynman.run_aifeynman(
                str(data_path),
                str(Path(tmp) / "results"),
                bf_ops=["+", "-", "*", "/", "pow", "exp", "log", "sin", "cos"],
                max_time=30,
            )
        # Package writes discovered equations to results; fall back if none parsed
    except Exception as exc:
        _LOG.warning("aifeynman run failed: %s", exc)
        return _engine_outcome(
            None,
            f"fit_exception:{type(exc).__name__}: {str(exc)[:160]}",
            return_outcome=_return_outcome,
        )
    _LOG.warning("aifeynman produced no parseable equation")
    return _engine_outcome(
        None,
        "no_parseable_equation",
        return_outcome=_return_outcome,
    )


def _rank_latent_dims(
    latent_codes: np.ndarray,
    main_target: np.ndarray,
    names: list[str],
) -> list[tuple[int, float]]:
    scored: list[tuple[int, float]] = []
    for j in range(latent_codes.shape[1]):
        col = latent_codes[:, j]
        mask = np.isfinite(col) & np.isfinite(main_target)
        if mask.sum() < 3:
            continue
        if np.std(col[mask]) <= 0 or np.std(main_target[mask]) <= 0:
            continue
        corr = safe_pearson_r(col[mask], main_target[mask])
        scored.append((j, corr))
    scored.sort(key=lambda t: abs(t[1]), reverse=True)
    return scored


def discover_latent_interpretations(
    rows: list[dict[str, Any]],
    latent_sources: list[LatentSource],
    *,
    main_target: str,
    feature_columns: list[str],
    top_latents: int = 3,
    min_abs_correlation: float = 0.3,
    use_pysr: bool = True,
    use_operon: bool = True,
    use_physics_templates: bool = True,
    use_native_gp: bool = True,
    use_ai_feynman: bool = False,
    extrapolation: bool = True,
    gp_generations: int = 20,
    gp_population: int = 48,
    pysr_iterations: int = 40,
    pysr_maxsize: int = 20,
    pysr_populations: int = 15,
    operon_generations: int = 50,
    operon_population: int = 300,
    eval_backend: str | None = None,
    require_gpu: bool = False,
    on_progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Close Phase 4→5 loop: express top predictive latents as closed forms over descriptors."""
    if not rows or not latent_sources:
        return []

    main = np.asarray(
        [row.get(main_target, float("nan")) for row in rows],
        dtype=float,
    )
    interp_rows = [
        {
            **{c: row.get(c, float("nan")) for c in feature_columns},
            **({"size": row["size"]} if row.get("size") is not None else {}),
        }
        for row in rows
    ]
    candidates: list[tuple[float, int, LatentSource, int]] = []
    for src in latent_sources:
        if src.codes.size == 0 or src.codes.shape[0] != len(rows):
            continue
        k = src.codes.shape[1]
        names = src.columns or [f"{src.source}_{i}" for i in range(k)]
        for j, corr in _rank_latent_dims(src.codes, main, names):
            if abs(corr) >= min_abs_correlation:
                candidates.append((abs(corr), j, src, corr))

    candidates.sort(key=lambda t: t[0], reverse=True)
    seen_cols: set[str] = set()
    results: list[dict[str, Any]] = []

    for _, j, src, corr in candidates:
        names = src.columns or [f"{src.source}_{i}" for i in range(src.codes.shape[1])]
        latent_target = names[j]
        if latent_target in seen_cols:
            continue
        seen_cols.add(latent_target)

        for interp_row, value in zip(interp_rows, src.codes[:, j]):
            interp_row[latent_target] = float(value)

        sym = discover_equation(
            interp_rows,
            latent_target,
            features=feature_columns,
            use_pysr=use_pysr,
            use_operon=use_operon,
            use_physics_templates=use_physics_templates,
            use_native_gp=use_native_gp,
            use_ai_feynman=use_ai_feynman,
            extrapolation=extrapolation,
            gp_generations=gp_generations,
            gp_population=gp_population,
            pysr_iterations=pysr_iterations,
            pysr_maxsize=pysr_maxsize,
            pysr_populations=pysr_populations,
            operon_generations=operon_generations,
            operon_population=operon_population,
            eval_backend=eval_backend,
            require_gpu=require_gpu,
            on_progress=on_progress,
        )
        results.append(
            {
                "latent_column": latent_target,
                "latent_index": j,
                "latent_source": src.source,
                "main_target": main_target,
                "main_target_correlation": corr,
                "method": sym.method,
                "equation": sym.equation,
                "r_squared": sym.r_squared,
                **sym.metadata,
            }
        )
        if len(results) >= top_latents:
            break
    return results


def _select_symbolic_features(
    rows: list[dict[str, Any]],
    target: str,
    candidates: list[str],
    *,
    top_features: int,
) -> tuple[list[str], dict[str, int], dict[str, int], int]:
    """Choose finite, target-relevant columns before joint symbolic fitting."""
    unique_candidates = list(dict.fromkeys(c for c in candidates if c != target))
    if not unique_candidates or top_features <= 0:
        return [], {}, {}, 0

    from rde.analyze.feature_table import FeatureTable

    table = FeatureTable.from_rows(rows, columns=[*unique_candidates, target])
    y = table.column(target)
    target_finite = np.isfinite(y)
    finite_counts = {
        column: int((np.isfinite(table.column(column)) & np.isfinite(y)).sum())
        for column in unique_candidates
    }
    dropped = {
        column: count
        for column, count in finite_counts.items()
        if count < 10
    }
    usable = [column for column in unique_candidates if column not in dropped]
    if not usable:
        return [], dropped, finite_counts, int(target_finite.sum())

    hits = table.correlate_with_target(target, min_abs_r=0.0)
    by_col = {
        hit["column"]: abs(float(hit["pearson_r"]))
        for hit in hits
        if np.isfinite(hit.get("pearson_r", float("nan")))
    }
    order = {column: index for index, column in enumerate(unique_candidates)}
    ranked = sorted(
        usable,
        key=lambda column: (
            -int(column in by_col and by_col[column] > 0.0),
            -by_col.get(column, 0.0),
            order[column],
        ),
    )
    selected: list[str] = []
    joint_mask = target_finite.copy()
    for column in ranked:
        candidate_mask = joint_mask & np.isfinite(table.column(column))
        if int(candidate_mask.sum()) < 10:
            continue
        selected.append(column)
        joint_mask = candidate_mask
        if len(selected) >= top_features:
            break
    return selected, dropped, finite_counts, int(joint_mask.sum())


def run_symbolic_discovery(
    rows: list[dict[str, Any]],
    target: str,
    *,
    feature_columns: list[str] | None = None,
    latent_sources: list[LatentSource] | None = None,
    top_features: int = 8,
    top_latents: int = 3,
    use_pysr: bool = True,
    use_operon: bool = True,
    use_physics_templates: bool = True,
    use_native_gp: bool = True,
    use_ai_feynman: bool = False,
    extrapolation: bool = True,
    symbolic_min_r_squared: float = 0.5,
    gp_generations: int = 20,
    gp_population: int = 48,
    pysr_iterations: int = 40,
    pysr_maxsize: int = 20,
    pysr_populations: int = 15,
    operon_generations: int = 50,
    operon_population: int = 300,
    eval_backend: str | None = None,
    require_gpu: bool = False,
    on_stage: Callable[[str], None] | None = None,
    on_progress: ProgressCallback | None = None,
) -> SymbolicDiscoveryResult:
    """Phase 5: fit target directly and interpret promising Phase-4 latents."""
    backends = symbolic_backends()

    candidate_features = (
        list(feature_columns)
        if feature_columns is not None
        else numeric_columns(rows, exclude={target})
    )
    feats, dropped_features, finite_feature_counts, joint_finite_rows = _select_symbolic_features(
        rows,
        target,
        candidate_features,
        top_features=top_features,
    )

    target_fit = discover_equation(
        rows,
        target,
        features=feats,
        use_pysr=use_pysr,
        use_operon=use_operon,
        use_physics_templates=use_physics_templates,
        use_native_gp=use_native_gp,
        use_ai_feynman=use_ai_feynman,
        extrapolation=extrapolation,
        symbolic_min_r_squared=symbolic_min_r_squared,
        gp_generations=gp_generations,
        gp_population=gp_population,
        pysr_iterations=pysr_iterations,
        pysr_maxsize=pysr_maxsize,
        pysr_populations=pysr_populations,
        operon_generations=operon_generations,
        operon_population=operon_population,
        eval_backend=eval_backend,
        require_gpu=require_gpu,
        on_stage=on_stage,
        on_progress=on_progress,
    )
    target_fit.metadata["dropped_feature_columns"] = dropped_features
    target_fit.metadata["feature_finite_counts"] = {
        column: finite_feature_counts[column]
        for column in feats
    }
    target_fit.metadata["joint_finite_rows"] = joint_finite_rows

    latent_interps: list[dict[str, Any]] = []
    if latent_sources:
        latent_interps = discover_latent_interpretations(
            rows,
            latent_sources,
            main_target=target,
            feature_columns=feats,
            top_latents=top_latents,
            use_pysr=use_pysr,
            use_operon=use_operon,
            use_physics_templates=use_physics_templates,
            use_native_gp=use_native_gp,
            use_ai_feynman=use_ai_feynman,
            extrapolation=extrapolation,
            gp_generations=gp_generations,
            gp_population=gp_population,
            pysr_iterations=pysr_iterations,
            pysr_maxsize=pysr_maxsize,
            pysr_populations=pysr_populations,
            operon_generations=operon_generations,
            operon_population=operon_population,
            eval_backend=eval_backend,
            require_gpu=require_gpu,
            on_progress=on_progress,
        )

    return SymbolicDiscoveryResult(
        target_fit=target_fit,
        latent_interpretations=latent_interps,
        backends=backends,
    )


def discover_latent_interpretation(
    rows: list[dict[str, Any]],
    latent_codes: np.ndarray,
    *,
    main_target: str,
    feature_columns: list[str],
    latent_columns: list[str] | None = None,
    latent_source: str = "latent",
    use_pysr: bool = True,
    use_operon: bool = True,
    extrapolation: bool = True,
) -> dict[str, Any] | None:
    """Phase 4→5 bridge: express the best latent coordinate as a closed form over descriptors."""
    cols = latent_columns
    if cols is None and latent_codes.ndim == 2:
        cols = [f"latent.ae_{i}" for i in range(latent_codes.shape[1])]
    interps = discover_latent_interpretations(
        rows,
        [LatentSource(codes=latent_codes, columns=cols or [], source=latent_source)],
        main_target=main_target,
        feature_columns=feature_columns,
        top_latents=1,
        min_abs_correlation=0.0,
        use_pysr=use_pysr,
        use_operon=use_operon,
        extrapolation=extrapolation,
    )
    return interps[0] if interps else None

