"""Walsh generating-function search + DreamCoder-style program library (Phase 6)."""

from __future__ import annotations

import itertools
import random
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rde.analyze.feature_table import FeatureTable
from rde.discovery.programs import ProgramResult, evolve_programs
from rde.expression.batch import (
    EvalBackend,
    normalize_expr_backend,
    prepare_device_envs,
    score_population,
)
from rde.expression import Expr
from rde.features.fourier import walsh_hadamard, walsh_hadamard_batch
from rde.features.numeric import safe_pearson_r
from rde.runtime.heartbeat import ProgressCallback, throttled_progress


@dataclass
class WalshProgram:
    ansatz: str
    expression: str
    support_indices: list[int]
    fitness: float
    pearson_r: float
    reconstruction_r2: float = float("nan")
    mdl_bits: float = 0.0


@dataclass
class WalshSynthesisResult:
    method: str
    n_candidates: int
    walsh_variables: list[str]
    top_programs: list[WalshProgram] = field(default_factory=list)
    library_fragments: list[dict[str, Any]] = field(default_factory=list)
    dreamcoder_programs: list[ProgramResult] = field(default_factory=list)


def _walsh_descriptor_columns(rows: list[dict[str, Any]]) -> list[str]:
    cols: set[str] = set()
    for row in rows:
        for key in row:
            if not isinstance(row.get(key), (int, float)):
                continue
            if key.startswith(("fourier.", "walsh_log.")):
                cols.add(key)
    return sorted(cols)


def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _pad_pm1(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return arr
    n = _next_power_of_two(arr.size)
    out = np.zeros(n, dtype=float)
    out[: arr.size] = arr
    if arr.size > 1 and np.all((arr >= 0) & (arr <= 1)):
        out[: arr.size] = 2.0 * arr - 1.0
    return out


def _fit_walsh_support(
    C: np.ndarray,
    t: np.ndarray,
    support: list[int],
    *,
    reg: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Ridge fit for one support set; returns beta, pred, r2, pearson_r."""
    if not support:
        X_aug = np.ones((C.shape[0], 1))
    else:
        X = C[:, support]
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
    reg_mat = reg * np.eye(X_aug.shape[1])
    reg_mat[0, 0] = 0.0
    beta = np.linalg.solve(X_aug.T @ X_aug + reg_mat, X_aug.T @ t)
    pred = X_aug @ beta
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = float(1.0 - np.sum((t - pred) ** 2) / max(ss_tot, 1e-12))
    pr = safe_pearson_r(pred, t) if np.std(pred) > 0 else 0.0
    return beta, pred, r2, pr


def _fit_walsh_supports_batch(
    C: np.ndarray,
    t: np.ndarray,
    supports: list[list[int]],
    *,
    reg: float = 1e-3,
) -> list[tuple[list[int], np.ndarray, float, float]]:
    """Fit equal-width Walsh supports with one batched ridge solve."""
    if not supports:
        return []
    support_array = np.asarray(supports, dtype=int)
    selected = C[:, support_array].transpose(1, 0, 2)
    design = np.concatenate(
        [
            np.ones((len(supports), C.shape[0], 1), dtype=float),
            selected,
        ],
        axis=2,
    )
    gram = np.einsum("bni,bnj->bij", design, design)
    reg_matrix = reg * np.eye(design.shape[2])
    reg_matrix[0, 0] = 0.0
    gram += reg_matrix[None, :, :]
    rhs = np.einsum("bni,n->bi", design, t)
    beta = np.linalg.solve(gram, rhs[..., None])[..., 0]
    pred = np.einsum("bni,bi->bn", design, beta)
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = 1.0 - np.sum((t[None, :] - pred) ** 2, axis=1) / max(ss_tot, 1e-12)
    centered_pred = pred - pred.mean(axis=1, keepdims=True)
    centered_target = t - t.mean()
    covariance = np.sum(centered_pred * centered_target[None, :], axis=1)
    denominator = np.sqrt(
        np.sum(centered_pred**2, axis=1) * np.sum(centered_target**2)
    )
    pearson = np.zeros(len(supports), dtype=float)
    valid = denominator > 0.0
    pearson[valid] = covariance[valid] / denominator[valid]
    return [
        (list(support), beta[i], float(r2[i]), float(pearson[i]))
        for i, support in enumerate(supports)
    ]


def _walsh_program_from_support(
    support: list[int],
    beta: np.ndarray,
    *,
    pr: float,
    r2: float,
) -> WalshProgram:
    expr = " + ".join(f"{beta[i + 1]:+.4g}*W[{support[i]}]" for i in range(len(support)))
    expr = f"{beta[0]:+.4g} + {expr}" if support else f"{beta[0]:+.4g}"
    fitness = abs(pr) - 0.005 * len(support)
    return WalshProgram(
        ansatz="sparse_walsh",
        expression=expr,
        support_indices=support,
        fitness=fitness,
        pearson_r=pr,
        reconstruction_r2=r2,
        mdl_bits=float(len(support) * 2.5 + 4.0),
    )


def _record_walsh(results: dict[str, WalshProgram], program: WalshProgram) -> None:
    key = ",".join(map(str, program.support_indices))
    prev = results.get(key)
    if prev is None or program.fitness > prev.fitness:
        results[key] = program


def _omp_walsh_search(
    C: np.ndarray,
    t: np.ndarray,
    *,
    max_support: int,
) -> list[list[int]]:
    """Greedy OMP supports (one path, length 1..max_support)."""
    n = C.shape[1]
    residual = t - float(np.mean(t))
    support: list[int] = []
    paths: list[list[int]] = []
    for _ in range(min(max_support, n)):
        scores = C.T @ residual
        if support:
            scores[support] = 0.0
        j = int(np.argmax(np.abs(scores)))
        if j in support or abs(float(scores[j])) < 1e-12:
            break
        support.append(j)
        paths.append(sorted(support))
        _, pred, _, _ = _fit_walsh_support(C, t, support)
        residual = t - pred
    return paths


def _sparse_walsh_fit(
    coeffs: np.ndarray,
    target: np.ndarray,
    *,
    max_support: int,
    max_candidates: int,
    seed: int,
    on_progress: ProgressCallback | None = None,
) -> list[WalshProgram]:
    """Search sparse Walsh subsets via OMP + capped random/combination search."""
    n = coeffs.shape[1]
    y = target.copy()
    mask = np.isfinite(y) & np.all(np.isfinite(coeffs), axis=1)
    if mask.sum() < 4:
        return []
    C = coeffs[mask]
    t = y[mask]
    if np.std(t) < 1e-12:
        return []

    results: dict[str, WalshProgram] = {}
    indices = list(range(n))

    for support in _omp_walsh_search(C, t, max_support=max_support):
        beta, _, r2, pr = _fit_walsh_support(C, t, support)
        _record_walsh(results, _walsh_program_from_support(support, beta, pr=pr, r2=r2))

    rng = random.Random(seed)
    random_budget = min(max_candidates, max(200, max_support * 40))
    tries = 0
    candidate_groups: dict[int, list[list[int]]] = {}
    while tries < random_budget:
        tries += 1
        k = rng.randint(1, min(max_support, n))
        support = sorted(rng.sample(indices, k))
        candidate_groups.setdefault(k, []).append(support)

    for k in range(1, min(max_support, 3) + 1):
        for support in itertools.combinations(indices[: min(n, 12)], k):
            candidate_groups.setdefault(k, []).append(list(support))

    total_candidates = sum(len(supports) for supports in candidate_groups.values())
    report_progress = throttled_progress(
        on_progress,
        min_interval_s=1.0,
        every_n=200,
    )
    if report_progress is not None:
        report_progress(0, total_candidates, "OMP supports complete; fitting candidates")

    evaluated = 0
    for supports in candidate_groups.values():
        for support, beta, r2, pr in _fit_walsh_supports_batch(C, t, supports):
            _record_walsh(results, _walsh_program_from_support(support, beta, pr=pr, r2=r2))
        evaluated += len(supports)
        if report_progress is not None:
            report_progress(
                evaluated,
                total_candidates,
                f"backend=numpy supports={evaluated}",
            )

    ranked = sorted(results.values(), key=lambda p: -p.fitness)
    if report_progress is not None:
        report_progress(
            total_candidates,
            total_candidates,
            f"done programs={len(ranked[:10])}",
        )
    return ranked[:10]


def _state_walsh_columns(
    states: np.ndarray,
    *,
    compute_backend: str = "numpy",
) -> np.ndarray | None:
    if states.size == 0:
        return None
    arr = np.asarray(states, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    state_dim = arr.shape[1]
    padded_len = _next_power_of_two(state_dim)
    if padded_len < 2 or (padded_len & (padded_len - 1)) != 0:
        return None
    padded = np.zeros((arr.shape[0], padded_len), dtype=float)
    padded[:, :state_dim] = arr
    if state_dim > 1 and np.all((arr >= 0) & (arr <= 1)):
        padded[:, :state_dim] = 2.0 * arr - 1.0
    if compute_backend.lower() == "mlx":
        import mlx.core as mx

        transformed = mx.array(padded, dtype=mx.float32)
        h = 1
        while h < padded_len:
            pairs = mx.reshape(transformed, (transformed.shape[0], -1, 2, h))
            top = pairs[:, :, 0, :]
            bottom = pairs[:, :, 1, :]
            transformed = mx.reshape(
                mx.stack([top + bottom, top - bottom], axis=2),
                transformed.shape,
            )
            h *= 2
        transformed = transformed / mx.sqrt(mx.array(float(padded_len)))
        mx.eval(transformed)
        return np.asarray(transformed, dtype=float)
    try:
        return walsh_hadamard_batch(padded)
    except ValueError:
        return None


def _extract_library_fragments(programs: list[ProgramResult], *, min_count: int = 2) -> list[dict[str, Any]]:
    """DreamCoder-style: fragments that recur across top programs."""
    counts: dict[str, int] = {}
    for prog in programs:
        for token in re.findall(r"[a-zA-Z_][\w.]*", prog.expression):
            if token in {"add", "sub", "mul", "div"}:
                continue
            counts[token] = counts.get(token, 0) + 1
    return [
        {"fragment": frag, "count": count}
        for frag, count in sorted(counts.items(), key=lambda t: -t[1])
        if count >= min_count
    ]


def search_walsh_generating_functions(
    rows: list[dict[str, Any]],
    target: str,
    *,
    state_vectors: np.ndarray | None = None,
    max_support: int = 6,
    max_candidates: int = 2000,
    seed: int = 0,
    compute_backend: str = "numpy",
    on_progress: ProgressCallback | None = None,
) -> WalshSynthesisResult:
    """Search sparse Walsh-basis ansätze for target prediction."""
    walsh_cols = _walsh_descriptor_columns(rows)
    table_columns = list(walsh_cols)
    if target not in table_columns:
        table_columns.append(target)
    table = FeatureTable.from_rows(rows, columns=table_columns)
    y = table.column(target)

    coeff_matrix: np.ndarray | None = None
    if walsh_cols:
        coeff_matrix = table.data[:, : len(walsh_cols)]
    elif state_vectors is not None and state_vectors.size:
        coeff_matrix = _state_walsh_columns(
            state_vectors,
            compute_backend=compute_backend,
        )

    top: list[WalshProgram] = []
    if coeff_matrix is not None and coeff_matrix.size:
        top = _sparse_walsh_fit(
            coeff_matrix,
            y,
            max_support=max_support,
            max_candidates=max_candidates,
            seed=seed,
            on_progress=on_progress,
        )

    return WalshSynthesisResult(
        method="sparse_walsh_generating_function",
        n_candidates=max_candidates,
        walsh_variables=walsh_cols,
        top_programs=top,
    )


def dreamcoder_style_search(
    rows: list[dict[str, Any]],
    target: str,
    variables: list[str],
    *,
    generations: int = 12,
    population: int = 40,
    seed: int = 0,
    eval_backend: EvalBackend | None = None,
    walsh_variables: list[str] | None = None,
    require_gpu: bool = False,
    on_progress: ProgressCallback | None = None,
) -> tuple[list[ProgramResult], list[dict[str, Any]]]:
    """GP with Walsh-aware seed library + fragment extraction."""
    backend = normalize_expr_backend(eval_backend)
    extended = list(dict.fromkeys([*variables, *(walsh_variables or [])]))
    progs = evolve_programs(
        rows,
        target,
        extended,
        generations=generations,
        population=population,
        seed=seed,
        top_k=8,
        eval_backend=backend,
        seed_expressions=True,
        require_gpu=require_gpu,
        on_progress=on_progress,
    )
    library = _extract_library_fragments(progs)
    # Re-seed GP with discovered library fragments as unary leaves
    if library:
        rng = random.Random(seed + 1)
        table = FeatureTable.from_rows(rows, columns=[*extended, target])
        env = table.env(extended)
        y = table.column(target)
        backend, mlx_env, torch_env, target_mlx, _ = prepare_device_envs(
            env,
            backend,
            len(rows),
            y,
            require_gpu=require_gpu,
            requested_backend=backend,
        )
        library_exprs: list[Expr] = []
        for item in library[:4]:
            frag = item["fragment"]
            if frag not in extended:
                expr = Expr.var(frag) if frag in env else Expr.var(rng.choice(extended))
            else:
                expr = Expr.var(frag)
            library_exprs.append(expr)
        scores = score_population(
            library_exprs,
            env,
            y,
            backend=backend,
            mlx_env=mlx_env,
            torch_env=torch_env,
            target_mlx=target_mlx,
        )
        for expr, scored in zip(library_exprs, scores):
            if scored is not None:
                _, pr, _ = scored
                progs.append(
                    ProgramResult(
                        expression=str(expr),
                        fitness=abs(pr) - 0.01,
                        pearson_r=float(pr),
                        generation=-1,
                    )
                )
        progs = sorted({p.expression: p for p in progs}.values(), key=lambda p: -p.fitness)[:8]
    return progs, library


def run_walsh_synthesis(
    rows: list[dict[str, Any]],
    target: str,
    variables: list[str],
    *,
    state_vectors: np.ndarray | None = None,
    max_support: int = 6,
    max_candidates: int = 2000,
    gp_generations: int = 10,
    gp_population: int = 40,
    seed: int = 0,
    eval_backend: EvalBackend | None = None,
    require_gpu: bool = False,
    compute_backend: str = "numpy",
    on_progress: ProgressCallback | None = None,
) -> WalshSynthesisResult:
    """Combined Walsh generating-function + DreamCoder-style GP search."""
    walsh = search_walsh_generating_functions(
        rows,
        target,
        state_vectors=state_vectors,
        max_support=max_support,
        max_candidates=max_candidates,
        seed=seed,
        compute_backend=compute_backend,
        on_progress=on_progress,
    )
    dc_progs, library = dreamcoder_style_search(
        rows,
        target,
        variables,
        generations=gp_generations,
        population=gp_population,
        seed=seed,
        eval_backend=eval_backend,
        walsh_variables=walsh.walsh_variables[:12],
        require_gpu=require_gpu,
        on_progress=on_progress,
    )
    walsh.library_fragments = library
    walsh.dreamcoder_programs = dc_progs
    return walsh
