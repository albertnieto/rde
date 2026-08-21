"""Genetic programming over metric expression AST (Phase 6)."""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from rde.analyze.feature_table import FeatureTable
from rde.expression import Expr, enumerate_expressions
from rde.expression.batch import (
    EvalBackend,
    ProgressCallback,
    normalize_expr_backend,
    prepare_device_envs,
    score_expression,
    score_population,
)


@dataclass
class ProgramResult:
    expression: str
    fitness: float
    pearson_r: float
    generation: int


def _random_expr(variables: list[str], depth: int, rng: random.Random) -> Expr:
    if depth <= 1 or not variables:
        return Expr.var(rng.choice(variables))
    if rng.random() < 0.4:
        return Expr.var(rng.choice(variables))
    op = rng.choice(["add", "sub", "mul", "div"])
    left = _random_expr(variables, depth - 1, rng)
    right = _random_expr(variables, depth - 1, rng)
    return Expr(op, left=left, right=right)


def _mutate(expr: Expr, variables: list[str], rng: random.Random) -> Expr:
    if rng.random() < 0.3:
        return _random_expr(variables, 2, rng)
    if expr.kind == "var" and rng.random() < 0.5:
        return Expr.var(rng.choice(variables))
    if expr.left and rng.random() < 0.5:
        return Expr(expr.kind, left=_mutate(expr.left, variables, rng), right=expr.right)
    if expr.right:
        return Expr(expr.kind, left=expr.left, right=_mutate(expr.right, variables, rng))
    return expr


def _graft(base: Expr, donor: Expr, rng: random.Random) -> Expr:
    if base.kind in {"const", "var"}:
        return donor
    if rng.random() < 0.5 and base.left:
        return Expr(base.kind, left=_graft(base.left, donor, rng), right=base.right)
    if base.right:
        return Expr(base.kind, left=base.left, right=_graft(base.right, donor, rng))
    return donor


def _crossover(a: Expr, b: Expr, rng: random.Random) -> Expr:
    if rng.random() < 0.5:
        return _graft(a, b, rng)
    return _graft(b, a, rng)


def _tournament_winners(
    scored: list[tuple[tuple[float, float], Expr]],
    k: int,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select tournament winners in one vectorized NumPy reduction."""
    if count <= 0 or not scored:
        return np.empty(0, dtype=int)
    fitness = np.asarray([item[0][0] for item in scored], dtype=float)
    picks = _tournament_indices(len(scored), k, count, rng)
    winner_columns = np.argmax(fitness[picks], axis=1)
    return picks[np.arange(count), winner_columns]


def _tournament_indices(
    n_items: int,
    tournament_size: int,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw unique indices per tournament row in a vectorized pass."""
    if count <= 0 or n_items <= 0:
        return np.empty((0, 0), dtype=int)
    size = max(1, min(tournament_size, n_items))
    if size == n_items:
        picks = np.broadcast_to(
            np.arange(n_items, dtype=int), (count, n_items)
        )
    else:
        # Draw in bulk, then repair duplicate rows.  With the small default
        # tournament size this normally needs zero repair passes, while still
        # guaranteeing the without-replacement invariant.
        picks = rng.integers(
            0,
            n_items,
            size=(count, size),
        )
        for column in range(1, size):
            duplicate = np.any(
                picks[:, column, None] == picks[:, :column],
                axis=1,
            )
            while np.any(duplicate):
                picks[duplicate, column] = rng.integers(
                    0,
                    n_items,
                    size=int(np.count_nonzero(duplicate)),
                )
                duplicate = np.any(
                    picks[:, column, None] == picks[:, :column],
                    axis=1,
                )
    return picks


def evolve_programs(
    rows: list[dict[str, Any]],
    target: str,
    variables: list[str],
    *,
    population: int = 40,
    generations: int = 15,
    seed: int = 0,
    top_k: int = 5,
    tournament_k: int = 3,
    eval_backend: EvalBackend | None = None,
    seed_expressions: bool = True,
    parallel_workers: int = 0,
    on_progress: ProgressCallback | None = None,
    force_gpu: bool = False,
    require_gpu: bool = False,
) -> list[ProgramResult]:
    """Genetic search for expressions predicting target; returns top_k programs."""
    rng = random.Random(seed)
    # Selection draws use a vectorized stream separate from tree mutation
    # draws; seeded runs are reproducible, but intentionally not bit-identical
    # to the historical per-child Random.sample trajectory.
    selection_rng = np.random.default_rng(seed + 0x5EED)
    table = FeatureTable.from_rows(rows, columns=[*variables, target])
    y = table.column(target)
    env = table.env(variables)
    requested = normalize_expr_backend(eval_backend)
    backend, mlx_env, torch_env, target_mlx, _ = prepare_device_envs(
        env,
        requested,
        len(rows),
        y,
        force_gpu=force_gpu,
        require_gpu=require_gpu,
        requested_backend=requested,
    )

    def fitness(expr: Expr) -> tuple[float, float]:
        scored = score_expression(
            expr,
            env,
            y,
            backend=backend,
            mlx_env=mlx_env,
            torch_env=torch_env,
            target_mlx=target_mlx,
            need_values=False,
        )
        if scored is None:
            return -1.0, 0.0
        _, pr, _ = scored
        if abs(pr) < 1e-12:
            return -1.0, 0.0
        return abs(pr) - 0.01 * expr.size(), pr

    pool: list[Expr] = []
    if seed_expressions:
        pool.extend(list(enumerate_expressions(variables, max_depth=2, max_candidates=population))[: population // 2])
    while len(pool) < population:
        pool.append(_random_expr(variables, 3, rng))
    hall_of_fame: dict[Expr, ProgramResult] = {}

    for gen in range(generations):
        workers = parallel_workers if parallel_workers > 0 else 1
        if backend == "mlx" and mlx_env is not None and target_mlx is not None:
            batch = score_population(
                pool,
                env,
                y,
                backend=backend,
                mlx_env=mlx_env,
                torch_env=torch_env,
                target_mlx=target_mlx,
            )
            scored: list[tuple[tuple[float, float], Expr]] = []
            for expr, parts in zip(pool, batch):
                if parts is None:
                    scored.append(((-1.0, 0.0), expr))
                else:
                    _r2, pr, _values = parts
                    fit = abs(pr) - 0.01 * expr.size()
                    if abs(pr) < 1e-12:
                        fit = -1.0
                    scored.append(((fit, pr), expr))
        elif workers > 1 and len(pool) >= 16:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                scored = [(fit, expr) for fit, expr in zip(executor.map(fitness, pool), pool)]
        else:
            scored = [(fitness(e), e) for e in pool]
        scored.sort(key=lambda t: -t[0][0])
        for (fit_score, pr), expr in scored[:top_k]:
            prev = hall_of_fame.get(expr)
            if prev is None or fit_score > prev.fitness:
                hall_of_fame[expr] = ProgramResult(
                    str(expr), float(fit_score), float(pr), gen
                )
        survivors = [e for _, e in scored[: population // 2]]
        next_pool = list(survivors)
        children = max(0, population - len(next_pool))
        branch_rolls = selection_rng.random(children)
        tournament_winners = _tournament_winners(
            scored, tournament_k, children * 3, selection_rng
        ).reshape(children, 3)
        for child_index in range(children):
            if branch_rolls[child_index] < 0.6 and len(survivors) >= 2:
                a = scored[tournament_winners[child_index, 0]][1]
                b = scored[tournament_winners[child_index, 1]][1]
                child = _crossover(a, b, rng)
            else:
                parent = scored[tournament_winners[child_index, 2]][1]
                child = _mutate(parent, variables, rng)
            next_pool.append(child)
        pool = next_pool
        if on_progress is not None:
            best_fit = scored[0][0][0] if scored else -1.0
            on_progress(gen + 1, generations, f"backend={backend} best_fitness={best_fit:.4f}")

    ranked = sorted(hall_of_fame.values(), key=lambda p: -p.fitness)
    return ranked[:top_k]
