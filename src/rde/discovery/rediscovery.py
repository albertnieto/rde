"""Level-5 rediscovery assessment — same hidden object across Q, N, r splits."""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rde.analyze.feature_table import FeatureTable
from rde.discovery.programs import ProgramResult, evolve_programs
from rde.discovery.walsh_synthesis import WalshProgram
from rde.expression.batch import (
    EvalBackend,
    ProgressCallback,
    normalize_expr_backend,
    prepare_device_envs,
    score_population,
)
from rde.expression.parse import parse_expr


@dataclass
class RediscoveryAssessment:
    g5_met: bool
    g5_hint: bool
    dominant_object: str | None
    rediscovery_frequency: float
    n_splits: int
    splits_hit: int
    cross_n_stable: bool
    cross_generator_stable: bool
    canonical_forms: list[dict[str, Any]] = field(default_factory=list)
    criteria: dict[str, Any] = field(default_factory=dict)


def canonicalize_expression(expr: str) -> str:
    """Normalize expression strings for cross-run comparison."""
    s = str(expr).strip().lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace("**", "^")
    s = re.sub(r"\+0(\.0+)?(?=[+\-*\)/]|$)", "", s)
    s = re.sub(r"(?<=[+\-*/(])0(\.0+)?(?=[+\-*\)/]|$)", "", s)
    return s or "empty"


def _collect_seed_objects(
    programs: list[dict[str, Any]] | None,
    walsh_programs: list[WalshProgram] | None,
    symbolic_equation: str | None,
) -> list[tuple[str, str, str, float]]:
    """Return (canonical, original_expr, source, fitness) tuples."""
    seeds: list[tuple[str, str, str, float]] = []
    for prog in programs or []:
        expr = str(prog.get("expression", ""))
        if not expr:
            continue
        seeds.append(
            (
                canonicalize_expression(expr),
                expr,
                "gp",
                float(prog.get("fitness", prog.get("pearson_r", 0.0))),
            )
        )
    for wp in walsh_programs or []:
        seeds.append((canonicalize_expression(wp.expression), wp.expression, "walsh", float(wp.fitness)))
    if symbolic_equation:
        seeds.append((canonicalize_expression(symbolic_equation), symbolic_equation, "symbolic", 1.0))
    best: dict[str, tuple[str, str, float]] = {}
    for canon, expr, source, fit in seeds:
        prev = best.get(canon)
        if prev is None or fit > prev[2]:
            best[canon] = (expr, source, fit)
    return [(c, e, s, f) for c, (e, s, f) in best.items()]


def _score_seed_programs(
    rows: list[dict[str, Any]],
    target: str,
    variables: list[str],
    seeds: list[tuple[str, str, str, float]],
    *,
    min_fitness: float,
    eval_backend: EvalBackend | None = None,
    require_gpu: bool = False,
    table: FeatureTable | None = None,
    on_progress: ProgressCallback | None = None,
) -> ProgramResult | None:
    """Re-score known seed expressions on a row subset before spawning GP."""
    if not rows or not seeds or not variables:
        return None
    if table is None:
        table = FeatureTable.from_rows(rows, columns=[*variables, target])
    env = table.env(variables)
    y = table.column(target)
    backend, mlx_env, torch_env, target_mlx, _ = prepare_device_envs(
        env,
        normalize_expr_backend(eval_backend),
        len(rows),
        y,
        require_gpu=require_gpu,
        requested_backend=normalize_expr_backend(eval_backend),
    )
    parsed: list[tuple[Any, Any]] = []
    from rde.runtime.resilience import SOFT_FAIL_EXCEPTIONS

    for _canon, expr_str, _source, _fit in seeds:
        if _source == "walsh":
            continue
        try:
            tree = parse_expr(expr_str)
        except (ValueError, RecursionError, MemoryError, TypeError):
            continue
        parsed.append((tree, expr_str))
    if not parsed:
        return None

    try:
        scores = score_population(
            [tree for tree, _expr_str in parsed],
            env,
            y,
            backend=backend,
            mlx_env=mlx_env,
            torch_env=torch_env,
            target_mlx=target_mlx,
        )
    except SOFT_FAIL_EXCEPTIONS:
        # Score seeds individually so one deep AST cannot wipe the seed set.
        from rde.expression.batch import score_expression

        scores = []
        for tree, _expr_str in parsed:
            try:
                scores.append(
                    score_expression(
                        tree,
                        env,
                        y,
                        backend=backend,
                        mlx_env=mlx_env,
                        torch_env=torch_env,
                        target_mlx=target_mlx,
                        need_values=False,
                    )
                )
            except SOFT_FAIL_EXCEPTIONS:
                scores.append(None)
    best: ProgramResult | None = None
    for (tree, _expr_str), scored in zip(parsed, scores):
        if scored is None:
            continue
        _r2, pr, _ = scored
        fitness = abs(pr) - 0.01 * tree.size()
        if abs(pr) < min_fitness and fitness < min_fitness:
            continue
        candidate = ProgramResult(str(tree), float(fitness), float(pr), generation=-1)
        if best is None or candidate.fitness > best.fitness:
            best = candidate
    return best


def _winner_on_subset(
    rows: list[dict[str, Any]],
    target: str,
    variables: list[str],
    seeds: list[tuple[str, str, str, float]],
    *,
    min_fitness: float,
    seed: int,
    eval_backend: EvalBackend | None = None,
    gp_generations: int = 6,
    gp_population: int = 24,
    allow_gp: bool = True,
    require_gpu: bool = False,
    table: FeatureTable | None = None,
    on_progress: ProgressCallback | None = None,
) -> ProgramResult | None:
    seeded = _score_seed_programs(
        rows,
        target,
        variables,
        seeds,
        min_fitness=min_fitness,
        eval_backend=eval_backend,
        require_gpu=require_gpu,
        table=table,
    )
    if seeded is not None:
        return seeded
    if not allow_gp:
        return None
    progs = evolve_programs(
        rows,
        target,
        variables,
        generations=gp_generations,
        population=gp_population,
        seed=seed,
        top_k=1,
        seed_expressions=True,
        eval_backend=eval_backend,
        require_gpu=require_gpu,
        on_progress=on_progress,
    )
    return progs[0] if progs else None


def assess_rediscovery(
    rows: list[dict[str, Any]],
    target: str,
    *,
    programs: list[dict[str, Any]] | None = None,
    walsh_programs: list[WalshProgram] | None = None,
    symbolic_equation: str | None = None,
    variables: list[str] | None = None,
    n_splits: int = 5,
    min_frequency: float = 0.6,
    min_fitness: float = 0.7,
    seed: int = 0,
    eval_backend: EvalBackend | None = None,
    require_gpu: bool = False,
    on_progress: ProgressCallback | None = None,
) -> RediscoveryAssessment:
    """Bootstrap rediscovery: does the same object win on random row subsets?"""
    from rde.runtime.resilience import soft_call

    empty = RediscoveryAssessment(
        g5_met=False,
        g5_hint=False,
        dominant_object=None,
        rediscovery_frequency=0.0,
        n_splits=n_splits,
        splits_hit=0,
        cross_n_stable=False,
        cross_generator_stable=False,
    )
    return soft_call(
        "rediscovery",
        lambda: _assess_rediscovery_inner(
            rows,
            target,
            programs=programs,
            walsh_programs=walsh_programs,
            symbolic_equation=symbolic_equation,
            variables=variables,
            n_splits=n_splits,
            min_frequency=min_frequency,
            min_fitness=min_fitness,
            seed=seed,
            eval_backend=eval_backend,
            require_gpu=require_gpu,
            on_progress=on_progress,
            empty=empty,
        ),
        fallback=RediscoveryAssessment(
            g5_met=False,
            g5_hint=False,
            dominant_object=None,
            rediscovery_frequency=0.0,
            n_splits=n_splits,
            splits_hit=0,
            cross_n_stable=False,
            cross_generator_stable=False,
            criteria={"soft_failed": True},
        ),
    )


def _assess_rediscovery_inner(
    rows: list[dict[str, Any]],
    target: str,
    *,
    programs: list[dict[str, Any]] | None,
    walsh_programs: list[WalshProgram] | None,
    symbolic_equation: str | None,
    variables: list[str] | None,
    n_splits: int,
    min_frequency: float,
    min_fitness: float,
    seed: int,
    eval_backend: EvalBackend | None,
    require_gpu: bool,
    on_progress: ProgressCallback | None,
    empty: RediscoveryAssessment,
) -> RediscoveryAssessment:
    if len(rows) < max(8, n_splits * 2):
        return empty

    vars_ = variables or [
        c
        for c in sorted({k for row in rows for k in row if isinstance(row.get(k), (int, float))})
        if c != target and not str(c).startswith("metric.")
    ][:8]
    if not vars_:
        return empty

    seed_objects = _collect_seed_objects(programs, walsh_programs, symbolic_equation)
    backend = normalize_expr_backend(eval_backend)
    table = FeatureTable.from_rows(rows, columns=[*vars_, target])

    def subset_progress(scope: str) -> ProgressCallback | None:
        if on_progress is None:
            return None

        def _callback(done: int, total: int, detail: str) -> None:
            on_progress(done, total, f"{scope}: {detail}")

        return _callback

    full_seed_winner = _score_seed_programs(
        rows,
        target,
        vars_,
        seed_objects,
        min_fitness=min_fitness,
        eval_backend=backend,
        require_gpu=require_gpu,
        table=table,
    )
    seeds_sufficient = (
        full_seed_winner is not None
        and (
            abs(full_seed_winner.pearson_r) >= min_fitness
            or full_seed_winner.fitness >= min_fitness
        )
    )

    from rde.runtime.resilience import soft_call

    winners: dict[str, int] = {}
    split_winners: list[str] = []
    for split_idx in range(n_splits):
        split_rng = random.Random(seed + 1000 + split_idx)
        idx = split_rng.sample(range(len(rows)), max(4, int(0.7 * len(rows))))
        sub_table = table.take(idx)
        winner = soft_call(
            f"rediscovery_split_{split_idx + 1}",
            lambda idx=split_idx, sub=sub_table: _winner_on_subset(
                sub.rows,
                target,
                vars_,
                seed_objects,
                min_fitness=min_fitness,
                seed=seed + idx,
                eval_backend=backend,
                require_gpu=require_gpu,
                allow_gp=not seeds_sufficient,
                table=sub,
                on_progress=subset_progress(f"rediscovery split {idx + 1}"),
            ),
            fallback=None,
        )
        if on_progress is not None:
            on_progress(
                split_idx + 1,
                n_splits,
                f"rediscovery split {split_idx + 1}/{n_splits} complete",
            )
        if winner is None:
            continue
        canon = canonicalize_expression(winner.expression)
        if abs(winner.pearson_r) >= min_fitness or winner.fitness >= min_fitness:
            winners[canon] = winners.get(canon, 0) + 1
            split_winners.append(canon)

    if not winners:
        return empty

    dominant, hits = max(winners.items(), key=lambda t: t[1])
    frequency = hits / max(n_splits, 1)
    seed_canons = {c for c, _, _, _ in seed_objects}
    matches_seed = dominant in seed_canons

    size_values = np.asarray(
        [int(r["size"]) if r.get("size") is not None else -1 for r in rows],
        dtype=int,
    )
    sizes = sorted(set(size_values[size_values >= 0].tolist()))
    cross_n = False
    g5_hint = frequency >= min_frequency and matches_seed
    if g5_hint and len(sizes) >= 2:
        per_size: dict[int, str | None] = {}
        for size in sizes:
            size_indices = np.flatnonzero(size_values == size)
            if size_indices.size < 4:
                continue
            sub_table = table.take(size_indices)
            winner = _winner_on_subset(
                sub_table.rows,
                target,
                vars_,
                seed_objects,
                min_fitness=min_fitness,
                seed=seed + size,
                gp_generations=4,
                gp_population=16,
                eval_backend=backend,
                allow_gp=not seeds_sufficient,
                table=sub_table,
                on_progress=subset_progress(f"rediscovery N={size}"),
            )
            per_size[size] = canonicalize_expression(winner.expression) if winner else None
        canon_values = [v for v in per_size.values() if v]
        cross_n = len(canon_values) >= 2 and len(set(canon_values)) == 1

    generator_values = np.asarray(
        [str(r.get("generator", "")) for r in rows],
        dtype=str,
    )
    generators = sorted(set(generator_values[generator_values != ""].tolist()))
    cross_gen = False
    if g5_hint and len(generators) >= 2:
        per_gen: dict[str, str | None] = {}
        for gen in generators[:4]:
            generator_indices = np.flatnonzero(generator_values == gen)
            if generator_indices.size < 4:
                continue
            sub_table = table.take(generator_indices)
            winner = _winner_on_subset(
                sub_table.rows,
                target,
                vars_,
                seed_objects,
                min_fitness=min_fitness,
                seed=int.from_bytes(
                    hashlib.sha256(gen.encode("utf-8")).digest()[:8], "little"
                )
                % 997,
                gp_generations=4,
                gp_population=16,
                eval_backend=backend,
                allow_gp=not seeds_sufficient,
                table=sub_table,
                on_progress=subset_progress(f"rediscovery generator={gen}"),
            )
            per_gen[gen] = canonicalize_expression(winner.expression) if winner else None
        gen_values = [v for v in per_gen.values() if v]
        cross_gen = len(gen_values) >= 2 and len(set(gen_values)) == 1

    g5_met = g5_hint and (cross_n or cross_gen) and hits >= max(2, int(min_frequency * n_splits))

    canonical_forms = [
        {"expression": canon, "split_wins": count, "in_seed_pool": canon in seed_canons}
        for canon, count in sorted(winners.items(), key=lambda t: -t[1])
    ]

    return RediscoveryAssessment(
        g5_met=g5_met,
        g5_hint=g5_hint,
        dominant_object=dominant,
        rediscovery_frequency=frequency,
        n_splits=n_splits,
        splits_hit=hits,
        cross_n_stable=cross_n,
        cross_generator_stable=cross_gen,
        canonical_forms=canonical_forms[:10],
        criteria={
            "min_frequency": min_frequency,
            "min_fitness": min_fitness,
            "matches_seed_pool": matches_seed,
            "n_seed_objects": len(seed_objects),
            "seeds_sufficient": seeds_sufficient,
        },
    )
