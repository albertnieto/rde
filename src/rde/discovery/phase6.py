"""Ship 6 orchestration — representation discovery (programs, coordinates, G4–G5)."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rde.discovery.context import DiscoveryContext, StageTimer
from rde.discovery.coordinates import CoordinateResult, search_linear_coordinates
from rde.discovery.datasets import load_rows_from_source
from rde.discovery.operator import OperatorResult, learn_update_predictor
from rde.expression.batch import ProgressCallback, normalize_expr_backend
from rde.backends.resolve import is_gpu_expr_backend
from rde.discovery.programs import ProgramResult, evolve_programs
from rde.discovery.rediscovery import RediscoveryAssessment, assess_rediscovery
from rde.discovery.representation import RepresentationResult, discover_representation_encoder_decoder
from rde.discovery.walsh_synthesis import WalshSynthesisResult, run_walsh_synthesis
from rde.io.json_util import json_default
from rde.runtime.targets import resolve_target_for_run


@dataclass
class Phase6Report:
    run_id: str
    target: str
    n_rows: int
    representation: RepresentationResult | None = None
    walsh_synthesis: WalshSynthesisResult | None = None
    programs: list[ProgramResult] = field(default_factory=list)
    coordinates: CoordinateResult | None = None
    operator: OperatorResult | None = None
    rediscovery: RediscoveryAssessment | None = None
    g4_met: bool = False
    g5_met: bool = False
    g5_hint: bool = False
    compute_backend: str = "numpy"
    timings: dict[str, float] = field(default_factory=dict)
    stage_errors: list[dict[str, Any]] = field(default_factory=list)

    def to_phase6_dict(self) -> dict[str, Any]:
        rep = self.representation
        walsh = self.walsh_synthesis
        redisc = self.rediscovery
        return {
            "g4_met": self.g4_met,
            "g5_met": self.g5_met,
            "g5_hint": self.g5_hint,
            "compute_backend": self.compute_backend,
            "representation": {
                "method": rep.method if rep else None,
                "n_samples": rep.n_samples if rep else 0,
                "latent_dim": rep.latent_dim if rep else 0,
                "state_reconstruction_r2": rep.state_reconstruction_r2 if rep else None,
                "latent_update_r2": rep.latent_update_r2 if rep else None,
                "target_correlation": rep.target_correlation if rep else None,
                "g4_met": rep.g4_met if rep else False,
                "latent_descriptor_hits": rep.latent_descriptor_hits[:5] if rep else [],
            },
            "walsh_synthesis": {
                "method": walsh.method if walsh else None,
                "n_candidates": walsh.n_candidates if walsh else 0,
                "walsh_variables": walsh.walsh_variables[:12] if walsh else [],
                "top_programs": [
                    {
                        "ansatz": p.ansatz,
                        "expression": p.expression,
                        "fitness": p.fitness,
                        "pearson_r": p.pearson_r,
                        "reconstruction_r2": p.reconstruction_r2,
                    }
                    for p in (walsh.top_programs if walsh else [])
                ],
                "library_fragments": walsh.library_fragments if walsh else [],
                "dreamcoder_programs": [
                    {"expression": p.expression, "fitness": p.fitness, "pearson_r": p.pearson_r}
                    for p in (walsh.dreamcoder_programs if walsh else [])
                ],
            },
            "programs": [
                {"expression": p.expression, "fitness": p.fitness, "pearson_r": p.pearson_r}
                for p in self.programs
            ],
            "coordinates": {
                "method": self.coordinates.method,
                "n_components": self.coordinates.n_components,
                "target_correlation": self.coordinates.target_correlation,
                "update_simplicity": self.coordinates.update_simplicity,
            }
            if self.coordinates
            else {},
            "operator": {
                "method": self.operator.method,
                "train_r_squared": self.operator.train_r_squared,
                "n_rows": self.operator.n_rows,
            }
            if self.operator
            else {},
            "rediscovery": {
                "g5_met": redisc.g5_met if redisc else False,
                "g5_hint": redisc.g5_hint if redisc else False,
                "dominant_object": redisc.dominant_object if redisc else None,
                "rediscovery_frequency": redisc.rediscovery_frequency if redisc else 0.0,
                "splits_hit": redisc.splits_hit if redisc else 0,
                "cross_n_stable": redisc.cross_n_stable if redisc else False,
                "cross_generator_stable": redisc.cross_generator_stable if redisc else False,
                "canonical_forms": redisc.canonical_forms[:5] if redisc else [],
                "criteria": redisc.criteria if redisc else {},
            },
            "stage_errors": self.stage_errors,
            "timings": self.timings,
        }


def _phase6_walsh_worker(payload: dict[str, Any]) -> WalshSynthesisResult:
    return run_walsh_synthesis(**payload)


def _phase6_program_worker(payload: dict[str, Any]) -> list[ProgramResult]:
    return evolve_programs(**payload)


def _phase6_parallel_gp_enabled(generations: int, population: int) -> bool:
    setting = os.environ.get("RDE_GP_PARALLEL", "auto").lower()
    if setting in {"0", "false", "no", "off"}:
        return False
    if setting in {"1", "true", "yes", "on"}:
        return True
    return generations * population >= 100_000


def _phase6_gpu_locked(
    expression_backend: str | None,
    compute_backend: str | None,
    require_gpu: bool,
) -> bool:
    """Whether Phase 6 must keep GPU-owning stages in one process."""
    return (
        require_gpu
        or (compute_backend or "numpy").lower() == "mlx"
        or is_gpu_expr_backend(expression_backend or "numpy")
    )


def run_phase6(
    run_id: str,
    store_root: Path | str,
    *,
    target: str | None = None,
    gp_generations: int = 15,
    gp_population: int = 48,
    walsh_max_candidates: int = 2000,
    walsh_max_support: int = 6,
    rediscovery_splits: int = 5,
    expr_eval_backend: str | None = None,
    rows: list[dict[str, Any]] | None = None,
    symbolic_equation: str | None = None,
    on_stage: Callable[[str], None] | None = None,
    on_progress: ProgressCallback | None = None,
    require_gpu: bool = False,
    compute_backend: str | None = None,
    context: DiscoveryContext | None = None,
) -> Phase6Report:
    """Run Phase 6 representation discovery on one run ledger."""
    timer = StageTimer(
        on_stage=on_stage,
        on_progress=on_progress if on_stage is None else None,
    )

    store_root = Path(store_root)
    if context is not None:
        resolved_target = context.target
        rows = context.rows
    else:
        resolved_target = resolve_target_for_run(run_id, store_root, override=target)
        if rows is None:
            rows = load_rows_from_source(run_id=run_id, store_root=store_root)
    report = Phase6Report(run_id=run_id, target=resolved_target, n_rows=len(rows))
    if not rows:
        timer.finish()
        report.timings = timer.timings
        return report

    if context is not None:
        X, cols = context.feature_matrix()
    else:
        from rde.discovery.datasets import load_feature_matrix

        X, cols, _ = load_feature_matrix(run_id, store_root, target=resolved_target)
    target_arr = X[:, cols.index(resolved_target)]
    feat_cols = [c for c in cols if c != resolved_target][:32]
    feat_idx = [cols.index(c) for c in feat_cols]

    rep_batch = context.representation_batch() if context is not None else None
    state_backend = compute_backend or (
        "mlx"
        if require_gpu or normalize_expr_backend(expr_eval_backend) == "mlx"
        else "numpy"
    )
    report.compute_backend = state_backend
    from rde.runtime.resilience import soft_call

    timer.begin("representation encoder/decoder")
    report.representation = soft_call(
        "phase6.representation",
        lambda: discover_representation_encoder_decoder(
            run_id,
            store_root,
            target=resolved_target,
            rows=rows,
            checkpoint=True,
            batch=rep_batch,
            compute_backend=state_backend,
        ),
        fallback=None,
        errors=report.stage_errors,
        on_error=on_stage,
    )
    report.g4_met = bool(report.representation and report.representation.g4_met)

    if rep_batch is None:
        from rde.discovery.datasets import load_representation_batch

        rep_batch = load_representation_batch(run_id, store_root, target=resolved_target)
    state_vectors = rep_batch.state_vectors if rep_batch.n_samples else None

    timer.begin("walsh generating-function search")
    feature_matrix = X[:, feat_idx]
    paired = (
        np.isfinite(feature_matrix)
        & (np.abs(feature_matrix) < 1e100)
        & np.isfinite(target_arr)[:, None]
        & (np.abs(target_arr) < 1e100)[:, None]
    )
    counts = paired.sum(axis=0)
    safe_counts = np.maximum(counts, 1)
    x_values = np.where(paired, feature_matrix, 0.0)
    y_values = np.where(paired, target_arr[:, None], 0.0)
    x_mean = x_values.sum(axis=0) / safe_counts
    y_mean = y_values.sum(axis=0) / safe_counts
    x_centered = np.where(paired, feature_matrix - x_mean, 0.0)
    y_centered = np.where(paired, target_arr[:, None] - y_mean, 0.0)
    covariance = np.sum(x_centered * y_centered, axis=0)
    denominator = np.sqrt(
        np.sum(x_centered * x_centered, axis=0)
        * np.sum(y_centered * y_centered, axis=0)
    )
    correlations = np.full(len(feat_cols), np.nan, dtype=float)
    valid = (counts >= 2) & (denominator > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlations[valid] = covariance[valid] / denominator[valid]
    corr_hits = [
        {"column": c, "r": abs(float(pr))}
        for c, pr in zip(feat_cols, correlations)
        if np.isfinite(pr) and abs(pr) > 0
    ]
    gp_vars = [h["column"] for h in sorted(corr_hits, key=lambda d: -d["r"])[:8]] or feat_cols[:8]
    parallel_gp = _phase6_parallel_gp_enabled(gp_generations, gp_population)
    parallel_backend = os.environ.get("RDE_GP_PARALLEL_BACKEND", expr_eval_backend)
    parallel_backend_resolved = normalize_expr_backend(parallel_backend)
    parallel_gpu_locked = _phase6_gpu_locked(
        parallel_backend,
        state_backend,
        require_gpu,
    )
    parallel_require_gpu = require_gpu or parallel_backend_resolved == "mlx"
    walsh_payload = {
        "rows": rows,
        "target": resolved_target,
        "variables": gp_vars,
        "state_vectors": state_vectors,
        "max_support": walsh_max_support,
        "max_candidates": walsh_max_candidates,
        "gp_generations": gp_generations,
        "gp_population": gp_population,
        "eval_backend": parallel_backend,
        "require_gpu": parallel_require_gpu,
        "compute_backend": state_backend,
    }
    program_payload = {
        "rows": rows,
        "target": resolved_target,
        "variables": gp_vars,
        "generations": gp_generations,
        "population": gp_population,
        "top_k": 5,
        "eval_backend": parallel_backend,
        "require_gpu": parallel_require_gpu,
    }

    def progress_for(label: str) -> ProgressCallback | None:
        if on_progress is None:
            return None

        def _callback(done: int, total: int, detail: str) -> None:
            on_progress(done, total, f"{label}: {detail}")

        return _callback

    empty_walsh = WalshSynthesisResult(
        method="skipped",
        n_candidates=0,
        walsh_variables=[],
        top_programs=[],
        library_fragments=[],
        dreamcoder_programs=[],
    )

    def _run_walsh_gp_sequential() -> tuple[WalshSynthesisResult, list[ProgramResult]]:
        walsh_local = soft_call(
            "phase6.walsh",
            lambda: run_walsh_synthesis(
                **walsh_payload,
                on_progress=progress_for("walsh search"),
            ),
            fallback=empty_walsh,
            errors=report.stage_errors,
            on_error=on_stage,
        )
        programs_local = soft_call(
            "phase6.genetic_programs",
            lambda: evolve_programs(
                **program_payload,
                on_progress=progress_for("genetic programs"),
            ),
            fallback=[],
            errors=report.stage_errors,
            on_error=on_stage,
        )
        return walsh_local, programs_local

    if parallel_gp and not parallel_gpu_locked:
        timer.begin("walsh + genetic programs (parallel)")
        try:
            with ProcessPoolExecutor(max_workers=2) as executor:
                walsh_future = executor.submit(_phase6_walsh_worker, walsh_payload)
                programs_future = executor.submit(_phase6_program_worker, program_payload)
                walsh = walsh_future.result()
                report.programs = programs_future.result()
        except Exception as exc:  # noqa: BLE001 — preserve a sequential fallback
            timer.begin("walsh + genetic programs (sequential fallback)")
            if on_stage is not None:
                on_stage(f"parallel GP unavailable: {exc}")
            walsh, report.programs = _run_walsh_gp_sequential()
    else:
        if parallel_gp and parallel_gpu_locked and on_stage is not None:
            on_stage("GPU expression backend selected; running Walsh and GP sequentially")
        timer.begin("genetic programs")
        walsh, report.programs = _run_walsh_gp_sequential()
    report.walsh_synthesis = walsh

    update_proxy = np.array([row.get("update.step_ratio_mean", float("nan")) for row in rows], dtype=float)
    timer.begin("coordinates + operator")
    report.coordinates = soft_call(
        "phase6.coordinates",
        lambda: search_linear_coordinates(
            X[:, feat_idx],
            target=target_arr,
            update_proxy=update_proxy,
        ),
        fallback=None,
        errors=report.stage_errors,
        on_error=on_stage,
    )
    report.operator = soft_call(
        "phase6.operator",
        lambda: learn_update_predictor(rows, feat_cols, update_key="update.step_ratio_mean"),
        fallback=None,
        errors=report.stage_errors,
        on_error=on_stage,
    )

    timer.begin("level-5 rediscovery")
    sym_equation = symbolic_equation
    program_dicts = [
        {"expression": p.expression, "fitness": p.fitness, "pearson_r": p.pearson_r}
        for p in report.programs
    ]
    walsh_programs = walsh.top_programs if walsh is not None else []
    report.rediscovery = soft_call(
        "phase6.rediscovery",
        lambda: assess_rediscovery(
            rows,
            resolved_target,
            programs=program_dicts,
            walsh_programs=walsh_programs,
            symbolic_equation=sym_equation,
            variables=gp_vars,
            n_splits=rediscovery_splits,
            eval_backend=expr_eval_backend,
            require_gpu=require_gpu,
            on_progress=progress_for("rediscovery"),
        ),
        fallback=RediscoveryAssessment(
            g5_met=False,
            g5_hint=False,
            dominant_object=None,
            rediscovery_frequency=0.0,
            n_splits=rediscovery_splits,
            splits_hit=0,
            cross_n_stable=False,
            cross_generator_stable=False,
            criteria={"soft_failed": True},
        ),
        errors=report.stage_errors,
        on_error=on_stage,
    )
    report.g5_met = bool(report.rediscovery and report.rediscovery.g5_met)
    report.g5_hint = bool(report.rediscovery and report.rediscovery.g5_hint)
    timer.finish()
    report.timings = timer.timings
    return report


def write_phase6_report(report: Phase6Report, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": report.run_id,
        "target": report.target,
        "n_rows": report.n_rows,
        "phase6": report.to_phase6_dict(),
        "timings": report.timings,
    }
    out.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    return out

