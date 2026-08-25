"""Full discovery loop — Phases 3–6 orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rde.analyze.query import correlate_with_target
from rde.discovery.coordinates import search_linear_coordinates
from rde.discovery.context import DiscoveryContext, StageTimer
from rde.discovery.datasets import broadcast_instance_latents
from rde.discovery.phase4 import run_phase4
from rde.discovery.phase5 import run_phase5
from rde.discovery.phase6 import run_phase6
from rde.discovery.promote import promote_top_conjectures
from rde.discovery.symbolic import LatentSource
from rde.expression.batch import ProgressCallback, normalize_expr_backend
from rde.expression.enumerate import enumerate_expressions
from rde.expression.generators import enumerate_metric_candidates, metric_variable_columns
from rde.discovery.operator import learn_update_predictor
from rde.analyze.outcome import assess_outcome
from rde.analyze.ranker import ConjectureRanker, candidate_universe_fingerprint
from rde.descriptor_gen.enumerate import enumerate_descriptor_templates
from rde.descriptor_gen.materialize import load_row_arrays
from rde.descriptor_gen.rank import candidates_to_records, rank_descriptor_generators
from rde.runtime.targets import resolve_target_for_run


@dataclass
class DiscoveryReport:
    run_id: str
    target: str
    n_rows: int
    top_correlations: list[dict[str, Any]] = field(default_factory=list)
    metric_candidates: list[dict[str, Any]] = field(default_factory=list)
    descriptor_candidates: list[dict[str, Any]] = field(default_factory=list)
    latent: dict[str, Any] = field(default_factory=dict)
    symbolic: dict[str, Any] = field(default_factory=dict)
    promoted_conjectures: list[dict[str, Any]] = field(default_factory=list)
    cross_n_rule: list[dict[str, Any]] = field(default_factory=list)
    programs: list[dict[str, Any]] = field(default_factory=list)
    coordinates: dict[str, Any] = field(default_factory=dict)
    operator: dict[str, Any] = field(default_factory=dict)
    phase6: dict[str, Any] = field(default_factory=dict)
    outcome: dict[str, Any] = field(default_factory=dict)
    outcome_grade_hint: int = 0
    leak_audit_summary: dict[str, Any] = field(default_factory=dict)
    obstruct_summary: dict[str, Any] = field(default_factory=dict)
    lower_bound_conjectures: list[dict[str, Any]] = field(default_factory=list)
    certify_results: list[dict[str, Any]] = field(default_factory=list)
    promotion_rejections: list[dict[str, Any]] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    expr_eval_backend_requested: str | None = None
    expr_eval_backend_effective: str | None = None
    expr_eval_fallback_reason: str | None = None
    massive_catalog_skipped_no_arrays: bool = False
    stage_errors: list[dict[str, Any]] = field(default_factory=list)


def run_discovery(
    run_id: str,
    store_root: Path | str,
    *,
    target: str | None = None,
    max_expr_candidates: int = 20_000,
    max_expr_depth: int = 3,
    max_descriptor_templates: int | None = None,
    max_derived_descriptors: int = 3000,
    max_derived_depth: int = 2,
    gp_generations: int = 15,
    gp_population: int = 48,
    use_pysr: bool = True,
    use_operon: bool = True,
    use_native_gp: bool = True,
    use_ai_feynman: bool = False,
    use_massive_catalog: bool = False,
    top_features: int = 8,
    top_latents: int = 3,
    max_promoted_conjectures: int = 10,
    prefer_torch_autoencoder: bool = False,
    on_stage: Callable[[str], None] | None = None,
    on_progress: ProgressCallback | None = None,
    expr_eval_backend: str | None = None,
    prefer_mlx_autoencoder: bool = False,
    run_obstruct: bool = False,
    operon_generations: int = 15,
    operon_population: int = 48,
    walsh_max_candidates: int = 5000,
    walsh_max_support: int = 12,
    rediscovery_splits: int = 3,
    promotion_blocked: bool = False,
    force_gpu: bool = False,
    require_gpu: bool = False,
) -> DiscoveryReport:
    """Run Phases 3–6; return a partial report instead of raising on fatal errors."""
    from rde.runtime.heartbeat import throttled_progress
    from rde.runtime.progress import default_discovery_progress
    from rde.runtime.resilience import SOFT_FAIL_EXCEPTIONS, failure_record

    if on_stage is None or on_progress is None:
        default_progress = default_discovery_progress()
        if on_stage is None:
            on_stage = default_progress.on_stage
        if on_progress is None:
            on_progress = throttled_progress(
                default_progress.on_progress,
                min_interval_s=2.0,
            )

    try:
        return _run_discovery_body(
            run_id,
            store_root,
            target=target,
            max_expr_candidates=max_expr_candidates,
            max_expr_depth=max_expr_depth,
            max_descriptor_templates=max_descriptor_templates,
            max_derived_descriptors=max_derived_descriptors,
            max_derived_depth=max_derived_depth,
            gp_generations=gp_generations,
            gp_population=gp_population,
            use_pysr=use_pysr,
            use_operon=use_operon,
            use_native_gp=use_native_gp,
            use_ai_feynman=use_ai_feynman,
            use_massive_catalog=use_massive_catalog,
            top_features=top_features,
            top_latents=top_latents,
            max_promoted_conjectures=max_promoted_conjectures,
            prefer_torch_autoencoder=prefer_torch_autoencoder,
            on_stage=on_stage,
            on_progress=on_progress,
            expr_eval_backend=expr_eval_backend,
            prefer_mlx_autoencoder=prefer_mlx_autoencoder,
            run_obstruct=run_obstruct,
            operon_generations=operon_generations,
            operon_population=operon_population,
            walsh_max_candidates=walsh_max_candidates,
            walsh_max_support=walsh_max_support,
            rediscovery_splits=rediscovery_splits,
            promotion_blocked=promotion_blocked,
            force_gpu=force_gpu,
            require_gpu=require_gpu,
        )
    except SOFT_FAIL_EXCEPTIONS as exc:
        report = DiscoveryReport(
            run_id=run_id,
            target=target or "",
            n_rows=0,
        )
        report.stage_errors.append(failure_record(stage="run_discovery_fatal", exc=exc))
        if on_stage is not None:
            on_stage(f"run_discovery_fatal: {type(exc).__name__}: {exc}")
        _LOG = __import__("logging").getLogger(__name__)
        _LOG.exception("run_discovery fatal for run_id=%s", run_id)
        return report


def _run_discovery_body(
    run_id: str,
    store_root: Path | str,
    *,
    target: str | None = None,
    max_expr_candidates: int = 20_000,
    max_expr_depth: int = 3,
    max_descriptor_templates: int | None = None,
    max_derived_descriptors: int = 3000,
    max_derived_depth: int = 2,
    gp_generations: int = 15,
    gp_population: int = 48,
    use_pysr: bool = True,
    use_operon: bool = True,
    use_native_gp: bool = True,
    use_ai_feynman: bool = False,
    use_massive_catalog: bool = False,
    top_features: int = 32,
    top_latents: int = 3,
    max_promoted_conjectures: int = 10,
    promote_min_r_squared: float = 0.5,
    prefer_torch_autoencoder: bool = False,
    prefer_mlx_autoencoder: bool | None = None,
    expr_eval_backend: str | None = None,
    run_obstruct: bool = False,
    pysr_iterations: int = 40,
    pysr_maxsize: int = 20,
    pysr_populations: int = 15,
    operon_generations: int = 50,
    operon_population: int = 300,
    walsh_max_candidates: int = 2000,
    walsh_max_support: int = 6,
    rediscovery_splits: int = 5,
    promotion_blocked: bool = False,
    on_stage: Callable[[str], None] | None = None,
    on_progress: ProgressCallback | None = None,
    force_gpu: bool = False,
    require_gpu: bool = False,
) -> DiscoveryReport:
    from rde.discovery.checkpoint import (
        load_stage_progress,
        population_fingerprint,
        save_stage_progress,
        stage_completed,
    )
    from rde.runtime.resilience import soft_call

    timer = StageTimer(on_stage=on_stage, on_progress=on_progress)
    store_root = Path(store_root)

    timer.begin("load features")
    ctx = DiscoveryContext.load(run_id, store_root, target=target)
    resolved_target = ctx.target
    domain_id: str | None = None

    def _domain_id() -> str:
        nonlocal domain_id
        if domain_id is None:
            from rde.io.store import Store

            domain_id = Store(store_root).read_manifest(run_id).domain_id
        return domain_id

    def _domain_id_or_none() -> str | None:
        """Best-effort `_domain_id()` for optional, fail-open contract filtering.

        Candidate-variable-selection call sites (`domain_id=` on
        `metric_variable_columns`/`rank_descriptor_generators`) must not
        turn a missing/malformed manifest into a fatal error — `domain_id`
        there is already fail-open (`None` just skips contract filtering);
        only the call sites that genuinely require a domain_id use
        `_domain_id()` directly.
        """
        try:
            return _domain_id()
        except (FileNotFoundError, OSError, ValueError, KeyError):
            return None

    rows = ctx.rows
    report = DiscoveryReport(run_id=run_id, target=resolved_target, n_rows=len(rows))
    # Keyed on the population, not just the run_id: re-running with a changed
    # population reuses the run_id, and resuming on it alone would score stale
    # stage output against the new rows.
    pop_fingerprint = population_fingerprint(rows, resolved_target)
    stage_progress = load_stage_progress(store_root, run_id, fingerprint=pop_fingerprint)
    report.stage_errors = list(stage_progress.get("errors") or [])
    partial = dict(stage_progress.get("partial") or {})
    completed_stages: list[str] = list(stage_progress.get("completed") or [])

    def _persist_stage(stage: str) -> None:
        if stage not in completed_stages:
            completed_stages.append(stage)
        payload = {
            "top_correlations": report.top_correlations,
            "descriptor_candidates": report.descriptor_candidates,
            "metric_candidates": report.metric_candidates,
            "latent": report.latent,
            "symbolic": report.symbolic,
            "programs": report.programs,
            "coordinates": report.coordinates,
            "operator": report.operator,
            "phase6": report.phase6,
            "promoted_conjectures": report.promoted_conjectures,
            "cross_n_rule": report.cross_n_rule,
            "outcome": report.outcome,
            "obstruct_summary": report.obstruct_summary,
            "certify_results": report.certify_results,
            "expr_eval_backend_requested": report.expr_eval_backend_requested,
            "expr_eval_backend_effective": report.expr_eval_backend_effective,
            "expr_eval_fallback_reason": report.expr_eval_fallback_reason,
            "massive_catalog_skipped_no_arrays": report.massive_catalog_skipped_no_arrays,
        }
        partial.update(payload)
        save_stage_progress(
            store_root,
            run_id,
            completed=completed_stages,
            errors=report.stage_errors,
            partial=partial,
            fingerprint=pop_fingerprint,
        )
        # Durable partial report so a crash never leaves zero discovery artifacts.
        write_discovery_report(
            report,
            store_root / "discovery" / f"{run_id}.partial.json",
        )

    report.parameters = {
        "max_expr_candidates": max_expr_candidates,
        "max_expr_depth": max_expr_depth,
        "max_descriptor_templates": max_descriptor_templates,
        "max_derived_descriptors": max_derived_descriptors,
        "max_derived_depth": max_derived_depth,
        "gp_generations": gp_generations,
        "gp_population": gp_population,
        "use_pysr": use_pysr,
        "use_operon": use_operon,
        "use_native_gp": use_native_gp,
        "use_ai_feynman": use_ai_feynman,
        "use_massive_catalog": use_massive_catalog,
        "top_features": top_features,
        "top_latents": top_latents,
        "max_promoted_conjectures": max_promoted_conjectures,
        "promote_min_r_squared": promote_min_r_squared,
        "prefer_torch_autoencoder": prefer_torch_autoencoder,
        "prefer_mlx_autoencoder": prefer_mlx_autoencoder,
        "expr_eval_backend": expr_eval_backend,
        "run_obstruct": run_obstruct,
        "pysr_iterations": pysr_iterations,
        "pysr_maxsize": pysr_maxsize,
        "pysr_populations": pysr_populations,
        "operon_generations": operon_generations,
        "operon_population": operon_population,
        "walsh_max_candidates": walsh_max_candidates,
        "walsh_max_support": walsh_max_support,
        "rediscovery_splits": rediscovery_splits,
        "promotion_blocked": promotion_blocked,
        "force_gpu": force_gpu,
        "require_gpu": require_gpu,
    }
    if not rows:
        timer.finish()
        report.timings = timer.timings
        return report

    backend = normalize_expr_backend(expr_eval_backend)
    report.expr_eval_backend_requested = backend
    if backend == "mlx":
        require_gpu = True
        prefer_mlx_autoencoder = True
    state_compute_backend = "mlx" if (prefer_mlx_autoencoder or require_gpu) else None

    def progress_cb(label: str) -> ProgressCallback | None:
        if on_progress is None:
            return None

        def _cb(done: int, total: int, detail: str) -> None:
            on_progress(done, total, f"{label}: {detail}")

        return _cb

    # Fail before constructing feature matrices: massive mode is not a
    # best-effort ranker path and cannot proceed without persisted arrays.
    massive_catalog_skipped_no_arrays = False
    include_templates = True
    if use_massive_catalog:
        array_payloads = load_row_arrays(run_id, store_root)
        include_templates = bool(array_payloads)
        massive_catalog_skipped_no_arrays = not include_templates
        if not include_templates:
            raise ValueError(
                "massive descriptor catalog requires persisted slice arrays; "
                "rerun the campaign with save_arrays=True (missing array_ref/arrays payloads)"
            )

    vars_ = metric_variable_columns(
        rows, resolved_target, max_vars=max(top_features, 8), domain_id=_domain_id_or_none()
    )
    X, cols = ctx.feature_matrix()
    target_arr = X[:, cols.index(resolved_target)]
    feat_cols = [c for c in cols if c != resolved_target][:top_features]
    symbolic_feat_cols = [c for c in cols if c != resolved_target]
    feat_idx = [cols.index(c) for c in feat_cols]

    if stage_completed(stage_progress, "rankers"):
        report.top_correlations = list(partial.get("top_correlations") or [])
        report.descriptor_candidates = list(partial.get("descriptor_candidates") or [])
        report.metric_candidates = list(partial.get("metric_candidates") or [])
        report.expr_eval_backend_effective = partial.get("expr_eval_backend_effective", backend)
        report.expr_eval_fallback_reason = partial.get("expr_eval_fallback_reason")
        report.massive_catalog_skipped_no_arrays = bool(
            partial.get("massive_catalog_skipped_no_arrays", False)
        )
        if on_stage is not None:
            on_stage("resume: skip rankers (checkpoint)")
    else:
        timer.begin("correlations")
        report.top_correlations = soft_call(
            "correlations",
            lambda: correlate_with_target(rows, resolved_target, min_abs_r=0.2)[:20],
            fallback=[],
            errors=report.stage_errors,
            on_error=on_stage,
        )

        templates = (
            __import__(
                "rde.descriptor_gen.enumerate_massive",
                fromlist=["enumerate_massive_templates"],
            ).enumerate_massive_templates(max_templates=max_descriptor_templates)
            if use_massive_catalog and include_templates
            else enumerate_descriptor_templates(max_templates=max_descriptor_templates)
        )

        timer.begin("descriptor generators")
        report.descriptor_candidates = soft_call(
            "descriptor_generators",
            lambda: candidates_to_records(
                rank_descriptor_generators(
                    rows,
                    run_id=run_id,
                    store_root=store_root,
                    target=resolved_target,
                    templates=templates,
                    max_derived=max_derived_descriptors,
                    max_derived_depth=max_derived_depth,
                    min_abs_r=0.25,
                    max_results=20,
                    eval_backend=backend,  # type: ignore[arg-type]
                    include_templates=include_templates,
                    on_progress=progress_cb("descriptor generators"),
                    force_gpu=force_gpu,
                    require_gpu=require_gpu,
                    domain_id=_domain_id_or_none(),
                )
            ),
            fallback=[],
            errors=report.stage_errors,
            on_error=on_stage,
        )

        cand_iter = enumerate_metric_candidates(
            vars_,
            max_depth=max_expr_depth,
            max_candidates=max_expr_candidates,
        )
        ranker = ConjectureRanker(target_column=resolved_target, min_abs_r=0.25, max_results=20)
        timer.begin("expression ranker")
        expr_hits = soft_call(
            "expression_ranker",
            lambda: ranker.rank_expressions_streaming(
                rows,
                cand_iter,
                variables=vars_,
                max_evaluated=max_expr_candidates,
                eval_backend=backend,  # type: ignore[arg-type]
                env=ctx.env_for(vars_),
                target=ctx.target_array(),
                on_progress=progress_cb("expression ranker"),
                force_gpu=force_gpu,
                require_gpu=require_gpu,
                checkpoint_path=(
                    store_root
                    / "discovery"
                    / "checkpoints"
                    / run_id
                    / "expression_ranker.json"
                ),
                resume=True,
                checkpoint_every=10_000,
                candidate_universe_id=candidate_universe_fingerprint(
                    vars_,
                    max_depth=max_expr_depth,
                    max_candidates=max_expr_candidates,
                ),
            ),
            fallback=[],
            errors=report.stage_errors,
            on_error=on_stage,
        )
        report.metric_candidates = ranker.to_records(expr_hits)
        report.expr_eval_backend_effective = getattr(ranker, "last_eval_backend", backend)
        report.expr_eval_fallback_reason = getattr(ranker, "last_eval_fallback_reason", None)
        report.massive_catalog_skipped_no_arrays = massive_catalog_skipped_no_arrays
        _persist_stage("rankers")

    def ae_epoch_progress(label: str) -> Callable[[int, int], None] | None:
        if on_progress is None:
            return None

        def _cb(epoch: int, total: int) -> None:
            on_progress(epoch, total, f"{label}: epoch {epoch}/{total}")

        return _cb

    latent_sources: list[LatentSource] = []
    if stage_completed(stage_progress, "phase5"):
        # Symbolic already finished — restore both latent summary + symbolic.
        report.latent = dict(partial.get("latent") or {})
        report.symbolic = dict(partial.get("symbolic") or {})
        if on_stage is not None:
            on_stage("resume: skip phase4/phase5 (checkpoint)")
    else:
        # Re-run phase4 when phase5 is incomplete so in-memory latent sources
        # are available; phase4's own latent checkpoints keep this cheap.
        timer.begin("latent models")
        phase4 = soft_call(
            "phase4",
            lambda: run_phase4(
                run_id,
                store_root,
                target=resolved_target,
                prefer_torch=prefer_torch_autoencoder,
                prefer_mlx=prefer_mlx_autoencoder,
                require_gpu=require_gpu,
                compute_backend=state_compute_backend,
                on_ae_epoch=ae_epoch_progress("feature autoencoder"),
                on_instance_ae_epoch=ae_epoch_progress("instance Q autoencoder"),
                checkpoint=True,
                context=ctx,
                on_stage=timer.begin,
            ),
            fallback=None,
            errors=report.stage_errors,
            on_error=on_stage,
        )
        report.latent = phase4.to_latent_dict() if phase4 is not None else {}
        if phase4 is not None:
            if phase4.pca is not None and phase4.pca.latent_codes.size:
                pca_cols = [f"latent.pca_{i}" for i in range(phase4.pca.latent_dim)]
                latent_sources.append(
                    LatentSource(codes=phase4.pca.latent_codes, columns=pca_cols, source="pca")
                )
            auto = phase4.feature_autoencoder
            if auto is not None and auto.latent_codes.size:
                latent_sources.append(
                    LatentSource(
                        codes=auto.latent_codes,
                        columns=auto.latent_columns,
                        source="autoencoder",
                    )
                )
            iq = phase4.instance_q_autoencoder
            if iq is not None and iq.latent_codes.size and iq.instance_ids:
                q_codes, q_cols = broadcast_instance_latents(
                    rows,
                    iq.instance_ids,
                    iq.latent_codes,
                    [f"latent.q_{i}" for i in range(iq.hidden_dim)],
                )
                if q_codes.shape[0] == len(rows):
                    latent_sources.append(
                        LatentSource(codes=q_codes, columns=q_cols, source="instance_q")
                    )
        _persist_stage("phase4")

        timer.begin("symbolic regression")
        phase5 = soft_call(
            "phase5",
            lambda: run_phase5(
                run_id,
                store_root,
                target=resolved_target,
                rows=rows,
                feature_columns=symbolic_feat_cols,
                latent_sources=latent_sources,
                metric_candidates=report.metric_candidates,
                top_latents=top_latents,
                use_pysr=use_pysr,
                use_operon=use_operon,
                use_native_gp=use_native_gp,
                use_ai_feynman=use_ai_feynman,
                top_features=top_features,
                symbolic_min_r_squared=promote_min_r_squared,
                promote_min_r_squared=promote_min_r_squared,
                gp_generations=gp_generations,
                gp_population=gp_population,
                pysr_iterations=pysr_iterations,
                pysr_maxsize=pysr_maxsize,
                pysr_populations=pysr_populations,
                operon_generations=operon_generations,
                operon_population=operon_population,
                expr_eval_backend=backend,
                require_gpu=require_gpu,
                promote=False,
                on_stage=timer.begin,
                on_symbolic_stage=timer.begin,
                on_progress=progress_cb("symbolic regression"),
            ),
            fallback=None,
            errors=report.stage_errors,
            on_error=on_stage,
        )
        report.symbolic = phase5.symbolic if phase5 is not None else {}
        _persist_stage("phase5")

    if stage_completed(stage_progress, "phase6"):
        report.phase6 = dict(partial.get("phase6") or {})
        report.programs = list(partial.get("programs") or report.phase6.get("programs", []))
        report.coordinates = dict(partial.get("coordinates") or report.phase6.get("coordinates") or {})
        report.operator = dict(partial.get("operator") or report.phase6.get("operator") or {})
        if on_stage is not None:
            on_stage("resume: skip phase6 (checkpoint)")
    else:
        gp_vars = [h["column"] for h in report.top_correlations[:8]] or vars_[:8]
        timer.begin("phase 6 representation discovery")
        phase6 = soft_call(
            "phase6",
            lambda: run_phase6(
                run_id,
                store_root,
                target=resolved_target,
                gp_generations=gp_generations,
                gp_population=gp_population,
                walsh_max_candidates=walsh_max_candidates,
                walsh_max_support=walsh_max_support,
                rediscovery_splits=rediscovery_splits,
                expr_eval_backend=backend,
                rows=rows,
                symbolic_equation=report.symbolic.get("equation"),
                context=ctx,
                on_stage=timer.begin,
                on_progress=progress_cb("genetic programs"),
                require_gpu=require_gpu,
                compute_backend=state_compute_backend,
            ),
            fallback=None,
            errors=report.stage_errors,
            on_error=on_stage,
        )
        if phase6 is not None:
            report.phase6 = phase6.to_phase6_dict()
            for err in getattr(phase6, "stage_errors", None) or []:
                report.stage_errors.append(err)
        else:
            report.phase6 = {
                "g4_met": False,
                "g5_met": False,
                "g5_hint": False,
                "rediscovery": {"soft_failed": True},
                "stage_errors": [{"stage": "phase6", "error": "soft-failed"}],
            }
        report.programs = report.phase6.get("programs", [])
        report.coordinates = report.phase6.get("coordinates", {}) or {}
        report.operator = report.phase6.get("operator", {}) or {}
        _persist_stage("phase6")

    if stage_completed(stage_progress, "promote"):
        report.promoted_conjectures = list(partial.get("promoted_conjectures") or [])
        report.cross_n_rule = list(partial.get("cross_n_rule") or [])
        report.obstruct_summary = dict(partial.get("obstruct_summary") or {})
        report.certify_results = list(partial.get("certify_results") or [])
        if on_stage is not None:
            on_stage("resume: skip promote (checkpoint)")
    else:
        timer.begin("promote conjectures")
        from rde.analyze.leak_audit import audit_conjectures

        def _promote() -> tuple[list[dict[str, Any]], dict[str, Any]]:
            raw_promoted = promote_top_conjectures(
                target=resolved_target,
                metric_candidates=report.metric_candidates,
                symbolic=report.symbolic,
                latent_interpretations=report.symbolic.get("latent_interpretations"),
                programs=report.programs,
                max_results=max_promoted_conjectures,
                min_r_squared=promote_min_r_squared,
                leak_audit=False,
                rejected=report.promotion_rejections,
            )
            return audit_conjectures(
                raw_promoted,
                target=resolved_target,
                domain_id=_domain_id(),
            )

        promoted, leak_summary = soft_call(
            "promote",
            _promote,
            fallback=([], {}),
            errors=report.stage_errors,
            on_error=on_stage,
        )
        report.promoted_conjectures = promoted
        report.leak_audit_summary = leak_summary
        if promotion_blocked:
            report.leak_audit_summary = {
                **report.leak_audit_summary,
                "promotion_blocked": True,
                "n2_calibration_only": True,
            }
            report.promoted_conjectures = []

        timer.begin("cross-N generative rule")
        from rde.discovery.scaling_rule import fit_cross_n_generative_rule

        report.cross_n_rule = soft_call(
            "cross_n_rule",
            lambda: [
                r.to_dict()
                for r in fit_cross_n_generative_rule(
                    rows, resolved_target, report.promoted_conjectures
                )
            ],
            fallback=[],
            errors=report.stage_errors,
            on_error=on_stage,
        )

        if run_obstruct:
            timer.begin("obstruction witness panel")
            from rde.analyze.obstructions import panel_summary, witness_panel
            from rde.discovery.promote_lb import build_lower_bound_conjectures

            report.obstruct_summary = soft_call(
                "obstructions",
                lambda: panel_summary(witness_panel(rows, resolved_target)),
                fallback={},
                errors=report.stage_errors,
                on_error=on_stage,
            )
            report.lower_bound_conjectures = soft_call(
                "lower_bound_conjectures",
                lambda: build_lower_bound_conjectures(
                    rows, target=resolved_target, run_id=run_id
                ),
                fallback=[],
                errors=report.stage_errors,
                on_error=on_stage,
            )

        certify_results: list[dict[str, Any]] = []
        if report.promoted_conjectures:
            timer.begin("certify top conjectures")
            from rde.analyze.certify import certify_representation_candidate

            for conj in report.promoted_conjectures[:3]:
                cr = soft_call(
                    "certify",
                    lambda c=conj: certify_representation_candidate(
                        c,
                        rows=rows,
                        target=resolved_target,
                        domain_id=_domain_id(),
                    ),
                    fallback=None,
                    errors=report.stage_errors,
                    on_error=on_stage,
                )
                if cr is None:
                    continue
                certify_results.append(
                    {
                        "expression": conj.get("expression"),
                        "passed": cr.passed,
                        "checks": cr.checks,
                        "violations": cr.violations,
                        "poly_gates": cr.poly_gates,
                        "poly_shots": cr.poly_shots,
                    }
                )
        report.certify_results = certify_results

        update_proxy = np.array(
            [row.get("update.step_ratio_mean", float("nan")) for row in rows], dtype=float
        )
        if not report.coordinates:
            timer.begin("coordinates + operator")

            def _coord_op() -> tuple[dict[str, Any], dict[str, Any]]:
                coord = search_linear_coordinates(
                    X[:, feat_idx], target=target_arr, update_proxy=update_proxy
                )
                op = learn_update_predictor(
                    rows, feat_cols, update_key="update.step_ratio_mean"
                )
                return (
                    {
                        "method": coord.method,
                        "n_components": coord.n_components,
                        "target_correlation": coord.target_correlation,
                        "update_simplicity": coord.update_simplicity,
                    },
                    {
                        "method": op.method,
                        "train_r_squared": op.train_r_squared,
                        "n_rows": op.n_rows,
                        "feature_columns": op.feature_columns,
                        "update_key": op.update_key,
                    },
                )

            coords, operator = soft_call(
                "coordinates_operator",
                _coord_op,
                fallback=({}, {}),
                errors=report.stage_errors,
                on_error=on_stage,
            )
            report.coordinates = coords
            report.operator = operator
        _persist_stage("promote")

    timer.begin("outcome assessment")
    top_certify = report.certify_results[0] if report.certify_results else None

    def _outcome() -> dict[str, Any]:
        assessment = assess_outcome(
            rows,
            resolved_target,
            metric_candidates=report.metric_candidates,
            latent=report.latent,
            phase6=report.phase6,
            discovery={
                "metric_candidates": report.metric_candidates,
                "latent": report.latent,
                "symbolic": report.symbolic,
                "phase6": report.phase6,
            },
            leak_audit_summary=report.leak_audit_summary,
            certify_result=top_certify,
            obstruct_summary=report.obstruct_summary,
            domain_id=_domain_id(),
        )
        report.outcome_grade_hint = _infer_grade(report)
        if assessment.g5_met:
            report.outcome_grade_hint = 5
        elif assessment.g4_met:
            report.outcome_grade_hint = max(report.outcome_grade_hint, 4)
        elif assessment.g1_met:
            report.outcome_grade_hint = max(report.outcome_grade_hint, 1)
        elif assessment.g0_met:
            report.outcome_grade_hint = 0
        if getattr(assessment, "target_degenerate", False):
            # Unconditional override, independent of the elif chain above:
            # `_infer_grade` reads Phase 6's raw g5_met/g5_hint
            # flags directly, bypassing assess_outcome's gating entirely --
            # confirmed root cause of a false outcome_grade_hint=5 (Direction
            # E, 2026-08-19: a numerically constant target still produced a
            # G5 rediscovery because every stage rediscovers a constant
            # trivially, and nothing checked the target had real variance
            # first). A degenerate target can never legitimately support any
            # grade above G0, so this cannot be overridden by any upstream
            # stage's claim.
            report.outcome_grade_hint = 0
        payload = assessment.to_payload()
        payload.update(
            {
                "leak_audit_summary": report.leak_audit_summary,
                "obstruct_summary": report.obstruct_summary,
                "certify_results": report.certify_results,
                "promotion_rejections": report.promotion_rejections,
                "parameters": report.parameters,
                "stage_errors": report.stage_errors,
            }
        )
        return payload

    report.outcome = soft_call(
        "outcome",
        _outcome,
        fallback={
            "grade": 0,
            "g0_met": True,
            "soft_failed": True,
            "stage_errors": report.stage_errors,
            "parameters": report.parameters,
        },
        errors=report.stage_errors,
        on_error=on_stage,
    )
    _persist_stage("outcome")
    timer.finish()
    report.timings = {**ctx.timings, **timer.timings}
    return report


def _infer_grade(report: DiscoveryReport) -> int:
    phase6 = report.phase6 or {}
    if phase6.get("g5_met"):
        return 5
    if phase6.get("g5_hint"):
        return 5
    if phase6.get("g4_met") or (phase6.get("representation") or {}).get("g4_met"):
        return 4
    best_r2 = 0.0
    if report.metric_candidates:
        best_r2 = max(c.get("r_squared", 0.0) for c in report.metric_candidates)
    if report.symbolic.get("r_squared", 0.0) > best_r2:
        best_r2 = report.symbolic["r_squared"]
    if best_r2 >= 0.95:
        return 1
    if best_r2 >= 0.75:
        return 1
    lat = report.latent.get("target_correlations", {})
    if lat and max(abs(v) for v in lat.values()) >= 0.9:
        return 4
    if report.programs and report.programs[0].get("fitness", 0) >= 0.8:
        return 5
    return 0


def write_discovery_report(report: DiscoveryReport, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": report.run_id,
        "target": report.target,
        "n_rows": report.n_rows,
        "outcome_grade_hint": report.outcome_grade_hint,
        "top_correlations": report.top_correlations,
        "descriptor_candidates": report.descriptor_candidates,
        "metric_candidates": report.metric_candidates,
        "latent": report.latent,
        "symbolic": report.symbolic,
        "promoted_conjectures": report.promoted_conjectures,
        "cross_n_rule": report.cross_n_rule,
        "programs": report.programs,
        "coordinates": report.coordinates,
        "operator": report.operator,
        "phase6": report.phase6,
        "outcome": report.outcome,
        "leak_audit_summary": report.leak_audit_summary,
        "obstruct_summary": report.obstruct_summary,
        "lower_bound_conjectures": report.lower_bound_conjectures,
        "certify_results": report.certify_results,
        "promotion_rejections": report.promotion_rejections,
        "parameters": report.parameters,
        "timings": report.timings,
        "expr_eval_backend_requested": report.expr_eval_backend_requested,
        "expr_eval_backend_effective": report.expr_eval_backend_effective,
        "expr_eval_fallback_reason": report.expr_eval_fallback_reason,
        "massive_catalog_skipped_no_arrays": report.massive_catalog_skipped_no_arrays,
        "stage_errors": report.stage_errors,
    }
    from rde.io.json_util import write_json

    write_json(out, payload)
    return out


def write_descriptor_conjectures_jsonl(report: DiscoveryReport, path: Path | str) -> Path:
    """Write ranked descriptor generator candidates as JSONL."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in report.descriptor_candidates:
            handle.write(json.dumps(record) + "\n")
    return out


def write_conjectures_jsonl(report: DiscoveryReport, path: Path | str) -> Path:
    """Write top-ranked promoted conjectures as JSONL (handoff rule: top-N only)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = report.promoted_conjectures
    if not records and report.metric_candidates:
        records = report.metric_candidates
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return out
