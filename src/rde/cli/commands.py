"""Command-line interface for RDE."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from rde.analyze.calibration import complexity_by_group, separation_score
from rde.analyze.outcome import assess_outcome
from rde.analyze.query import (
    correlate_with_target,
    cross_n_report,
    distribution_summary,
    flatten_features,
    outlier_rows,
    summarize_run,
    top_correlation_matrix,
)
from rde.analyze.ranker import ConjectureRanker
from rde.discovery import (
    discover_autoencoder,
    discover_latent_from_run,
    load_feature_matrix,
    promote_top_conjectures,
    run_discovery,
    run_phase4,
    run_phase5,
    run_phase6,
    symbolic_backends,
    write_conjectures_jsonl,
    write_descriptor_conjectures_jsonl,
    write_discovery_report,
    write_phase4_report,
    write_phase5_report,
    write_phase6_report,
)
from rde.discovery.loop import DiscoveryReport
from rde.discovery.datasets import broadcast_instance_latents
from rde.discovery.symbolic import LatentSource
from rde.descriptor_gen.enumerate import enumerate_descriptor_templates
from rde.descriptor_gen.rank import candidates_to_records, rank_descriptor_generators
from rde.discovery.report import format_discovery_summary, format_phase4_summary, format_phase5_summary
from rde.backends import (
    available_backends,
    default_backend,
    default_expr_backend,
    dev_machine_profile,
    expr_backend_choices,
    format_dev_machine_profile,
    resolve_compute_backend,
)
from rde.expression import enumerate_expressions, normalize_expr_backend
from rde.expression.generators import enumerate_metric_candidates, metric_variable_columns
from rde.generators import list_generators
from rde.runtime.campaign import CampaignConfig, run_campaign
from rde.runtime.pipeline import RunConfig, run_pipeline
from rde.core.plugins import build_registry, list_domain_ids
from rde.core.schema import validate_features_file, validate_instance_features_file
from rde.io.console import RdeConsole, get_console, resolve_plain, set_console
from rde.io.events import log_progress
from rde.io.progress_ui import ConsolePipelineProgress
from rde.io.seal import (
    estimate_seal_cleanup,
    is_run_sealed,
    list_campaign_run_ids,
    require_seal_dependencies,
    seal_run,
)
from rde.io.topk_retention import (
    compact_topk_summary,
    estimate_topk_retention,
    retain_topk_campaign,
    retain_topk_run,
)
from rde.io.shutdown import install_signal_handlers, run_cli_command
from rde.io.store import Store
from rde.runtime.admission import (
    configuration_fingerprint,
    directory_size_bytes,
    peak_rss_mb,
)
from rde.runtime.targets import resolve_target_for_run


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _storage_bytes_from_gb(value: float | None) -> int | None:
    if value is None:
        return None
    if not math.isfinite(value) or value <= 0:
        raise ValueError("--max-storage-gb must be finite and positive")
    return int(value * (1024**3))


def _add_coined_walk_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--coin-shift-grammar",
        action="store_true",
        help="coined_walk only: search the typed coin/shift grammar instead of fixed harness families",
    )


def _coin_shift_grammar_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "coin_shift_grammar", False))


def _add_perf_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        default=default_backend(),
        choices=available_backends(),
        help="Compute backend: auto (size+batch-aware from crossover table), mlx, numpy; torch_* opt-in.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes")
    parser.add_argument(
        "--compute-batch-size",
        type=int,
        default=None,
        help="Fused QUBO batch size for workers=1 (default: auto from crossover table)",
    )
    parser.add_argument("--generator", default=None, help="Instance generator id (optional)")


def _console(args: argparse.Namespace) -> RdeConsole:
    """Return the module console (configured in main from --plain / TTY / env)."""
    return get_console()


def _is_plain(args: argparse.Namespace) -> bool:
    return resolve_plain(flag=getattr(args, "plain", False))


def cmd_run(args: argparse.Namespace) -> int:
    ui = _console(args)
    plain = _is_plain(args)

    def _run():
        if not plain:
            ui.banner(subtitle="single pipeline run")
        else:
            ui.rule("pipeline run")
        ui.kv("domain", args.domain)
        ui.kv("size", args.size)
        ui.kv("instances", args.n_instances)
        ui.kv("backend", args.backend)
        if args.compute_batch_size is not None:
            ui.kv("compute_batch_size", args.compute_batch_size)
        ui.kv("workers", args.workers)
        gen = args.generator
        if gen:
            ui.kv("generator", gen)
        if _coin_shift_grammar_enabled(args):
            ui.kv("coin_shift_grammar", True)
        reg = build_registry(
            args.domain,
            compute_backend=args.backend,
            loader_kwargs={
                "coin_shift_grammar": _coin_shift_grammar_enabled(args)
            },
        )
        progress = None if plain else ConsolePipelineProgress(ui)
        return run_pipeline(
            RunConfig(
                domain_id=args.domain,
                n_instances=args.n_instances,
                size=args.size,
                seed=args.seed,
                indices=_parse_int_list(args.indices),
                store_root=args.store_root,
                run_id=args.run_id,
                resume=args.resume,
                save_arrays=not args.no_arrays,
                workers=args.workers,
                compute_backend=args.backend,
                compute_batch_size=args.compute_batch_size,
                generator_id=gen,
                coin_shift_grammar=_coin_shift_grammar_enabled(args),
            ),
            registry=reg,
            progress=progress,
        )

    if plain:
        result = _run()
    else:
        with ui.sticky_session():
            result = _run()
    Store(args.store_root).close()
    if plain:
        log_progress(
            f"run_id={result.run_id} feature_rows={result.n_feature_rows} "
            f"skipped_instances={result.n_skipped_instances} skipped_slices={result.n_skipped_slices}"
        )
    else:
        ui.kv("run_id", result.run_id)
        ui.kv("feature_rows", result.n_feature_rows)
        ui.kv("skipped_instances", result.n_skipped_instances)
        ui.kv("skipped_slices", result.n_skipped_slices)
    return 0


def cmd_campaign(args: argparse.Namespace) -> int:
    ui = _console(args)
    plain = _is_plain(args)

    def _run():
        if not plain:
            ui.banner(subtitle="multi-size campaign")
        backend = args.backend
        ui.set_campaign_config(
            campaign_id=args.campaign_id or "auto",
            domain=args.domain,
            sizes=args.sizes,
            n_per_size=args.n_per_size,
            backend=backend,
            workers=args.workers,
            generator=args.generator or "default",
        )
        reg = build_registry(
            args.domain,
            compute_backend=backend,
            loader_kwargs={
                "coin_shift_grammar": _coin_shift_grammar_enabled(args)
            },
        )
        n_per_size: int | dict[int, int] = getattr(args, "n_per_size_map", None) or args.n_per_size
        extra: dict = {}
        if getattr(args, "descriptor_names", None):
            extra["descriptor_panel"] = getattr(args, "descriptor_names", None)
        return run_campaign(
            CampaignConfig(
                domain_id=args.domain,
                sizes=_parse_int_list(args.sizes),
                n_per_size=n_per_size,
                seed_base=args.seed,
                indices=_parse_int_list(args.indices),
                store_root=args.store_root,
                campaign_id=args.campaign_id,
                resume=args.resume,
                save_arrays=not args.no_arrays
                if args.max_storage_gb is None
                else False,
                workers=args.workers,
                compute_backend=backend,
                compute_batch_size=args.compute_batch_size,
                generator_id=args.generator,
                plain_output=plain,
                descriptor_names=getattr(args, "descriptor_names", None),
                metric_names=getattr(args, "metric_names", None),
                instance_descriptor_modules=getattr(
                    args, "instance_descriptor_modules", None
                ),
                enable_cross_slice=getattr(args, "enable_cross_slice", True),
                coin_shift_grammar=_coin_shift_grammar_enabled(args),
                extra=extra,
                max_storage_bytes=_storage_bytes_from_gb(args.max_storage_gb),
                seal_batches=args.seal_batches,
                seal_keep_arrays=args.seal_keep_arrays,
            ),
            registry=reg,
        )

    if plain:
        result = _run()
    else:
        with ui.sticky_session():
            result = _run()
    total_rows = sum(r.n_feature_rows for r in result.run_results)
    if plain:
        log_progress(
            f"campaign_id={result.campaign_id} runs={len(result.run_results)} "
            f"total_feature_rows={total_rows} "
            f"batch_index={result.batch_summary.get('batch_index')} "
            f"stop_reason={result.batch_summary.get('stop_reason')} "
            f"instances_completed={result.batch_summary.get('instances_completed_this_session', 0)} "
            f"features_completed={result.batch_summary.get('features_completed_this_session', 0)} "
            f"bytes_written={result.batch_summary.get('bytes_written_this_session', 0)}"
        )
    else:
        ui.rule("campaign complete")
        ui.success(f"campaign_id={result.campaign_id}  runs={len(result.run_results)}  total_feature_rows={total_rows}")
        ui.kv("batch_index", result.batch_summary.get("batch_index"))
        ui.kv("stop_reason", result.batch_summary.get("stop_reason"))
        ui.kv(
            "instances_completed",
            result.batch_summary.get("instances_completed_this_session", 0),
        )
        ui.kv(
            "features_completed",
            result.batch_summary.get("features_completed_this_session", 0),
        )
        ui.kv("bytes_written", result.batch_summary.get("bytes_written_this_session", 0))
    return 0


def cmd_seal(args: argparse.Namespace) -> int:
    """Seal one run or every run belonging to a campaign."""
    ui = _console(args)
    store = Store(args.store_root)
    if args.run_id is not None:
        run_ids = [args.run_id]
    else:
        run_ids = list_campaign_run_ids(args.store_root, args.campaign_id, store=store)
        if not run_ids:
            store.close()
            raise ValueError(
                f"no runs found for campaign {args.campaign_id!r}"
            )

    if args.dry_run:
        try:
            for run_id in run_ids:
                plan = estimate_seal_cleanup(
                    args.store_root,
                    run_id,
                    delete_arrays=not args.keep_arrays,
                    delete_jsonl=not args.keep_jsonl,
                    store=store,
                )
                ui.kv("run_id", plan["run_id"])
                ui.kv("action", plan["action"])
                ui.kv("status", plan["status"])
                ui.kv("feature_rows", plan.get("feature_rows"))
                ui.kv("bytes_freed", plan.get("bytes_freed", 0))
                if plan.get("would_remove"):
                    ui.kv("would_remove", plan["would_remove"])
                if plan.get("shard_path"):
                    ui.kv("shard_path", plan["shard_path"])
        finally:
            store.close()
        return 0

    records: list[dict] = []
    try:
        require_seal_dependencies()
        for run_id in run_ids:
            records.append(
                seal_run(
                    args.store_root,
                    run_id,
                    campaign_id=args.campaign_id,
                    batch_index=args.batch_index,
                    delete_arrays=not args.keep_arrays,
                    delete_jsonl=not args.keep_jsonl,
                    store=store,
                )
            )
    finally:
        store.close()
    for record in records:
        ui.success(
            f"sealed run={record['run_id']} "
            f"features={record['rows']['sealed_features']} "
            f"bytes_freed={record['cleanup']['bytes_freed']}"
        )
    return 0


def cmd_retain_topk(args: argparse.Namespace) -> int:
    """Retain targets, ids, and ranked columns in a compact top-K shard."""
    ui = _console(args)
    store = Store(args.store_root)
    if args.run_id is not None:
        run_ids = [args.run_id]
    else:
        run_ids = list_campaign_run_ids(args.store_root, args.campaign_id, store=store)
        if not run_ids:
            store.close()
            raise ValueError(f"no runs found for campaign {args.campaign_id!r}")

    ranked_columns = None
    if args.ranked_columns:
        ranked_columns = [part.strip() for part in args.ranked_columns.split(",") if part.strip()]

    if args.dry_run:
        try:
            for run_id in run_ids:
                resolved_target = args.target or resolve_target_for_run(
                    run_id,
                    args.store_root,
                    override=None,
                )
                report_path = args.discovery_report
                if report_path is None and args.campaign_id is not None:
                    candidate = Path(args.store_root) / "discovery" / f"{run_id}.json"
                    if candidate.is_file():
                        report_path = str(candidate)
                plan = estimate_topk_retention(
                    args.store_root,
                    run_id,
                    target=resolved_target,
                    top_k=args.top_k,
                    discovery_report_path=report_path,
                    ranked_columns=ranked_columns,
                )
                ui.kv("run_id", plan["run_id"])
                ui.kv("action", plan["action"])
                ui.kv("target", plan.get("target"))
                ui.kv("retained_columns", len(plan.get("retained_columns") or []))
                ui.kv("bytes_freed", plan.get("bytes_freed", 0))
                if plan.get("ranked_columns_kept"):
                    ui.kv("ranked_columns_kept", plan["ranked_columns_kept"])
        finally:
            store.close()
        return 0

    records: list[dict] = []
    try:
        require_seal_dependencies()
        if args.run_id is not None:
            resolved_target = args.target or resolve_target_for_run(
                args.run_id,
                args.store_root,
                override=None,
            )
            report_path = args.discovery_report
            if report_path is None:
                candidate = Path(args.store_root) / "discovery" / f"{args.run_id}.json"
                if candidate.is_file():
                    report_path = str(candidate)
            records.append(
                retain_topk_run(
                    args.store_root,
                    args.run_id,
                    target=resolved_target,
                    top_k=args.top_k,
                    discovery_report_path=report_path,
                    ranked_columns=ranked_columns,
                    delete_full_shard=not args.keep_full_shard,
                )
            )
        else:
            records.extend(
                retain_topk_campaign(
                    args.store_root,
                    args.campaign_id,
                    target=args.target,
                    top_k=args.top_k,
                    discovery_report_dir=Path(args.store_root) / "discovery",
                    delete_full_shard=not args.keep_full_shard,
                )
            )
    finally:
        store.close()

    for record in records:
        summary = compact_topk_summary(record)
        ui.success(
            f"retained top-K run={summary['run_id']} "
            f"columns={len(record.get('retained_columns') or [])} "
            f"bytes_freed={record.get('bytes_freed', 0)}"
        )
        ui.info(record.get("non_claim", ""))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    ui = _console(args)
    store = Store(args.store_root)
    out = Path(args.output)
    if args.format == "parquet" or out.suffix == ".parquet":
        path = store.export_features_parquet(args.run_id, out)
    else:
        path = store.export_features_csv(args.run_id, out)
    store.close()
    ui.success(f"exported {path}")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    ui = _console(args)
    summary = summarize_run(args.run_id, args.store_root)
    ui.rule(f"summary  {args.run_id}")
    for key, value in summary.items():
        ui.kv(key, value)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    ui = _console(args)
    rows = flatten_features(args.run_id, args.store_root)
    target = resolve_target_for_run(args.run_id, args.store_root, override=args.target)
    output: dict | list | None = None
    ui.rule(f"analyze  {args.mode}")
    ui.kv("target", target)
    ui.kv("n_rows", len(rows))
    if args.mode == "correlations":
        hits = correlate_with_target(rows, target, min_abs_r=args.min_abs_r)
        ui.kv("n_hits", len(hits))
        table_rows = [
            [h["column"][:36], f"{h['pearson_r']:+.4f}", f"{h['r_squared']:.4f}", h["n"]]
            for h in hits[: args.top]
        ]
        ui.table("top correlations", ["column", "r", "R2", "n"], table_rows)
        output = hits[: args.top]
    elif args.mode == "distribution":
        summary = distribution_summary(rows, target, group_by="size")
        table_rows = [[str(r.get("size")), f"{r['mean']:.4f}", f"{r['std']:.4f}", r["count"]] for r in summary]
        ui.table("distribution by size", ["size", "mean", "std", "n"], table_rows)
        output = summary
    elif args.mode == "outliers":
        found = outlier_rows(rows, target, z_threshold=args.z_threshold)
        ui.kv("n_outliers", len(found))
        table_rows = [
            [row.get("instance_id", ""), f"{row.get('z_score', 0):.2f}", row.get(target)]
            for row in found[: args.top]
        ]
        ui.table("outliers", ["instance", "z", "value"], table_rows)
        output = found[: args.top]
    elif args.mode == "cross_n":
        report = cross_n_report(rows, target, min_abs_r=args.min_abs_r)
        ui.kv("n_columns", len(report))
        table_rows = [
            [
                h["column"][:32],
                f"{h.get('cross_n_sign_stability', float('nan')):.3f}",
                f"{h['pearson_r']:+.4f}",
                f"{h['r_squared']:.4f}",
            ]
            for h in report[: args.top]
        ]
        ui.table("cross-N stability", ["column", "cross_N", "r", "R2"], table_rows)
        output = report[: args.top]
    elif args.mode == "matrix":
        mat = top_correlation_matrix(rows, target, top_k=args.top, min_abs_r=args.min_abs_r)
        ui.kv("matrix_size", len(mat["columns"]))
        for col, row_vals in zip(mat["columns"], mat["matrix"]):
            vals = " ".join(f"{v:+.2f}" if isinstance(v, float) and v == v else " nan" for v in row_vals)
            ui.info(f"{col:30s} {vals}")
        output = mat
    if args.output and output is not None:
        Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
        ui.success(f"written {args.output}")
    return 0


def cmd_rank_expr(args: argparse.Namespace) -> int:
    ui = _console(args)
    rows = flatten_features(args.run_id, args.store_root)
    target = resolve_target_for_run(args.run_id, args.store_root, override=args.target)
    ui.rule("rank expressions")
    ui.kv("target", target)
    ui.kv("n_rows", len(rows))
    vars_ = [
        c
        for c in correlate_with_target(rows, target, min_abs_r=0.0)
        if c["column"] != target
    ]
    var_names = [v["column"] for v in vars_[: args.max_vars]]
    if not var_names:
        from rde.analyze.query import numeric_columns

        var_names = numeric_columns(rows, exclude={target})[: args.max_vars]
    backend = normalize_expr_backend(args.expr_backend or args.backend)
    ui.kv("backend", backend)
    ui.kv("max_candidates", args.max_candidates)
    cand_iter = enumerate_expressions(var_names, max_depth=args.expr_depth, max_candidates=args.max_candidates)
    ranker = ConjectureRanker(target_column=target, min_abs_r=args.min_abs_r, max_results=args.top)
    with ui.spinner("ranking expressions"):
        ranked = ranker.to_records(
            ranker.rank_expressions_streaming(
                rows,
                cand_iter,
                variables=var_names,
                max_evaluated=args.max_candidates,
                eval_backend=backend,  # type: ignore[arg-type]
            )
        )
    table_rows = [
        [i, f"{hit['r_squared']:.4f}", f"{hit['pearson_r']:+.4f}", hit["expression"][:48]]
        for i, hit in enumerate(ranked, start=1)
    ]
    ui.table("top expressions", ["#", "R2", "r", "expression"], table_rows)
    if args.output:
        Path(args.output).write_text(json.dumps(ranked, indent=2), encoding="utf-8")
        ui.success(f"written {args.output}")
    return 0


def cmd_rank_metrics(args: argparse.Namespace) -> int:
    ui = _console(args)
    rows = flatten_features(args.run_id, args.store_root)
    target = resolve_target_for_run(args.run_id, args.store_root, override=args.target)
    ui.rule("rank metric generators")
    ui.kv("target", target)
    ui.kv("n_rows", len(rows))
    domain_id = Store(args.store_root).read_manifest(args.run_id).domain_id
    var_names = metric_variable_columns(rows, target, max_vars=args.max_vars, domain_id=domain_id)
    backend = normalize_expr_backend(args.expr_backend or args.backend)
    ui.kv("backend", backend)
    ui.kv("variables", len(var_names))
    ui.kv("max_candidates", args.max_candidates)
    cand_iter = enumerate_metric_candidates(
        var_names,
        max_depth=args.expr_depth,
        max_candidates=args.max_candidates,
        include_ratios=not args.no_ratios,
        include_products=not args.no_products,
    )
    ranker = ConjectureRanker(target_column=target, min_abs_r=args.min_abs_r, max_results=args.top)
    with ui.spinner("ranking metric generators"):
        ranked = ranker.to_records(
            ranker.rank_expressions_streaming(
                rows,
                cand_iter,
                variables=var_names,
                max_evaluated=args.max_candidates,
                eval_backend=backend,  # type: ignore[arg-type]
                chunk_size=args.chunk_size,
            )
        )
    table_rows = [
        [i, f"{hit['r_squared']:.4f}", f"{hit['pearson_r']:+.4f}", hit["expression"][:52]]
        for i, hit in enumerate(ranked, start=1)
    ]
    ui.table("top metric generators", ["#", "R2", "r", "expression"], table_rows)
    if args.output:
        Path(args.output).write_text(json.dumps(ranked, indent=2), encoding="utf-8")
        ui.success(f"written {args.output}")
    return 0


def cmd_rank_desc(args: argparse.Namespace) -> int:
    ui = _console(args)
    rows = flatten_features(args.run_id, args.store_root)
    target = resolve_target_for_run(args.run_id, args.store_root, override=args.target)
    ui.rule("rank descriptor generators")
    ui.kv("target", target)
    ui.kv("n_rows", len(rows))
    templates = enumerate_descriptor_templates(max_templates=args.max_templates)
    backend = normalize_expr_backend(args.expr_backend or args.backend)
    ui.kv("backend", backend)
    ui.kv("n_templates", len(templates))
    ui.kv("mode", args.mode)
    with ui.spinner("ranking descriptor generators"):
        ranked = candidates_to_records(
            rank_descriptor_generators(
                rows,
                run_id=args.run_id,
                store_root=args.store_root,
                target=target,
                templates=templates,
                max_derived=args.max_derived,
                max_derived_depth=args.derived_depth,
                min_abs_r=args.min_abs_r,
                max_results=args.top,
                eval_backend=backend,  # type: ignore[arg-type]
                include_templates=args.mode in {"templates", "both"},
                include_derived=args.mode in {"derived", "both"},
                domain_id=Store(args.store_root).read_manifest(args.run_id).domain_id,
            )
        )
    table_rows = [
        [
            i,
            hit.get("kind", "?"),
            f"{hit['r_squared']:.4f}",
            f"{hit['pearson_r']:+.4f}",
            hit["key"][:52],
        ]
        for i, hit in enumerate(ranked, start=1)
    ]
    ui.table("top descriptor generators", ["#", "kind", "R2", "r", "key"], table_rows)
    if args.output:
        Path(args.output).write_text(json.dumps(ranked, indent=2), encoding="utf-8")
        ui.success(f"written {args.output}")
    return 0


def cmd_discover(args: argparse.Namespace) -> int:
    ui = _console(args)
    plain = _is_plain(args)
    if not plain:
        ui.banner(subtitle="discovery loop  phases 3-6")
    target = resolve_target_for_run(args.run_id, args.store_root, override=args.target)
    ui.kv("run_id", args.run_id)
    ui.kv("target", target)

    discovery_stages = 10
    expr_backend = normalize_expr_backend(args.expr_backend or args.backend)
    require_gpu = expr_backend == "mlx"

    def _run_discovery(on_stage, on_progress=None):
        max_candidates = args.max_candidates
        if args.massive_catalog and max_candidates == 20_000:
            max_candidates = 1_000_000
        return run_discovery(
            args.run_id,
            args.store_root,
            target=target,
            max_expr_candidates=max_candidates,
            max_expr_depth=args.expr_depth,
            max_descriptor_templates=args.max_descriptor_templates,
            use_massive_catalog=args.massive_catalog,
            gp_generations=args.gp_generations,
            gp_population=args.gp_population,
            use_pysr=not args.no_pysr,
            use_operon=not args.no_operon,
            top_latents=getattr(args, "top_latents", 3),
            max_promoted_conjectures=getattr(args, "promote_top", 10),
            prefer_torch_autoencoder=args.torch_ae,
            prefer_mlx_autoencoder=args.mlx_ae or require_gpu,
            expr_eval_backend=expr_backend,
            require_gpu=require_gpu,
            run_obstruct=getattr(args, "obstruct", False),
            on_stage=on_stage,
            on_progress=on_progress,
        )

    from rde.runtime.heartbeat import throttled_progress

    on_progress_cb = None
    if plain or ui.log_progress:

        def _emit_progress(done: int, total: int, detail: str) -> None:
            if total > 0:
                pct = 100.0 * float(done) / float(total)
                log_progress(f"  [{done}/{total}] ({pct:.1f}%) {detail}")
            else:
                log_progress(f"  [progress] {detail}")

        on_progress_cb = throttled_progress(_emit_progress, min_interval_s=2.0)

    if plain:
        def on_stage(name: str) -> None:
            log_progress(f"  -> {name}")

        report = _run_discovery(on_stage, on_progress_cb)
    else:
        with ui.task_progress("discovery loop", discovery_stages) as prog:
            def on_stage(name: str) -> None:
                prog.update(1, detail=name)

            report = _run_discovery(on_stage, on_progress_cb)

    out = Path(args.output) if args.output else Path(args.store_root) / "discovery" / f"{args.run_id}.json"
    write_discovery_report(report, out)
    conjectures_path = out.with_name("conjectures.jsonl")
    write_conjectures_jsonl(report, conjectures_path)
    desc_path = out.with_name("descriptor_conjectures.jsonl")
    write_descriptor_conjectures_jsonl(report, desc_path)
    if report.lower_bound_conjectures:
        from rde.discovery.promote_lb import write_lower_bound_conjectures_jsonl

        lb_path = Path(args.store_root) / "discovery" / "lower_bound_conjectures.jsonl"
        write_lower_bound_conjectures_jsonl(report.lower_bound_conjectures, lb_path)
        ui.kv("lower_bound_conjectures", lb_path)
    ui.rule("discovery complete")
    ui.kv("outcome_grade_hint", report.outcome_grade_hint)
    outcome = report.outcome or {}
    if outcome.get("target_degenerate"):
        reason = (outcome.get("criteria", {}) or {}).get("target_degeneracy", {}).get("reason", "")
        ui.warn(
            f"TARGET DEGENERATE -- outcome_grade_hint forced to 0. Any R^2/rediscovery "
            f"claim from this run is fitting noise, not signal. {reason}"
        )
    ui.kv("report", out)
    ui.kv("conjectures", conjectures_path)
    ui.kv("descriptor_conjectures", desc_path)
    if report.promoted_conjectures:
        top = report.promoted_conjectures[0]
        ui.kv("best_promoted_R2", f"{top.get('r_squared', 0):.4f}")
        ui.info(f"[{top.get('source')}] {str(top.get('expression', ''))[:80]}")
    elif report.metric_candidates:
        top = report.metric_candidates[0]
        ui.kv("best_metric_R2", f"{top.get('r_squared', 0):.4f}")
        ui.info(str(top.get("expression", ""))[:80])
    if report.descriptor_candidates:
        top_desc = report.descriptor_candidates[0]
        ui.kv("best_descriptor_R2", f"{top_desc.get('r_squared', 0):.4f}")
        ui.info(str(top_desc.get("key", ""))[:80])
    if not plain:
        log_progress(format_discovery_summary(report))

    if getattr(args, "retain_topk", False):
        require_seal_dependencies()
        if not is_run_sealed(args.store_root, args.run_id):
            seal_run(args.store_root, args.run_id)
        topk_record = retain_topk_run(
            args.store_root,
            args.run_id,
            target=target,
            top_k=getattr(args, "top_k_retention", 64),
            discovery_report_path=out,
            delete_full_shard=not getattr(args, "keep_full_shard", False),
        )
        ui.success(
            f"retained top-K columns={len(topk_record.get('retained_columns') or [])} "
            f"bytes_freed={topk_record.get('bytes_freed', 0)}"
        )
        ui.info(topk_record.get("non_claim", ""))
    return 0


def cmd_discover_symbolic(args: argparse.Namespace) -> int:
    """Phase 5 only: symbolic regression on target + Phase-4 latent interpretation."""
    ui = _console(args)
    plain = _is_plain(args)
    if not plain:
        ui.banner(subtitle="phase 5  equation search")

    backends = symbolic_backends()
    if plain:
        log_progress(f"symbolic backends: {backends}")
    else:
        ui.kv("polynomial_fallback", backends["polynomial_fallback"])
        ui.kv("physics_templates", backends["physics_templates"])
        ui.kv("native_gp", backends["native_gp"])
        ui.kv("pysr", backends["pysr"])
        ui.kv("operon", backends["operon"])
        ui.kv("ai_feynman", backends["ai_feynman"])
        if backends.get("errors"):
            ui.warn(str(backends["errors"]))

    target = resolve_target_for_run(args.run_id, args.store_root, override=args.target)
    ui.kv("run_id", args.run_id)
    ui.kv("target", target)

    phase4 = run_phase4(
        args.run_id,
        args.store_root,
        target=target,
        prefer_torch=args.torch_ae,
        checkpoint=True,
    )
    rows = flatten_features(args.run_id, args.store_root)
    ui.kv("n_rows", len(rows))
    if not rows:
        ui.error("no feature rows")
        return 1

    latent_sources: list[LatentSource] = []
    if phase4.pca is not None and phase4.pca.latent_codes.size:
        latent_sources.append(
            LatentSource(
                codes=phase4.pca.latent_codes,
                columns=[f"latent.pca_{i}" for i in range(phase4.pca.latent_dim)],
                source="pca",
            )
        )
    auto = phase4.feature_autoencoder
    if auto is not None and auto.latent_codes.size:
        latent_sources.append(
            LatentSource(codes=auto.latent_codes, columns=auto.latent_columns, source="autoencoder")
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
            latent_sources.append(LatentSource(codes=q_codes, columns=q_cols, source="instance_q"))

    X, cols, _ = load_feature_matrix(args.run_id, args.store_root, target=target)
    feat_cols = [c for c in cols if c != target][:32]

    report = run_phase5(
        args.run_id,
        args.store_root,
        target=target,
        rows=rows,
        feature_columns=feat_cols,
        latent_sources=latent_sources,
        top_latents=args.top_latents,
        max_promoted=args.top,
        use_pysr=not args.no_pysr,
        use_operon=not args.no_operon,
        symbolic_min_r_squared=args.min_r_squared,
        promote_min_r_squared=args.min_r_squared,
        gp_generations=getattr(args, "gp_generations", 20),
        gp_population=getattr(args, "gp_population", 48),
        promote=True,
    )

    out = Path(args.output) if args.output else Path(args.store_root) / "discovery" / f"{args.run_id}_symbolic.json"
    write_phase5_report(report, out)

    conjectures_path = out.with_name("conjectures.jsonl")
    stub_report = DiscoveryReport(
        run_id=args.run_id,
        target=target,
        n_rows=len(rows),
        promoted_conjectures=report.promoted_conjectures,
    )
    write_conjectures_jsonl(stub_report, conjectures_path)

    ui.rule("symbolic discovery complete")
    ui.kv("target_fit", f"{report.symbolic.get('method')} R²={report.symbolic.get('r_squared'):.4f}")
    ui.kv("latent_interpretations", len(report.latent_interpretations))
    ui.kv("promoted", len(report.promoted_conjectures))
    ui.kv("report", out)
    ui.kv("conjectures", conjectures_path)
    if not plain:
        log_progress(format_phase5_summary(report))
    return 0


def cmd_latent(args: argparse.Namespace) -> int:
    ui = _console(args)
    plain = _is_plain(args)
    if not plain:
        ui.banner(subtitle="phase 4  latent discovery")
    target = resolve_target_for_run(args.run_id, args.store_root, override=args.target)
    ui.kv("run_id", args.run_id)
    ui.kv("target", target)

    phase4_stages = 8

    def _run(on_stage):
        return run_phase4(
            args.run_id,
            args.store_root,
            target=target,
            n_pca_components=args.pca_components,
            hidden_dim=args.hidden_dim,
            prefer_torch=args.torch_ae,
            prefer_mlx=args.mlx_ae,
            checkpoint=not args.no_checkpoint,
            on_stage=on_stage,
        )

    if plain:
        def on_stage(name: str) -> None:
            log_progress(f"  -> {name}")

        report = _run(on_stage)
    else:
        with ui.task_progress("phase 4 latent", phase4_stages) as prog:
            def on_stage(name: str) -> None:
                prog.update(1, detail=name)

            report = _run(on_stage)

    out = Path(args.output) if args.output else Path(args.store_root) / "discovery" / f"{args.run_id}_phase4.json"
    write_phase4_report(report, out)
    ui.rule("phase 4 complete")
    ui.kv("n_rows", report.n_rows)
    ui.kv("checkpoints", report.checkpoint_dir)
    ui.kv("report", out)
    if not plain:
        log_progress(format_phase4_summary(report))
    return 0


def cmd_represent(args: argparse.Namespace) -> int:
    ui = _console(args)
    plain = _is_plain(args)
    if not plain:
        ui.banner(subtitle="phase 6  representation discovery")
    target = resolve_target_for_run(args.run_id, args.store_root, override=args.target)
    ui.kv("run_id", args.run_id)
    ui.kv("target", target)

    phase6_stages = 6

    def _run(on_stage):
        return run_phase6(
            args.run_id,
            args.store_root,
            target=target,
            gp_generations=args.gp_generations,
            gp_population=args.gp_population,
            walsh_max_candidates=args.walsh_candidates,
            walsh_max_support=args.walsh_support,
            rediscovery_splits=args.rediscovery_splits,
            expr_eval_backend=args.expr_backend or args.backend,
            on_stage=on_stage,
        )

    if plain:
        def on_stage(name: str) -> None:
            log_progress(f"  -> {name}")

        report = _run(on_stage)
    else:
        with ui.task_progress("phase 6 representation", phase6_stages) as prog:
            def on_stage(name: str) -> None:
                prog.update(1, detail=name)

            report = _run(on_stage)

    out = Path(args.output) if args.output else Path(args.store_root) / "discovery" / f"{args.run_id}_phase6.json"
    write_phase6_report(report, out)
    ui.rule("phase 6 complete")
    ui.kv("n_rows", report.n_rows)
    ui.kv("g4_met", report.g4_met)
    ui.kv("g5_met", report.g5_met)
    ui.kv("g5_hint", report.g5_hint)
    ui.kv("report", out)
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    ui = _console(args)
    target = resolve_target_for_run(args.run_id, args.store_root, override=args.target)
    records: list[dict] = []
    if args.conjectures:
        path = Path(args.conjectures)
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        disc_path = Path(args.store_root) / "discovery" / f"{args.run_id}.json"
        if disc_path.exists():
            disc = json.loads(disc_path.read_text(encoding="utf-8"))
            records = disc.get("promoted_conjectures") or disc.get("metric_candidates") or []
        else:
            con_path = Path(args.store_root) / "discovery" / "conjectures.jsonl"
            if con_path.exists():
                records = [json.loads(line) for line in con_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    from rde.analyze.leak_audit import audit_conjectures
    domain_id = Store(args.store_root).read_manifest(args.run_id).domain_id

    clean, summary = audit_conjectures(
        records,
        target=target,
        domain_id=domain_id,
    )
    ui.rule("leak audit")
    ui.kv("target", target)
    ui.kv("n_total", summary["n_total"])
    ui.kv("n_passed", summary["n_passed"])
    ui.kv("n_blocked", summary["n_blocked"])
    ui.kv("promotion_blocked", summary["promotion_blocked"])
    if args.output:
        payload = {"summary": summary, "clean": clean, "target": target}
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        ui.success(f"written {args.output}")
    return 0


def cmd_obstruct(args: argparse.Namespace) -> int:
    ui = _console(args)
    rows = flatten_features(args.run_id, args.store_root)
    target = resolve_target_for_run(args.run_id, args.store_root, override=args.target)
    from rde.analyze.obstructions import panel_summary, witness_panel
    from rde.discovery.promote_lb import build_lower_bound_conjectures, write_lower_bound_conjectures_jsonl

    records = witness_panel(rows, target)
    summary = panel_summary(records)
    conjectures = build_lower_bound_conjectures(rows, target=target, run_id=args.run_id)
    lb_path = Path(args.store_root) / "discovery" / "lower_bound_conjectures.jsonl"
    write_lower_bound_conjectures_jsonl(conjectures, lb_path)
    ui.rule("obstruction panel")
    ui.kv("target", target)
    ui.kv("n_witnesses", summary["n_witnesses"])
    ui.kv("exponential_witnesses", len(summary["exponential_witnesses"]))
    ui.kv("negative_outcome", summary["negative_outcome"])
    ui.kv("lower_bound_conjectures", lb_path)
    if args.output:
        payload = {"summary": summary, "conjectures": conjectures, "witnesses": [r.name for r in records]}
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        ui.success(f"written {args.output}")
    return 0


def cmd_certify(args: argparse.Namespace) -> int:
    ui = _console(args)
    rows = flatten_features(args.run_id, args.store_root)
    target = resolve_target_for_run(args.run_id, args.store_root, override=args.target)
    records: list[dict] = []
    disc_path = Path(args.store_root) / "discovery" / f"{args.run_id}.json"
    if disc_path.exists():
        disc = json.loads(disc_path.read_text(encoding="utf-8"))
        records = disc.get("promoted_conjectures") or []
    if not records:
        con_path = Path(args.store_root) / "discovery" / "conjectures.jsonl"
        if con_path.exists():
            records = [json.loads(line) for line in con_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        ui.error("no conjectures found — run rde discover first")
        return 1
    idx = int(getattr(args, "conjecture_index", 0))
    idx = min(idx, len(records) - 1)
    from rde.analyze.certify import certify_representation_candidate

    domain_id = Store(args.store_root).read_manifest(args.run_id).domain_id
    result = certify_representation_candidate(
        records[idx],
        rows=rows,
        target=target,
        domain_id=domain_id,
    )
    ui.rule("certify candidate")
    ui.kv("target", target)
    ui.kv("passed", result.passed)
    ui.kv("poly_gates", result.poly_gates)
    ui.kv("poly_shots", result.poly_shots)
    for k, v in result.checks.items():
        ui.kv(k, v)
    if result.violations:
        ui.kv("violations", result.violations)
    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "passed": result.passed,
                    "checks": result.checks,
                    "violations": result.violations,
                    "poly_gates": result.poly_gates,
                    "poly_shots": result.poly_shots,
                    "resource_model": result.resource_model.to_dict(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        ui.success(f"written {args.output}")
    return 0 if result.passed else 1


def cmd_synthesize(args: argparse.Namespace) -> int:
    """Backward (target-first) algorithm synthesis (Mode 2, ALGO-057).

    Reverse of `discover`: instead of fitting a representation to already-
    generated data, declare a resource budget and search for an algorithm
    skeleton that meets it *and* is verified correct against the domain's own
    brute-force oracle — never a skeleton chosen because it happens to fit
    already-known output.
    """
    ui = _console(args)
    from rde.synthesis import default_skeleton_catalog, synthesize, write_synthesis_conjectures_jsonl

    reg = build_registry(args.domain, compute_backend=None, max_bruteforce_n=None)
    domain = reg.get_domain(args.domain)
    # "generate" is Domain's, not SynthesisDomain's, but cmd_synthesize needs
    # it to produce instances — check for it too so a domain missing only
    # this gets the same clean error instead of a raw AttributeError below.
    required = ("generate", "size_of", "brute_force", "decompose_flat", "decompose_divide", "combine", "cost")
    missing = [m for m in required if not hasattr(domain, m)]
    if missing:
        ui.error(f"domain {args.domain!r} does not implement SynthesisDomain (missing: {missing})")
        return 1

    instances = domain.generate(args.n_instances, args.size, args.seed)
    base_exponent = args.base_exponent
    if base_exponent is None:
        base_exponent = getattr(domain, "base_case_cost_exponent", lambda: 1.0)()

    catalog = default_skeleton_catalog(base_exponent=base_exponent)
    # Worst case (every skeleton needs domain verification): one unit per
    # instance while precomputing brute-force ground truth, plus one per
    # skeleton. If skeletons get pruned symbolically first, fewer updates
    # fire and the bar simply doesn't reach 100% — still real, not simulated.
    total_units = len(instances) + len(catalog)
    with ui.task_progress("algorithm synthesis", total_units) as prog:

        def on_progress(detail: str) -> None:
            prog.update(1, detail=detail)

        report = synthesize(
            domain,
            instances,
            target_degree=args.target_degree,
            base_threshold=args.base_threshold,
            base_exponent=base_exponent,
            catalog=catalog,
            on_progress=on_progress,
        )
    out_path = Path(args.store_root) / "discovery" / "synthesis_conjectures.jsonl"
    write_synthesis_conjectures_jsonl(report, out_path)

    summary = report.summary()
    ui.rule("algorithm synthesis")
    for key, value in summary.items():
        ui.kv(key, value)
    for candidate in report.candidates:
        ui.kv(f"  {candidate.name}", f"{candidate.status}: {candidate.cost_class} — {candidate.detail}")
    ui.kv("synthesis_conjectures", out_path)
    if args.output:
        Path(args.output).write_text(
            json.dumps({"summary": summary, "candidates": [c.to_dict() for c in report.candidates]}, indent=2),
            encoding="utf-8",
        )
        ui.success(f"written {args.output}")
    return 0 if report.accepted else 1


def cmd_outcome(args: argparse.Namespace) -> int:
    ui = _console(args)
    rows = flatten_features(args.run_id, args.store_root)
    target = resolve_target_for_run(args.run_id, args.store_root, override=args.target)
    from rde.io.store import Store

    domain_id = Store(args.store_root).read_manifest(args.run_id).domain_id
    discovery: dict | None = None
    if args.discovery:
        discovery = json.loads(Path(args.discovery).read_text(encoding="utf-8"))
    assessment = assess_outcome(
        rows,
        target,
        discovery=discovery,
        metric_candidates=discovery.get("metric_candidates") if discovery else None,
        latent=discovery.get("latent") if discovery else None,
        phase6=discovery.get("phase6") if discovery else None,
        leak_audit_summary=discovery.get("leak_audit_summary") if discovery else None,
        certify_result=(discovery.get("certify_results") or [None])[0] if discovery else None,
        obstruct_summary=discovery.get("obstruct_summary") if discovery else None,
        domain_id=domain_id,
    )
    ui.rule("outcome assessment")
    ui.kv("target", target)
    ui.kv("n_rows", len(rows))
    ui.kv("grade", assessment.grade)
    ui.kv("g0_met", assessment.g0_met)
    ui.kv("g1_met", assessment.g1_met)
    ui.kv("g2_met", assessment.g2_met)
    ui.kv("g3_met", assessment.g3_met)
    ui.kv("g4_met", assessment.g4_met)
    ui.kv("g5_met", assessment.g5_met)
    ui.kv("promotion_blocked", assessment.promotion_blocked)
    ui.kv("negative_outcome", assessment.negative_outcome)
    if assessment.g1_triggers:
        ui.kv("g1_triggers", assessment.g1_triggers)
    if assessment.g0_triggers:
        ui.kv("g0_triggers", assessment.g0_triggers)
    for key, val in assessment.criteria.items():
        ui.kv(key, val)
    output = assessment.to_payload()
    output["n_rows"] = len(rows)
    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
        ui.success(f"written {args.output}")
    return 0


def cmd_stress_store(args: argparse.Namespace) -> int:
    from rde.testing.store_stress import stress_store_features, stress_write_jsonl

    ui = _console(args)
    ui.rule("store stress benchmark")
    if args.via_store:
        with ui.task_progress("writing via Store", args.n_rows) as prog:
            stats = stress_store_features(
                args.store_root,
                args.run_id,
                args.n_rows,
                batch_size=args.batch_size,
                on_batch=lambda n: prog.update(n),
            )
        ui.kv("run_id", stats["run_id"])
        ui.kv("n_rows", stats["n_rows"])
        ui.kv("bytes", stats["bytes"])
        ui.kv("batch_size", stats["batch_size"])
        return 0
    if not args.output:
        ui.error("provide --output for raw mode or use --via-store")
        return 1
    out = Path(args.output)
    with ui.task_progress("writing JSONL", args.n_rows) as prog:
        nbytes = stress_write_jsonl(
            out,
            args.n_rows,
            batch_size=args.batch_size,
            on_batch=lambda n: prog.update(n),
        )
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    ui.kv("n_rows", len(lines))
    ui.kv("bytes", nbytes)
    ui.kv("batch_size", args.batch_size)
    ui.kv("path", out)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    ui = _console(args)
    rows = flatten_features(args.run_id, args.store_root)
    metric = args.metric
    group_by = args.group_by
    sep = separation_score(rows, metric=metric, label_key=group_by)
    summary = complexity_by_group(rows, metric=metric, group_by=group_by)
    ui.rule("calibrate")
    ui.kv("metric", metric)
    ui.kv("group_by", group_by)
    ui.kv("n_rows", len(rows))
    ui.kv("separation", f"{sep:.4f}")
    table_rows = [
        [str(r.get(group_by, "?")), r["count"], f"{r['mean']:.4f}", f"{r['std']:.4f}"]
        for r in summary[: args.top]
    ]
    ui.table("groups", [group_by, "n", "mean", "std"], table_rows)
    output = {"metric": metric, "group_by": group_by, "separation_score": sep, "groups": summary}
    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
        ui.success(f"written {args.output}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    ui = _console(args)
    store = Store(args.store_root)
    feat_errors = validate_features_file(store.read_features(args.run_id))
    inst_errors = validate_instance_features_file(store.read_instance_features(args.run_id))
    errors = feat_errors + inst_errors
    if errors:
        for idx, errs in errors[:10]:
            ui.error(f"row {idx}: {errs}")
        ui.error(f"validation_failed errors={len(errors)}")
        return 1
    n_feat = len(store.read_features(args.run_id))
    n_inst = len(store.read_instance_features(args.run_id))
    ui.success(f"validation_ok features={n_feat} instance_features={n_inst}")
    return 0


def cmd_power_plan(args: argparse.Namespace) -> int:
    from rde.analyze.power_plan import power_plan_to_dict, simulate_power
    from rde.discovery.datasets import load_campaign_matrix

    ui = _console(args)
    _, _, rows = load_campaign_matrix(
        args.campaign_id, args.store_root, target=args.target
    )
    if not rows:
        ui.error("no rows loaded (campaign may be missing or empty)")
        return 1
    plan = simulate_power(rows, target_col=args.target, max_instances_cap=args.max_instances)
    out = power_plan_to_dict(plan)
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    ui.success(json.dumps(out, indent=2))
    return 0


def cmd_calibrate_hardware(args: argparse.Namespace) -> int:
    import time

    from rde.core.plugins import build_registry
    from rde.features.catalog import PANEL_SMOKE_MINIMAL, resolve_descriptor_panel
    from rde.runtime.pipeline import RunConfig, run_pipeline

    ui = _console(args)
    sizes = _parse_int_list(args.sizes)
    effective_backend = resolve_compute_backend(
        args.backend,
        size=max(sizes, default=None),
        batch_size=args.compute_batch_size,
    )
    reg = build_registry(args.domain, compute_backend=effective_backend)
    desc, metrics = resolve_descriptor_panel(reg, PANEL_SMOKE_MINIMAL)
    indices = [0]
    save_arrays = False
    generator_id = args.generator
    calibration_store = (
        Path(args.store_root) / "_calibration" / f"{args.profile}_{effective_backend}"
    )
    calibration_store.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    storage_bytes: dict[str, int] = {}
    for size in sizes:
        before_bytes = directory_size_bytes(calibration_store)
        t0 = time.perf_counter()
        run_pipeline(
            RunConfig(
                domain_id=args.domain,
                n_instances=args.n_instances,
                size=size,
                seed=0,
                indices=indices,
                descriptor_names=desc,
                metric_names=metrics,
                store_root=calibration_store,
                run_id=f"calibration_{args.profile}_n{size}",
                save_arrays=save_arrays,
                resume=False,
                workers=args.workers,
                compute_backend=effective_backend,
                compute_batch_size=args.compute_batch_size,
                generator_id=generator_id,
            ),
            registry=reg,
        )
        timings[str(size)] = time.perf_counter() - t0
        storage_bytes[str(size)] = directory_size_bytes(calibration_store) - before_bytes
    total_seconds = sum(timings.values())
    per_unit = [
        elapsed / max(1, args.n_instances)
        for elapsed in timings.values()
    ]
    out = {
        "backend_requested": args.backend,
        "backend_effective": effective_backend,
        "workers": args.workers,
        "sizes": sizes,
        "n_instances_per_size": args.n_instances,
        "timings_s": timings,
        "units": len(timings) * args.n_instances,
        # Admission uses the slowest measured size, not the mean, so a
        # large-N profile cannot be admitted on an optimistic average.
        "seconds_per_unit": max(per_unit, default=0.0),
        "seconds_per_unit_mean": total_seconds / max(1, len(timings) * args.n_instances),
        "storage_bytes_per_unit": max(
            (
                storage_bytes[str(size)] / max(1, args.n_instances)
                for size in sizes
            ),
            default=0.0,
        ),
        "storage_bytes_per_unit_mean": sum(storage_bytes.values()) / max(
            1, len(timings) * args.n_instances
        ),
        "peak_rss_mb": peak_rss_mb(),
        "profile": args.profile,
        "descriptor_count": len(desc),
        "metric_count": len(metrics),
        "indices": indices,
        "save_arrays": save_arrays,
        "generator_id": generator_id,
        "calibration_store": str(calibration_store),
        "contract": {
            "profile": args.profile,
            "domain": args.domain,
            "sizes": sizes,
            "indices": indices,
            "save_arrays": save_arrays,
            "generator_id": generator_id,
            "backend_effective": effective_backend,
            "workers": args.workers,
            "compute_batch_size": args.compute_batch_size,
            "descriptor_names_sha256": configuration_fingerprint({"names": desc}),
            "metric_names_sha256": configuration_fingerprint({"names": metrics}),
        },
    }
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    ui.success(json.dumps(out, indent=2))
    return 0


def cmd_machine_profile(args: argparse.Namespace) -> int:
    ui = _console(args)
    profile = dev_machine_profile()
    if getattr(args, "json", False):
        sys.stdout.write(format_dev_machine_profile(as_json=True))
        sys.stdout.flush()
        return 0
    ui.rule("dev machine profile")
    for key, value in profile.to_dict().items():
        ui.kv(key, value)
    if profile.profile_id == "apple_silicon_mac":
        ui.info("RDE compute: prefer --backend mlx (or auto) on this machine.")
    elif profile.profile_id == "intel_mac":
        ui.info("RDE compute: use --backend numpy; MLX is not installed on Intel Macs.")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    ui = _console(args)
    if args.what == "domains":
        items = list_domain_ids()
        ui.table("domains", ["id"], [[d] for d in items])
    elif args.what == "generators":
        items = list_generators(args.domain)
        ui.table("generators", ["id"], [[g] for g in items])
    elif args.what == "backends":
        items = available_backends()
        ui.table("backends", ["name"], [[b] for b in items])
    elif args.what == "descriptors":
        from rde.core.plugins import build_registry
        from rde.features.catalog import estimate_keys_per_slice, generated_template_count

        reg = build_registry("synthetic_poly")
        modules = reg.list_descriptors()
        counts = estimate_keys_per_slice()
        rows = [
            ["builtin_modules", ", ".join(modules)],
            ["hand_keys_estimate", str(counts["hand_descriptors_estimate"])],
            ["gen_pipeline_templates", str(counts["generated_templates_pipeline"])],
            ["gen_full_catalog", str(generated_template_count())],
            ["total_per_slice_estimate", str(counts["total_per_slice_estimate"])],
        ]
        ui.table("descriptor catalog", ["key", "value"], rows)
    return 0


def _repr_demo_batch(pattern: str, n: int, samples: int, seed: int):
    """Generic (domain-agnostic) demo batches for `repr-rank --input` omitted.

    Never imports `rde_domains` — core CLI must not depend on domain
    plugins. Real domain data (hsp_functions/tsp) is reachable through
    `rde_domains.hsp_functions.representations` /
    `rde_domains.tsp.representations`'s own Python API, not through this
    command; those modules have no CLI of their own to attach to (they are
    plugin libraries, not entry points), and forcing a domain import into
    `rde/cli/` would violate the same core/domain boundary
    `tests/rde/integration/test_no_rde_domains_import.py` enforces.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    if pattern == "random":
        return rng.normal(size=(samples, n))
    if pattern == "periodic":
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        return np.stack([np.sin(3 * t + phase) for phase in rng.normal(size=samples)])
    if pattern == "polynomial":
        nodes = np.arange(n, dtype=float)
        slopes = rng.normal(size=samples)
        intercepts = rng.normal(size=samples)
        return intercepts[:, None] + slopes[:, None] * nodes[None, :]
    raise ValueError(f"unknown --pattern: {pattern!r}")


def cmd_repr_rank(args: argparse.Namespace) -> int:
    ui = _console(args)
    plain = _is_plain(args)
    import numpy as np

    from rde.representation import rank_representations, write_search_report

    if args.input:
        batch = np.load(args.input)
        n = batch.shape[1]
    else:
        n = args.n
        batch = _repr_demo_batch(args.pattern, n, args.samples, args.seed)

    ranked = rank_representations(batch, n=n, tolerance=args.tolerance)

    if not plain:
        ui.banner(subtitle="representation search")
    ui.kv("n", n)
    ui.kv("samples", batch.shape[0])
    rows = [
        [
            c.representation_id,
            f"{c.complexity:.4g}",
            f"{c.conversion_cost:.4g}",
            c.certificate.status,
            f"{c.certificate.error:.3e}",
        ]
        for c in ranked
    ]
    ui.table(
        "representation ranking",
        ["representation_id", "complexity", "conversion_cost", "status", "roundtrip_error"],
        rows,
    )

    if args.output:
        out = write_search_report(ranked, args.output)
        ui.kv("report", out)
    return 0


def cmd_repr_rank_run(args: argparse.Namespace) -> int:
    """Rank the grammar against one already-materialized array field from a real run.

    Purely additive: reads `instance_features.jsonl` + `arrays/` that
    `run`/`campaign` already wrote via `Store`, does not touch or modify
    them, and does not change what `run`/`campaign`/`discover` themselves
    do. `--array-key` names an array-valued `primitive_features()` key
    (e.g. hsp_functions' `diff_profile`, tsp's `D`) — this command never
    imports `rde_domains` (same core/domain boundary as `repr-rank`'s demo
    batches); the caller supplies the key as a plain string, so it works
    for any domain without this command knowing what the key means.
    """
    ui = _console(args)
    plain = _is_plain(args)
    import numpy as np

    from rde.io.store import Store
    from rde.representation import rank_representations, write_search_report_to_store

    store = Store(args.store_root)
    rows = store.read_instance_features(args.run_id)
    if not rows:
        ui.warn(f"no instance_features found for run_id={args.run_id!r}")
        return 1

    by_size: dict[int, list[np.ndarray]] = {}
    for row in rows:
        array_refs = row.get("array_refs") or {}
        if args.array_key not in array_refs:
            continue
        array = store.load_array(args.run_id, row["instance_id"], f"primitive_{args.array_key}")
        by_size.setdefault(int(row["size"]), []).append(np.asarray(array, dtype=float))

    if not by_size:
        ui.warn(
            f"no instance_features rows had array_key={args.array_key!r} saved "
            f"(checked {len(rows)} rows)"
        )
        return 1

    if not plain:
        ui.banner(subtitle="representation search over a stored run")
    ui.kv("run_id", args.run_id)
    ui.kv("array_key", args.array_key)

    any_written = False
    for size in sorted(by_size):
        arrays = by_size[size]
        lengths = {a.shape[-1] for a in arrays}
        if len(lengths) != 1:
            ui.warn(f"size={size}: inconsistent array lengths {sorted(lengths)}; skipping")
            continue
        n = lengths.pop()
        batch = np.stack(arrays, axis=0)
        ranked = rank_representations(batch, n=n, tolerance=args.tolerance)
        table_rows = [[c.representation_id, f"{c.complexity:.4g}", c.certificate.status] for c in ranked]
        ui.table(
            f"size={size} (n={n}, samples={batch.shape[0]})",
            ["representation_id", "complexity", "status"],
            table_rows,
        )
        write_search_report_to_store(ranked, store, args.run_id)
        any_written = True

    store.flush(args.run_id)
    if any_written:
        ui.kv("representation_reports", store.run_dir(args.run_id) / "representation_reports.jsonl")
        return 0
    return 1


def _add_store_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store-root", default="rde_runs", help="Store root directory")


def build_parser() -> argparse.ArgumentParser:
    """Build the full `rde` argument parser (all subcommands registered).

    Split out from ``main`` so tests can reflect on subparser wiring
    (e.g. verifying every ``cmd_*`` handler only reads ``args.<x>`` for an
    ``x`` actually registered on its own subparser) without going through
    ``parse_args``/dispatch.
    """
    parser = argparse.ArgumentParser(
        prog="rde",
        description="Representation Discovery Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Tip: set RDE_PLAIN=1 or pass --plain for CI / log files.",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Plain text output (no colors, no live progress)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        dest="plain",
        help="Alias for --plain",
    )
    parser.add_argument(
        "--log-progress",
        action="store_true",
        help="Append newline progress lines to stdout/log (for tee); default is in-place TTY UI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a single pipeline job")
    _add_store_root(run_p)
    _add_perf_flags(run_p)
    run_p.add_argument("--domain", required=True)
    run_p.add_argument("--n-instances", type=int, default=4)
    run_p.add_argument("--size", type=int, required=True)
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument("--indices", default="1,2,4")
    run_p.add_argument("--run-id", default=None)
    run_p.add_argument("--resume", action="store_true")
    run_p.add_argument("--no-arrays", action="store_true")
    _add_coined_walk_flags(run_p)
    run_p.set_defaults(func=cmd_run)

    camp_p = sub.add_parser("campaign", help="Run a multi-size campaign")
    _add_store_root(camp_p)
    _add_perf_flags(camp_p)
    camp_p.add_argument("--domain", required=True)
    camp_p.add_argument("--sizes", required=True, help="Comma-separated N values")
    camp_p.add_argument("--n-per-size", type=int, default=100)
    camp_p.add_argument("--seed", type=int, default=0)
    camp_p.add_argument("--indices", default="1,2,4,8,16")
    camp_p.add_argument("--campaign-id", default=None)
    camp_p.add_argument("--resume", action="store_true", default=True)
    camp_p.add_argument("--no-resume", action="store_false", dest="resume")
    camp_p.add_argument("--no-arrays", action="store_true", help="Skip NPZ array sidecars")
    camp_p.add_argument(
        "--max-storage-gb",
        type=float,
        default=None,
        help="Stop this campaign invocation after approximately this many GB of new artifacts",
    )
    camp_p.add_argument(
        "--seal-batches",
        action="store_true",
        help="After the session, verify a compact Parquet snapshot and remove SQLite/arrays",
    )
    camp_p.add_argument(
        "--seal-keep-arrays",
        action="store_true",
        help="With --seal-batches, retain NPZ arrays for representation/massive-catalog work",
    )
    _add_coined_walk_flags(camp_p)
    camp_p.set_defaults(func=cmd_campaign)

    seal_p = sub.add_parser("seal", help="Verify and compact one run or campaign")
    _add_store_root(seal_p)
    seal_group = seal_p.add_mutually_exclusive_group(required=True)
    seal_group.add_argument("--run-id", default=None)
    seal_group.add_argument("--campaign-id", default=None)
    seal_p.add_argument("--batch-index", type=int, default=None)
    seal_p.add_argument(
        "--keep-arrays",
        action="store_true",
        help="Retain NPZ arrays; default removes arrays after verification",
    )
    seal_p.add_argument(
        "--keep-jsonl",
        action="store_true",
        help="Retain features.jsonl and instance_features.jsonl after verification",
    )
    seal_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be sealed/deleted and expected bytes freed",
    )
    seal_p.set_defaults(func=cmd_seal)

    topk_p = sub.add_parser(
        "retain-topk",
        help="After discovery ranking, keep targets + ranked columns only",
    )
    _add_store_root(topk_p)
    topk_group = topk_p.add_mutually_exclusive_group(required=True)
    topk_group.add_argument("--run-id", default=None)
    topk_group.add_argument("--campaign-id", default=None)
    topk_p.add_argument("--target", default=None, help="Primary target column (default: domain-specific)")
    topk_p.add_argument(
        "--top-k",
        type=int,
        default=64,
        help="Maximum ranked feature columns to retain beyond ids/target",
    )
    topk_p.add_argument(
        "--discovery-report",
        default=None,
        help="Discovery report JSON used to choose ranked columns",
    )
    topk_p.add_argument(
        "--ranked-columns",
        default=None,
        help="Comma-separated ranked columns (overrides discovery report)",
    )
    topk_p.add_argument(
        "--keep-full-shard",
        action="store_true",
        help="Retain the full sealed Parquet shard after writing features_topk.parquet",
    )
    topk_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show retained columns and expected bytes freed without writing/deleting",
    )
    topk_p.set_defaults(func=cmd_retain_topk)

    export_p = sub.add_parser("export", help="Export features to CSV or Parquet")
    _add_store_root(export_p)
    export_p.add_argument("--run-id", required=True)
    export_p.add_argument("--output", required=True)
    export_p.add_argument("--format", choices=["csv", "parquet"], default="csv")
    export_p.set_defaults(func=cmd_export)

    summary_p = sub.add_parser("summary", help="Print run summary")
    _add_store_root(summary_p)
    summary_p.add_argument("--run-id", required=True)
    summary_p.set_defaults(func=cmd_summary)

    analyze_p = sub.add_parser("analyze", help="Analyze feature tables")
    _add_store_root(analyze_p)
    analyze_p.add_argument("--run-id", required=True)
    analyze_p.add_argument("--target", default=None, help="Target column (default: domain-specific)")
    analyze_p.add_argument("--mode", choices=["correlations", "distribution", "outliers", "cross_n", "matrix"], default="correlations")
    analyze_p.add_argument("--min-abs-r", type=float, default=0.3)
    analyze_p.add_argument("--z-threshold", type=float, default=2.5)
    analyze_p.add_argument("--top", type=int, default=20)
    analyze_p.add_argument("--output", default=None, help="Write JSON results to this path")
    analyze_p.set_defaults(func=cmd_analyze)

    discover_p = sub.add_parser("discover", help="Run full discovery loop (Phases 3–6)")
    _add_store_root(discover_p)
    discover_p.add_argument("--run-id", required=True)
    discover_p.add_argument("--target", default=None, help="Target column (default: domain-specific)")
    discover_p.add_argument("--output", default=None)
    discover_p.add_argument("--max-candidates", type=int, default=20_000)
    discover_p.add_argument(
        "--massive-catalog",
        action="store_true",
        help=(
            "Enumerate the executable massive descriptor catalog; requires "
            "campaign slice arrays and defaults expression search to 1,000,000"
        ),
    )
    discover_p.add_argument(
        "--max-descriptor-templates",
        type=int,
        default=None,
        help="Optional cap for descriptor templates (massive mode uses its full registered budget)",
    )
    discover_p.add_argument("--expr-depth", type=int, default=3, help="Expression DSL max depth")
    discover_p.add_argument("--gp-generations", type=int, default=10)
    discover_p.add_argument("--gp-population", type=int, default=40)
    discover_p.add_argument(
        "--backend",
        default=None,
        choices=available_backends(),
        help="Unified eval backend alias (mlx default); same as --expr-backend when set",
    )
    discover_p.add_argument(
        "--expr-backend",
        default=None,
        choices=expr_backend_choices(),
        help="Expression DSL eval backend (default: mlx on Apple Silicon when Metal is available)",
    )
    discover_p.add_argument("--no-pysr", action="store_true")
    discover_p.add_argument("--no-operon", action="store_true")
    discover_p.add_argument("--top-latents", type=int, default=3, help="Phase 5: interpret top-N predictive latents")
    discover_p.add_argument("--promote-top", type=int, default=10, help="Max promoted conjectures in conjectures.jsonl")
    discover_p.add_argument(
        "--torch-ae",
        action="store_true",
        help="Opt-in: try PyTorch MLP autoencoder before numpy random features",
    )
    discover_p.add_argument(
        "--mlx-ae",
        action="store_true",
        help="Opt-in: try MLX MLP autoencoder before numpy random features",
    )
    discover_p.add_argument(
        "--obstruct",
        action="store_true",
        help="Run obstruction witness panel and emit lower_bound_conjectures.jsonl",
    )
    discover_p.add_argument(
        "--retain-topk",
        action="store_true",
        help="After discovery, write features_topk.parquet and drop the full sealed shard",
    )
    discover_p.add_argument(
        "--top-k-retention",
        type=int,
        default=64,
        help="With --retain-topk, maximum ranked columns to keep beyond ids/target",
    )
    discover_p.add_argument(
        "--keep-full-shard",
        action="store_true",
        help="With --retain-topk, retain the full sealed Parquet shard",
    )
    discover_p.set_defaults(func=cmd_discover)

    latent_p = sub.add_parser("latent", help="Run Phase 4 latent discovery only")
    _add_store_root(latent_p)
    latent_p.add_argument("--run-id", required=True)
    latent_p.add_argument("--target", default=None)
    latent_p.add_argument("--output", default=None)
    latent_p.add_argument("--pca-components", type=int, default=8)
    latent_p.add_argument("--hidden-dim", type=int, default=16)
    latent_p.add_argument("--torch-ae", action="store_true")
    latent_p.add_argument("--mlx-ae", action="store_true")
    latent_p.add_argument("--no-checkpoint", action="store_true")
    latent_p.set_defaults(func=cmd_latent)

    rep_p = sub.add_parser("represent", help="Run Phase 6 representation discovery only")
    _add_store_root(rep_p)
    _add_perf_flags(rep_p)
    rep_p.add_argument(
        "--expr-backend",
        default=None,
        choices=expr_backend_choices(),
        help="Expression DSL eval backend (default: mlx on Apple Silicon when Metal is available)",
    )
    rep_p.add_argument("--run-id", required=True)
    rep_p.add_argument("--target", default=None)
    rep_p.add_argument("--output", default=None)
    rep_p.add_argument("--gp-generations", type=int, default=12)
    rep_p.add_argument("--gp-population", type=int, default=40)
    rep_p.add_argument("--walsh-candidates", type=int, default=2000)
    rep_p.add_argument("--walsh-support", type=int, default=6)
    rep_p.add_argument("--rediscovery-splits", type=int, default=5)
    rep_p.set_defaults(func=cmd_represent)

    sym_p = sub.add_parser("discover-symbolic", help="Phase 5: symbolic regression + latent interpretation")
    _add_store_root(sym_p)
    sym_p.add_argument("--run-id", required=True)
    sym_p.add_argument("--target", default=None)
    sym_p.add_argument("--output", default=None)
    sym_p.add_argument("--top-latents", type=int, default=3)
    sym_p.add_argument("--top", type=int, default=10, help="Max promoted conjectures")
    sym_p.add_argument("--min-r-squared", type=float, default=0.5)
    sym_p.add_argument("--no-pysr", action="store_true")
    sym_p.add_argument("--no-operon", action="store_true")
    sym_p.add_argument("--torch-ae", action="store_true")
    sym_p.set_defaults(func=cmd_discover_symbolic)

    rank_p = sub.add_parser("rank-expr", help="Rank expression DSL candidates against a target")
    _add_store_root(rank_p)
    rank_p.add_argument("--run-id", required=True)
    rank_p.add_argument("--target", default=None)
    rank_p.add_argument("--max-candidates", type=int, default=5000)
    rank_p.add_argument("--expr-depth", type=int, default=3)
    rank_p.add_argument("--max-vars", type=int, default=16)
    rank_p.add_argument("--min-abs-r", type=float, default=0.25)
    rank_p.add_argument("--top", type=int, default=10)
    rank_p.add_argument(
        "--backend",
        default=None,
        choices=available_backends(),
        help="Unified eval backend alias (mlx default); same as --expr-backend when set",
    )
    rank_p.add_argument(
        "--expr-backend",
        default=None,
        choices=expr_backend_choices(),
        help="Expression DSL eval backend (default: mlx on Apple Silicon when Metal is available)",
    )
    rank_p.add_argument("--output", default=None)
    rank_p.set_defaults(func=cmd_rank_expr)

    rank_metrics_p = sub.add_parser(
        "rank-metrics",
        help="Rank metric generator candidates (ratios, products, expression DSL)",
    )
    _add_store_root(rank_metrics_p)
    rank_metrics_p.add_argument("--run-id", required=True)
    rank_metrics_p.add_argument("--target", default=None)
    rank_metrics_p.add_argument("--max-candidates", type=int, default=20_000)
    rank_metrics_p.add_argument("--expr-depth", type=int, default=3)
    rank_metrics_p.add_argument("--max-vars", type=int, default=24)
    rank_metrics_p.add_argument("--min-abs-r", type=float, default=0.25)
    rank_metrics_p.add_argument("--top", type=int, default=15)
    rank_metrics_p.add_argument("--chunk-size", type=int, default=1024)
    rank_metrics_p.add_argument("--no-ratios", action="store_true")
    rank_metrics_p.add_argument("--no-products", action="store_true")
    rank_metrics_p.add_argument(
        "--backend",
        default=None,
        choices=available_backends(),
    )
    rank_metrics_p.add_argument(
        "--expr-backend",
        default=None,
        choices=expr_backend_choices(),
    )
    rank_metrics_p.add_argument("--output", default=None)
    rank_metrics_p.set_defaults(func=cmd_rank_metrics)

    rank_desc_p = sub.add_parser(
        "rank-desc",
        help="Rank parameterized and derived descriptor generator candidates",
    )
    _add_store_root(rank_desc_p)
    rank_desc_p.add_argument("--run-id", required=True)
    rank_desc_p.add_argument("--target", default=None)
    rank_desc_p.add_argument("--mode", choices=["templates", "derived", "both"], default="both")
    rank_desc_p.add_argument("--max-templates", type=int, default=200)
    rank_desc_p.add_argument("--max-derived", type=int, default=3000)
    rank_desc_p.add_argument("--derived-depth", type=int, default=2)
    rank_desc_p.add_argument("--min-abs-r", type=float, default=0.25)
    rank_desc_p.add_argument("--top", type=int, default=15)
    rank_desc_p.add_argument(
        "--backend",
        default=None,
        choices=available_backends(),
        help="Unified eval backend alias (mlx default); same as --expr-backend when set",
    )
    rank_desc_p.add_argument(
        "--expr-backend",
        default=None,
        choices=expr_backend_choices(),
        help="Expression DSL eval backend for derived descriptors",
    )
    rank_desc_p.add_argument("--output", default=None)
    rank_desc_p.set_defaults(func=cmd_rank_desc)

    calibrate_p = sub.add_parser("calibrate", help="Calibrate complexity metric across instance groups")
    _add_store_root(calibrate_p)
    calibrate_p.add_argument("--run-id", required=True)
    calibrate_p.add_argument("--metric", default="metric.representation_complexity")
    calibrate_p.add_argument("--group-by", default="generator", help="Row key to group by (generator, size, ...)")
    calibrate_p.add_argument("--top", type=int, default=20)
    calibrate_p.add_argument("--output", default=None)
    calibrate_p.set_defaults(func=cmd_calibrate)

    stress_p = sub.add_parser("stress-store", help="Write synthetic JSONL rows to benchmark store I/O")
    stress_p.add_argument("--output", default=None, help="Output JSONL path (raw mode)")
    stress_p.add_argument("--via-store", action="store_true", help="Write schema-valid rows through Store")
    _add_store_root(stress_p)
    stress_p.add_argument("--run-id", default="stress_run")
    stress_p.add_argument("--n-rows", type=int, default=100_000)
    stress_p.add_argument("--batch-size", type=int, default=512)
    stress_p.set_defaults(func=cmd_stress_store)

    outcome_p = sub.add_parser("outcome", help="Assess pre-registered G0 vs G1 outcome")
    _add_store_root(outcome_p)
    outcome_p.add_argument("--run-id", required=True)
    outcome_p.add_argument("--target", default=None)
    outcome_p.add_argument("--discovery", default=None, help="Optional discovery JSON from rde discover")
    outcome_p.add_argument("--output", default=None)
    outcome_p.set_defaults(func=cmd_outcome)

    audit_p = sub.add_parser("audit", help="Leak/tautology audit on promoted conjectures")
    _add_store_root(audit_p)
    audit_p.add_argument("--run-id", required=True)
    audit_p.add_argument("--target", default=None)
    audit_p.add_argument("--conjectures", default=None, help="Path to conjectures JSONL")
    audit_p.add_argument("--output", default=None)
    audit_p.set_defaults(func=cmd_audit)

    obstruct_p = sub.add_parser("obstruct", help="Obstruction witness panel + lower-bound conjectures")
    _add_store_root(obstruct_p)
    obstruct_p.add_argument("--run-id", required=True)
    obstruct_p.add_argument("--target", default=None)
    obstruct_p.add_argument("--output", default=None)
    obstruct_p.set_defaults(func=cmd_obstruct)

    certify_p = sub.add_parser("certify", help="G5 candidate resource/query pre-check")
    _add_store_root(certify_p)
    certify_p.add_argument("--run-id", required=True)
    certify_p.add_argument("--target", default=None)
    certify_p.add_argument("--conjecture-index", type=int, default=0)
    certify_p.add_argument("--output", default=None)
    certify_p.set_defaults(func=cmd_certify)

    synthesize_p = sub.add_parser(
        "synthesize",
        help="Backward target-first algorithm synthesis (Mode 2, ALGO-057)",
    )
    _add_store_root(synthesize_p)
    synthesize_p.add_argument(
        "--domain",
        required=True,
        help="Must implement SynthesisDomain (e.g. block_separable, qubo_separable, qubo_dense_control)",
    )
    synthesize_p.add_argument("--size", type=int, required=True, help="Domain size parameter (e.g. n_blocks)")
    synthesize_p.add_argument("--n-instances", type=int, default=8)
    synthesize_p.add_argument("--seed", type=int, default=0)
    synthesize_p.add_argument(
        "--target-degree",
        type=float,
        default=None,
        help="Max accepted polynomial degree (default: any polynomial)",
    )
    synthesize_p.add_argument(
        "--base-exponent",
        type=float,
        default=None,
        help="Exponential rate of the direct-brute-force skeleton; default asks the domain",
    )
    synthesize_p.add_argument("--base-threshold", type=int, default=1, help="Recursion stops at this instance size")
    synthesize_p.add_argument("--output", default=None)
    synthesize_p.set_defaults(func=cmd_synthesize)

    validate_p = sub.add_parser("validate", help="Validate features.jsonl schema")
    _add_store_root(validate_p)
    validate_p.add_argument("--run-id", required=True)
    validate_p.set_defaults(func=cmd_validate)

    list_p = sub.add_parser("list", help="List domains, generators, or backends")
    list_p.add_argument("what", choices=["domains", "generators", "backends", "descriptors"])
    list_p.add_argument("--domain", default=None, help="Filter generators by domain")
    list_p.set_defaults(func=cmd_list)

    power_p = sub.add_parser("power-plan", help="Prospective power planning from campaign data")
    _add_store_root(power_p)
    power_p.add_argument("--campaign-id", required=True)
    power_p.add_argument("--target", required=True)
    power_p.add_argument("--max-instances", type=int, default=500)
    power_p.add_argument("--output", default=None)
    power_p.set_defaults(func=cmd_power_plan)

    cal_hw_p = sub.add_parser("calibrate-hardware", help="Benchmark a declared RDE resource profile")
    _add_store_root(cal_hw_p)
    _add_perf_flags(cal_hw_p)
    cal_hw_p.add_argument("--domain", required=True)
    cal_hw_p.add_argument("--sizes", required=True)
    cal_hw_p.add_argument("--n-instances", type=int, default=4)
    cal_hw_p.add_argument(
        "--profile",
        choices=("minimal",),
        default="minimal",
        help="Generic descriptor panel to benchmark",
    )
    cal_hw_p.add_argument("--output", default=None)
    cal_hw_p.set_defaults(func=cmd_calibrate_hardware)

    backends_p = sub.add_parser("backends", help="List compute backends")
    backends_p.set_defaults(func=lambda args: cmd_list(argparse.Namespace(what="backends", domain=None)))

    machine_p = sub.add_parser(
        "machine-profile",
        help="Detect Apple Silicon vs Intel Mac and recommended RDE backends",
    )
    machine_p.add_argument("--json", action="store_true", help="Emit JSON for agents/scripts")
    machine_p.add_argument("--plain", action="store_true")
    machine_p.set_defaults(func=cmd_machine_profile)

    repr_rank_p = sub.add_parser(
        "repr-rank",
        help="Rank rde.representation's grammar against a batch (Representation Core, not campaign features)",
    )
    repr_rank_p.add_argument("--n", type=int, default=8, help="Vector length (ignored with --input)")
    repr_rank_p.add_argument("--samples", type=int, default=8, help="Batch size (ignored with --input)")
    repr_rank_p.add_argument(
        "--pattern",
        choices=("random", "periodic", "polynomial"),
        default="random",
        help="Demo batch generator (ignored with --input)",
    )
    repr_rank_p.add_argument("--seed", type=int, default=0)
    repr_rank_p.add_argument(
        "--input", default=None, help="Path to an .npy file holding a (samples, n) batch instead of a demo pattern"
    )
    repr_rank_p.add_argument("--tolerance", type=float, default=1e-6)
    repr_rank_p.add_argument("--output", default=None, help="Write a JSON search report to this path")
    repr_rank_p.set_defaults(func=cmd_repr_rank)

    repr_rank_run_p = sub.add_parser(
        "repr-rank-run",
        help="Rank rde.representation's grammar against an array field already stored by run/campaign",
    )
    _add_store_root(repr_rank_run_p)
    repr_rank_run_p.add_argument("--run-id", required=True)
    repr_rank_run_p.add_argument(
        "--array-key",
        required=True,
        help="Array-valued primitive_features() key to load (e.g. hsp_functions' diff_profile, tsp's D)",
    )
    repr_rank_run_p.add_argument("--tolerance", type=float, default=1e-6)
    repr_rank_run_p.set_defaults(func=cmd_repr_rank_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    import os

    log_progress = (
        args.log_progress
        or os.environ.get("RDE_LOG_PROGRESS", "").lower() in {"1", "true", "yes"}
    )
    set_console(
        RdeConsole(
            plain=resolve_plain(flag=args.plain),
            log_progress=log_progress,
        )
    )
    install_signal_handlers()
    return run_cli_command(args.func, args)


if __name__ == "__main__":
    sys.exit(main())
