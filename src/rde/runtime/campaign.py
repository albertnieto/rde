"""Multi-size batch campaign runner with resume support."""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rde.backends.resolve import default_compute_backend
from rde.io.json_util import utc_now_iso
from rde.runtime.pipeline import (
    RunConfig,
    RunResult,
    StorageBudgetExceeded,
    run_pipeline,
)
from rde.core.registry import Registry
from rde.io.seal import (
    SEAL_SCHEMA_VERSION,
    compact_seal_summary,
    is_run_generation_complete,
    read_sealed_metadata,
    require_seal_dependencies,
    seal_run,
)
from rde.io.store import Store


def resolve_n_per_size(size: int, n_per_size: int | dict[int, int], *, default: int = 100) -> int:
    if isinstance(n_per_size, dict):
        return int(n_per_size.get(size, default))
    return int(n_per_size)


@dataclass
class CampaignConfig:
    """Run the same domain across multiple sizes and instance counts."""

    domain_id: str
    sizes: list[int]
    n_per_size: int | dict[int, int] = 100
    seed_base: int = 0
    indices: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    store_root: Path | str = "rde_runs"
    campaign_id: str | None = None
    resume: bool = True
    save_arrays: bool = True
    workers: int = 1
    compute_backend: str = field(default_factory=default_compute_backend)
    compute_batch_size: int | None = None
    generator_id: str | None = None
    coin_shift_grammar: bool = False
    descriptor_names: list[str] | None = None
    metric_names: list[str] | None = None
    instance_descriptor_modules: list[str] | None = None
    enable_cross_slice: bool = True
    extra: dict[str, Any] = field(default_factory=dict)
    plain_output: bool = False
    strict: bool = False
    max_wall_seconds: float | None = None
    max_storage_bytes: int | None = None
    seal_batches: bool = False
    seal_keep_arrays: bool = False


def _resolve_generator_id(config: CampaignConfig) -> str | None:
    return config.generator_id


@dataclass
class CampaignResult:
    campaign_id: str
    run_results: list[RunResult]
    store_root: Path
    timings: list[dict[str, Any]] = field(default_factory=list)
    batch_summary: dict[str, Any] = field(default_factory=dict)


def _append_batch_record(path: Path, record: dict[str, Any]) -> None:
    """Append and fsync one durable session checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _next_batch_index(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _normalize_size_map(value: Any) -> dict[int, int]:
    if not isinstance(value, dict):
        raise TypeError(f"expected size map dict, got {type(value)!r}")
    return {int(key): int(item) for key, item in value.items()}


def _campaign_contract_values_equal(key: str, expected: Any, existing: Any) -> bool:
    if key in {"n_per_size", "n_per_size_map"}:
        if isinstance(expected, dict) and isinstance(existing, dict):
            return _normalize_size_map(expected) == _normalize_size_map(existing)
    return existing == expected


def run_campaign(
    config: CampaignConfig,
    registry: Registry | None = None,
) -> CampaignResult:
    if config.max_wall_seconds is not None and (
        not math.isfinite(float(config.max_wall_seconds))
        or float(config.max_wall_seconds) < 0
    ):
        raise ValueError("max_wall_seconds must be a finite non-negative number")
    if config.max_storage_bytes is not None:
        storage_value = float(config.max_storage_bytes)
        if (
            not math.isfinite(storage_value)
            or storage_value < 0
            or storage_value != math.floor(storage_value)
        ):
            raise ValueError("max_storage_bytes must be a finite non-negative integer")
    campaign_start = time.monotonic()
    campaign_id = config.campaign_id or uuid.uuid4().hex[:12]
    generator_id = _resolve_generator_id(config)
    store_root = Path(config.store_root)
    campaign_dir = store_root / "campaigns" / campaign_id
    campaign_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = campaign_dir / "manifest.json"
    n_map = config.n_per_size if isinstance(config.n_per_size, dict) else None
    campaign_contract = {
        "campaign_id": campaign_id,
        "domain_id": config.domain_id,
        "sizes": list(config.sizes),
        "n_per_size": config.n_per_size,
        "n_per_size_map": n_map,
        "seed_base": config.seed_base,
        "indices": list(config.indices),
        "generator_id": generator_id,
        "save_arrays": config.save_arrays,
        "descriptor_names": config.descriptor_names,
        "metric_names": config.metric_names,
        "instance_descriptor_modules": config.instance_descriptor_modules,
        "enable_cross_slice": config.enable_cross_slice,
        "workers": config.workers,
        "compute_backend": config.compute_backend,
        "compute_batch_size": config.compute_batch_size,
        "strict": config.strict,
        "max_wall_seconds": config.max_wall_seconds,
        "max_storage_bytes": config.max_storage_bytes,
        "seal_batches": config.seal_batches,
        "seal_keep_arrays": config.seal_keep_arrays,
    }
    if config.seal_batches:
        campaign_contract["storage_mode"] = "compact"
        campaign_contract["seal_schema_version"] = SEAL_SCHEMA_VERSION
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(campaign_contract, indent=2), encoding="utf-8")
    else:
        existing_contract = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = {
            key: (expected, existing_contract.get(key))
            for key, expected in campaign_contract.items()
            if key
            not in {
                "max_storage_bytes",
                "seal_batches",
                "seal_keep_arrays",
                "storage_mode",
                "seal_schema_version",
            }
            and not _campaign_contract_values_equal(
                key,
                expected,
                existing_contract.get(
                    key,
                    expected if key in {
                        "strict",
                        "max_wall_seconds",
                        "max_storage_bytes",
                        "seal_batches",
                        "seal_keep_arrays",
                    } else None,
                ),
            )
        }
        if mismatches:
            raise ValueError(f"campaign resume configuration mismatch: {mismatches}")

    timings_path = campaign_dir / "timings.json"
    timings: list[dict[str, Any]] = []
    if timings_path.exists():
        try:
            timings = json.loads(timings_path.read_text(encoding="utf-8")).get("sizes", [])
        except (json.JSONDecodeError, OSError):
            timings = []

    def _write_timings(*, campaign_duration_s: float | None = None) -> None:
        timings_path.write_text(
            json.dumps(
                {
                    "campaign_id": campaign_id,
                    "sizes": timings,
                    "campaign_duration_s": campaign_duration_s,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    results: list[RunResult] = []
    store = Store(store_root)
    total = len(config.sizes)
    batch_started_at = utc_now_iso()
    session_completed_instances = 0
    session_feature_rows = 0
    session_instance_feature_rows = 0
    batch_stop_reason = "complete"
    touched_run_ids: list[str] = []

    if registry is None:
        from rde.core.plugins import build_registry

        registry = build_registry(
            config.domain_id,
            compute_backend=config.compute_backend,
            loader_kwargs={"coin_shift_grammar": config.coin_shift_grammar},
        )

    if config.plain_output:
        from rde.io.progress_ui import PlainCampaignProgress

        ui = PlainCampaignProgress(campaign_id, total)
    else:
        from rde.io.console import get_console
        from rde.io.progress_ui import ConsoleCampaignProgress

        ui = ConsoleCampaignProgress(get_console(), campaign_id, total)

    def check_deadline() -> None:
        if (
            config.max_wall_seconds is not None
            and time.monotonic() - campaign_start >= config.max_wall_seconds
        ):
            raise TimeoutError(
                f"campaign exceeded max_wall_seconds={config.max_wall_seconds}"
            )

    for i, size in enumerate(config.sizes, start=1):
        t0 = time.monotonic()
        size_started_at = utc_now_iso()
        run_id = f"{campaign_id}_n{size}"
        try:
            check_deadline()
        except TimeoutError:
            if config.max_storage_bytes is None:
                raise
            batch_stop_reason = "wall"
            touched_run_ids.append(run_id)
            elapsed = time.monotonic() - t0
            timings.append(
                {
                    "size": size,
                    "run_id": run_id,
                    "started_at": size_started_at,
                    "elapsed_s": elapsed,
                    "feature_rows": 0,
                    "wall_budget_exceeded": True,
                }
            )
            _write_timings(campaign_duration_s=time.monotonic() - campaign_start)
            break
        progress = ui.begin_size(size, i, run_id)
        expected_instances = resolve_n_per_size(size, config.n_per_size)
        if config.resume and is_run_generation_complete(
            store_root,
            run_id,
            expected_instances=expected_instances,
            expected_slices=len(config.indices),
        ):
            metadata = read_sealed_metadata(store_root, run_id) or {}
            rows = metadata.get("rows") or {}
            feature_rows = int(rows.get("sealed_features", rows.get("features", 0)))
            if feature_rows <= 0:
                feature_rows = expected_instances * len(config.indices)
            elapsed = time.monotonic() - t0
            avg = (time.monotonic() - campaign_start) / i
            eta = avg * (total - i)
            timings.append(
                {
                    "size": size,
                    "run_id": run_id,
                    "started_at": size_started_at,
                    "elapsed_s": elapsed,
                    "feature_rows": feature_rows,
                    "skipped_complete": True,
                }
            )
            _write_timings(campaign_duration_s=time.monotonic() - campaign_start)
            ui.end_size(feature_rows=feature_rows, elapsed_s=elapsed, eta_s=eta)
            manifest = store.read_manifest(run_id)
            instance_feature_rows = int(
                rows.get("instance_features", rows.get("instances", expected_instances))
            )
            results.append(
                RunResult(
                    run_id=run_id,
                    manifest=manifest,
                    n_feature_rows=feature_rows,
                    n_instance_feature_rows=instance_feature_rows,
                    n_skipped_instances=expected_instances,
                    n_skipped_slices=expected_instances * len(config.indices),
                    instance_ids=[],
                    duration_s=elapsed,
                    n_completed_instances=0,
                    n_new_feature_rows=0,
                    n_new_instance_feature_rows=0,
                )
            )
            continue
        from rde.backends.resolve import resolve_compute_backend
        from rde.core.plugins import build_registry as _build_registry

        size_backend = resolve_compute_backend(
            config.compute_backend,
            size=size,
            batch_size=config.compute_batch_size or 1,
        )
        size_registry = _build_registry(
            config.domain_id,
            compute_backend=size_backend,
            loader_kwargs={"coin_shift_grammar": config.coin_shift_grammar},
        )
        from rde.runtime.resilience import soft_call

        def run_size() -> RunResult:
            check_deadline()
            remaining_wall = (
                None
                if config.max_wall_seconds is None
                else config.max_wall_seconds - (time.monotonic() - campaign_start)
            )
            if remaining_wall is not None and remaining_wall <= 0:
                raise TimeoutError(
                    f"campaign exceeded max_wall_seconds={config.max_wall_seconds}"
                )
            try:
                return run_pipeline(
                    RunConfig(
                        domain_id=config.domain_id,
                        n_instances=resolve_n_per_size(size, config.n_per_size),
                        size=size,
                        seed=config.seed_base + size * 1000,
                        indices=list(config.indices),
                        store_root=store_root,
                        run_id=run_id,
                        save_arrays=config.save_arrays,
                        resume=config.resume,
                        workers=config.workers,
                        compute_backend=config.compute_backend,
                        compute_batch_size=config.compute_batch_size,
                        generator_id=generator_id,
                        coin_shift_grammar=config.coin_shift_grammar,
                        descriptor_names=config.descriptor_names,
                        metric_names=config.metric_names,
                        instance_descriptor_modules=config.instance_descriptor_modules,
                        enable_cross_slice=config.enable_cross_slice,
                        extra={**config.extra, "campaign_id": campaign_id},
                        max_wall_seconds=remaining_wall,
                        max_storage_bytes=config.max_storage_bytes,
                    ),
                    registry=size_registry,
                    progress=progress,
                    store=store,
                )
            except Exception as exc:
                store.flush(run_id)
                if not isinstance(exc, StorageBudgetExceeded) or config.max_storage_bytes is None:
                    store.close()
                raise

        # A deadline is a campaign contract, not an optional soft-failure.
        try:
            result = (
                run_size()
                if (
                    config.strict
                    or config.max_wall_seconds is not None
                    or config.max_storage_bytes is not None
                )
                else soft_call(
                    f"campaign_size_n{size}",
                    run_size,
                    fallback=None,
                )
            )
        except StorageBudgetExceeded as exc:
            batch_stop_reason = "storage_budget"
            session_completed_instances += exc.completed_jobs
            session_feature_rows += exc.new_feature_rows
            session_instance_feature_rows += exc.new_instance_feature_rows
            touched_run_ids.append(run_id)
            elapsed = time.monotonic() - t0
            timings.append(
                {
                    "size": size,
                    "run_id": run_id,
                    "started_at": size_started_at,
                    "elapsed_s": elapsed,
                    "feature_rows": exc.new_feature_rows,
                    "completed_instances": exc.completed_jobs,
                    "storage_budget_exceeded": True,
                }
            )
            _write_timings(campaign_duration_s=time.monotonic() - campaign_start)
            ui.end_size(feature_rows=exc.new_feature_rows, elapsed_s=elapsed, eta_s=0.0)
            break
        except TimeoutError:
            if config.max_storage_bytes is None:
                raise
            batch_stop_reason = "wall"
            touched_run_ids.append(run_id)
            elapsed = time.monotonic() - t0
            timings.append(
                {
                    "size": size,
                    "run_id": run_id,
                    "started_at": size_started_at,
                    "elapsed_s": elapsed,
                    "feature_rows": 0,
                    "wall_budget_exceeded": True,
                }
            )
            _write_timings(campaign_duration_s=time.monotonic() - campaign_start)
            ui.end_size(feature_rows=0, elapsed_s=elapsed, eta_s=0.0)
            break
        elapsed = time.monotonic() - t0
        avg = (time.monotonic() - campaign_start) / i
        eta = avg * (total - i)
        if result is None:
            timings.append(
                {
                    "size": size,
                    "run_id": run_id,
                    "started_at": size_started_at,
                    "elapsed_s": elapsed,
                    "feature_rows": 0,
                    "soft_failed": True,
                }
            )
            _write_timings(campaign_duration_s=time.monotonic() - campaign_start)
            ui.end_size(feature_rows=0, elapsed_s=elapsed, eta_s=eta)
            if config.strict:
                raise RuntimeError(f"campaign size N={size} produced no result")
            continue
        store.flush(run_id)
        touched_run_ids.append(run_id)
        session_completed_instances += result.n_completed_instances
        session_feature_rows += result.n_new_feature_rows
        session_instance_feature_rows += result.n_new_instance_feature_rows
        timings.append(
            {
                "size": size,
                "run_id": run_id,
                "started_at": size_started_at,
                "elapsed_s": elapsed,
                "feature_rows": result.n_feature_rows,
            }
        )
        _write_timings(campaign_duration_s=time.monotonic() - campaign_start)
        ui.end_size(feature_rows=result.n_feature_rows, elapsed_s=elapsed, eta_s=eta)
        results.append(result)
        if (
            config.max_storage_bytes is not None
            and store.session_bytes_written >= int(config.max_storage_bytes)
        ):
            batch_stop_reason = "storage_budget"
            break

    store.close()
    batch_index = _next_batch_index(campaign_dir / "batches.jsonl")
    seal_records: list[dict[str, Any]] = []
    seal_error: Exception | None = None
    if config.seal_batches:
        require_seal_dependencies()
        runs_to_seal = list(dict.fromkeys(touched_run_ids))
        if batch_stop_reason == "complete":
            for size in config.sizes:
                runs_to_seal.append(f"{campaign_id}_n{size}")
            runs_to_seal = list(dict.fromkeys(runs_to_seal))
        for sealed_run_id in runs_to_seal:
            try:
                seal_records.append(
                    seal_run(
                        store_root,
                        sealed_run_id,
                        campaign_id=campaign_id,
                        batch_index=batch_index,
                        delete_arrays=not config.seal_keep_arrays,
                    )
                )
            except Exception as exc:
                seal_records.append(
                    {
                        "status": "failed",
                        "run_id": sealed_run_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                if seal_error is None:
                    seal_error = exc
    seal_summary = [compact_seal_summary(record) for record in seal_records]
    batch_record = {
        "batch_index": batch_index,
        "campaign_id": campaign_id,
        "started_at": batch_started_at,
        "ended_at": utc_now_iso(),
        "instances_completed_this_session": session_completed_instances,
        "features_completed_this_session": session_feature_rows,
        "instance_features_completed_this_session": session_instance_feature_rows,
        "bytes_written_this_session": store.session_bytes_written,
        "max_storage_bytes": config.max_storage_bytes,
        "stop_reason": batch_stop_reason,
        "run_ids": touched_run_ids,
        "seal_batches": config.seal_batches,
        "seal_keep_arrays": config.seal_keep_arrays,
        "sealed": bool(seal_summary)
        and seal_error is None
        and all(entry.get("sealed") for entry in seal_summary),
        "seal_records": seal_records,
        "seal_summary": seal_summary,
    }
    _append_batch_record(campaign_dir / "batches.jsonl", batch_record)
    if seal_error is not None:
        raise RuntimeError(
            "campaign batch completed, but requested sealing failed; "
            "inspect seal_records for the affected runs"
        ) from seal_error
    return CampaignResult(
        campaign_id=campaign_id,
        run_results=results,
        store_root=store_root,
        timings=timings,
        batch_summary=batch_record,
    )
