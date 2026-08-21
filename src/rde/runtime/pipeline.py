"""Pipeline runner: generate → materialize → descriptors → metrics → persist."""

from __future__ import annotations

import sys
import uuid
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any, Callable, Iterator

from rde.generators import get_generator, list_generators
from rde.core.instance import InstanceRecord
from rde.core.limits import validate_bruteforce_size
from rde.core.registry import Registry, get_registry
from rde.io.store import RunManifest, Store
from rde.backends.resolve import default_compute_backend, recommended_compute_batch_size, resolve_compute_backend
from rde.runtime.worker import process_instance, process_instance_worker
from rde.runtime.progress import (
    InstanceProgress,
    PipelineProgressCallback,
    default_pipeline_progress,
)
from rde.io.shutdown import abort_process_pool
from rde.io.json_util import json_default, utc_now_iso
from rde.runtime.heartbeat import ProgressHeartbeat
from rde.io.events import configure_run_logging, event
from rde.io.provenance import collect_provenance


class StorageBudgetExceeded(RuntimeError):
    """Raised after a durable pipeline checkpoint reaches its byte budget."""

    def __init__(
        self,
        message: str,
        *,
        run_id: str | None = None,
        completed_jobs: int = 0,
        new_feature_rows: int = 0,
        new_instance_feature_rows: int = 0,
        bytes_written: int = 0,
    ) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.completed_jobs = int(completed_jobs)
        self.new_feature_rows = int(new_feature_rows)
        self.new_instance_feature_rows = int(new_instance_feature_rows)
        self.bytes_written = int(bytes_written)


@dataclass
class RunConfig:
    """Configuration for one RDE pipeline run."""

    domain_id: str
    n_instances: int
    size: int
    seed: int = 0
    indices: list[int] = field(default_factory=lambda: [1, 2, 4])
    descriptor_names: list[str] | None = None
    metric_names: list[str] | None = None
    instance_descriptor_modules: list[str] | None = None
    enable_cross_slice: bool = True
    store_root: Path | str = "rde_runs"
    run_id: str | None = None
    save_arrays: bool = True
    resume: bool = False
    max_bruteforce_n: int | None = 14
    workers: int = 1
    compute_backend: str = field(default_factory=default_compute_backend)
    compute_batch_size: int | None = None
    generator_id: str | None = None
    coin_shift_grammar: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
    max_wall_seconds: float | None = None
    max_storage_bytes: int | None = None


@dataclass
class RunResult:
    """Summary returned after a pipeline run."""

    run_id: str
    manifest: RunManifest
    n_feature_rows: int
    n_instance_feature_rows: int
    n_skipped_instances: int
    n_skipped_slices: int
    instance_ids: list[str]
    duration_s: float = 0.0
    n_completed_instances: int = 0
    n_new_feature_rows: int = 0
    n_new_instance_feature_rows: int = 0


@dataclass
class _JobPlan:
    """Bounded execution plan for one generated instance."""

    instance: InstanceRecord
    pending_indices: list[int]
    write_instance: bool
    write_instance_features: bool
    cached_instance_scalars: dict[str, Any] | None


@dataclass
class _PlanningStats:
    """Counters collected while the instance stream is consumed."""

    instance_ids: list[str] = field(default_factory=list)
    collect_instance_ids: bool = True
    skipped_instances: int = 0
    skipped_slices: int = 0


def _config_fingerprint(
    config: RunConfig,
    *,
    descriptor_names: list[str],
    metric_names: list[str],
    batch_size: int,
) -> str:
    """Fingerprint every setting that can change persisted run rows."""
    payload = {
        "domain_id": config.domain_id,
        "n_instances": config.n_instances,
        "size": config.size,
        "seed": config.seed,
        "indices": list(config.indices),
        "descriptor_names": list(descriptor_names),
        "metric_names": list(metric_names),
        "instance_descriptor_modules": config.instance_descriptor_modules,
        "enable_cross_slice": config.enable_cross_slice,
        "save_arrays": config.save_arrays,
        "max_bruteforce_n": config.max_bruteforce_n,
        "workers": config.workers,
        "compute_backend": config.compute_backend,
        "compute_batch_size": batch_size,
        "generator_id": config.generator_id,
        "coin_shift_grammar": config.coin_shift_grammar,
        "extra": config.extra,
        "max_wall_seconds": config.max_wall_seconds,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resume_science_contract_compatible(
    existing: RunManifest,
    config: RunConfig,
    *,
    descriptor_names: list[str],
    metric_names: list[str],
    batch_size: int,
) -> bool:
    """Allow resume when session budgets drift but science rows stay compatible."""
    existing_extra = existing.extra or {}
    return (
        existing.domain_id == config.domain_id
        and existing.n_instances == config.n_instances
        and existing.size == config.size
        and existing.seed == config.seed
        and list(existing.indices) == list(config.indices)
        and list(existing.descriptor_names) == list(descriptor_names)
        and list(existing.metric_names) == list(metric_names)
        and existing_extra.get("generator_id") == config.generator_id
        and existing_extra.get("compute_backend") == config.compute_backend
        and existing_extra.get("workers") == config.workers
        and int(existing_extra.get("compute_batch_size", batch_size)) == int(batch_size)
        and bool(existing_extra.get("save_arrays", config.save_arrays)) == bool(config.save_arrays)
        and existing_extra.get("coin_shift_grammar", False) == config.coin_shift_grammar
        and existing_extra.get("campaign_id") == (config.extra or {}).get("campaign_id")
    )


def _generate_instances(domain, config: RunConfig) -> Iterator[InstanceRecord]:
    if config.generator_id:
        gen = get_generator(config.generator_id)
        if gen.domain_id != config.domain_id:
            raise ValueError(
                f"Generator {config.generator_id!r} is for domain {gen.domain_id!r}, "
                f"not {config.domain_id!r}"
            )
        stream = getattr(gen, "generate_iter", None)
        generated = (
            stream(config.n_instances, config.size, config.seed)
            if callable(stream)
            else gen.generate(config.n_instances, config.size, config.seed)
        )
    else:
        stream = getattr(domain, "generate_iter", None)
        generated = (
            stream(config.n_instances, config.size, config.seed)
            if callable(stream)
            else domain.generate(config.n_instances, config.size, config.seed)
        )
    yield from generated


def _iter_job_plans(
    store: Store,
    run_id: str,
    instances: Iterator[InstanceRecord],
    indices: list[int],
    stats: _PlanningStats,
    check_deadline: Callable[[], None],
    *,
    recorded_instance_ids: set[str] | None = None,
) -> Iterator[_JobPlan]:
    """Convert generated instances to bounded resume-aware work plans."""
    known_instance_ids = (
        set(recorded_instance_ids) if recorded_instance_ids is not None else None
    )
    for instance in instances:
        check_deadline()
        if stats.collect_instance_ids:
            stats.instance_ids.append(instance.instance_id)
        completed_indices = store.completed_feature_indices(
            run_id, instance.instance_id, indices
        )
        pending_indices = [
            index for index in indices if index not in completed_indices
        ]
        completed_instance = store.has_completed_instance(run_id, instance.instance_id)
        if not pending_indices and completed_instance:
            if (
                known_instance_ids is not None
                and instance.instance_id not in known_instance_ids
            ):
                store.append_instance(run_id, instance)
                known_instance_ids.add(instance.instance_id)
            stats.skipped_instances += 1
            stats.skipped_slices += len(indices)
            continue
        stats.skipped_slices += len(indices) - len(pending_indices)
        cached_scalars = None
        if completed_instance:
            cached_scalars = store.instance_scalars(run_id, instance.instance_id)
            if cached_scalars is None and pending_indices:
                cached_scalars = store.instance_feature_scalars(
                    run_id, instance.instance_id
                )
        yield _JobPlan(
            instance=instance,
            pending_indices=pending_indices,
            write_instance=not completed_instance,
            write_instance_features=not completed_instance,
            cached_instance_scalars=cached_scalars,
        )


def _persist_result(store: Store, run_id: str, result) -> None:
    if result.write_instance:
        store.append_instance(run_id, result.instance)
    if result.instance_features_row is not None:
        store.append_instance_features(run_id, result.instance_features_row)
    for _iid, name, array in result.array_payloads:
        store.save_array(run_id, _iid, name, array)
    for row in result.feature_rows:
        store.append_features(run_id, row)
    if result.all_slices_complete:
        store.purge_instance_resume_scalars(run_id, result.instance.instance_id)


def run_pipeline(
    config: RunConfig,
    registry: Registry | None = None,
    *,
    progress: PipelineProgressCallback | None = None,
    store: Store | None = None,
) -> RunResult:
    """Execute a full pipeline run."""
    run_started_at = utc_now_iso()
    t_start = perf_counter()
    if progress is None:
        progress = default_pipeline_progress()
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
    deadline = (
        monotonic() + float(config.max_wall_seconds)
        if config.max_wall_seconds is not None
        else None
    )

    def check_deadline() -> None:
        if deadline is not None and monotonic() >= deadline:
            raise TimeoutError(
                f"pipeline exceeded max_wall_seconds={config.max_wall_seconds}"
            )

    check_deadline()
    batch_size = config.compute_batch_size
    if batch_size is None:
        from rde.backends.resolve import mlx_usable

        prefer = None
        if config.compute_backend in {"auto", None} and mlx_usable():
            prefer = "mlx"
        elif config.compute_backend not in {"auto", None}:
            prefer = config.compute_backend
        batch_size = recommended_compute_batch_size(config.size, prefer=prefer)
    batch_size = max(1, int(batch_size))

    config.compute_backend = resolve_compute_backend(
        config.compute_backend,
        size=config.size,
        batch_size=batch_size,
    )
    from rde.core.plugins import build_registry, registry_loader_kwargs

    loader_kwargs = registry_loader_kwargs(
        config.domain_id,
        compute_backend=config.compute_backend,
        max_bruteforce_n=config.max_bruteforce_n,
        loader_kwargs={"coin_shift_grammar": config.coin_shift_grammar},
    )
    reg = registry
    if reg is None:
        reg = build_registry(config.domain_id, **loader_kwargs)
    else:
        domain = reg.get_domain(config.domain_id)
        backend_name = getattr(domain, "backend_name", None)
        rebuild = backend_name is not None and backend_name != config.compute_backend
        if rebuild:
            reg = build_registry(config.domain_id, **loader_kwargs)
    domain = reg.get_domain(config.domain_id)

    check_deadline()
    if getattr(domain, "bruteforce_enumeration", True):
        validate_bruteforce_size(config.size, config.max_bruteforce_n)

    descriptor_names = (
        list(config.descriptor_names)
        if config.descriptor_names is not None
        else reg.list_descriptors()
    )
    metric_names = (
        list(config.metric_names)
        if config.metric_names is not None
        else reg.list_metrics()
    )
    config_fingerprint = _config_fingerprint(
        config,
        descriptor_names=descriptor_names,
        metric_names=metric_names,
        batch_size=batch_size,
    )

    run_id = config.run_id or uuid.uuid4().hex[:12]
    active_store = store if store is not None else Store(config.store_root)

    existing_manifest: RunManifest | None = None
    if config.resume and active_store.run_dir(run_id).exists():
        existing = active_store.read_manifest(run_id)
        existing_manifest = existing
        if existing.domain_id != config.domain_id:
            raise ValueError(
                f"Resume domain mismatch: manifest {existing.domain_id!r} vs {config.domain_id!r}"
            )
        if existing.size != config.size:
            raise ValueError(f"Resume size mismatch: manifest {existing.size} vs {config.size}")
        if list(existing.indices) != list(config.indices):
            raise ValueError(
                f"Resume indices mismatch: manifest {existing.indices} vs {config.indices}"
            )
        existing_fingerprint = existing.extra.get("config_fingerprint")
        if existing_fingerprint != config_fingerprint and not _resume_science_contract_compatible(
            existing,
            config,
            descriptor_names=descriptor_names,
            metric_names=metric_names,
            batch_size=batch_size,
        ):
            raise ValueError(
                "Resume configuration mismatch: persisted rows were produced by "
                f"{existing_fingerprint!r}, requested {config_fingerprint!r}"
            )
        # Completion is backed by the store's SQLite index. It is rebuilt
        # once from legacy JSONL when necessary and queried per pending key,
        # avoiding a multi-gigabyte in-memory set on million-instance runs.
        active_store.ensure_completion_index(run_id)

    manifest = RunManifest(
        run_id=run_id,
        domain_id=config.domain_id,
        created_at=(
            existing_manifest.created_at
            if existing_manifest is not None
            else utc_now_iso()
        ),
        n_instances=config.n_instances,
        size=config.size,
        seed=config.seed,
        indices=list(config.indices),
        descriptor_names=list(descriptor_names),
        metric_names=list(metric_names),
        extra={
            **config.extra,
            "resume": config.resume,
            "max_bruteforce_n": config.max_bruteforce_n,
            "workers": config.workers,
            "compute_backend": config.compute_backend,
            "compute_batch_size": batch_size,
            "generator_id": config.generator_id,
            "coin_shift_grammar": config.coin_shift_grammar,
            "max_wall_seconds": config.max_wall_seconds,
            "max_storage_bytes": config.max_storage_bytes,
            "config_fingerprint": config_fingerprint,
            "feature_row_layout": "slice-only-v1",
            "save_arrays": config.save_arrays,
        },
        provenance=(
            (existing_manifest.provenance or collect_provenance())
            if existing_manifest is not None
            else collect_provenance()
        ),
    )
    if not config.resume or not active_store.run_dir(run_id).exists():
        if not config.resume and active_store.run_dir(run_id).exists():
            from rde.io.seal import is_run_sealed

            if is_run_sealed(config.store_root, run_id):
                raise ValueError(
                    f"run {run_id!r} is sealed; refusing resume=False because it would "
                    "regenerate completed science rows. Use resume=True or a new run_id."
                )
            active_store.reset_run(run_id)
        active_store.write_manifest(manifest)
    # Configure after a fresh-run reset so the handler is never left pointing
    # at an unlinked events.jsonl file.
    logger = configure_run_logging(config.store_root, run_id)
    event(logger, "pipeline_started", run_id=run_id, backend=config.compute_backend)

    check_deadline()
    instances = _generate_instances(domain, config)
    recorded_instance_ids = active_store.recorded_instance_ids(run_id)
    # Returning every ID is convenient for small interactive runs but defeats
    # the streaming contract for million-instance campaigns. Keep the
    # compatibility field bounded and persist the authoritative completion
    # index/manifest for large runs.
    planning_stats = _PlanningStats(collect_instance_ids=config.n_instances <= 100_000)
    job_plans = _iter_job_plans(
        active_store,
        run_id,
        instances,
        list(config.indices),
        planning_stats,
        check_deadline,
        recorded_instance_ids=recorded_instance_ids,
    )
    workers = max(1, int(config.workers))
    # The requested instance count is known without materializing a job list.
    total_jobs = max(0, int(config.n_instances))
    if progress is not None:
        progress.on_run_start(
            run_id=run_id,
            total_jobs=total_jobs,
            size=config.size,
            backend=str(config.compute_backend),
        )

    from rde.io.store import count_jsonl_lines

    feat_path = active_store.run_dir(run_id) / "features.jsonl"
    running_feature_rows = count_jsonl_lines(feat_path) if feat_path.exists() else 0
    instance_feat_path = active_store.run_dir(run_id) / "instance_features.jsonl"
    running_instance_feature_rows = (
        count_jsonl_lines(instance_feat_path) if instance_feat_path.exists() else 0
    )
    completed_jobs = 0
    new_feature_rows = 0
    new_instance_feature_rows = 0
    heartbeat: ProgressHeartbeat | None = None
    if progress is not None and total_jobs > 0:
        heartbeat = ProgressHeartbeat(progress, total_jobs=total_jobs, interval_s=5.0)
        heartbeat.start()

    def check_storage_budget() -> None:
        if config.max_storage_bytes is None:
            return
        if active_store.storage_bytes_including_pending < int(config.max_storage_bytes):
            return
        active_store.flush(run_id)
        if active_store.session_bytes_written < int(config.max_storage_bytes):
            return
        raise StorageBudgetExceeded(
            "pipeline reached max_storage_bytes="
            f"{config.max_storage_bytes} after a durable checkpoint",
            run_id=run_id,
            completed_jobs=completed_jobs,
            new_feature_rows=new_feature_rows,
            new_instance_feature_rows=new_instance_feature_rows,
            bytes_written=active_store.session_bytes_written,
        )

    try:
        check_storage_budget()
        if workers == 1:
            use_batch = batch_size > 1 and hasattr(domain, "prepare_instances_batch")
            indices = list(config.indices)
            while True:
                check_deadline()
                try:
                    first_plan = next(job_plans)
                except StopIteration:
                    break
                chunk = [first_plan]
                if use_batch:
                    for _ in range(batch_size - 1):
                        check_deadline()
                        try:
                            chunk.append(next(job_plans))
                        except StopIteration:
                            break
                caches: list[dict[str, Any] | None]
                if use_batch and len(chunk) > 1:
                    caches = domain.prepare_instances_batch(
                        [plan.instance for plan in chunk], indices
                    )
                else:
                    caches = [None] * len(chunk)
                for plan, prebuilt_cache in zip(chunk, caches):
                    check_deadline()
                    result = process_instance(
                        domain,
                        reg,
                        plan.instance,
                        indices=indices,
                        pending_indices=plan.pending_indices,
                        descriptor_names=descriptor_names,
                        metric_names=metric_names,
                        run_id=run_id,
                        write_instance=plan.write_instance,
                        write_instance_features=plan.write_instance_features,
                        save_arrays=config.save_arrays,
                        cached_instance_scalars=plan.cached_instance_scalars,
                        prebuilt_cache=prebuilt_cache,
                        instance_descriptor_modules=config.instance_descriptor_modules,
                        enable_cross_slice=config.enable_cross_slice,
                    )
                    check_deadline()
                    _persist_result(active_store, run_id, result)
                    completed_jobs += 1
                    if heartbeat:
                        heartbeat.mark_done()
                    running_feature_rows += len(result.feature_rows)
                    new_feature_rows += len(result.feature_rows)
                    if result.instance_features_row is not None:
                        running_instance_feature_rows += 1
                        new_instance_feature_rows += 1
                    if progress is not None:
                        progress.on_instance_done(
                            InstanceProgress(
                                run_id=run_id,
                                completed=completed_jobs,
                                total=total_jobs,
                                feature_rows=running_feature_rows,
                                instance_id=plan.instance.instance_id,
                                pending_slices=len(plan.pending_indices),
                            )
                        )
                    check_storage_budget()
        else:
            # Hybrid: parent owns GPU batching; workers receive prebuilt caches.
            use_batch = batch_size > 1 and hasattr(domain, "prepare_instances_batch")
            indices = list(config.indices)

            def _iter_worker_payloads() -> Iterator[dict[str, Any]]:
                while True:
                    check_deadline()
                    try:
                        first_plan = next(job_plans)
                    except StopIteration:
                        return
                    chunk = [first_plan]
                    if use_batch:
                        for _ in range(batch_size - 1):
                            check_deadline()
                            try:
                                chunk.append(next(job_plans))
                            except StopIteration:
                                break
                    if use_batch and len(chunk) > 1:
                        caches = domain.prepare_instances_batch(
                            [plan.instance for plan in chunk], indices
                        )
                    else:
                        caches = [None] * len(chunk)
                    for plan, prebuilt_cache in zip(chunk, caches):
                        yield {
                            "domain_id": config.domain_id,
                            "compute_backend": config.compute_backend,
                            "coin_shift_grammar": config.coin_shift_grammar,
                            "generator_id": config.generator_id,
                            "max_bruteforce_n": config.max_bruteforce_n,
                            "instance": plan.instance.to_dict(),
                            "indices": indices,
                            "pending_indices": plan.pending_indices,
                            "descriptor_names": descriptor_names,
                            "metric_names": metric_names,
                            "run_id": run_id,
                            "write_instance": plan.write_instance,
                            "write_instance_features": plan.write_instance_features,
                            "save_arrays": config.save_arrays,
                            "cached_instance_scalars": plan.cached_instance_scalars,
                            "prebuilt_cache": prebuilt_cache,
                            "instance_descriptor_modules": config.instance_descriptor_modules,
                            "enable_cross_slice": config.enable_cross_slice,
                        }

            from rde.runtime.config import ResourceLimits

            limits = ResourceLimits.from_env()
            pool_workers = min(workers, limits.max_workers)
            max_inflight = max(
                1,
                min(pool_workers * 2, limits.max_inflight_descriptor_jobs),
            )
            max_buffered_results = max_inflight
            pool = ProcessPoolExecutor(max_workers=max(1, pool_workers))
            payload_iter = enumerate(_iter_worker_payloads())
            in_flight: dict[Any, tuple[int, dict[str, Any]]] = {}
            results_buffer: dict[int, Any] = {}
            next_emit = 0
            pool_aborted = False

            def _submit_next() -> bool:
                check_deadline()
                try:
                    seq, payload = next(payload_iter)
                except StopIteration:
                    return False
                future = pool.submit(process_instance_worker, payload)
                in_flight[future] = (seq, payload)
                return True

            try:
                while len(in_flight) < max_inflight and _submit_next():
                    pass
                while in_flight:
                    remaining = (
                        None if deadline is None else deadline - monotonic()
                    )
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError(
                            f"pipeline exceeded max_wall_seconds={config.max_wall_seconds}"
                        )
                    done, _ = wait(
                        in_flight,
                        return_when=FIRST_COMPLETED,
                        timeout=remaining,
                    )
                    if not done:
                        raise TimeoutError(
                            f"pipeline exceeded max_wall_seconds={config.max_wall_seconds}"
                        )
                    for fut in done:
                        check_deadline()
                        seq, payload = in_flight.pop(fut)
                        try:
                            results_buffer[seq] = fut.result()
                        except Exception:
                            event(
                                logger,
                                "worker_failed",
                                run_id=run_id,
                                instance_id=payload["instance"]["instance_id"],
                                exc_info=True,
                            )
                            raise
                    while next_emit in results_buffer:
                        result = results_buffer.pop(next_emit)
                        check_deadline()
                        _persist_result(active_store, run_id, result)
                        completed_jobs += 1
                        if heartbeat:
                            heartbeat.mark_done()
                        running_feature_rows += len(result.feature_rows)
                        new_feature_rows += len(result.feature_rows)
                        if result.instance_features_row is not None:
                            running_instance_feature_rows += 1
                            new_instance_feature_rows += 1
                        if progress is not None:
                            progress.on_instance_done(
                                InstanceProgress(
                                    run_id=run_id,
                                    completed=completed_jobs,
                                    total=total_jobs,
                                    feature_rows=running_feature_rows,
                                    instance_id=result.instance.instance_id,
                                    pending_slices=result.pending_count,
                                )
                            )
                        check_storage_budget()
                        next_emit += 1
                    while (
                        len(in_flight) < max_inflight
                        and len(results_buffer) < max_buffered_results
                        and _submit_next()
                    ):
                        pass
            except BaseException:
                abort_process_pool(pool, list(in_flight.keys()))
                pool_aborted = True
                raise
            finally:
                if not pool_aborted:
                    pool.shutdown(wait=True)
    except StorageBudgetExceeded as exc:
        event(
            logger,
            "pipeline_batch_stopped",
            run_id=run_id,
            completed_jobs=exc.completed_jobs,
            new_feature_rows=exc.new_feature_rows,
            bytes_written=exc.bytes_written,
        )
        raise
    except Exception:
        event(logger, "pipeline_failed", run_id=run_id, exc_info=True)
        raise
    finally:
        if heartbeat:
            heartbeat.stop()
        active_store.flush(run_id)
        if sys.exc_info()[0] is not None and progress is not None and hasattr(progress, "abort"):
            progress.abort()

    # Avoid a full JSON re-parse of both files just to count rows: feature
    # rows are already tracked incrementally above, and instance rows only
    # need a line count, not deserialization.
    n_inst_rows = count_jsonl_lines(
        active_store.run_dir(run_id) / "instance_features.jsonl"
    )
    n_feat_rows = running_feature_rows
    run_completed_at = utc_now_iso()
    duration_s = perf_counter() - t_start
    manifest.extra.update(
        {
            "completed_jobs": completed_jobs,
            "skipped_instances": planning_stats.skipped_instances,
            "skipped_slices": planning_stats.skipped_slices,
            "persisted_feature_rows": n_feat_rows,
            "persisted_instance_feature_rows": n_inst_rows,
            "started_at": run_started_at,
            "completed_at": run_completed_at,
            "duration_s": duration_s,
        }
    )
    active_store.write_manifest(manifest)
    event(
        logger,
        "pipeline_completed",
        run_id=run_id,
        completed_jobs=completed_jobs,
        skipped_instances=planning_stats.skipped_instances,
        skipped_slices=planning_stats.skipped_slices,
        feature_rows=n_feat_rows,
        instance_feature_rows=n_inst_rows,
        duration_s=duration_s,
    )

    if progress is not None:
        progress.on_run_complete(
            run_id=run_id,
            feature_rows=n_feat_rows,
            skipped_instances=planning_stats.skipped_instances,
        )

    return RunResult(
        run_id=run_id,
        manifest=manifest,
        n_feature_rows=n_feat_rows,
        n_instance_feature_rows=n_inst_rows,
        n_skipped_instances=planning_stats.skipped_instances,
        n_skipped_slices=planning_stats.skipped_slices,
        instance_ids=planning_stats.instance_ids,
        duration_s=duration_s,
        n_completed_instances=completed_jobs,
        n_new_feature_rows=new_feature_rows,
        n_new_instance_feature_rows=new_instance_feature_rows,
    )
