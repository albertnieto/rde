"""Post-discovery long-term retention: keep targets, ids, and ranked columns only."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rde.io.json_util import utc_now_iso, write_json
from rde.io.seal import (
    _sha256,
    _verify_parquet,
    is_run_sealed,
    list_campaign_run_ids,
    read_sealed_metadata,
    require_seal_dependencies,
    sealed_dir_for_run,
    sealed_marker_path,
    sealed_parquet_path,
)

TOPK_SCHEMA_VERSION = "rde-topk-v1"
TOPK_PARQUET_NAME = "features_topk.parquet"
TOPK_MARKER_NAME = "topk.json"

TOPK_NON_CLAIM = (
    "Columns dropped by top-K retention are not recoverable from this artifact. "
    "Recompute from campaign seeds and the measurement panel to restore the full catalog."
)

IDENTITY_COLUMNS = (
    "run_id",
    "instance_id",
    "domain_id",
    "size",
    "seed",
    "family_index",
    "slice_kind",
    "generator",
)


def topk_parquet_path(store_root: Path | str, run_id: str) -> Path:
    return sealed_dir_for_run(store_root, run_id) / TOPK_PARQUET_NAME


def topk_marker_path(store_root: Path | str, run_id: str) -> Path:
    return sealed_dir_for_run(store_root, run_id) / TOPK_MARKER_NAME


def read_topk_metadata(store_root: Path | str, run_id: str) -> dict[str, Any] | None:
    marker = topk_marker_path(store_root, run_id)
    if not marker.is_file():
        return None
    return json.loads(marker.read_text(encoding="utf-8"))


def is_run_topk_retained(store_root: Path | str, run_id: str) -> bool:
    metadata = read_topk_metadata(store_root, run_id)
    if metadata is None or metadata.get("status") != "retained":
        return False
    parquet = topk_parquet_path(store_root, run_id)
    return parquet.is_file() and parquet.stat().st_size > 0


def effective_feature_parquet_path(store_root: Path | str, run_id: str) -> Path:
    """Prefer the retained top-K shard when present."""
    if is_run_topk_retained(store_root, run_id):
        return topk_parquet_path(store_root, run_id)
    return sealed_parquet_path(store_root, run_id)


def _parquet_column_names(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(path).schema_arrow.names)


def ranked_columns_from_discovery(
    report: dict[str, Any],
    *,
    top_k: int,
    available_columns: set[str] | None = None,
) -> list[str]:
    """Extract ranked feature column names from a discovery report payload."""
    seen: set[str] = set()
    ranked: list[str] = []

    def add(column: str | None) -> None:
        if not column or column in seen:
            return
        if available_columns is not None and column not in available_columns:
            return
        seen.add(column)
        ranked.append(column)

    for hit in report.get("top_correlations") or []:
        add(str(hit.get("column")) if hit.get("column") is not None else None)
    for hit in report.get("descriptor_candidates") or []:
        add(hit.get("column"))
        add(hit.get("descriptor_key"))
        add(hit.get("key"))
    for hit in report.get("metric_candidates") or []:
        variables = hit.get("variables")
        if isinstance(variables, list):
            for variable in variables:
                add(str(variable))
    latent = report.get("latent") or {}
    autoencoder = latent.get("autoencoder") or {}
    for column in autoencoder.get("latent_columns") or []:
        add(str(column))
    instance_q = latent.get("instance_q_autoencoder") or {}
    for column in instance_q.get("latent_columns") or []:
        add(str(column))
    latent_dim = latent.get("latent_dim")
    if latent_dim is not None:
        try:
            for index in range(int(latent_dim)):
                add(f"latent.pca_{index}")
        except (TypeError, ValueError):
            pass
    for conjecture in report.get("promoted_conjectures") or []:
        add(conjecture.get("latent_column"))
        variables = conjecture.get("variables")
        if isinstance(variables, list):
            for variable in variables:
                add(str(variable))
    return ranked[: max(0, int(top_k))]


def retention_column_plan(
    parquet_columns: list[str],
    *,
    target: str,
    ranked_columns: list[str],
    top_k: int,
) -> tuple[list[str], list[str], list[str]]:
    """Return (selected_columns, ranked_kept, ranked_missing)."""
    available = set(parquet_columns)
    identity = [name for name in IDENTITY_COLUMNS if name in available]
    targets = [target] if target in available else []
    ranked_kept: list[str] = []
    ranked_missing: list[str] = []
    seen: set[str] = set(identity) | set(targets)
    for column in ranked_columns:
        if len(ranked_kept) >= max(0, int(top_k)):
            break
        if column in seen:
            continue
        if column not in available:
            ranked_missing.append(column)
            continue
        seen.add(column)
        ranked_kept.append(column)
    selected = [*identity, *targets, *ranked_kept]
    return selected, ranked_kept, ranked_missing


def _load_discovery_report(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"discovery report must be a JSON object: {path}")
    return payload


def _resolve_ranked_columns(
    *,
    store_root: Path,
    run_id: str,
    target: str,
    top_k: int,
    discovery_report_path: Path | str | None,
    ranked_columns: list[str] | None,
) -> tuple[list[str], str]:
    full_path = sealed_parquet_path(store_root, run_id)
    available = set(_parquet_column_names(full_path))
    if ranked_columns is not None:
        return list(ranked_columns)[: max(0, int(top_k))], "explicit"
    if discovery_report_path is not None:
        report = _load_discovery_report(discovery_report_path)
        return (
            ranked_columns_from_discovery(
                report,
                top_k=top_k,
                available_columns=available,
            ),
            "discovery_report",
        )
    from rde.analyze.query import correlate_with_target, load_rows_from_shard

    rows = load_rows_from_shard(full_path)
    hits = correlate_with_target(rows, target, min_abs_r=0.0)
    return [hit["column"] for hit in hits[: max(0, int(top_k))]], "correlation_fallback"


def _project_parquet(
    source: Path,
    destination: Path,
    columns: list[str],
    *,
    expected_rows: int,
    compression: str = "zstd",
) -> tuple[int, str, int]:
    import pyarrow.parquet as pq

    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        table = pq.read_table(source, columns=columns)
        if int(table.num_rows) != int(expected_rows):
            raise RuntimeError(
                f"top-K projection row count mismatch for {source}: "
                f"expected {expected_rows}, got {table.num_rows}"
            )
        pq.write_table(table, temporary, compression=compression)
        parquet_rows = _verify_parquet(temporary, expected_rows)
        parquet_sha256 = _sha256(temporary)
        os.replace(temporary, destination)
        if _sha256(destination) != parquet_sha256:
            raise RuntimeError(
                f"top-K Parquet checksum changed during publication: {destination}"
            )
        return parquet_rows, parquet_sha256, destination.stat().st_size
    finally:
        temporary.unlink(missing_ok=True)


def estimate_topk_retention(
    store_root: Path | str,
    run_id: str,
    *,
    target: str,
    top_k: int = 64,
    discovery_report_path: Path | str | None = None,
    ranked_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Dry-run plan for ``retain_topk_run`` without mutating artifacts."""
    root = Path(store_root)
    if not is_run_sealed(root, run_id):
        raise FileNotFoundError(f"run {run_id!r} is not sealed; seal before top-K retention")
    existing = read_topk_metadata(root, run_id)
    if existing is not None and existing.get("status") == "retained":
        return {
            "run_id": run_id,
            "action": "already_retained",
            "status": "retained",
            **compact_topk_summary(existing),
            "bytes_freed": 0,
            "would_remove_full_shard": False,
        }

    sealed_metadata = read_sealed_metadata(root, run_id) or {}
    expected_rows = int(
        sealed_metadata.get("rows", {}).get(
            "sealed_features",
            sealed_metadata.get("rows", {}).get("features", 0),
        )
    )
    full_path = sealed_parquet_path(root, run_id)
    ranked, ranked_source = _resolve_ranked_columns(
        store_root=root,
        run_id=run_id,
        target=target,
        top_k=top_k,
        discovery_report_path=discovery_report_path,
        ranked_columns=ranked_columns,
    )
    selected, ranked_kept, ranked_missing = retention_column_plan(
        _parquet_column_names(full_path),
        target=target,
        ranked_columns=ranked,
        top_k=top_k,
    )
    full_bytes = full_path.stat().st_size if full_path.is_file() else 0
    return {
        "run_id": run_id,
        "action": "would_retain_topk",
        "status": "unretained",
        "target": target,
        "top_k": int(top_k),
        "ranked_source": ranked_source,
        "expected_rows": expected_rows,
        "full_shard_path": str(full_path.relative_to(root)),
        "full_shard_bytes": full_bytes,
        "topk_shard_path": str(topk_parquet_path(root, run_id).relative_to(root)),
        "retained_columns": selected,
        "ranked_columns_requested": ranked,
        "ranked_columns_kept": ranked_kept,
        "ranked_columns_missing": ranked_missing,
        "would_remove_full_shard": True,
        "bytes_freed": full_bytes,
        "non_claim": TOPK_NON_CLAIM,
    }


def compact_topk_summary(record: dict[str, Any]) -> dict[str, Any]:
    parquet = record.get("parquet") or {}
    return {
        "run_id": record.get("run_id"),
        "status": record.get("status"),
        "retained": record.get("status") == "retained",
        "shard_path": parquet.get("path"),
        "feature_rows": record.get("rows", {}).get("retained_features"),
        "checksum": parquet.get("sha256"),
        "target": record.get("target"),
        "ranked_columns_kept": record.get("ranked_columns_kept"),
        "full_shard_removed": record.get("full_shard_removed", False),
    }


def retain_topk_run(
    store_root: Path | str,
    run_id: str,
    *,
    target: str,
    top_k: int = 64,
    discovery_report_path: Path | str | None = None,
    ranked_columns: list[str] | None = None,
    delete_full_shard: bool = True,
) -> dict[str, Any]:
    """Write a verified top-K Parquet shard and optionally drop the full sealed shard."""
    root = Path(store_root)
    if not is_run_sealed(root, run_id):
        raise FileNotFoundError(f"run {run_id!r} is not sealed; seal before top-K retention")

    require_seal_dependencies()

    existing = read_topk_metadata(root, run_id)
    if existing is not None and existing.get("status") == "retained":
        parquet_path = topk_parquet_path(root, run_id)
        expected_rows = int(existing.get("rows", {}).get("retained_features", 0))
        if parquet_path.is_file() and expected_rows > 0:
            _verify_parquet(parquet_path, expected_rows)
            if existing.get("parquet", {}).get("sha256") == _sha256(parquet_path):
                full_path = sealed_parquet_path(root, run_id)
                full_removed = not full_path.exists()
                if delete_full_shard and full_path.exists():
                    full_removed = False
                elif not delete_full_shard and full_path.exists():
                    full_removed = False
                else:
                    full_removed = bool(existing.get("full_shard_removed"))
                if full_removed == bool(existing.get("full_shard_removed")):
                    return existing

    sealed_metadata = read_sealed_metadata(root, run_id) or {}
    expected_rows = int(
        sealed_metadata.get("rows", {}).get(
            "sealed_features",
            sealed_metadata.get("rows", {}).get("features", 0),
        )
    )
    if expected_rows <= 0:
        raise RuntimeError(f"sealed run {run_id!r} has no feature row count")

    full_path = sealed_parquet_path(root, run_id)
    if not full_path.is_file() or full_path.stat().st_size == 0:
        if is_run_topk_retained(root, run_id):
            return read_topk_metadata(root, run_id) or {}
        raise FileNotFoundError(f"sealed Parquet shard missing for run {run_id!r}")

    ranked, ranked_source = _resolve_ranked_columns(
        store_root=root,
        run_id=run_id,
        target=target,
        top_k=top_k,
        discovery_report_path=discovery_report_path,
        ranked_columns=ranked_columns,
    )
    selected, ranked_kept, ranked_missing = retention_column_plan(
        _parquet_column_names(full_path),
        target=target,
        ranked_columns=ranked,
        top_k=top_k,
    )
    if len(selected) <= len([name for name in IDENTITY_COLUMNS if name in selected]) + (
        1 if target in selected else 0
    ):
        raise RuntimeError(
            f"top-K retention selected no ranked columns for run {run_id!r}; "
            f"provide --discovery-report or --ranked-columns"
        )

    sealed_dir = sealed_dir_for_run(root, run_id)
    sealed_dir.mkdir(parents=True, exist_ok=True)
    topk_path = topk_parquet_path(root, run_id)
    full_bytes = full_path.stat().st_size
    full_sha256 = _sha256(full_path)

    metadata = {
        "schema_version": TOPK_SCHEMA_VERSION,
        "status": "pending_cleanup",
        "run_id": run_id,
        "target": target,
        "top_k": int(top_k),
        "ranked_source": ranked_source,
        "ranked_columns_requested": ranked,
        "ranked_columns_kept": ranked_kept,
        "ranked_columns_missing": ranked_missing,
        "retained_columns": selected,
        "rows": {"retained_features": expected_rows},
        "source_full_shard": {
            "path": str(full_path.relative_to(root)),
            "bytes": full_bytes,
            "sha256": full_sha256,
        },
        "non_claim": TOPK_NON_CLAIM,
        "delete_full_shard": delete_full_shard,
    }
    marker_path = topk_marker_path(root, run_id)
    write_json(marker_path, metadata)

    try:
        parquet_rows, parquet_sha256, parquet_bytes = _project_parquet(
            full_path,
            topk_path,
            selected,
            expected_rows=expected_rows,
        )
    except Exception:
        marker_path.unlink(missing_ok=True)
        topk_path.unlink(missing_ok=True)
        raise

    metadata["rows"]["retained_features"] = parquet_rows
    metadata["parquet"] = {
        "path": str(topk_path.relative_to(root)),
        "bytes": parquet_bytes,
        "sha256": parquet_sha256,
        "compression": "zstd",
        "column_count": len(selected),
    }
    write_json(marker_path, metadata)

    full_shard_removed = False
    full_shard_bytes_removed = 0
    if delete_full_shard:
        if not full_path.is_file():
            raise RuntimeError(f"full sealed shard disappeared before cleanup: {full_path}")
        full_shard_bytes_removed = full_path.stat().st_size
        full_path.unlink()
        full_shard_removed = True

    metadata["status"] = "retained"
    metadata["retained_at"] = utc_now_iso()
    metadata["full_shard_removed"] = full_shard_removed
    metadata["full_shard_bytes_removed"] = full_shard_bytes_removed
    metadata["bytes_freed"] = full_shard_bytes_removed
    write_json(marker_path, metadata)

    sealed_marker = read_sealed_metadata(root, run_id) or {}
    sealed_marker["topk"] = {
        "schema_version": TOPK_SCHEMA_VERSION,
        "status": "retained",
        "retained_at": metadata["retained_at"],
        "target": target,
        "shard_path": metadata["parquet"]["path"],
        "ranked_columns_kept": ranked_kept,
        "full_shard_removed": full_shard_removed,
        "non_claim": TOPK_NON_CLAIM,
    }
    write_json(sealed_marker_path(root, run_id), sealed_marker)
    return metadata


def retain_topk_campaign(
    store_root: Path | str,
    campaign_id: str,
    *,
    target: str | None = None,
    top_k: int = 64,
    discovery_report_dir: Path | str | None = None,
    delete_full_shard: bool = True,
) -> list[dict[str, Any]]:
    """Retain top-K shards for every sealed run in a campaign."""
    root = Path(store_root)
    records: list[dict[str, Any]] = []
    for run_id in list_campaign_run_ids(root, campaign_id):
        if not is_run_sealed(root, run_id):
            continue
        resolved_target = target
        if resolved_target is None:
            from rde.runtime.targets import resolve_target_for_run

            resolved_target = resolve_target_for_run(run_id, root)
        report_path = None
        if discovery_report_dir is not None:
            candidate = Path(discovery_report_dir) / f"{run_id}.json"
            if candidate.is_file():
                report_path = candidate
        records.append(
            retain_topk_run(
                root,
                run_id,
                target=resolved_target,
                top_k=top_k,
                discovery_report_path=report_path,
                delete_full_shard=delete_full_shard,
            )
        )
    return records
