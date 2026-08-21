"""Verified compaction and cleanup of completed RDE run artifacts."""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from rde.io.json_util import utc_now_iso, write_json
from rde.io.store import Store


SEAL_SCHEMA_VERSION = "rde-sealed-v1"

SEALED_MARKER_NAME = "sealed.json"
SEALED_PARQUET_NAME = "features.parquet"

_COMPLETION_DB_NAMES = (
    "completion.sqlite3",
    "completion.sqlite3-wal",
    "completion.sqlite3-shm",
    "completion.sqlite3-journal",
)
_FEATURE_JSONL_NAMES = ("features.jsonl", "instance_features.jsonl")


def effective_instance_count(stats: dict[str, int | str]) -> int:
    """Best-effort instance count when ``instances.jsonl`` lags ``instance_features.jsonl``.

    Resume can skip fully indexed instances without re-appending the instance
    metadata row; ``instance_features.jsonl`` is the durable per-instance anchor.
    """
    instances = int(stats.get("instances_lines", 0))
    instance_features = int(stats.get("instance_features_lines", 0))
    return max(instances, instance_features)


def require_seal_dependencies() -> None:
    """Fail fast with an actionable message when Parquet export is unavailable."""
    try:
        import pyarrow.parquet  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "RDE sealing requires pyarrow. Install with: "
            "pip install pyarrow  (or pip install -e '.[rde-parquet]')"
        ) from exc


def list_campaign_run_ids(
    store_root: Path | str,
    campaign_id: str,
    *,
    store: Store | None = None,
) -> list[str]:
    """Return run IDs whose manifest belongs to ``campaign_id``."""
    active_store = store or Store(store_root)
    owns_store = store is None
    run_ids: list[str] = []
    try:
        for run_id in active_store.list_runs(active_only=False):
            try:
                manifest = active_store.read_manifest(run_id)
            except (FileNotFoundError, OSError, ValueError, TypeError):
                continue
            if manifest.extra.get("campaign_id") == campaign_id:
                run_ids.append(run_id)
    finally:
        if owns_store:
            active_store.close()
    return sorted(run_ids)


def compact_seal_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize one seal record for ``batches.jsonl`` checkpoints."""
    parquet = record.get("parquet") or {}
    rows = record.get("rows") or {}
    return {
        "run_id": record.get("run_id"),
        "status": record.get("status"),
        "sealed": record.get("status") == "sealed",
        "shard_path": parquet.get("path"),
        "feature_rows": rows.get("sealed_features", rows.get("features")),
        "checksum": parquet.get("sha256"),
        "bytes_freed": (record.get("cleanup") or {}).get("bytes_freed", 0),
    }


def _run_completion_bytes(run_dir: Path) -> int:
    return sum(_path_bytes(run_dir / name) for name in _COMPLETION_DB_NAMES)


def _run_jsonl_bytes(run_dir: Path) -> int:
    return sum(_path_bytes(run_dir / name) for name in _FEATURE_JSONL_NAMES)


def _run_row_counts(
    store: Store,
    run_id: str,
    *,
    run_manifest,
) -> tuple[dict[str, int], dict[str, int], bool]:
    stats = store.run_stats(run_id)
    expected_rows = {
        "instances": int(run_manifest.n_instances),
        "instance_features": int(run_manifest.n_instances),
        "features": int(run_manifest.n_instances) * len(run_manifest.indices),
    }
    metadata = read_sealed_metadata(store.root, run_id) if is_run_sealed(store.root, run_id) else None
    sealed_rows = (metadata or {}).get("rows") or {}
    feature_rows = int(stats["features_lines"])
    instance_features = int(stats["instance_features_lines"])
    if metadata is not None:
        sealed_features = int(sealed_rows.get("sealed_features", sealed_rows.get("features", 0)))
        sealed_instance_features = int(sealed_rows.get("instance_features", 0))
        if feature_rows > 0:
            feature_rows = max(sealed_features, feature_rows)
        else:
            feature_rows = sealed_features
        if instance_features > 0:
            instance_features = max(sealed_instance_features, instance_features)
        elif sealed_instance_features > 0:
            instance_features = sealed_instance_features
    effective_instances = effective_instance_count(stats)
    if (
        metadata is not None
        and feature_rows >= expected_rows["features"]
        and instance_features < expected_rows["instance_features"]
    ):
        # Feature JSONL may already be gone while the sealed shard is full.
        instance_features = max(instance_features, effective_instances, sealed_instance_features)
    actual_rows = {
        "instances": effective_instances,
        "instance_features": instance_features,
        "features": feature_rows,
    }
    run_complete = (
        effective_instances >= expected_rows["instances"]
        and actual_rows["instance_features"] >= expected_rows["instance_features"]
        and actual_rows["features"] >= expected_rows["features"]
    )
    return expected_rows, actual_rows, run_complete


def estimate_seal_cleanup(
    store_root: Path | str,
    run_id: str,
    *,
    delete_arrays: bool = True,
    delete_jsonl: bool = True,
    store: Store | None = None,
) -> dict[str, Any]:
    """Dry-run plan for ``seal_run`` without mutating artifacts."""
    root = Path(store_root)
    run_dir = root / "runs" / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"RDE run does not exist: {run_id!r}")

    existing = read_sealed_metadata(root, run_id)
    if existing is not None and existing.get("status") == "sealed":
        return {
            "run_id": run_id,
            "action": "already_sealed",
            "status": "sealed",
            **compact_seal_summary(existing),
            "bytes_freed": 0,
            "would_remove": {},
        }

    active_store = store or Store(root)
    owns_store = store is None
    try:
        run_manifest = active_store.read_manifest(run_id)
        expected_rows, actual_rows, run_complete = _run_row_counts(
            active_store,
            run_id,
            run_manifest=run_manifest,
        )
        completion_bytes = _run_completion_bytes(run_dir)
        arrays_bytes = _path_bytes(run_dir / "arrays") if delete_arrays else 0
        jsonl_bytes = _run_jsonl_bytes(run_dir) if delete_jsonl and run_complete else 0
        would_remove = {
            "completion_index_bytes": completion_bytes,
            "arrays_bytes": arrays_bytes if delete_arrays else 0,
            "jsonl_bytes": jsonl_bytes,
        }
        return {
            "run_id": run_id,
            "action": "would_seal",
            "status": "unsealed",
            "sealed": False,
            "complete": run_complete,
            "feature_rows": actual_rows["features"],
            "expected_rows": expected_rows,
            "actual_rows": actual_rows,
            "shard_path": _relative_to_root(sealed_parquet_path(root, run_id), root),
            "would_retain_jsonl": not (delete_jsonl and run_complete),
            "would_remove": would_remove,
            "bytes_freed": sum(would_remove.values()),
        }
    finally:
        if owns_store:
            active_store.close()


def _path_bytes(path: Path) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    if not path.is_dir():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _verify_parquet(path: Path, expected_rows: int) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "RDE sealing requires pyarrow: install the rde-parquet extra"
        ) from exc
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"sealed Parquet artifact is missing or empty: {path}")
    try:
        actual_rows = int(pq.ParquetFile(path).metadata.num_rows)
    except Exception as exc:
        raise RuntimeError(f"cannot verify sealed Parquet artifact: {path}") from exc
    if actual_rows != int(expected_rows):
        raise RuntimeError(
            f"sealed Parquet row count mismatch for {path}: "
            f"expected {expected_rows}, got {actual_rows}"
        )
    return actual_rows


def sealed_dir_for_run(store_root: Path | str, run_id: str) -> Path:
    return Path(store_root) / "runs" / run_id / "sealed"


def sealed_marker_path(store_root: Path | str, run_id: str) -> Path:
    return sealed_dir_for_run(store_root, run_id) / SEALED_MARKER_NAME


def sealed_parquet_path(store_root: Path | str, run_id: str) -> Path:
    return sealed_dir_for_run(store_root, run_id) / SEALED_PARQUET_NAME


def read_sealed_metadata(store_root: Path | str, run_id: str) -> dict[str, Any] | None:
    marker = sealed_marker_path(store_root, run_id)
    if not marker.is_file():
        return None
    import json

    return json.loads(marker.read_text(encoding="utf-8"))


def is_run_generation_complete(
    store_root: Path | str,
    run_id: str,
    *,
    expected_instances: int,
    expected_slices: int,
) -> bool:
    """Return True when a run already has its full instance/slice contract."""
    root = Path(store_root)
    run_dir = root / "runs" / run_id
    if not run_dir.is_dir():
        return False
    expected_features = int(expected_instances) * int(expected_slices)
    metadata = read_sealed_metadata(root, run_id)
    if metadata is not None and metadata.get("status") == "sealed":
        rows = metadata.get("rows") or {}
        expected = metadata.get("expected_rows") or {}
        instances = max(
            int(rows.get("instances", 0)),
            int(rows.get("instance_features", 0)),
        )
        features = int(rows.get("sealed_features", rows.get("features", 0)))
        exp_instances = int(expected.get("instances", expected_instances))
        exp_features = int(expected.get("features", expected_features))
        return instances >= exp_instances and features >= exp_features

    from rde.io.store import Store

    active_store = Store(root)
    try:
        stats = active_store.run_stats(run_id)
    finally:
        active_store.close()
    return (
        effective_instance_count(stats) >= expected_instances
        and int(stats["features_lines"]) >= expected_features
    )


def refresh_sealed_metadata_counts(
    store_root: Path | str,
    run_id: str,
    *,
    store: Store | None = None,
) -> dict[str, Any] | None:
    """Update ``sealed.json`` row counts after backfilling instance metadata."""
    root = Path(store_root)
    metadata = read_sealed_metadata(root, run_id)
    if metadata is None or metadata.get("status") != "sealed":
        return None

    active_store = store or Store(root)
    owns_store = store is None
    try:
        run_manifest = active_store.read_manifest(run_id)
        expected_rows, actual_rows, run_complete = _run_row_counts(
            active_store,
            run_id,
            run_manifest=run_manifest,
        )
    finally:
        if owns_store:
            active_store.close()

    rows = dict(metadata.get("rows") or {})
    rows["instances"] = actual_rows["instances"]
    rows["instance_features"] = actual_rows["instance_features"]
    rows["features"] = actual_rows["features"]
    if "sealed_features" in rows:
        rows["sealed_features"] = max(int(rows["sealed_features"]), actual_rows["features"])
    metadata["rows"] = rows
    metadata["complete"] = run_complete
    metadata["expected_rows"] = expected_rows
    marker_path = root / "runs" / run_id / "sealed" / SEALED_MARKER_NAME
    write_json(marker_path, metadata)
    return metadata


def is_run_sealed(store_root: Path | str, run_id: str) -> bool:
    metadata = read_sealed_metadata(store_root, run_id)
    if metadata is None or metadata.get("status") != "sealed":
        return False
    from rde.io.topk_retention import effective_feature_parquet_path

    parquet = effective_feature_parquet_path(store_root, run_id)
    return parquet.is_file() and parquet.stat().st_size > 0


def _delete_jsonl_artifacts(run_dir: Path, *, include_instances_catalog: bool = False) -> dict[str, Any]:
    removed: dict[str, int] = {}
    bytes_freed = 0
    names = list(_FEATURE_JSONL_NAMES)
    if include_instances_catalog:
        names.append("instances.jsonl")
    for name in names:
        path = run_dir / name
        if not path.exists():
            continue
        bytes_freed += _path_bytes(path)
        path.unlink()
        removed[name] = 1
    return {
        "jsonl_removed": bool(removed),
        "jsonl_files_removed": sorted(removed),
        "jsonl_bytes_removed": bytes_freed,
    }


def _cleanup_run(
    store: Store,
    run_id: str,
    *,
    delete_arrays: bool,
    delete_jsonl: bool,
    run_complete: bool,
) -> dict[str, Any]:
    run_dir = store.run_dir(run_id)
    completion_paths = [run_dir / name for name in _COMPLETION_DB_NAMES]
    completion_bytes_before = sum(_path_bytes(path) for path in completion_paths)
    completion_bytes_removed = store.remove_completion_index(run_id)
    completion_bytes_removed = max(completion_bytes_removed, completion_bytes_before)

    arrays_dir = run_dir / "arrays"
    arrays_bytes = _path_bytes(arrays_dir)
    arrays_removed = False
    if delete_arrays and arrays_dir.exists():
        if not arrays_dir.is_dir():
            raise RuntimeError(f"refusing to remove non-directory arrays path: {arrays_dir}")
        shutil.rmtree(arrays_dir)
        arrays_removed = True

    cleanup = {
        "completion_index_removed": completion_bytes_removed > 0
        or not any(path.exists() for path in completion_paths),
        "completion_bytes_removed": completion_bytes_removed,
        "arrays_removed": arrays_removed,
        "arrays_bytes_removed": arrays_bytes if arrays_removed else 0,
        "arrays_retained": arrays_dir.exists(),
        "bytes_freed": completion_bytes_removed + (arrays_bytes if arrays_removed else 0),
        "source_jsonl_retained": True,
        "jsonl_removed": False,
        "jsonl_files_removed": [],
        "jsonl_bytes_removed": 0,
    }
    if delete_jsonl and run_complete:
        jsonl_cleanup = _delete_jsonl_artifacts(
            run_dir,
            include_instances_catalog=True,
        )
        cleanup.update(jsonl_cleanup)
        cleanup["source_jsonl_retained"] = not jsonl_cleanup["jsonl_removed"]
        cleanup["bytes_freed"] += int(jsonl_cleanup["jsonl_bytes_removed"])
    return cleanup


def _hot_operational_artifacts(
    run_dir: Path,
    *,
    delete_arrays: bool,
    delete_jsonl: bool,
    run_complete: bool,
) -> bool:
    hot_jsonl = delete_jsonl and run_complete and (
        _run_jsonl_bytes(run_dir) > 0 or (run_dir / "instances.jsonl").exists()
    )
    hot_arrays = delete_arrays and (run_dir / "arrays").exists()
    hot_completion = _run_completion_bytes(run_dir) > 0
    return hot_jsonl or hot_arrays or hot_completion


def _finalize_sealed_run(
    root: Path,
    run_id: str,
    existing: dict[str, Any],
    *,
    expected_rows: dict[str, int],
    actual_rows: dict[str, int],
    run_complete: bool,
    delete_arrays: bool,
    delete_jsonl: bool,
    store: Store,
) -> dict[str, Any]:
    """Verify an existing shard and remove operational artifacts without re-export."""
    run_dir = root / "runs" / run_id
    parquet_path = sealed_parquet_path(root, run_id)
    expected_features = int(existing.get("rows", {}).get("sealed_features", actual_rows["features"]))
    _verify_parquet(parquet_path, expected_features)
    if existing.get("parquet", {}).get("sha256") != _sha256(parquet_path):
        raise RuntimeError(f"sealed Parquet checksum mismatch for {run_id!r}")

    metadata = dict(existing)
    metadata["complete"] = run_complete
    metadata["rows"] = {
        **actual_rows,
        "sealed_features": expected_features,
    }
    metadata["expected_rows"] = expected_rows
    marker_path = run_dir / "sealed" / SEALED_MARKER_NAME
    write_json(marker_path, {**metadata, "status": "pending_cleanup"})

    cleanup = _cleanup_run(
        store,
        run_id,
        delete_arrays=delete_arrays,
        delete_jsonl=delete_jsonl,
        run_complete=run_complete,
    )
    metadata["status"] = "sealed"
    metadata["cleanup"] = {
        **dict(metadata.get("cleanup") or {}),
        **cleanup,
    }
    write_json(marker_path, metadata)
    return metadata


def seal_run(
    store_root: Path | str,
    run_id: str,
    *,
    campaign_id: str | None = None,
    batch_index: int | None = None,
    delete_arrays: bool = True,
    delete_jsonl: bool = True,
    store: Store | None = None,
) -> dict[str, Any]:
    """Write a verified compact snapshot, then remove operational artifacts.

    Discovery can reload the flat feature table from the sealed Parquet shard.
    When ``delete_jsonl`` is true and the run is complete, ``features.jsonl``
    and ``instance_features.jsonl`` are removed after verification.  Incomplete
    runs always retain JSONL so resume can continue.  The SQLite completion index
    is rebuildable from JSONL or the sealed shard and is therefore removed.
    Arrays are removed by default because standard discovery uses scalar feature
    rows; pass ``delete_arrays=False`` for representation or massive-catalog
    workflows that require them.

    Cleanup is fail-closed: no source artifact is removed until the Parquet
    row count and checksum have been verified and a pending-cleanup marker has
    been durably written.
    """
    root = Path(store_root)
    run_dir = root / "runs" / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"RDE run does not exist: {run_id!r}")

    require_seal_dependencies()

    existing = read_sealed_metadata(root, run_id)
    if existing is not None and existing.get("status") == "sealed":
        parquet_path = sealed_parquet_path(root, run_id)
        expected_rows_count = int(existing.get("rows", {}).get("sealed_features", 0))
        if parquet_path.is_file() and expected_rows_count > 0:
            _verify_parquet(parquet_path, expected_rows_count)
            if existing.get("parquet", {}).get("sha256") == _sha256(parquet_path):
                probe_store = store or Store(root)
                owns_probe = store is None
                try:
                    probe_store.flush(run_id)
                    run_manifest = probe_store.read_manifest(run_id)
                    expected_rows, actual_rows, run_complete = _run_row_counts(
                        probe_store,
                        run_id,
                        run_manifest=run_manifest,
                    )
                    hot = _hot_operational_artifacts(
                        run_dir,
                        delete_arrays=delete_arrays,
                        delete_jsonl=delete_jsonl,
                        run_complete=run_complete,
                    )
                    up_to_date = (
                        int(actual_rows["features"]) == expected_rows_count
                        and existing.get("complete") is run_complete
                        and not hot
                    )
                    if up_to_date:
                        return existing
                    if (
                        run_complete
                        and int(actual_rows["features"]) == expected_rows_count
                    ):
                        return _finalize_sealed_run(
                            root,
                            run_id,
                            existing,
                            expected_rows=expected_rows,
                            actual_rows=actual_rows,
                            run_complete=run_complete,
                            delete_arrays=delete_arrays,
                            delete_jsonl=delete_jsonl,
                            store=probe_store,
                        )
                finally:
                    if owns_probe:
                        probe_store.close()

    active_store = store or Store(root)
    owns_store = store is None
    try:
        active_store.flush(run_id)
        run_manifest = active_store.read_manifest(run_id)
        resolved_campaign_id = campaign_id or run_manifest.extra.get("campaign_id")
        expected_rows, actual_rows, run_complete = _run_row_counts(
            active_store,
            run_id,
            run_manifest=run_manifest,
        )

        sealed_dir = run_dir / "sealed"
        sealed_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = sealed_dir / "features.parquet"
        temporary_path = sealed_dir / f".features.parquet.{uuid.uuid4().hex}.tmp"
        try:
            active_store.export_features_parquet(
                run_id,
                temporary_path,
                flat=True,
                compression="zstd",
            )
            parquet_rows = _verify_parquet(
                temporary_path,
                int(actual_rows["features"]),
            )
            parquet_sha256 = _sha256(temporary_path)
            parquet_bytes = temporary_path.stat().st_size
            os.replace(temporary_path, parquet_path)
            if _sha256(parquet_path) != parquet_sha256:
                raise RuntimeError(
                    f"sealed Parquet checksum changed during publication: {parquet_path}"
                )
        finally:
            temporary_path.unlink(missing_ok=True)

        overrun = {
            key: (expected_rows[key], actual_rows[key])
            for key in expected_rows
            if actual_rows[key] > expected_rows[key]
        }
        if overrun:
            raise RuntimeError(
                f"sealed run row counts exceed its manifest contract: {overrun}"
            )
        source_sizes = {
            name: _path_bytes(run_dir / name)
            for name in ("instances.jsonl", *_FEATURE_JSONL_NAMES)
        }
        metadata = {
            "schema_version": SEAL_SCHEMA_VERSION,
            "status": "pending_cleanup",
            "run_id": run_id,
            "campaign_id": resolved_campaign_id,
            "batch_index": batch_index,
            "sealed_at": utc_now_iso(),
            "complete": run_complete,
            "rows": {
                **actual_rows,
                "sealed_features": parquet_rows,
            },
            "expected_rows": expected_rows,
            "source_bytes": source_sizes,
            "parquet": {
                "path": _relative_to_root(parquet_path, root),
                "bytes": parquet_bytes,
                "sha256": parquet_sha256,
                "compression": "zstd",
            },
            "cleanup": {
                "source_jsonl_retained": not (delete_jsonl and run_complete),
                "delete_arrays": delete_arrays,
                "delete_jsonl": delete_jsonl,
            },
        }
        marker_path = sealed_dir / "sealed.json"
        write_json(marker_path, metadata)

        cleanup = _cleanup_run(
            active_store,
            run_id,
            delete_arrays=delete_arrays,
            delete_jsonl=delete_jsonl,
            run_complete=run_complete,
        )
        metadata["status"] = "sealed"
        metadata["cleanup"].update(cleanup)
        write_json(marker_path, metadata)
        return metadata
    finally:
        if owns_store:
            active_store.close()
