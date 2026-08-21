"""Append-only run ledger: JSONL catalog + NumPy NPZ sidecars."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from rde.core.schema import validate_feature_row, validate_instance_features_row
from rde.io.completion_cache import extract_cross_slice_scalars, resume_scalars_json

from rde.core.instance import InstanceRecord
from rde.io.json_util import atomic_write, json_default, utc_now_iso, write_json


def _json_default(obj: Any) -> Any:
    return json_default(obj)


def count_jsonl_lines(path: Path) -> int:
    """Stream-count non-empty JSONL lines without loading the file."""
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


class JsonlWriter:
    """Buffered JSONL writer — flushes every batch_size records."""

    def __init__(
        self,
        path: Path,
        batch_size: int = 64,
        *,
        on_flush: Callable[[int], None] | None = None,
    ) -> None:
        self.path = path
        self.batch_size = max(1, batch_size)
        self._buffer: list[str] = []
        self._buffered_bytes = 0
        self._on_flush = on_flush
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        text = json.dumps(record, sort_keys=True, default=_json_default) + "\n"
        self._buffer.append(text)
        self._buffered_bytes += len(text.encode("utf-8"))
        if len(self._buffer) >= self.batch_size:
            self.flush()

    @property
    def buffered_bytes(self) -> int:
        return self._buffered_bytes

    def flush(self) -> int:
        if not self._buffer:
            return 0
        bytes_written = 0
        with self.path.open("a", encoding="utf-8") as fh:
            for text in self._buffer:
                fh.write(text)
                bytes_written += len(text.encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())
        self._buffer.clear()
        self._buffered_bytes = 0
        if self._on_flush is not None:
            self._on_flush(bytes_written)
        return bytes_written

    def close(self) -> None:
        self.flush()


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return iter(())
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream JSONL rows without loading the full file."""
    return _read_jsonl(path)


@dataclass
class RunManifest:
    """Metadata for one pipeline run."""

    run_id: str
    domain_id: str
    created_at: str = field(default_factory=utc_now_iso)
    n_instances: int = 0
    size: int = 0
    seed: int = 0
    indices: list[int] = field(default_factory=list)
    descriptor_names: list[str] = field(default_factory=list)
    metric_names: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunManifest:
        return cls(**data)


class Store:
    """Filesystem-backed store for RDE runs."""

    def __init__(self, root: Path | str, *, jsonl_batch_size: int | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if jsonl_batch_size is None:
            import os

            env_batch = os.environ.get("RDE_JSONL_BATCH")
            jsonl_batch_size = int(env_batch) if env_batch else 256
        self._jsonl_batch_size = max(1, int(jsonl_batch_size))
        self._writers: dict[str, JsonlWriter] = {}
        self._completion_connections: dict[str, sqlite3.Connection] = {}
        self._completion_pending: dict[
            str, dict[str, list[tuple[str, int] | tuple[str, str]]]
        ] = {}
        self._session_bytes_written = 0

    def run_dir(self, run_id: str) -> Path:
        return self.root / "runs" / run_id

    @property
    def session_bytes_written(self) -> int:
        """Bytes durably added by this Store instance since construction."""
        return self._session_bytes_written

    @property
    def pending_bytes(self) -> int:
        """Approximate buffered JSONL bytes not yet durable on disk."""
        return sum(writer.buffered_bytes for writer in self._writers.values())

    @property
    def storage_bytes_including_pending(self) -> int:
        return self.session_bytes_written + self.pending_bytes

    def _record_jsonl_flush(self, bytes_written: int) -> None:
        self._session_bytes_written += max(0, int(bytes_written))

    def _record_file_growth(self, path: Path, before_size: int) -> None:
        try:
            after_size = path.stat().st_size
        except OSError:
            return
        self._session_bytes_written += max(0, after_size - before_size)

    def reset_run(self, run_id: str) -> None:
        """Remove all persisted rows/arrays for a run (fresh non-resume restart)."""
        self.flush(run_id)
        connection = self._completion_connections.pop(run_id, None)
        if connection is not None:
            connection.close()
        self._completion_pending.pop(run_id, None)
        prefix = f"{run_id}:"
        for key in list(self._writers):
            if key.startswith(prefix):
                del self._writers[key]
        run_dir = self.run_dir(run_id)
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

    def _writer(self, run_id: str, filename: str) -> JsonlWriter:
        key = f"{run_id}:{filename}"
        if key not in self._writers:
            self._writers[key] = JsonlWriter(
                self.run_dir(run_id) / filename,
                batch_size=self._jsonl_batch_size,
                on_flush=self._record_jsonl_flush,
            )
        return self._writers[key]

    def flush(self, run_id: str | None = None) -> None:
        if run_id is None:
            for writer in self._writers.values():
                writer.flush()
            for current_run_id in list(self._completion_pending):
                self._flush_completion_index(current_run_id)
            return
        prefix = f"{run_id}:"
        for key, writer in self._writers.items():
            if key.startswith(prefix):
                writer.flush()
        self._flush_completion_index(run_id)

    def close(self) -> None:
        self.flush()
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()
        for connection in self._completion_connections.values():
            connection.close()
        self._completion_connections.clear()

    def remove_completion_index(self, run_id: str) -> int:
        """Close and remove the operational SQLite resume index for a run.

        The JSONL rows remain the source of truth, so a later resume can
        rebuild this index through ``ensure_completion_index``.  SQLite
        sidecars are removed together to avoid leaving a WAL or journal
        consuming space after a seal.
        """
        self.flush(run_id)
        connection = self._completion_connections.pop(run_id, None)
        if connection is not None:
            connection.close()
        self._completion_pending.pop(run_id, None)
        removed_bytes = 0
        for suffix in ("", "-wal", "-shm", "-journal"):
            path = self.run_dir(run_id) / f"completion.sqlite3{suffix}"
            try:
                removed_bytes += path.stat().st_size
                path.unlink()
            except FileNotFoundError:
                continue
        return removed_bytes

    def _completion_connection(self, run_id: str) -> sqlite3.Connection:
        connection = self._completion_connections.get(run_id)
        if connection is not None:
            return connection
        run_dir = self.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        db_path = run_dir / "completion.sqlite3"
        before_size = db_path.stat().st_size if db_path.exists() else 0
        connection = sqlite3.connect(db_path)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS feature_completion "
            "(instance_id TEXT NOT NULL, family_index INTEGER NOT NULL, "
            "PRIMARY KEY (instance_id, family_index))"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS instance_completion "
            "(instance_id TEXT PRIMARY KEY, scalars_json TEXT)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS completion_meta "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._reconcile_completion_index(connection, run_dir)
        self._completion_connections[run_id] = connection
        self._completion_pending.setdefault(
            run_id, {"features": [], "instances": []}
        )
        self._record_file_growth(db_path, before_size)
        return connection

    def _reconcile_completion_index(
        self, connection: sqlite3.Connection, run_dir: Path
    ) -> None:
        """Reconcile only JSONL bytes not covered by the completion index.

        The offsets make reopening a large run bounded in memory and recover
        rows that reached JSONL before a process crashed before the SQLite
        transaction committed.  A final, incomplete JSONL line is left for the
        next reconciliation rather than being indexed prematurely.

        When JSONL has been removed after sealing, rebuild the index from the
        verified Parquet shard instead.
        """
        features_path = run_dir / "features.jsonl"
        instance_features_path = run_dir / "instance_features.jsonl"
        if not features_path.exists() and not instance_features_path.exists():
            if self._reconcile_completion_from_sealed(connection, run_dir):
                return
        sources = (
            ("features.jsonl", "features_offset", "feature_completion"),
            ("instance_features.jsonl", "instance_features_offset", "instance_completion"),
        )
        for filename, meta_key, table in sources:
            path = run_dir / filename
            if not path.exists():
                continue
            stored = connection.execute(
                "SELECT value FROM completion_meta WHERE key=?", (meta_key,)
            ).fetchone()
            offset = int(stored[0]) if stored is not None else 0
            file_size = path.stat().st_size
            if offset > file_size:
                connection.execute(f"DELETE FROM {table}")
                offset = 0
            with path.open("rb") as handle:
                handle.seek(offset)
                while True:
                    line_start = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    terminated = line.endswith(b"\n")
                    try:
                        row = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        if not terminated:
                            handle.seek(line_start)
                            break
                        raise
                    if table == "feature_completion":
                        instance_id = row.get("instance_id")
                        family_index = row.get("family_index")
                        if instance_id is not None and family_index is not None:
                            connection.execute(
                                "INSERT OR IGNORE INTO feature_completion VALUES (?, ?)",
                                (str(instance_id), int(family_index)),
                            )
                    else:
                        instance_id = row.get("instance_id")
                        if instance_id is not None:
                            connection.execute(
                                "INSERT OR REPLACE INTO instance_completion VALUES (?, ?)",
                                (
                                    str(instance_id),
                                    resume_scalars_json(
                                        row.get("scalars", {}),
                                        json_default=_json_default,
                                    ),
                                ),
                            )
                    offset = handle.tell()
                    if not terminated:
                        break
            connection.execute(
                "INSERT OR REPLACE INTO completion_meta VALUES (?, ?)",
                (meta_key, str(offset)),
            )
        connection.commit()

    def _reconcile_completion_from_sealed(
        self, connection: sqlite3.Connection, run_dir: Path
    ) -> bool:
        from rde.analyze.query import _FLAT_META_KEYS, load_rows_from_shard

        marker = run_dir / "sealed" / "sealed.json"
        parquet = run_dir / "sealed" / "features.parquet"
        if not marker.is_file() or not parquet.is_file():
            return False
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if metadata.get("status") != "sealed":
            return False
        parquet_sha256 = metadata.get("parquet", {}).get("sha256")
        if not parquet_sha256:
            return False
        stored = connection.execute(
            "SELECT value FROM completion_meta WHERE key=?", ("sealed_sha256",)
        ).fetchone()
        if stored is not None and stored[0] == parquet_sha256:
            return True

        rows = load_rows_from_shard(parquet)
        instance_scalars: dict[str, dict[str, Any]] = {}
        for row in rows:
            instance_id = row.get("instance_id")
            family_index = row.get("family_index")
            if instance_id is None or family_index is None:
                continue
            connection.execute(
                "INSERT OR IGNORE INTO feature_completion VALUES (?, ?)",
                (str(instance_id), int(family_index)),
            )
            if str(instance_id) in instance_scalars:
                continue
            scalars = extract_cross_slice_scalars(
                {
                    key: value
                    for key, value in row.items()
                    if key not in _FLAT_META_KEYS and not str(key).startswith("metric.")
                }
            )
            if not scalars:
                instance_scalars[str(instance_id)] = {}
                continue
            instance_scalars[str(instance_id)] = scalars
        for instance_id, scalars in instance_scalars.items():
            connection.execute(
                "INSERT OR REPLACE INTO instance_completion VALUES (?, ?)",
                (
                    instance_id,
                    resume_scalars_json(scalars, json_default=_json_default),
                ),
            )
        connection.execute(
            "INSERT OR REPLACE INTO completion_meta VALUES (?, ?)",
            ("sealed_sha256", parquet_sha256),
        )
        connection.execute(
            "INSERT OR REPLACE INTO completion_meta VALUES (?, ?)",
            ("features_offset", str(parquet.stat().st_size)),
        )
        connection.execute(
            "INSERT OR REPLACE INTO completion_meta VALUES (?, ?)",
            ("instance_features_offset", str(parquet.stat().st_size)),
        )
        connection.commit()
        return True

    def _flush_completion_index(self, run_id: str) -> None:
        pending = self._completion_pending.get(run_id)
        if not pending or not any(pending.values()):
            return
        connection = self._completion_connection(run_id)
        db_path = self.run_dir(run_id) / "completion.sqlite3"
        before_size = db_path.stat().st_size if db_path.exists() else 0
        connection.executemany(
            "INSERT OR IGNORE INTO feature_completion VALUES (?, ?)",
            pending["features"],
        )
        connection.executemany(
            "INSERT OR REPLACE INTO instance_completion VALUES (?, ?)",
            pending["instances"],
        )
        for filename, meta_key in (
            ("features.jsonl", "features_offset"),
            ("instance_features.jsonl", "instance_features_offset"),
        ):
            path = self.run_dir(run_id) / filename
            offset = path.stat().st_size if path.exists() else 0
            connection.execute(
                "INSERT OR REPLACE INTO completion_meta VALUES (?, ?)",
                (meta_key, str(offset)),
            )
        connection.commit()
        self._record_file_growth(db_path, before_size)
        pending["features"].clear()
        pending["instances"].clear()

    def _checkpoint_completion_index(self, run_id: str) -> None:
        """Flush JSONL before committing its bounded completion checkpoint."""
        for filename in ("features.jsonl", "instance_features.jsonl"):
            writer = self._writers.get(f"{run_id}:{filename}")
            if writer is not None:
                writer.flush()
        self._flush_completion_index(run_id)

    def run_stats(self, run_id: str) -> dict[str, int | str]:
        """Lightweight run size stats (streaming line counts)."""
        run_dir = self.run_dir(run_id)
        return {
            "run_id": run_id,
            "features_lines": count_jsonl_lines(run_dir / "features.jsonl"),
            "instance_features_lines": count_jsonl_lines(run_dir / "instance_features.jsonl"),
            "instances_lines": count_jsonl_lines(run_dir / "instances.jsonl"),
        }

    def write_manifest(self, manifest: RunManifest) -> Path:
        run_dir = self.run_dir(manifest.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "manifest.json"
        before_size = path.stat().st_size if path.exists() else 0
        write_json(path, manifest.to_dict())
        self._record_file_growth(path, before_size)
        return path

    def read_manifest(self, run_id: str) -> RunManifest:
        path = self.run_dir(run_id) / "manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return RunManifest.from_dict(data)

    def append_instance(self, run_id: str, instance: InstanceRecord) -> None:
        self._writer(run_id, "instances.jsonl").append(instance.to_dict())

    def recorded_instance_ids(self, run_id: str) -> set[str]:
        path = self.run_dir(run_id) / "instances.jsonl"
        if not path.is_file():
            return set()
        return {str(row["instance_id"]) for row in iter_jsonl(path)}

    def backfill_missing_instance_records(
        self,
        run_id: str,
        instances: Iterator[InstanceRecord],
        *,
        anchor_ids: set[str] | None = None,
    ) -> int:
        """Append ``instances.jsonl`` rows present in anchor set but missing on disk."""
        recorded = self.recorded_instance_ids(run_id)
        if anchor_ids is None:
            anchor_path = self.run_dir(run_id) / "instance_features.jsonl"
            if not anchor_path.is_file():
                return 0
            anchor_ids = {
                str(row["instance_id"]) for row in iter_jsonl(anchor_path)
            }
        missing = anchor_ids - recorded
        if not missing:
            return 0
        appended = 0
        for instance in instances:
            if instance.instance_id in missing:
                self.append_instance(run_id, instance)
                appended += 1
        self.flush(run_id)
        return appended

    def append_features(self, run_id: str, row: dict[str, Any]) -> None:
        self._writer(run_id, "features.jsonl").append(row)
        instance_id = row.get("instance_id")
        family_index = row.get("family_index")
        if instance_id is not None and family_index is not None:
            pending = self._completion_pending.setdefault(
                run_id, {"features": [], "instances": []}
            )
            pending["features"].append((str(instance_id), int(family_index)))
            if sum(len(rows) for rows in pending.values()) >= self._jsonl_batch_size:
                self._checkpoint_completion_index(run_id)

    def append_instance_features(self, run_id: str, row: dict[str, Any]) -> None:
        scalars_json = resume_scalars_json(
            row.get("scalars", {}),
            json_default=_json_default,
        )
        self._writer(run_id, "instance_features.jsonl").append(row)
        instance_id = row.get("instance_id")
        if instance_id is not None:
            pending = self._completion_pending.setdefault(
                run_id, {"features": [], "instances": []}
            )
            pending["instances"].append((str(instance_id), scalars_json))
            if sum(len(rows) for rows in pending.values()) >= self._jsonl_batch_size:
                self._checkpoint_completion_index(run_id)

    def read_instance_features(
        self, run_id: str, *, validate: bool = False
    ) -> list[dict[str, Any]]:
        rows = list(iter_jsonl(self.run_dir(run_id) / "instance_features.jsonl"))
        if validate:
            for i, row in enumerate(rows):
                errs = validate_instance_features_row(row)
                if errs:
                    raise ValueError(f"instance_features row {i}: {errs}")
        return rows

    def completed_instance_ids(self, run_id: str) -> set[str]:
        self.flush(run_id)
        connection = self._completion_connection(run_id)
        return {
            str(row[0])
            for row in connection.execute("SELECT instance_id FROM instance_completion")
        }

    def ensure_completion_index(self, run_id: str) -> None:
        """Build or validate the scalable completion index without loading rows."""
        self._completion_connection(run_id)

    def completed_feature_keys(self, run_id: str) -> set[tuple[str, int]]:
        self.flush(run_id)
        connection = self._completion_connection(run_id)
        return {
            (str(row[0]), int(row[1]))
            for row in connection.execute(
                "SELECT instance_id, family_index FROM feature_completion"
            )
        }

    def completed_feature_indices(
        self, run_id: str, instance_id: str, indices: Iterable[int]
    ) -> set[int]:
        """Return only the completed indices requested for one instance."""
        requested = tuple(dict.fromkeys(int(index) for index in indices))
        if not requested:
            return set()
        connection = self._completion_connection(run_id)
        if len(requested) <= 900:
            placeholders = ",".join("?" for _ in requested)
            rows = connection.execute(
                "SELECT family_index FROM feature_completion "
                f"WHERE instance_id=? AND family_index IN ({placeholders})",
                (str(instance_id), *requested),
            )
        else:
            rows = connection.execute(
                "SELECT family_index FROM feature_completion WHERE instance_id=?",
                (str(instance_id),),
            )
        requested_set = set(requested)
        return {int(row[0]) for row in rows if int(row[0]) in requested_set}

    def get_instance_completion(
        self, run_id: str, instance_id: str
    ) -> tuple[bool, dict[str, Any] | None]:
        """Return completion state and one instance's cached scalar map."""
        connection = self._completion_connection(run_id)
        row = connection.execute(
            "SELECT scalars_json FROM instance_completion WHERE instance_id=?",
            (str(instance_id),),
        ).fetchone()
        if row is None:
            return False, None
        if row[0] is None:
            return True, None
        scalars = json.loads(row[0])
        return True, dict(scalars) if scalars else None

    def purge_instance_resume_scalars(self, run_id: str, instance_id: str) -> None:
        """Drop cross-slice resume payload once every slice for an instance is durable."""
        iid = str(instance_id)
        connection = self._completion_connection(run_id)
        connection.execute(
            "INSERT OR REPLACE INTO instance_completion (instance_id, scalars_json) "
            "VALUES (?, NULL)",
            (iid,),
        )
        connection.commit()
        pending = self._completion_pending.get(run_id)
        if pending is None:
            return
        replaced = False
        updated: list[tuple[str, str | None]] = []
        for pending_id, payload in pending["instances"]:
            if pending_id == iid:
                updated.append((iid, None))
                replaced = True
            else:
                updated.append((pending_id, payload))
        if not replaced:
            updated.append((iid, None))
        pending["instances"] = updated

    def has_completed_feature(self, run_id: str, instance_id: str, family_index: int) -> bool:
        connection = self._completion_connection(run_id)
        return (
            connection.execute(
                "SELECT 1 FROM feature_completion WHERE instance_id=? AND family_index=?",
                (str(instance_id), int(family_index)),
            ).fetchone()
            is not None
        )

    def has_completed_instance(self, run_id: str, instance_id: str) -> bool:
        connection = self._completion_connection(run_id)
        return (
            connection.execute(
                "SELECT 1 FROM instance_completion WHERE instance_id=?",
                (str(instance_id),),
            ).fetchone()
            is not None
        )

    def instance_scalars(self, run_id: str, instance_id: str) -> dict[str, Any] | None:
        _completed, scalars = self.get_instance_completion(run_id, instance_id)
        return scalars

    def instance_feature_scalars(
        self, run_id: str, instance_id: str
    ) -> dict[str, Any] | None:
        """Load durable instance scalars from JSONL when resume cache was purged."""
        target = str(instance_id)
        for row in self.iter_instance_features(run_id):
            if str(row.get("instance_id")) == target:
                scalars = row.get("scalars")
                return dict(scalars) if scalars else None
        return None

    def save_array(
        self,
        run_id: str,
        instance_id: str,
        name: str,
        array: np.ndarray,
    ) -> Path:
        path = self.run_dir(run_id) / "arrays" / instance_id / f"{name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        before_size = path.stat().st_size if path.exists() else 0
        atomic_write(path, lambda temporary: np.savez_compressed(temporary, values=array))
        self._record_file_growth(path, before_size)
        return path

    def load_array(self, run_id: str, instance_id: str, name: str) -> np.ndarray:
        path = self.run_dir(run_id) / "arrays" / instance_id / f"{name}.npz"
        with np.load(path) as data:
            return np.asarray(data["values"])

    def read_instances(self, run_id: str) -> list[InstanceRecord]:
        return [InstanceRecord.from_dict(row) for row in iter_jsonl(self.run_dir(run_id) / "instances.jsonl")]

    def iter_instances(self, run_id: str) -> Iterator[InstanceRecord]:
        for row in iter_jsonl(self.run_dir(run_id) / "instances.jsonl"):
            yield InstanceRecord.from_dict(row)

    def iter_instance_features(self, run_id: str) -> Iterator[dict[str, Any]]:
        yield from iter_jsonl(self.run_dir(run_id) / "instance_features.jsonl")

    def iter_features(self, run_id: str, *, validate: bool = False) -> Iterator[dict[str, Any]]:
        for i, row in enumerate(iter_jsonl(self.run_dir(run_id) / "features.jsonl")):
            if validate:
                errs = validate_feature_row(row)
                if errs:
                    raise ValueError(f"features row {i}: {errs}")
            yield row

    def read_features(self, run_id: str, *, validate: bool = False) -> list[dict[str, Any]]:
        rows = list(iter_jsonl(self.run_dir(run_id) / "features.jsonl"))
        if validate:
            for i, row in enumerate(rows):
                errs = validate_feature_row(row)
                if errs:
                    raise ValueError(f"features row {i}: {errs}")
        return rows

    def list_runs(self, *, active_only: bool = True) -> list[str]:
        runs_root = self.root / "runs"
        if not runs_root.exists():
            return []
        runs = sorted(p.name for p in runs_root.iterdir() if p.is_dir())
        return runs

    def is_active_run(self, run_id: str) -> bool:
        return self.run_dir(run_id).is_dir()

    def export_features_csv(self, run_id: str, path: Path | str, *, flat: bool = True) -> Path:
        """Flatten features into a CSV file (includes instance-level scalars when flat=True)."""
        rows = self._rows_for_export(run_id, flat=flat)
        return self._export_features_flat(rows, path, fmt="csv")

    def export_features_parquet(
        self,
        run_id: str,
        path: Path | str,
        *,
        flat: bool = True,
        compression: str | None = "zstd",
        batch_size: int = 8192,
    ) -> Path:
        """Export features to Parquet without loading a large run into RAM.

        Flat exports are the durable discovery representation.  They are
        scanned twice: once to establish a stable column set and once to
        write bounded Arrow record batches.  The two-pass scan keeps peak
        memory proportional to the instance-scalar join map plus one batch,
        rather than to the complete feature catalog.
        """
        if not flat:
            rows = self._rows_for_export(run_id, flat=False)
            return self._export_features_flat(rows, path, fmt="parquet")
        return self._export_features_parquet_streaming(
            run_id,
            path,
            compression=compression,
            batch_size=batch_size,
        )

    def _rows_for_export(self, run_id: str, *, flat: bool) -> list[dict[str, Any]]:
        if flat:
            from rde.analyze.query import flatten_features

            return flatten_features(run_id, self.root)
        return self.read_features(run_id)

    def _flatten_feature_rows(self, rows: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
        descriptor_keys: set[str] = set()
        metric_keys: set[str] = set()
        meta_keys: set[str] = set()
        for row in rows:
            for key, value in row.items():
                if key.startswith("metric."):
                    metric_keys.add(key.removeprefix("metric."))
                elif key in {
                    "run_id",
                    "instance_id",
                    "domain_id",
                    "size",
                    "seed",
                    "family_index",
                    "slice_kind",
                }:
                    continue
                elif isinstance(value, (int, float)) or value is None:
                    descriptor_keys.add(key)
                else:
                    meta_keys.add(key)
        desc_sorted = sorted(descriptor_keys)
        metric_sorted = sorted(metric_keys)
        fieldnames = [
            "run_id",
            "instance_id",
            "domain_id",
            "size",
            "seed",
            "family_index",
            "slice_kind",
            *desc_sorted,
            *[f"metric.{k}" for k in metric_sorted],
        ]
        flat_rows: list[dict[str, Any]] = []
        for row in rows:
            flat: dict[str, Any] = {
                "run_id": row.get("run_id"),
                "instance_id": row.get("instance_id"),
                "domain_id": row.get("domain_id"),
                "size": row.get("size"),
                "seed": row.get("seed"),
                "family_index": row.get("family_index"),
                "slice_kind": row.get("slice_kind"),
            }
            if "descriptors" in row:
                for key in desc_sorted:
                    flat[key] = row.get("descriptors", {}).get(key, row.get(key))
                for key in metric_sorted:
                    flat[f"metric.{key}"] = row.get("metrics", {}).get(key)
            else:
                for key in desc_sorted:
                    flat[key] = row.get(key)
                for key in metric_sorted:
                    flat[f"metric.{key}"] = row.get(f"metric.{key}")
            flat_rows.append(flat)
        return fieldnames, flat_rows

    @staticmethod
    def _flat_fieldnames(rows: Iterable[dict[str, Any]]) -> list[str]:
        descriptor_keys: set[str] = set()
        metric_keys: set[str] = set()
        for row in rows:
            for key, value in row.items():
                if key.startswith("metric."):
                    metric_keys.add(key.removeprefix("metric."))
                elif key in {
                    "run_id",
                    "instance_id",
                    "domain_id",
                    "size",
                    "seed",
                    "family_index",
                    "slice_kind",
                }:
                    continue
                elif isinstance(value, (int, float)) or value is None:
                    descriptor_keys.add(key)
        return [
            "run_id",
            "instance_id",
            "domain_id",
            "size",
            "seed",
            "family_index",
            "slice_kind",
            *sorted(descriptor_keys),
            *[f"metric.{key}" for key in sorted(metric_keys)],
        ]

    @staticmethod
    def _arrow_schema(pa: Any, fieldnames: list[str]) -> Any:
        string_fields = {"run_id", "instance_id", "domain_id", "slice_kind"}
        integer_fields = {"size", "seed", "family_index"}
        fields = []
        for name in fieldnames:
            if name in string_fields:
                dtype = pa.string()
            elif name in integer_fields:
                dtype = pa.int64()
            else:
                dtype = pa.float64()
            fields.append(pa.field(name, dtype))
        return pa.schema(fields)

    def _export_features_parquet_streaming(
        self,
        run_id: str,
        path: Path | str,
        *,
        compression: str | None,
        batch_size: int,
    ) -> Path:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "Parquet export requires pyarrow: pip install pyarrow"
            ) from exc

        from rde.analyze.query import iter_flatten_features

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        batch_size = max(1, int(batch_size))
        fieldnames = self._flat_fieldnames(iter_flatten_features(run_id, self.root))
        schema = self._arrow_schema(pa, fieldnames)
        writer = pq.ParquetWriter(out_path, schema, compression=compression)
        batch: list[dict[str, Any]] = []
        try:
            for row in iter_flatten_features(run_id, self.root):
                batch.append(row)
                if len(batch) >= batch_size:
                    arrays = [
                        pa.array([row.get(name) for row in batch], type=schema.field(name).type)
                        for name in fieldnames
                    ]
                    writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
                    batch.clear()
            if batch:
                arrays = [
                    pa.array([row.get(name) for row in batch], type=schema.field(name).type)
                    for name in fieldnames
                ]
                writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
        finally:
            writer.close()
        return out_path

    def _export_features_flat(self, rows: list[dict[str, Any]], path: Path | str, *, fmt: str) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames, flat_rows = self._flatten_feature_rows(rows)

        if fmt == "csv":
            with out_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flat_rows)
            return out_path

        if fmt == "parquet":
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise ImportError(
                    "Parquet export requires pyarrow: pip install pyarrow"
                ) from exc
            table = pa.Table.from_pylist(flat_rows)
            pq.write_table(table, out_path)
            return out_path

        raise ValueError(f"Unknown export format: {fmt!r}")
