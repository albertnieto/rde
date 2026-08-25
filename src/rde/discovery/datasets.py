"""Load feature tables from RDE stores (Phase 4+)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from rde.analyze.query import flatten_features, load_rows_from_shard, numeric_columns
from rde.io.json_util import utc_now_iso, write_json
from rde.io.seal import is_run_sealed, read_sealed_metadata
from rde.io.topk_retention import effective_feature_parquet_path
from rde.io.store import Store, count_jsonl_lines


@dataclass
class DatasetManifest:
    """Pooled multi-run dataset manifest (v0.3)."""

    dataset_id: str
    campaign_id: str
    domain_id: str
    contract_version: str = "0.3"
    child_run_ids: list[str] = field(default_factory=list)
    sizes: list[int] = field(default_factory=list)
    expected_instances: int = 0
    actual_instances: int = 0
    expected_slices: int = 0
    actual_slices: int = 0
    feature_catalog_checksum: str = ""
    split_policy_checksum: str = ""
    child_checksums: dict[str, str] = field(default_factory=dict)
    status: str = "incomplete"
    created_at: str = field(default_factory=utc_now_iso)
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def checksum(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_checksum(store: Store, run_id: str) -> str:
    digest = hashlib.sha256()
    run_dir = store.run_dir(run_id)
    metadata = read_sealed_metadata(store.root, run_id)
    if metadata is not None and metadata.get("status") == "sealed":
        for name in ("manifest.json", "instances.jsonl", "sealed/sealed.json", "sealed/topk.json"):
            path = run_dir / name
            digest.update(name.encode("utf-8"))
            if not path.exists():
                continue
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        parquet_path = effective_feature_parquet_path(store.root, run_id)
        digest.update(str(parquet_path.relative_to(run_dir)).encode("utf-8"))
        if parquet_path.is_file():
            with parquet_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    for name in ("manifest.json", "instances.jsonl", "instance_features.jsonl", "features.jsonl"):
        path = run_dir / name
        digest.update(name.encode("utf-8"))
        if not path.exists():
            continue
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _run_row_counts(store: Store, run_id: str) -> tuple[int, int]:
    """Return (instance_feature_rows, feature_rows) for manifest accounting."""
    run_dir = store.run_dir(run_id)
    metadata = read_sealed_metadata(store.root, run_id)
    if metadata is not None and metadata.get("status") == "sealed":
        rows = metadata.get("rows", {})
        return (
            int(rows.get("instance_features", rows.get("instances", 0))),
            int(rows.get("sealed_features", rows.get("features", 0))),
        )
    return (
        count_jsonl_lines(run_dir / "instance_features.jsonl"),
        count_jsonl_lines(run_dir / "features.jsonl"),
    )


def load_sealed_run_rows(run_id: str, store_root: Path | str) -> list[dict[str, Any]]:
    """Load flat feature rows for a sealed run."""
    if not is_run_sealed(store_root, run_id):
        raise FileNotFoundError(f"run {run_id!r} has no sealed feature shard")
    return load_rows_from_shard(effective_feature_parquet_path(store_root, run_id))


def load_sealed_campaign(
    campaign_id: str,
    store_root: Path | str,
) -> list[dict[str, Any]]:
    """Concatenate feature rows for all campaign child runs (sealed or JSONL)."""
    root = Path(store_root)
    run_sources: list[tuple[Path, Path]] = [(root, root / "runs")]

    rows: list[dict[str, Any]] = []
    for store_base, runs_root in run_sources:
        if not runs_root.is_dir():
            continue
        for run_dir in sorted(runs_root.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith(f"{campaign_id}_n"):
                continue
            run_id = run_dir.name
            if is_run_sealed(store_base, run_id):
                rows.extend(load_sealed_run_rows(run_id, store_base))
            else:
                rows.extend(flatten_features(run_id, store_base))
    return rows


def build_dataset_manifest(
    campaign_id: str,
    store_root: Path | str,
    *,
    domain_id: str,
    contract_version: str = "0.3",
    feature_catalog_checksum: str = "",
) -> DatasetManifest:
    root = Path(store_root)
    store = Store(root)
    runs_dir = root / "runs"
    if runs_dir.is_dir():
        child_runs = [
            p.name
            for p in sorted(runs_dir.iterdir())
            if p.is_dir() and p.name.startswith(f"{campaign_id}_n")
        ]
    else:
        child_runs = []
    sizes: list[int] = []
    expected_instances = 0
    expected_slices = 0
    actual_instances = 0
    actual_slices = 0
    child_checksums: dict[str, str] = {}
    complete = True

    for run_id in child_runs:
        manifest = store.read_manifest(run_id)
        sizes.append(int(manifest.size))
        expected_instances += int(manifest.n_instances)
        expected_slices += int(manifest.n_instances) * len(manifest.indices)
        inst_lines, feat_lines = _run_row_counts(store, run_id)
        actual_instances += inst_lines
        actual_slices += feat_lines
        child_checksums[run_id] = _run_checksum(store, run_id)
        if inst_lines < manifest.n_instances or feat_lines < manifest.n_instances * len(manifest.indices):
            complete = False

    status = "complete" if complete and child_runs else "incomplete"
    dm = DatasetManifest(
        dataset_id=f"{campaign_id}_pooled",
        campaign_id=campaign_id,
        domain_id=domain_id,
        contract_version=contract_version,
        child_run_ids=child_runs,
        sizes=sorted(set(sizes)),
        expected_instances=expected_instances,
        actual_instances=actual_instances,
        expected_slices=expected_slices,
        actual_slices=actual_slices,
        feature_catalog_checksum=feature_catalog_checksum,
        child_checksums=child_checksums,
        status=status,
        completed_at=utc_now_iso() if status == "complete" else None,
    )
    return dm


def write_dataset_manifest(manifest: DatasetManifest, store_root: Path | str) -> Path:
    root = Path(store_root)
    path = root / "datasets" / f"{manifest.dataset_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, manifest.to_dict())
    return path


def load_dataset_manifest(dataset_id: str, store_root: Path | str) -> DatasetManifest:
    path = Path(store_root) / "datasets" / f"{dataset_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return DatasetManifest(**data)


def q_upper_flat(q: np.ndarray) -> np.ndarray:
    """Upper-triangular flatten of a symmetric Q matrix."""
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError(f"Q must be square 2D, got shape {q.shape}")
    idx = np.triu_indices(q.shape[0])
    return q[idx]


def build_feature_matrix_from_rows(
    rows: list[dict[str, Any]],
    *,
    columns: list[str] | None = None,
    target: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build (X, column_names) from already-flattened rows (no store I/O)."""
    from rde.analyze.feature_table import FeatureTable

    table = FeatureTable.from_rows(rows, columns=columns, target=target)
    return table.data, table.columns


def load_feature_matrix(
    run_id: str,
    store_root: Path | str,
    *,
    columns: list[str] | None = None,
    target: str | None = None,
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    """Return (X, column_names, raw_rows) with NaN for missing values."""
    rows = flatten_features(run_id, store_root)
    mat, cols = build_feature_matrix_from_rows(rows, columns=columns, target=target)
    return mat, cols, rows


def load_campaign_matrix(
    campaign_id: str,
    store_root: Path | str,
    *,
    target: str = "metric.representation_complexity",
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    """Concatenate all runs whose directory name starts with campaign_id_n."""
    all_rows = load_sealed_campaign(campaign_id, store_root)
    cols = numeric_columns(all_rows, exclude=set()) if all_rows else []
    if target not in cols and all_rows:
        cols.append(target)
    if not all_rows:
        return np.zeros((0, 0)), [], []
    mat, _ = build_feature_matrix_from_rows(all_rows, columns=cols)
    return mat, cols, all_rows


def load_parquet_features(path: Path | str) -> list[dict[str, Any]]:
    """Load a flattened feature table exported by ``rde export --format parquet``."""
    return load_rows_from_shard(path)


def load_rows_from_source(
    *,
    run_id: str | None = None,
    store_root: Path | str | None = None,
    campaign_id: str | None = None,
    parquet_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Unified loader: single run, campaign concat, or Parquet file."""
    if parquet_path is not None:
        return load_rows_from_shard(parquet_path)
    if campaign_id is not None:
        if store_root is None:
            raise ValueError("store_root required with campaign_id")
        return load_sealed_campaign(campaign_id, store_root)
    if run_id is not None and store_root is not None:
        return flatten_features(run_id, store_root)
    raise ValueError("Provide run_id+store_root, campaign_id+store_root, or parquet_path")


def split_rows_by_size(
    rows: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        size = row.get("size")
        if size is not None:
            groups.setdefault(int(size), []).append(row)
    return groups


@dataclass
class InstanceQBatch:
    """One Q vector per instance with an aggregated complexity target."""

    instance_ids: list[str]
    q_vectors: np.ndarray
    targets: np.ndarray
    sizes: np.ndarray


def load_instance_q_batch(
    run_id: str,
    store_root: Path | str,
    *,
    target: str = "metric.representation_complexity",
) -> InstanceQBatch:
    """Load flat upper-triangular Q per instance; target = mean metric over r slices."""
    from rde.io.store import Store

    store = Store(store_root)
    instances = {inst.instance_id: inst for inst in store.read_instances(run_id)}
    feature_rows = store.read_features(run_id)

    target_by_instance: dict[str, list[float]] = {}
    for row in feature_rows:
        iid = str(row.get("instance_id", ""))
        val = row.get("metrics", {}).get(target.removeprefix("metric."), None)
        if val is None:
            val = row.get(target)
        if iid and val is not None and np.isfinite(float(val)):
            target_by_instance.setdefault(iid, []).append(float(val))

    q_vectors: list[np.ndarray] = []
    targets: list[float] = []
    sizes: list[int] = []
    instance_ids: list[str] = []

    for iid, inst in instances.items():
        q: np.ndarray | None = None
        try:
            q = store.load_array(run_id, iid, "primitive_Q")
        except (FileNotFoundError, OSError):
            q_param = inst.params.get("Q")
            if q_param is not None:
                q = np.asarray(q_param, dtype=float)
        if q is None:
            continue
        if iid not in target_by_instance:
            continue
        q_vectors.append(q_upper_flat(q))
        targets.append(float(np.mean(target_by_instance[iid])))
        sizes.append(int(inst.size))
        instance_ids.append(iid)

    if not q_vectors:
        return InstanceQBatch([], np.zeros((0, 0)), np.zeros(0), np.zeros(0, dtype=int))

    max_dim = max(v.size for v in q_vectors)
    padded = np.full((len(q_vectors), max_dim), np.nan, dtype=float)
    for i, vec in enumerate(q_vectors):
        padded[i, : vec.size] = vec
    return InstanceQBatch(
        instance_ids=instance_ids,
        q_vectors=padded,
        targets=np.array(targets, dtype=float),
        sizes=np.array(sizes, dtype=int),
    )


@dataclass
class StatePairBatch:
    """Consecutive bound-normalized state vectors Ψ_r → Ψ_{r+1}."""

    prev_states: np.ndarray
    next_states: np.ndarray
    instance_ids: list[str]
    r_indices: list[int]


@dataclass
class RepresentationBatch:
    """Per-slice (Q, r) inputs with bound-normalized state Ψ_r for encoder/decoder search."""

    instance_ids: list[str]
    r_indices: np.ndarray
    sizes: np.ndarray
    q_vectors: np.ndarray
    state_vectors: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.state_vectors.shape[0]) if self.state_vectors.size else 0

    @property
    def state_dim(self) -> int:
        return int(self.state_vectors.shape[1]) if self.state_vectors.ndim == 2 and self.state_vectors.size else 0


def _array_name_from_ref(array_ref: str) -> str:
    name = Path(array_ref).name
    if name.endswith(".npz"):
        name = name[: -len(".npz")]
    return name


def load_state_pair_batch(
    run_id: str,
    store_root: Path | str,
) -> StatePairBatch:
    """Load consecutive slice state vectors from NPZ sidecars."""
    from rde.io.store import Store

    store = Store(store_root)
    by_instance: dict[str, list[tuple[int, str]]] = {}
    for row in store.read_features(run_id):
        iid = str(row.get("instance_id", ""))
        r = row.get("family_index")
        ref = row.get("array_ref")
        if not iid or r is None or not ref:
            continue
        by_instance.setdefault(iid, []).append((int(r), str(ref)))

    prev_list: list[np.ndarray] = []
    next_list: list[np.ndarray] = []
    instance_ids: list[str] = []
    r_indices: list[int] = []

    for iid, entries in by_instance.items():
        entries.sort(key=lambda t: t[0])
        for i in range(len(entries) - 1):
            r0, ref0 = entries[i]
            r1, ref1 = entries[i + 1]
            try:
                name0 = _array_name_from_ref(ref0)
                name1 = _array_name_from_ref(ref1)
                s0 = store.load_array(run_id, iid, name0)
                s1 = store.load_array(run_id, iid, name1)
            except (FileNotFoundError, OSError):
                continue
            if s0.shape != s1.shape:
                continue
            prev_list.append(np.asarray(s0, dtype=float).ravel())
            next_list.append(np.asarray(s1, dtype=float).ravel())
            instance_ids.append(iid)
            r_indices.append(r0)

    if not prev_list:
        return StatePairBatch(np.zeros((0, 0)), np.zeros((0, 0)), [], [])

    dim = max(v.size for v in prev_list)
    prev = np.full((len(prev_list), dim), np.nan, dtype=float)
    nxt = np.full((len(next_list), dim), np.nan, dtype=float)
    for i, (a, b) in enumerate(zip(prev_list, next_list)):
        prev[i, : a.size] = a
        nxt[i, : b.size] = b
    return StatePairBatch(prev, nxt, instance_ids, r_indices)


def load_representation_batch(
    run_id: str,
    store_root: Path | str,
    *,
    target: str = "metric.representation_complexity",
) -> RepresentationBatch:
    """Load (Q, r) encoder inputs and Ψ_r state vectors for Phase 6 representation search."""
    from rde.io.store import Store

    store = Store(store_root)
    instances = {inst.instance_id: inst for inst in store.read_instances(run_id)}

    q_cache: dict[str, np.ndarray] = {}
    q_vectors: list[np.ndarray] = []
    states: list[np.ndarray] = []
    instance_ids: list[str] = []
    r_indices: list[int] = []
    sizes: list[int] = []

    for row in store.read_features(run_id):
        iid = str(row.get("instance_id", ""))
        r = row.get("family_index")
        ref = row.get("array_ref")
        inst = instances.get(iid)
        if not iid or r is None or not ref or inst is None:
            continue
        q = q_cache.get(iid)
        if q is None:
            try:
                q = store.load_array(run_id, iid, "primitive_Q")
            except (FileNotFoundError, OSError):
                q_param = inst.params.get("Q")
                if q_param is not None:
                    q = np.asarray(q_param, dtype=float)
            if q is not None:
                q_cache[iid] = q
        if q is None:
            continue
        try:
            name = _array_name_from_ref(str(ref))
            state = np.asarray(store.load_array(run_id, iid, name), dtype=float).ravel()
        except (FileNotFoundError, OSError):
            continue
        q_vectors.append(q_upper_flat(q))
        states.append(state)
        instance_ids.append(iid)
        r_indices.append(int(r))
        sizes.append(int(inst.size))

    if not states:
        return RepresentationBatch([], np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros((0, 0)), np.zeros((0, 0)))

    q_dim = max(v.size for v in q_vectors)
    state_dim = max(v.size for v in states)
    q_mat = np.full((len(states), q_dim), np.nan, dtype=float)
    s_mat = np.full((len(states), state_dim), np.nan, dtype=float)
    for i, (qv, sv) in enumerate(zip(q_vectors, states)):
        q_mat[i, : qv.size] = qv
        s_mat[i, : sv.size] = sv
    return RepresentationBatch(
        instance_ids=instance_ids,
        r_indices=np.array(r_indices, dtype=int),
        sizes=np.array(sizes, dtype=int),
        q_vectors=q_mat,
        state_vectors=s_mat,
    )


@dataclass
class FeatureDataset:
    """Simple in-memory dataset for Phase 4 trainers."""

    features: np.ndarray
    target: np.ndarray
    columns: list[str]
    rows: list[dict[str, Any]]

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def batches(self, batch_size: int = 64) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(self)
        if n == 0:
            return iter(())
        bs = max(1, batch_size)
        for start in range(0, n, bs):
            end = min(start + bs, n)
            yield self.features[start:end], self.target[start:end]


def build_feature_dataset(
    run_id: str,
    store_root: Path | str,
    *,
    target: str,
    feature_columns: list[str] | None = None,
    exclude_prefixes: tuple[str, ...] = ("metric.", "run_id"),
    rows: list[dict[str, Any]] | None = None,
    X: np.ndarray | None = None,
    cols: list[str] | None = None,
    domain_id: str | None = None,
) -> FeatureDataset:
    from rde.analyze.tables import contract_excluded_columns

    if rows is None or X is None or cols is None:
        X, cols, rows = load_feature_matrix(run_id, store_root, target=target)
    if domain_id is None:
        try:
            domain_id = Store(store_root).read_manifest(run_id).domain_id
        except (FileNotFoundError, OSError, ValueError, KeyError):
            domain_id = None
    excluded = contract_excluded_columns(rows, domain_id)
    use_cols = feature_columns or [
        c
        for c in cols
        if c != target and c not in excluded and not any(c.startswith(p) for p in exclude_prefixes)
    ]
    idx = [cols.index(c) for c in use_cols if c in cols]
    target_arr = np.array([row.get(target, float("nan")) for row in rows], dtype=float)
    return FeatureDataset(X[:, idx], target_arr, use_cols, rows)


def broadcast_instance_latents(
    rows: list[dict[str, Any]],
    instance_ids: list[str],
    latent_codes: np.ndarray,
    columns: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Expand per-instance latent codes to one row per (instance, r) feature row."""
    if not rows or latent_codes.size == 0:
        return np.zeros((len(rows), latent_codes.shape[1] if latent_codes.ndim == 2 else 0)), columns
    lookup = {iid: latent_codes[i] for i, iid in enumerate(instance_ids)}
    dim = latent_codes.shape[1]
    out = np.full((len(rows), dim), np.nan, dtype=float)
    for i, row in enumerate(rows):
        code = lookup.get(str(row.get("instance_id", "")))
        if code is not None:
            out[i] = code
    return out, columns


def build_torch_dataset(
    dataset: FeatureDataset,
) -> Any | None:
    """Optional PyTorch ``TensorDataset`` wrapper; None if torch unavailable."""
    try:
        import torch
        from torch.utils.data import TensorDataset
    except ImportError:
        return None
    x = torch.tensor(np.nan_to_num(dataset.features, nan=0.0), dtype=torch.float32)
    y = torch.tensor(np.nan_to_num(dataset.target, nan=0.0), dtype=torch.float32)
    return TensorDataset(x, y)
