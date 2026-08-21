"""Load, summarize, and analyze RDE run data."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from rde.analyze.tables import group_indices_by_size, numeric_columns
from rde.expression import pearson_r, regression_r_squared
from rde.features.numeric import safe_pearson_r

_FLAT_META_KEYS = frozenset(
    {
        "run_id",
        "instance_id",
        "domain_id",
        "size",
        "seed",
        "family_index",
        "slice_kind",
        "generator",
    }
)


def load_rows_from_shard(path: Path | str) -> list[dict[str, Any]]:
    """Load flat feature rows from a sealed or exported Parquet shard."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Parquet load requires pyarrow: install the rde-parquet extra"
        ) from exc
    table = pq.read_table(Path(path))
    return table.to_pylist()


def _iter_jsonl_features(
    run_id: str, store_root: Path | str
) -> Iterator[dict[str, Any]]:
    from rde.io.store import Store

    store = Store(store_root)
    try:
        inst_scalars: dict[str, dict[str, Any]] = {}
        for row in store.iter_instance_features(run_id):
            inst_scalars[row["instance_id"]] = row.get("scalars", {})
        for row in store.iter_features(run_id):
            instance_id = row.get("instance_id", "")
            instance_values = inst_scalars.get(instance_id, {})
            flat = {
                "run_id": row.get("run_id"),
                "instance_id": instance_id,
                "domain_id": row.get("domain_id"),
                "size": row.get("size"),
                "seed": row.get("seed"),
                "family_index": row.get("family_index"),
                "slice_kind": row.get("slice_kind"),
            }
            flat.update(instance_values)
            flat.update(row.get("descriptors", {}))
            if row.get("generator") is not None:
                flat["generator"] = row["generator"]
            elif instance_values.get("generator") is not None:
                flat["generator"] = instance_values["generator"]
            for key, value in row.get("metrics", {}).items():
                flat[f"metric.{key}"] = value
            yield flat
    finally:
        store.close()


def iter_flatten_features(
    run_id: str, store_root: Path | str
) -> Iterator[dict[str, Any]]:
    """Yield flat rows while retaining only the instance-scalar join map.

    Sealed Parquet shards are preferred when present so discovery and export
    paths agree after JSONL cleanup.  When a sealed run still has incremental
    ``features.jsonl`` rows from a resumed session, shard rows and new JSONL
    rows are concatenated (disjoint by the completion contract).
    """
    from rde.io.seal import is_run_sealed
    from rde.io.topk_retention import effective_feature_parquet_path

    root = Path(store_root)
    run_dir = root / "runs" / run_id
    jsonl_path = run_dir / "features.jsonl"
    has_jsonl = jsonl_path.is_file() and jsonl_path.stat().st_size > 0
    if is_run_sealed(root, run_id):
        sealed_rows = list(load_rows_from_shard(effective_feature_parquet_path(root, run_id)))
        if not has_jsonl:
            yield from sealed_rows
            return
        sealed_keys = {
            (row.get("instance_id"), row.get("family_index")) for row in sealed_rows
        }
        yield from sealed_rows
        for row in _iter_jsonl_features(run_id, root):
            key = (row.get("instance_id"), row.get("family_index"))
            if key not in sealed_keys:
                yield row
        return
    yield from _iter_jsonl_features(run_id, root)


def flatten_features(run_id: str, store_root: Path | str) -> list[dict[str, Any]]:
    """Return flat dict rows merging descriptors, instance scalars, and metrics."""
    return list(iter_flatten_features(run_id, store_root))


def summarize_run(run_id: str, store_root: Path | str) -> dict[str, Any]:
    from rde.io.seal import is_run_sealed, read_sealed_metadata
    from rde.io.store import Store

    store = Store(store_root)
    manifest = store.read_manifest(run_id)
    if is_run_sealed(store_root, run_id):
        metadata = read_sealed_metadata(store_root, run_id) or {}
        rows = metadata.get("rows", {})
        n_feature_rows = int(rows.get("sealed_features", rows.get("features", 0)))
        n_instances = int(rows.get("instance_features", rows.get("instances", 0)))
    else:
        features = store.read_features(run_id)
        instance_features = store.read_instance_features(run_id)
        n_feature_rows = len(features)
        n_instances = len(instance_features)
    sizes = sorted({int(manifest.size)} if manifest.size is not None else set())
    return {
        "run_id": run_id,
        "domain_id": manifest.domain_id,
        "n_instances": n_instances,
        "n_feature_rows": n_feature_rows,
        "sizes": sizes,
        "indices": manifest.indices,
    }


def cross_n_sign_stability(
    rows: list[dict[str, Any]],
    column: str,
    target: str,
    *,
    groups: dict[Any, np.ndarray] | None = None,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
) -> float:
    """Fraction of size groups whose correlation sign matches the global sign.

    ``groups`` lets callers pass a precomputed ``group_indices_by_size(rows)``
    when scoring many columns/candidates against the same ``rows`` in one
    pass, instead of rescanning ``rows`` per call.

    ``x`` and ``y`` let columnar callers reuse arrays they already materialized
    instead of rebuilding both vectors from row dictionaries.
    """
    if x is None:
        x = np.array([row.get(column, float("nan")) for row in rows], dtype=float)
    else:
        x = np.asarray(x, dtype=float)
    if y is None:
        y = np.array([row.get(target, float("nan")) for row in rows], dtype=float)
    else:
        y = np.asarray(y, dtype=float)
    global_r = pearson_r(x, y)
    if not np.isfinite(global_r) or global_r == 0.0:
        return float("nan")
    if groups is None:
        groups = group_indices_by_size(rows)
    if len(groups) < 2:
        return float("nan")
    signs: list[float] = []
    global_sign = np.sign(global_r)
    for idxs in groups.values():
        if len(idxs) < 3:
            continue
        xv = x[idxs]
        yv = y[idxs]
        r = pearson_r(xv, yv)
        if np.isfinite(r) and r != 0.0:
            signs.append(float(np.sign(r) == global_sign))
    return float(np.mean(signs)) if signs else float("nan")


def correlate_with_target(
    rows: list[dict[str, Any]],
    target: str,
    *,
    min_abs_r: float = 0.3,
    exclude: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Regression R² and Pearson r of each numeric column against target."""
    from rde.analyze.feature_table import FeatureTable

    return FeatureTable.from_rows(rows, target=target).correlate_with_target(
        target,
        min_abs_r=min_abs_r,
        exclude=exclude,
    )


def cross_n_report(
    rows: list[dict[str, Any]],
    target: str,
    *,
    min_abs_r: float = 0.2,
) -> list[dict[str, Any]]:
    """Columns correlated with target, ranked by cross-size sign stability."""
    hits = correlate_with_target(rows, target, min_abs_r=min_abs_r)
    stable = [h for h in hits if np.isfinite(h.get("cross_n_sign_stability", float("nan")))]
    stable.sort(
        key=lambda d: (
            -float(d.get("cross_n_sign_stability", 0.0)),
            -abs(d["pearson_r"]),
        )
    )
    stable_ids = {id(h) for h in stable}
    unstable = [h for h in hits if id(h) not in stable_ids]
    return stable + unstable


def correlation_matrix(
    rows: list[dict[str, Any]],
    columns: list[str],
) -> dict[str, Any]:
    """Pearson correlation matrix for selected numeric columns."""
    if not rows or not columns:
        return {"columns": [], "matrix": []}
    arrays = {
        col: np.array([row.get(col, float("nan")) for row in rows], dtype=float) for col in columns
    }
    n = len(columns)
    mat = np.full((n, n), float("nan"))
    for i, ci in enumerate(columns):
        for j, cj in enumerate(columns):
            if j < i:
                mat[i, j] = mat[j, i]
                continue
            xi, xj = arrays[ci], arrays[cj]
            mask = np.isfinite(xi) & np.isfinite(xj)
            if mask.sum() < 3:
                continue
            xv, yv = xi[mask], xj[mask]
            if float(np.std(xv)) == 0.0 or float(np.std(yv)) == 0.0:
                mat[i, j] = 1.0 if i == j else float("nan")
            else:
                mat[i, j] = safe_pearson_r(xv, yv)
            if i != j:
                mat[j, i] = mat[i, j]
    return {"columns": columns, "matrix": mat.tolist()}


def top_correlation_matrix(
    rows: list[dict[str, Any]],
    target: str,
    *,
    top_k: int = 12,
    min_abs_r: float = 0.2,
) -> dict[str, Any]:
    """Correlation matrix among target and top correlated columns."""
    hits = correlate_with_target(rows, target, min_abs_r=min_abs_r)
    cols = [target] + [h["column"] for h in hits[:top_k] if h["column"] != target]
    return correlation_matrix(rows, cols)


def distribution_summary(
    rows: list[dict[str, Any]],
    column: str,
    *,
    group_by: str | None = "size",
) -> list[dict[str, Any]]:
    """Mean/std/min/max of column, optionally grouped."""
    if group_by:

        def key(row: dict[str, Any]) -> Any:
            return row.get(group_by)

        groups: dict[Any, list[float]] = {}
        for row in rows:
            val = row.get(column)
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                groups.setdefault(key(row), []).append(float(val))
        out: list[dict[str, Any]] = []
        for gkey in sorted(groups, key=lambda k: (str(k), k)):
            vals = np.array(groups[gkey], dtype=float)
            out.append(
                {
                    group_by: gkey,
                    "column": column,
                    "count": int(vals.size),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
            )
        return out

    vals = np.array(
        [float(row[column]) for row in rows if isinstance(row.get(column), (int, float))],
        dtype=float,
    )
    if vals.size == 0:
        return []
    return [
        {
            "column": column,
            "count": int(vals.size),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    ]


def outlier_rows(
    rows: list[dict[str, Any]],
    column: str,
    *,
    z_threshold: float = 2.5,
) -> list[dict[str, Any]]:
    """Return rows where column value exceeds z_threshold standard deviations."""
    vals = np.array([row.get(column, float("nan")) for row in rows], dtype=float)
    mask = np.isfinite(vals)
    if mask.sum() < 3:
        return []
    mu = float(np.mean(vals[mask]))
    sigma = float(np.std(vals[mask]))
    if sigma <= 0:
        return []
    out: list[dict[str, Any]] = []
    for row, val in zip(rows, vals):
        if not math.isfinite(float(val)):
            continue
        z = abs(float(val) - mu) / sigma
        if z >= z_threshold:
            tagged = dict(row)
            tagged["z_score"] = z
            tagged["outlier_column"] = column
            out.append(tagged)
    out.sort(key=lambda r: -float(r["z_score"]))
    return out


def kmeans_clusters(
    rows: list[dict[str, Any]],
    columns: list[str],
    *,
    k: int = 3,
    max_iter: int = 50,
    seed: int = 0,
) -> dict[str, Any]:
    """Simple k-means on selected columns (Phase 2 — hidden class discovery)."""
    if not rows or not columns or k < 1:
        return {"k": k, "labels": [], "centroids": []}
    rng = np.random.default_rng(seed)
    from rde.analyze.feature_table import FeatureTable
    from rde.discovery.impute import impute_matrix

    X = impute_matrix(FeatureTable.from_rows(rows, columns=columns).data)
    n = X.shape[0]
    k = min(k, n)
    if k == 0:
        return {"k": 0, "labels": [], "centroids": []}
    centroids = X[rng.choice(n, size=k, replace=False)]
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([X[labels == j].mean(axis=0) if (labels == j).any() else centroids[j] for j in range(k)])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return {
        "k": k,
        "labels": labels.tolist(),
        "centroids": centroids.tolist(),
        "columns": columns,
    }
