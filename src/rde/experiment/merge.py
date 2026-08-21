"""Assemble per-size pipeline runs into one cross-N run for discovery.

`run_discovery` loads a single ``run_id``. Cross-N checks (sign stability,
train-small / test-large extrapolation) need every size in one dataset, so this
concatenates the per-size shards into a combined run the discovery loop can
consume directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from rde.experiment.gate import ExperimentPreflightError
from rde.io.store import RunManifest, Store

_SHARDS = ("instances.jsonl", "instance_features.jsonl", "features.jsonl")

DEFAULT_PREDICTOR_PREFIXES = ("matrix.", "graph.")


def predictor_columns(
    rows: Sequence[dict[str, Any]],
    predictor_prefixes: Sequence[str],
) -> list[str]:
    """Column names in flat rows whose keys start with one of ``predictor_prefixes``."""
    prefixes = tuple(predictor_prefixes)
    cols: set[str] = set()
    for row in rows:
        for key in row:
            if any(key.startswith(prefix) for prefix in prefixes):
                cols.add(key)
    return sorted(cols)


def validate_clean_predictors(
    rows: Sequence[dict[str, Any]],
    *,
    predictor_prefixes: Sequence[str],
    target_metric: str,
) -> None:
    """Raise before discovery if leak-cleaning removed every predictor column."""
    if not rows:
        raise ExperimentPreflightError(
            "leak-clean discovery rows are empty — merge or write_clean_discovery_run failed."
        )
    preds = predictor_columns(rows, predictor_prefixes)
    if not preds:
        raise ExperimentPreflightError(
            "leak-clean copy has no predictor columns after filtering. "
            f"Expected keys matching {list(predictor_prefixes)!r}; got only metadata columns "
            f"like {sorted(rows[0])[:12]}. Pass the domain's predictor_prefixes to "
            "write_clean_discovery_run() — matrix./graph. is not universal "
            "(e.g. hsp_functions needs hsp_sample./landscape.)."
        )
    target_present = any(
        row.get(target_metric) is not None or row.get(target_metric.split("metric.", 1)[-1]) is not None
        for row in rows
    )
    if not target_present:
        raise ExperimentPreflightError(
            f"leak-clean rows are missing target {target_metric!r}."
        )


def prepare_leak_clean_discovery(
    store_root: Path | str,
    run_ids: Sequence[str],
    merged_run_id: str,
    clean_run_id: str,
    *,
    target_metric: str,
    predictor_prefixes: Sequence[str] = DEFAULT_PREDICTOR_PREFIXES,
    drop_substrings: Sequence[str] = (),
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge per-size runs, write a leak-clean copy, validate predictors survive."""
    from rde.analyze.query import flatten_features

    raw_run = merge_runs_for_discovery(store_root, run_ids, merged_run_id)
    clean_run = write_clean_discovery_run(
        store_root,
        raw_run,
        clean_run_id,
        target_metric=target_metric,
        predictor_prefixes=predictor_prefixes,
        drop_substrings=drop_substrings,
    )
    raw_rows = flatten_features(raw_run, store_root)
    clean_rows = flatten_features(clean_run, store_root)
    validate_clean_predictors(
        clean_rows,
        predictor_prefixes=predictor_prefixes,
        target_metric=target_metric,
    )
    return clean_run, raw_rows, clean_rows


def write_clean_discovery_run(
    store_root: Path | str,
    source_run_id: str,
    clean_run_id: str,
    *,
    target_metric: str,
    predictor_prefixes: Sequence[str] = DEFAULT_PREDICTOR_PREFIXES,
    drop_substrings: Sequence[str] = (),
    keep_scalar_keys: Sequence[str] = ("generator",),
) -> str:
    """Write a leak-clean copy of a run for the discovery loop.

    Keeps only contract-declared structural predictors (``matrix.*``/``graph.*``
    instance scalars) plus the target metric. Slice descriptors are dropped:
    they are computed on the outcome trajectory itself, so feeding them to
    discovery would rediscover the target rather than predict it. Likewise all
    non-target metrics, which are typically exact functions of the target.
    """
    store = Store(store_root)
    src_dir = store.run_dir(source_run_id)
    out_dir = store.run_dir(clean_run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefixes = tuple(predictor_prefixes)
    dropped = tuple(drop_substrings)
    keep_keys = tuple(keep_scalar_keys)
    target_key = target_metric.split("metric.", 1)[-1]

    def _keep_scalar(key: str) -> bool:
        if key in keep_keys:
            return True
        if not key.startswith(prefixes):
            return False
        return not any(token in key for token in dropped)

    # instances.jsonl passes through unchanged (provenance, not predictors).
    src_instances = src_dir / "instances.jsonl"
    if src_instances.is_file():
        (out_dir / "instances.jsonl").write_text(
            src_instances.read_text(encoding="utf-8"), encoding="utf-8"
        )

    with (out_dir / "instance_features.jsonl").open("w", encoding="utf-8") as dst:
        for line in _iter_lines(src_dir / "instance_features.jsonl"):
            row = json.loads(line)
            scalars = row.get("scalars", {}) or {}
            row["scalars"] = {k: v for k, v in scalars.items() if _keep_scalar(k)}
            dst.write(json.dumps(row) + "\n")

    with (out_dir / "features.jsonl").open("w", encoding="utf-8") as dst:
        for line in _iter_lines(src_dir / "features.jsonl"):
            row = json.loads(line)
            row["descriptors"] = {}
            metrics = row.get("metrics", {}) or {}
            row["metrics"] = (
                {target_key: metrics[target_key]} if target_key in metrics else {}
            )
            dst.write(json.dumps(row) + "\n")

    src_manifest = store.read_manifest(source_run_id)
    store.write_manifest(
        RunManifest(
            run_id=clean_run_id,
            domain_id=src_manifest.domain_id,
            n_instances=src_manifest.n_instances,
            size=src_manifest.size,
            seed=src_manifest.seed,
            indices=list(src_manifest.indices),
            descriptor_names=[],
            metric_names=[target_key],
            extra={
                **dict(src_manifest.extra or {}),
                "leak_clean_of": source_run_id,
                "predictor_prefixes": list(prefixes),
                "drop_substrings": list(dropped),
            },
            provenance={
                **dict(src_manifest.provenance or {}),
                "leak_clean_of": source_run_id,
            },
        )
    )
    return clean_run_id


def _iter_lines(path: Path):
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield line


def merge_runs_for_discovery(
    store_root: Path | str,
    run_ids: Sequence[str],
    combined_run_id: str,
) -> str:
    """Concatenate per-size runs into ``combined_run_id``; return that id."""
    if not run_ids:
        raise ValueError("merge_runs_for_discovery requires at least one run_id")

    store = Store(store_root)
    out_dir = store.run_dir(combined_run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    for shard in _SHARDS:
        with (out_dir / shard).open("w", encoding="utf-8") as dst:
            for run_id in run_ids:
                src = store.run_dir(run_id) / shard
                if not src.is_file():
                    continue
                with src.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            dst.write(line if line.endswith("\n") else line + "\n")

    manifests = [store.read_manifest(r) for r in run_ids]
    base = manifests[0]
    descriptor_names: list[str] = []
    metric_names: list[str] = []
    for m in manifests:
        for name in m.descriptor_names:
            if name not in descriptor_names:
                descriptor_names.append(name)
        for name in m.metric_names:
            if name not in metric_names:
                metric_names.append(name)

    merged = RunManifest(
        run_id=combined_run_id,
        domain_id=base.domain_id,
        n_instances=sum(int(m.n_instances) for m in manifests),
        size=max(int(m.size) for m in manifests),
        seed=int(base.seed),
        indices=list(base.indices),
        descriptor_names=descriptor_names,
        metric_names=metric_names,
        extra={
            **dict(base.extra or {}),
            "merged_from": list(run_ids),
            "merged_sizes": sorted({int(m.size) for m in manifests}),
        },
        provenance={
            **dict(base.provenance or {}),
            "merged_from": list(run_ids),
        },
    )
    store.write_manifest(merged)
    return combined_run_id


def write_run_subset(
    store_root: Path | str,
    source_run_id: str,
    dest_run_id: str,
    *,
    keep_instance_ids: Sequence[str],
) -> str:
    """Copy a run, keeping only the listed instance ids (discovery-fold shards)."""
    keep = set(keep_instance_ids)
    if not keep:
        raise ExperimentPreflightError(
            "write_run_subset requires at least one instance_id to keep."
        )
    store = Store(store_root)
    src_dir = store.run_dir(source_run_id)
    out_dir = store.run_dir(dest_run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    for shard in _SHARDS:
        src = src_dir / shard
        dst = out_dir / shard
        if not src.is_file():
            continue
        with src.open("r", encoding="utf-8") as fh, dst.open("w", encoding="utf-8") as out:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("instance_id", "")) in keep:
                    out.write(line if line.endswith("\n") else line + "\n")

    src_manifest = store.read_manifest(source_run_id)
    store.write_manifest(
        RunManifest(
            run_id=dest_run_id,
            domain_id=src_manifest.domain_id,
            n_instances=len(keep),
            size=src_manifest.size,
            seed=src_manifest.seed,
            indices=list(src_manifest.indices),
            descriptor_names=list(src_manifest.descriptor_names),
            metric_names=list(src_manifest.metric_names),
            extra={
                **dict(src_manifest.extra or {}),
                "subset_of": source_run_id,
                "n_kept_instances": len(keep),
            },
            provenance={
                **dict(src_manifest.provenance or {}),
                "subset_of": source_run_id,
            },
        )
    )
    return dest_run_id
