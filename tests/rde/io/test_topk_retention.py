"""Phase-5 long-term top-K retention tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rde.analyze.query import flatten_features, load_rows_from_shard, numeric_columns
from rde.discovery.context import DiscoveryContext
from rde.io.seal import seal_run
from rde.io.topk_retention import (
    estimate_topk_retention,
    is_run_topk_retained,
    read_topk_metadata,
    retain_topk_run,
    topk_parquet_path,
)
from rde.runtime.pipeline import RunConfig, run_pipeline
from rde.runtime.targets import resolve_target_for_run
from tests.rde.helpers import toy_registry


def _make_run(tmp_path: Path, run_id: str = "topk_run") -> None:
    run_pipeline(
        RunConfig(
            domain_id="synthetic_poly",
            n_instances=2,
            size=4,
            seed=7,
            indices=[1, 2],
            store_root=tmp_path,
            run_id=run_id,
            save_arrays=False,
        ),
        registry=toy_registry(),
    )


def _discovery_report(run_id: str, target: str, ranked_columns: list[str]) -> dict:
    return {
        "run_id": run_id,
        "target": target,
        "top_correlations": [{"column": column, "pearson_r": 0.9} for column in ranked_columns],
        "metric_candidates": [],
        "promoted_conjectures": [],
    }


def test_retain_topk_writes_verified_shard_and_drops_full_parquet(tmp_path: Path):
    pytest.importorskip("pyarrow")
    _make_run(tmp_path)
    run_id = "topk_run"
    target = resolve_target_for_run(run_id, tmp_path)
    full_rows = flatten_features(run_id, tmp_path)
    all_numeric = numeric_columns(full_rows, exclude=set())
    ranked = all_numeric[:3]
    report_path = tmp_path / "discovery" / f"{run_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(_discovery_report(run_id, target, ranked)),
        encoding="utf-8",
    )

    seal_run(tmp_path, run_id)
    run_dir = tmp_path / "runs" / run_id
    full_path = run_dir / "sealed" / "features.parquet"
    assert full_path.exists()

    record = retain_topk_run(
        tmp_path,
        run_id,
        target=target,
        top_k=3,
        discovery_report_path=report_path,
        delete_full_shard=True,
    )

    topk_path = topk_parquet_path(tmp_path, run_id)
    assert record["status"] == "retained"
    assert record["full_shard_removed"] is True
    assert not full_path.exists()
    assert topk_path.exists()
    assert record["non_claim"]
    assert is_run_topk_retained(tmp_path, run_id)

    topk_rows = load_rows_from_shard(topk_path)
    assert len(topk_rows) == len(full_rows)
    kept = set(record["retained_columns"])
    assert target in kept
    for column in ranked:
        assert column in kept
    for row in topk_rows:
        assert set(row).issubset(kept | {"generator"})


def test_retain_topk_failure_does_not_delete_full_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("pyarrow")
    _make_run(tmp_path, "topk_fail")
    run_id = "topk_fail"
    target = resolve_target_for_run(run_id, tmp_path)
    ranked = [
        column
        for column in numeric_columns(flatten_features(run_id, tmp_path), exclude=set())
        if column != target
    ][:1]
    assert ranked
    seal_run(tmp_path, run_id)
    full_path = tmp_path / "runs" / run_id / "sealed" / "features.parquet"

    def fail_project(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("projection failed")

    monkeypatch.setattr("rde.io.topk_retention._project_parquet", fail_project)
    with pytest.raises(RuntimeError, match="projection failed"):
        retain_topk_run(
            tmp_path,
            run_id,
            target=target,
            ranked_columns=ranked,
            delete_full_shard=True,
        )

    assert full_path.exists()
    metadata = read_topk_metadata(tmp_path, run_id)
    assert metadata is None or metadata.get("status") != "retained"


def test_discovery_context_loads_topk_only_run(tmp_path: Path):
    pytest.importorskip("pyarrow")
    _make_run(tmp_path, "topk_discover")
    run_id = "topk_discover"
    target = resolve_target_for_run(run_id, tmp_path)
    full_rows = flatten_features(run_id, tmp_path)
    ranked = numeric_columns(full_rows, exclude=set())[:2]
    report_path = tmp_path / "discovery" / f"{run_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(_discovery_report(run_id, target, ranked)),
        encoding="utf-8",
    )
    seal_run(tmp_path, run_id)
    retain_topk_run(
        tmp_path,
        run_id,
        target=target,
        top_k=2,
        discovery_report_path=report_path,
        delete_full_shard=True,
    )

    ctx = DiscoveryContext.load(run_id, tmp_path)
    assert len(ctx.rows) == 4
    assert target in ctx.feature_table().columns
    for column in ranked:
        assert column in ctx.feature_table().columns


def test_estimate_topk_retention_reports_columns_and_bytes(tmp_path: Path):
    pytest.importorskip("pyarrow")
    _make_run(tmp_path, "topk_plan")
    run_id = "topk_plan"
    target = resolve_target_for_run(run_id, tmp_path)
    seal_run(tmp_path, run_id)
    full_bytes = (tmp_path / "runs" / run_id / "sealed" / "features.parquet").stat().st_size
    plan = estimate_topk_retention(
        tmp_path,
        run_id,
        target=target,
        top_k=2,
        ranked_columns=numeric_columns(flatten_features(run_id, tmp_path), exclude=set())[:2],
    )
    assert plan["action"] == "would_retain_topk"
    assert plan["bytes_freed"] == full_bytes
    assert len(plan["retained_columns"]) >= 3


def test_retained_numeric_columns_match_full_shard_subset(tmp_path: Path):
    pytest.importorskip("pyarrow")
    _make_run(tmp_path, "topk_parity")
    run_id = "topk_parity"
    target = resolve_target_for_run(run_id, tmp_path)
    full_rows = flatten_features(run_id, tmp_path)
    ranked = numeric_columns(full_rows, exclude=set())[:2]
    seal_run(tmp_path, run_id)
    record = retain_topk_run(
        tmp_path,
        run_id,
        target=target,
        top_k=2,
        ranked_columns=ranked,
        delete_full_shard=False,
    )
    topk_rows = load_rows_from_shard(topk_parquet_path(tmp_path, run_id))
    kept_numeric = [
        column
        for column in record["retained_columns"]
        if column.startswith("metric.") or column.startswith("gen.")
    ]
    for column in kept_numeric:
        full_vals = np.asarray([row.get(column, float("nan")) for row in full_rows], dtype=float)
        topk_vals = np.asarray([row.get(column, float("nan")) for row in topk_rows], dtype=float)
        np.testing.assert_allclose(full_vals, topk_vals, equal_nan=True)
