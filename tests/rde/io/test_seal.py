"""Phase-1/2 run sealing, shard loading, and cleanup tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rde.analyze.query import flatten_features, load_rows_from_shard, numeric_columns
from rde.discovery.context import DiscoveryContext
from rde.discovery.datasets import load_campaign_matrix, load_sealed_campaign
from rde.io.seal import (
    effective_instance_count,
    estimate_seal_cleanup,
    is_run_generation_complete,
    refresh_sealed_metadata_counts,
    seal_run,
)
from rde.io.store import Store
from rde.runtime.campaign import CampaignConfig, run_campaign
from rde.runtime.pipeline import RunConfig, run_pipeline
from tests.rde.helpers import toy_registry


def _make_run(tmp_path: Path, run_id: str = "seal_run") -> None:
    run_pipeline(
        RunConfig(
            domain_id="synthetic_poly",
            n_instances=2,
            size=4,
            seed=7,
            indices=[1, 2],
            store_root=tmp_path,
            run_id=run_id,
            save_arrays=True,
        ),
        registry=toy_registry(),
    )


def _assert_rows_numeric_equal(
    left: list[dict],
    right: list[dict],
) -> None:
    assert len(left) == len(right)
    key = lambda row: (row.get("instance_id"), row.get("family_index"))
    left_sorted = sorted(left, key=key)
    right_sorted = sorted(right, key=key)
    for col in numeric_columns(left_sorted, exclude=set()):
        left_vals = np.asarray([row.get(col, float("nan")) for row in left_sorted], dtype=float)
        right_vals = np.asarray([row.get(col, float("nan")) for row in right_sorted], dtype=float)
        np.testing.assert_allclose(left_vals, right_vals, equal_nan=True)
    for col in ("run_id", "instance_id", "domain_id", "slice_kind"):
        assert [row.get(col) for row in left_sorted] == [row.get(col) for row in right_sorted]


def test_seal_writes_verified_snapshot_and_removes_operational_files(
    tmp_path: Path,
):
    _make_run(tmp_path)
    run_dir = tmp_path / "runs" / "seal_run"
    assert (run_dir / "completion.sqlite3").exists()
    assert (run_dir / "arrays").is_dir()

    record = seal_run(tmp_path, "seal_run")

    parquet_path = run_dir / "sealed" / "features.parquet"
    marker_path = run_dir / "sealed" / "sealed.json"
    assert record["status"] == "sealed"
    assert record["rows"]["sealed_features"] == 4
    assert record["parquet"]["sha256"]
    assert record["cleanup"]["jsonl_removed"] is True
    assert parquet_path.exists()
    assert json.loads(marker_path.read_text(encoding="utf-8"))["status"] == "sealed"
    assert not (run_dir / "completion.sqlite3").exists()
    assert not (run_dir / "arrays").exists()
    assert not (run_dir / "features.jsonl").exists()
    assert not (run_dir / "instance_features.jsonl").exists()
    assert not (run_dir / "instances.jsonl").exists()
    assert len(flatten_features("seal_run", tmp_path)) == 4

    resumed_store = Store(tmp_path)
    resumed_store.ensure_completion_index("seal_run")
    assert len(resumed_store.completed_feature_keys("seal_run")) == 4
    resumed_store.close()

    pyarrow = pytest.importorskip("pyarrow.parquet")
    assert pyarrow.ParquetFile(parquet_path).metadata.num_rows == 4


def test_effective_instance_count_uses_instance_features_when_instances_lag():
    stats = {
        "instances_lines": 111026,
        "instance_features_lines": 111111,
        "features_lines": 555555,
    }
    assert effective_instance_count(stats) == 111111


def test_is_run_generation_complete_when_instances_jsonl_lags(tmp_path: Path):
    _make_run(tmp_path, "lagging_instances")
    store = Store(tmp_path)
    run_dir = store.run_dir("lagging_instances")
    instances_path = run_dir / "instances.jsonl"
    instances_path.write_text("", encoding="utf-8")
    store.close()
    assert is_run_generation_complete(
        tmp_path,
        "lagging_instances",
        expected_instances=2,
        expected_slices=2,
    )


def test_refresh_sealed_metadata_counts_updates_complete_flag(tmp_path: Path):
    _make_run(tmp_path, "refresh_meta")
    seal_run(tmp_path, "refresh_meta", delete_jsonl=False)
    store = Store(tmp_path)
    run_dir = store.run_dir("refresh_meta")
    (run_dir / "instances.jsonl").write_text("", encoding="utf-8")
    metadata = refresh_sealed_metadata_counts(tmp_path, "refresh_meta", store=store)
    store.close()
    assert metadata is not None
    assert metadata["complete"] is True
    assert metadata["rows"]["instances"] == 2


def test_finalize_sealed_run_cleans_operational_artifacts_without_reexport(tmp_path: Path):
    _make_run(tmp_path, "finalize_sealed")
    first = seal_run(tmp_path, "finalize_sealed", delete_jsonl=False)
    assert first["complete"] is True
    run_dir = tmp_path / "runs" / "finalize_sealed"
    assert (run_dir / "features.jsonl").exists()
    (run_dir / "completion.sqlite3").write_bytes(b"stale")

    second = seal_run(tmp_path, "finalize_sealed")
    assert second["complete"] is True
    assert second["cleanup"]["jsonl_removed"] is True
    assert not (run_dir / "features.jsonl").exists()
    assert not (run_dir / "instance_features.jsonl").exists()
    assert not (run_dir / "instances.jsonl").exists()
    assert not (run_dir / "completion.sqlite3").exists()
    assert len(flatten_features("finalize_sealed", tmp_path)) == 4


def test_seal_can_retain_jsonl_when_requested(tmp_path: Path):
    _make_run(tmp_path, "keep_jsonl")
    run_dir = tmp_path / "runs" / "keep_jsonl"

    record = seal_run(tmp_path, "keep_jsonl", delete_jsonl=False)

    assert record["status"] == "sealed"
    assert record["cleanup"]["jsonl_removed"] is False
    assert (run_dir / "features.jsonl").exists()
    assert (run_dir / "instance_features.jsonl").exists()
    assert len(flatten_features("keep_jsonl", tmp_path)) == 4


def test_seal_can_retain_arrays_for_raw_array_consumers(tmp_path: Path):
    _make_run(tmp_path, "keep_arrays")
    run_dir = tmp_path / "runs" / "keep_arrays"

    record = seal_run(tmp_path, "keep_arrays", delete_arrays=False)

    assert record["status"] == "sealed"
    assert record["cleanup"]["arrays_retained"] is True
    assert (run_dir / "arrays").is_dir()
    assert not (run_dir / "completion.sqlite3").exists()


def test_seal_failure_does_not_remove_source_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _make_run(tmp_path, "failed_seal")
    run_dir = tmp_path / "runs" / "failed_seal"

    def fail_export(self, run_id, path, **kwargs):
        del self, run_id, kwargs
        Path(path).write_bytes(b"not parquet")
        return Path(path)

    monkeypatch.setattr(Store, "export_features_parquet", fail_export)
    with pytest.raises(RuntimeError, match="Parquet"):
        seal_run(tmp_path, "failed_seal")

    assert (run_dir / "completion.sqlite3").exists()
    assert (run_dir / "arrays").is_dir()
    assert not (run_dir / "sealed" / "sealed.json").exists()


def test_sealed_parquet_matches_jsonl_flatten(tmp_path: Path):
    pytest.importorskip("pyarrow")
    _make_run(tmp_path, "parity_run")
    run_dir = tmp_path / "runs" / "parity_run"
    jsonl_rows = flatten_features("parity_run", tmp_path)
    seal_run(tmp_path, "parity_run", delete_jsonl=False)
    shard_rows = load_rows_from_shard(run_dir / "sealed" / "features.parquet")
    _assert_rows_numeric_equal(jsonl_rows, shard_rows)
    _assert_rows_numeric_equal(flatten_features("parity_run", tmp_path), shard_rows)


def test_discovery_context_loads_sealed_only_run(tmp_path: Path):
    pytest.importorskip("pyarrow")
    _make_run(tmp_path, "discover_sealed")
    seal_run(tmp_path, "discover_sealed")
    run_dir = tmp_path / "runs" / "discover_sealed"
    assert not (run_dir / "features.jsonl").exists()

    ctx = DiscoveryContext.load("discover_sealed", tmp_path)
    assert len(ctx.rows) == 4
    X, cols = ctx.feature_matrix()
    assert X.shape[0] == 4
    assert cols


def test_campaign_concatenates_two_sealed_shards(tmp_path: Path):
    pytest.importorskip("pyarrow")
    reg = toy_registry()
    run_campaign(
        CampaignConfig(
            domain_id="synthetic_poly",
            sizes=[4, 8],
            n_per_size=2,
            indices=[1, 2],
            store_root=tmp_path,
            campaign_id="two_shard",
            resume=False,
            compute_backend="numpy",
            seal_batches=True,
        ),
        registry=reg,
    )
    _, _, rows = load_campaign_matrix("two_shard", tmp_path)
    assert len(rows) == 8
    assert sorted({int(row["size"]) for row in rows}) == [4, 8]
    assert len(load_sealed_campaign("two_shard", tmp_path)) == 8


def test_unsealed_run_still_loads_from_jsonl(tmp_path: Path):
    _make_run(tmp_path, "jsonl_only")
    rows = flatten_features("jsonl_only", tmp_path)
    assert len(rows) == 4
    assert (tmp_path / "runs" / "jsonl_only" / "features.jsonl").exists()


def test_sealed_shard_plus_incremental_jsonl_merges_for_export(tmp_path: Path):
    pytest.importorskip("pyarrow")
    _make_run(tmp_path, "merge_run")
    seal_run(tmp_path, "merge_run", delete_jsonl=False)
    run_dir = tmp_path / "runs" / "merge_run"
    assert (run_dir / "features.jsonl").exists()
    rows = flatten_features("merge_run", tmp_path)
    assert len(rows) == 4
    assert len(load_rows_from_shard(run_dir / "sealed" / "features.parquet")) == 4


def test_seal_dry_run_reports_expected_cleanup(tmp_path: Path):
    _make_run(tmp_path, "dry_run")
    plan = estimate_seal_cleanup(tmp_path, "dry_run")
    assert plan["action"] == "would_seal"
    assert plan["bytes_freed"] > 0
    assert plan["would_remove"]["completion_index_bytes"] > 0
    assert (tmp_path / "runs" / "dry_run" / "completion.sqlite3").exists()


def test_seal_missing_pyarrow_fails_without_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _make_run(tmp_path, "no_pyarrow")
    run_dir = tmp_path / "runs" / "no_pyarrow"

    def fail_require() -> None:
        raise ImportError(
            "RDE sealing requires pyarrow. Install with: "
            "pip install pyarrow  (or pip install -e '.[rde-parquet]')"
        )

    monkeypatch.setattr(
        "rde.io.seal.require_seal_dependencies",
        fail_require,
    )
    with pytest.raises(ImportError, match="pyarrow"):
        seal_run(tmp_path, "no_pyarrow")

    assert (run_dir / "completion.sqlite3").exists()
    assert (run_dir / "arrays").is_dir()
    assert not (run_dir / "sealed" / "sealed.json").exists()


def test_e2e_storage_capped_seal_resume_discovery(tmp_path: Path):
    pytest.importorskip("pyarrow")
    reg = toy_registry()
    campaign_id = "e2e_compact"
    partial = run_campaign(
        CampaignConfig(
            domain_id="synthetic_poly",
            sizes=[8],
            n_per_size=3,
            indices=[1, 2],
            store_root=tmp_path,
            campaign_id=campaign_id,
            resume=False,
            strict=True,
            compute_backend="numpy",
            save_arrays=True,
            max_storage_bytes=10_000,
            seal_batches=True,
        ),
        registry=reg,
    )
    run_id = f"{campaign_id}_n8"
    run_dir = tmp_path / "runs" / run_id
    assert partial.batch_summary["stop_reason"] == "storage_budget"
    assert partial.batch_summary["sealed"] is True
    assert partial.batch_summary["seal_summary"][0]["shard_path"]
    assert not (run_dir / "completion.sqlite3").exists()
    assert not (run_dir / "arrays").exists()
    assert (run_dir / "features.jsonl").exists()

    complete = run_campaign(
        CampaignConfig(
            domain_id="synthetic_poly",
            sizes=[8],
            n_per_size=3,
            indices=[1, 2],
            store_root=tmp_path,
            campaign_id=campaign_id,
            resume=True,
            strict=True,
            compute_backend="numpy",
            save_arrays=True,
            seal_batches=True,
        ),
        registry=reg,
    )
    assert complete.batch_summary["stop_reason"] == "complete"
    assert complete.batch_summary["sealed"] is True
    assert not (run_dir / "features.jsonl").exists()
    assert (run_dir / "sealed" / "features.parquet").exists()

    rows = flatten_features(run_id, tmp_path)
    assert len(rows) == 6
    ctx = DiscoveryContext.load(run_id, tmp_path)
    assert len(ctx.rows) == 6
    _, _, campaign_rows = load_campaign_matrix(campaign_id, tmp_path)
    assert len(campaign_rows) == 6
    assert len(load_sealed_campaign(campaign_id, tmp_path)) == 6

    manifest = json.loads(
        (tmp_path / "campaigns" / campaign_id / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["storage_mode"] == "compact"
    assert manifest["seal_schema_version"]
    journal = [
        json.loads(line)
        for line in (tmp_path / "campaigns" / campaign_id / "batches.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert len(journal) == 2
    assert journal[0]["sealed"] is True
    assert journal[1]["sealed"] is True
