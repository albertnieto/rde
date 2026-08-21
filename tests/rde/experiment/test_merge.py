"""Tests for cross-run merge and leak-clean discovery copies."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rde.experiment import ExperimentPreflightError
from rde.experiment.merge import merge_runs_for_discovery, write_clean_discovery_run
from rde.io.store import RunManifest, Store
from rde.runtime.pipeline import RunConfig, run_pipeline
from tests.rde.helpers import toy_registry


def test_merge_runs_for_discovery(tmp_path: Path):
    reg = toy_registry()
    for size in (4, 8):
        run_pipeline(
            RunConfig(
                domain_id="synthetic_poly",
                n_instances=2,
                size=size,
                seed=size,
                indices=[1, 2],
                store_root=tmp_path,
                run_id=f"merge_n{size}",
            ),
            registry=reg,
        )

    merged_id = merge_runs_for_discovery(
        tmp_path, ["merge_n4", "merge_n8"], "merged_cross_n"
    )
    assert merged_id == "merged_cross_n"

    store = Store(tmp_path)
    manifest = store.read_manifest("merged_cross_n")
    assert manifest.n_instances == 4
    assert set(manifest.extra.get("merged_from", [])) == {"merge_n4", "merge_n8"}
    assert store.run_stats("merged_cross_n")["features_lines"] == 8
    store.close()

    with pytest.raises(ValueError, match="at least one run_id"):
        merge_runs_for_discovery(tmp_path, [], "empty")


def test_write_clean_discovery_run_filters_leaky_columns(tmp_path: Path):
    store = Store(tmp_path)
    source_id = "leaky_source"
    run_dir = store.run_dir(source_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "instances.jsonl").write_text(
        json.dumps({"instance_id": "i1", "domain_id": "test"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "instance_features.jsonl").write_text(
        json.dumps(
            {
                "instance_id": "i1",
                "scalars": {
                    "matrix.trace": 1.0,
                    "dynamics.n_points": 3.0,
                    "graph.degree": 2.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "features.jsonl").write_text(
        json.dumps(
            {
                "instance_id": "i1",
                "family_index": 1,
                "descriptors": {"desc.leaky": 9.0},
                "metrics": {"y": 2.0, "leak_metric": 3.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    store.write_manifest(
        RunManifest(
            run_id=source_id,
            domain_id="synthetic_poly",
            n_instances=1,
            size=4,
            seed=0,
            indices=[1],
            descriptor_names=["desc.leaky"],
            metric_names=["y", "leak_metric"],
        )
    )
    store.close()

    clean_id = write_clean_discovery_run(
        tmp_path,
        source_id,
        "leaky_clean",
        target_metric="metric.y",
    )
    assert clean_id == "leaky_clean"

    clean_dir = Store(tmp_path).run_dir("leaky_clean")
    inst_row = json.loads((clean_dir / "instance_features.jsonl").read_text().strip())
    assert set(inst_row["scalars"]) == {"matrix.trace", "graph.degree"}

    feat_row = json.loads((clean_dir / "features.jsonl").read_text().strip())
    assert feat_row["descriptors"] == {}
    assert feat_row["metrics"] == {"y": 2.0}

    clean_manifest = Store(tmp_path).read_manifest("leaky_clean")
    assert clean_manifest.metric_names == ["y"]
    assert clean_manifest.extra.get("leak_clean_of") == source_id


def test_validate_clean_predictors_rejects_empty_predictor_set(tmp_path: Path):
    from rde.experiment.merge import validate_clean_predictors

    rows = [{"instance_id": "i1", "size": 8, "metric.y": 1.0}]
    with pytest.raises(ExperimentPreflightError, match="no predictor columns"):
        validate_clean_predictors(
            rows,
            predictor_prefixes=("matrix.", "graph."),
            target_metric="metric.y",
        )


def test_merge_skips_missing_shards(tmp_path: Path):
    reg = toy_registry()
    run_pipeline(
        RunConfig(
            domain_id="synthetic_poly",
            n_instances=1,
            size=4,
            seed=1,
            indices=[1],
            store_root=tmp_path,
            run_id="partial_merge",
        ),
        registry=reg,
    )
    run_dir = Store(tmp_path).run_dir("partial_merge")
    (run_dir / "instance_features.jsonl").unlink()

    merge_runs_for_discovery(tmp_path, ["partial_merge"], "merged_partial")
    merged_dir = Store(tmp_path).run_dir("merged_partial")
    assert (merged_dir / "instances.jsonl").exists()
    assert (merged_dir / "instance_features.jsonl").read_text(encoding="utf-8") == ""


def _write_toy_run(tmp_path: Path, run_id: str, rows: list[dict]) -> None:
    store = Store(tmp_path)
    run_dir = store.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "instances.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({"instance_id": row["instance_id"], "domain_id": "test"}) + "\n")
    with (run_dir / "instance_features.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(
                json.dumps(
                    {
                        "instance_id": row["instance_id"],
                        "scalars": row["scalars"],
                    }
                )
                + "\n"
            )
    with (run_dir / "features.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(
                json.dumps(
                    {
                        "instance_id": row["instance_id"],
                        "size": row["size"],
                        "descriptors": {},
                        "metrics": {"y": row["y"]},
                    }
                )
                + "\n"
            )
    store.write_manifest(
        RunManifest(
            run_id=run_id,
            domain_id="synthetic_poly",
            n_instances=len(rows),
            size=int(rows[0]["size"]),
            seed=0,
            indices=[0],
            descriptor_names=[],
            metric_names=["y"],
        )
    )
    store.close()


def test_write_clean_discovery_run_drop_substrings(tmp_path: Path):
    _write_toy_run(
        tmp_path,
        "count_source",
        [
            {
                "instance_id": "i1",
                "size": 8,
                "y": 1.0,
                "scalars": {
                    "hsp_sample.f.collision_rate": 0.2,
                    "hsp_sample.f.n_collisions_found": 12.0,
                    "hsp_sample.f.query_budget": 64.0,
                    "landscape.n_collisions_found": 12.0,
                    "landscape.collision_rate": 0.2,
                    "generator": "simon",
                },
            }
        ],
    )
    from rde.experiment.merge import write_clean_discovery_run

    write_clean_discovery_run(
        tmp_path,
        "count_source",
        "count_clean",
        target_metric="metric.y",
        predictor_prefixes=("hsp_sample.", "landscape."),
        drop_substrings=(
            "n_collisions_found",
            "query_budget",
            "difference_profile_query_cost",
        ),
    )
    clean = json.loads(
        (Store(tmp_path).run_dir("count_clean") / "instance_features.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert set(clean["scalars"]) == {
        "hsp_sample.f.collision_rate",
        "landscape.collision_rate",
        "generator",
    }


def test_write_run_subset_keeps_listed_ids(tmp_path: Path):
    from rde.experiment.merge import write_run_subset

    _write_toy_run(
        tmp_path,
        "subset_source",
        [
            {
                "instance_id": "keep_a",
                "size": 8,
                "y": 1.0,
                "scalars": {"matrix.trace": 1.0, "generator": "a"},
            },
            {
                "instance_id": "drop_b",
                "size": 8,
                "y": 0.0,
                "scalars": {"matrix.trace": 2.0, "generator": "b"},
            },
            {
                "instance_id": "keep_c",
                "size": 8,
                "y": 1.0,
                "scalars": {"matrix.trace": 3.0, "generator": "c"},
            },
        ],
    )
    write_run_subset(
        tmp_path,
        "subset_source",
        "subset_dest",
        keep_instance_ids=["keep_a", "keep_c"],
    )
    dest = Store(tmp_path).run_dir("subset_dest")
    kept = [
        json.loads(line)
        for line in dest.joinpath("features.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {row["instance_id"] for row in kept} == {"keep_a", "keep_c"}
    manifest = Store(tmp_path).read_manifest("subset_dest")
    assert manifest.n_instances == 2
    assert manifest.extra.get("subset_of") == "subset_source"
