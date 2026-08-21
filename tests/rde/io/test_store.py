"""Tests for RDE Store."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rde.core.instance import InstanceRecord
from rde.io.store import RunManifest, Store


def test_store_manifest_and_arrays(tmp_path: Path):
    store = Store(tmp_path)
    manifest = RunManifest(
        run_id="run001",
        domain_id="synthetic_poly",
        n_instances=2,
        size=4,
        seed=0,
        indices=[1, 2],
    )
    store.write_manifest(manifest)
    read = store.read_manifest("run001")
    assert read.run_id == "run001"
    assert read.domain_id == "synthetic_poly"

    inst = InstanceRecord(domain_id="synthetic_poly", size=4, seed=0, params={"a": 1.0})
    store.append_instance("run001", inst)
    store.save_array("run001", inst.instance_id, "test_r1", np.array([1.0, 2.0, 3.0]))
    store.flush("run001")

    instances = store.read_instances("run001")
    assert len(instances) == 1
    loaded = store.load_array("run001", inst.instance_id, "test_r1")
    assert np.allclose(loaded, [1.0, 2.0, 3.0])

    row = {"instance_id": inst.instance_id, "family_index": 1}
    store.append_instance_features("run001", {"instance_id": inst.instance_id, "scalars": {"x": 1.0}})
    store.flush("run001")
    instance_features = store.read_instance_features("run001")
    assert instance_features[0]["scalars"]["x"] == 1.0

    store.append_features("run001", row)
    store.flush("run001")
    features = store.read_features("run001")
    assert features[0]["family_index"] == 1

    manifest_path = store.run_dir("run001") / "manifest.json"
    assert json.loads(manifest_path.read_text())["run_id"] == "run001"


def test_reset_run_replaces_stale_rows(tmp_path: Path):
    store = Store(tmp_path)
    manifest = RunManifest(
        run_id="run002",
        domain_id="synthetic_poly",
        n_instances=1,
        size=4,
        seed=0,
        indices=[1],
    )
    store.write_manifest(manifest)
    stale = InstanceRecord(domain_id="synthetic_poly", size=4, seed=0, params={"a": 99.0})
    store.append_instance("run002", stale)
    store.flush("run002")
    assert len(store.read_instances("run002")) == 1

    store.reset_run("run002")
    store.write_manifest(manifest)
    fresh = InstanceRecord(domain_id="synthetic_poly", size=4, seed=0, params={"a": 1.0})
    store.append_instance("run002", fresh)
    store.flush("run002")

    instances = store.read_instances("run002")
    assert len(instances) == 1
    assert instances[0].params["a"] == 1.0


def test_completion_index_supports_resume_without_loading_jsonl(tmp_path: Path):
    store = Store(tmp_path)
    store.append_instance_features(
        "run003",
        {
            "instance_id": "i0",
            "scalars": {"dynamics.n_points": 2.0, "landscape.entropy": 1.5},
        },
    )
    store.append_features("run003", {"instance_id": "i0", "family_index": 4})
    store.flush("run003")

    assert store.has_completed_instance("run003", "i0")
    assert store.has_completed_feature("run003", "i0", 4)
    assert store.instance_scalars("run003", "i0") == {"dynamics.n_points": 2.0}
    assert (store.run_dir("run003") / "completion.sqlite3").exists()

    resumed = Store(tmp_path)
    resumed.ensure_completion_index("run003")
    assert resumed.has_completed_feature("run003", "i0", 4)
    store.close()
    resumed.close()


def test_completion_index_checkpoint_does_not_accumulate_rows(tmp_path: Path):
    store = Store(tmp_path, jsonl_batch_size=4)
    for i in range(40):
        instance_id = f"i{i}"
        store.append_instance_features(
            "run004",
            {"instance_id": instance_id, "scalars": {"x": float(i)}},
        )
        store.append_features(
            "run004",
            {"instance_id": instance_id, "family_index": 1},
        )
        pending = store._completion_pending["run004"]
        assert sum(len(rows) for rows in pending.values()) < 4

    store.flush("run004")
    assert len(store.completed_instance_ids("run004")) == 40
    assert len(store.completed_feature_keys("run004")) == 40
    store.close()


def test_completion_index_reconciles_jsonl_tail_after_partial_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "runs" / "run005"
    run_dir.mkdir(parents=True)
    (run_dir / "features.jsonl").write_text(
        json.dumps({"instance_id": "i0", "family_index": 2}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "instance_features.jsonl").write_text(
        json.dumps(
            {
                "instance_id": "i0",
                "scalars": {"dynamics.n_points": 3.0, "matrix.trace": 9.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    store = Store(tmp_path)
    store.ensure_completion_index("run005")
    assert store.has_completed_feature("run005", "i0", 2)
    assert store.instance_scalars("run005", "i0") == {"dynamics.n_points": 3.0}
    store.close()
