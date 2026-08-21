"""Phase 3 compact write-path tests: slice-only feature rows and slim completion DB."""

from __future__ import annotations

import inspect
import json
import sqlite3
from pathlib import Path

import numpy as np

from rde.analyze.query import flatten_features
from rde.core.plugins import build_registry
from rde.core.schema import (
    FEATURE_ROW_LAYOUT_SLICE_ONLY,
    SLICE_DESCRIPTOR_RESERVED_KEYS,
    duplicated_instance_scalar_keys,
)
from rde.io.store import Store
from rde.runtime.pipeline import RunConfig, run_pipeline


def _registry():
    return build_registry("synthetic_poly")


def _run(
    tmp_path: Path,
    run_id: str,
    *,
    save_arrays: bool = False,
    indices: list[int] | None = None,
) -> None:
    run_pipeline(
        RunConfig(
            domain_id="synthetic_poly",
            n_instances=2,
            size=8,
            seed=3,
            indices=indices or [1, 2, 4],
            store_root=tmp_path,
            run_id=run_id,
            save_arrays=save_arrays,
            generator_id="poly_narrow_coeff",
        ),
        registry=_registry(),
    )


def test_feature_rows_omit_instance_scalars(tmp_path: Path):
    _run(tmp_path, "slice_only_rows")
    store = Store(tmp_path)
    instance_scalars = {
        row["instance_id"]: row["scalars"]
        for row in store.read_instance_features("slice_only_rows")
    }
    for feature in store.read_features("slice_only_rows"):
        overlap = duplicated_instance_scalar_keys(
            feature["descriptors"],
            instance_scalars[feature["instance_id"]],
        )
        assert overlap == []
        assert set(feature["descriptors"]) >= SLICE_DESCRIPTOR_RESERVED_KEYS


def test_flattened_table_retains_instance_columns(tmp_path: Path):
    _run(tmp_path, "flatten_join")
    store = Store(tmp_path)
    inst = store.read_instance_features("flatten_join")[0]
    flat = flatten_features("flatten_join", tmp_path)[0]
    for key, value in inst["scalars"].items():
        assert key in flat
        if isinstance(value, (int, float)):
            np.testing.assert_allclose(float(flat[key]), float(value))


def test_save_arrays_false_creates_no_arrays_dir(tmp_path: Path):
    _run(tmp_path, "no_arrays", save_arrays=False)
    run_dir = Store(tmp_path).run_dir("no_arrays")
    assert not (run_dir / "arrays").exists()
    for row in Store(tmp_path).read_features("no_arrays"):
        assert "array_ref" not in row


def test_completion_db_stores_only_cross_slice_resume_payload(tmp_path: Path):
    store = Store(tmp_path)
    store.append_instance_features(
        "partial",
        {
            "instance_id": "i0",
            "scalars": {
                "dynamics.n_points": 3.0,
                "landscape.entropy": 1.25,
                "matrix.trace": 4.0,
            },
        },
    )
    store.append_features("partial", {"instance_id": "i0", "family_index": 1})
    store.flush("partial")

    db_path = store.run_dir("partial") / "completion.sqlite3"
    with sqlite3.connect(db_path) as connection:
        scalars_json = connection.execute(
            "SELECT scalars_json FROM instance_completion WHERE instance_id='i0'"
        ).fetchone()[0]
    assert json.loads(scalars_json) == {"dynamics.n_points": 3.0}

    store.append_features("partial", {"instance_id": "i0", "family_index": 2})
    store.append_features("partial", {"instance_id": "i0", "family_index": 4})
    store.purge_instance_resume_scalars("partial", "i0")
    store.flush("partial")
    with sqlite3.connect(db_path) as connection:
        scalars_json = connection.execute(
            "SELECT scalars_json FROM instance_completion WHERE instance_id='i0'"
        ).fetchone()[0]
    assert scalars_json is None
    assert store.has_completed_instance("partial", "i0")


def test_completion_db_purged_after_full_pipeline_run(tmp_path: Path):
    _run(tmp_path, "slim_completion")
    db_path = Store(tmp_path).run_dir("slim_completion") / "completion.sqlite3"
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT scalars_json FROM instance_completion"
        ).fetchall()
    assert rows
    for scalars_json, in rows:
        assert scalars_json is None


def test_manifest_records_slice_only_layout(tmp_path: Path):
    _run(tmp_path, "manifest_layout")
    manifest = Store(tmp_path).read_manifest("manifest_layout")
    assert manifest.extra.get("feature_row_layout") == FEATURE_ROW_LAYOUT_SLICE_ONLY
    assert manifest.extra.get("save_arrays") is False


def test_calibration_profiles_save_arrays_defaults():
    from rde.cli.commands import cmd_calibrate_hardware

    source = inspect.getsource(cmd_calibrate_hardware)
    assert 'phase1-wide' not in source
    assert "save_arrays = False" in source
