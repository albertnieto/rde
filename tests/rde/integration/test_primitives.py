"""Unit tests for primitive feature splitting and persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rde.core.instance import InstanceRecord
from rde.core.primitives import persist_instance_features, split_primitives
from rde.io.store import Store


def test_split_primitives_separates_scalars_and_arrays():
    scalars, arrays = split_primitives(
        {
            "Q_M": 3.0,
            "flag": True,
            "costs": np.array([1.0, 2.0]),
        }
    )
    assert scalars == {"Q_M": 3.0, "flag": 1.0}
    assert list(arrays.keys()) == ["costs"]
    assert np.allclose(arrays["costs"], [1.0, 2.0])


def test_persist_instance_features_writes_jsonl_and_npz(tmp_path: Path):
    store = Store(tmp_path)
    inst = InstanceRecord(domain_id="d", size=2, seed=1, params={})
    row = persist_instance_features(
        store,
        "run1",
        inst.instance_id,
        inst.domain_id,
        inst.size,
        inst.seed,
        {"Q_M": 2.0, "costs": np.array([0.0, 1.0, 2.0, 3.0])},
    )
    assert row["scalars"]["Q_M"] == 2.0
    assert "costs" in row["array_refs"]

    store.flush("run1")
    loaded = store.read_instance_features("run1")
    assert len(loaded) == 1
    arr = store.load_array("run1", inst.instance_id, "primitive_costs")
    assert np.allclose(arr, [0.0, 1.0, 2.0, 3.0])
