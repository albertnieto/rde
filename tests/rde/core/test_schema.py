"""Tests for JSONL row schema validation."""

from __future__ import annotations

from rde.core.schema import (
    duplicated_instance_scalar_keys,
    validate_feature_row,
    validate_features_file,
    validate_instance_features_file,
    validate_instance_features_row,
    validate_instance_row,
)


def test_validate_feature_row_errors():
    assert validate_feature_row({}) != []
    row = {
        "run_id": "r",
        "instance_id": "i",
        "domain_id": "d",
        "size": 4,
        "seed": 0,
        "family_index": 1,
        "slice_kind": "trajectory",
        "descriptors": {},
        "metrics": {},
    }
    assert validate_feature_row(row) == []
    assert validate_features_file([row]) == []


def test_validate_instance_rows():
    inst = {"domain_id": "d", "size": 4, "seed": 0, "params": {}}
    assert validate_instance_row(inst) == []
    feat = {
        "run_id": "r",
        "instance_id": "i",
        "domain_id": "d",
        "size": 4,
        "seed": 0,
        "scalars": {},
    }
    assert validate_instance_features_row(feat) == []
    assert validate_instance_features_file([feat]) == []


def test_duplicated_instance_scalar_keys():
    overlap = duplicated_instance_scalar_keys(
        {"matrix.trace": 1.0, "family_index": 2},
        {"matrix.trace": 1.0, "graph.degree": 3.0},
    )
    assert overlap == ["matrix.trace"]
    assert duplicated_instance_scalar_keys({"family_index": 1}, {"family_index": 1}) == []
