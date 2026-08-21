"""Tests for feature-table column helpers."""

from __future__ import annotations

import numpy as np

from rde.analyze.tables import (
    default_train_test_split,
    group_indices_by_size,
    numeric_columns,
)


def test_numeric_columns_skips_non_numeric():
    rows = [
        {"size": 4, "x": 1.0, "label": "a"},
        {"size": 8, "x": 2.0, "label": "b"},
    ]
    assert numeric_columns(rows) == ["size", "x"]
    assert numeric_columns(rows, exclude={"size"}) == ["x"]


def test_group_indices_by_size_and_default_split():
    rows = [{"size": 4}, {"size": 4}, {"size": 8}, {"size": 12}]
    groups = group_indices_by_size(rows)
    assert groups[4].tolist() == [0, 1]
    assert groups[8].tolist() == [2]

    split = default_train_test_split(rows)
    assert split is not None
    train_idx, test_idx = split
    assert len(train_idx) >= 1
    assert len(test_idx) >= 1
    assert default_train_test_split([{"size": 4}]) is None
