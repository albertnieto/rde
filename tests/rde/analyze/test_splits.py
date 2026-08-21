"""Tests for deterministic grouped split assignment."""

from __future__ import annotations

import pytest

from rde.analyze.splits import (
    SplitFold,
    SplitPolicy,
    assign_instance_fold,
    assign_splits,
    attach_splits_to_rows,
    filter_rows_by_folds,
)
from rde.core.instance import InstanceRecord


def _instance(
    *,
    size: int = 8,
    seed: int = 0,
    generator: str = "structure_break_abelian",
) -> InstanceRecord:
    return InstanceRecord(
        domain_id="hsp_functions",
        size=size,
        seed=seed,
        params={"generator": generator},
    )


def test_calibration_and_confirmatory_sizes():
    policy = SplitPolicy(calibration_sizes=(6,), confirmatory_sizes=(20,))
    assert assign_instance_fold(_instance(size=6), policy) == SplitFold.CALIBRATION
    assert assign_instance_fold(_instance(size=20), policy) == SplitFold.TEST


def test_held_out_size_and_generator():
    policy = SplitPolicy(
        held_out_sizes=(16,),
        held_out_generator_groups=("simon",),
    )
    assert assign_instance_fold(_instance(size=16), policy) == SplitFold.HELDOUT_N
    assert assign_instance_fold(_instance(size=8, generator="simon"), policy) == SplitFold.HELDOUT_GENERATOR


def test_assign_splits_and_row_helpers():
    policy = SplitPolicy(calibration_sizes=(6,), confirmatory_sizes=(20,))
    cal = _instance(size=6, seed=1)
    disc = _instance(size=8, seed=2)
    conf = _instance(size=20, seed=3)
    instances = [cal, disc, conf]
    assignments = assign_splits(instances, policy, campaign_seed=7)
    assert assignments[cal.instance_id] == SplitFold.CALIBRATION.value
    assert assignments[conf.instance_id] == SplitFold.TEST.value

    rows = [{"instance_id": cal.instance_id, "x": 1.0}, {"instance_id": disc.instance_id, "x": 2.0}]
    attached = attach_splits_to_rows(
        rows,
        {cal.instance_id: "calibration", disc.instance_id: "discovery"},
    )
    assert attached[0]["split_fold"] == "calibration"
    assert filter_rows_by_folds(attached, {"calibration"}) == [attached[0]]

    with pytest.raises(ValueError, match="no split assignment"):
        attach_splits_to_rows([{"instance_id": "missing"}], {"a": "discovery"})
