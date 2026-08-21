"""Tests for prospective power planning (v0.3)."""

from __future__ import annotations

import numpy as np

from rde.analyze.power_plan import (
    PowerPlan,
    estimate_target_variance,
    estimate_within_instance_correlation,
    power_plan_to_dict,
    simulate_power,
)
from tests.rde.helpers import analyze_rows


def test_estimate_target_variance():
    rows = analyze_rows()
    assert estimate_target_variance(rows, "metric.log_slice_rank") > 0
    assert np.isnan(estimate_target_variance([rows[0]], "metric.log_slice_rank"))


def test_estimate_within_instance_correlation():
    rows = analyze_rows()
    assert np.isfinite(estimate_within_instance_correlation(rows, "metric.log_slice_rank"))
    sparse = [{"instance_id": "only", "metric.log_slice_rank": 1.0}]
    assert np.isnan(estimate_within_instance_correlation(sparse, "metric.log_slice_rank"))


def test_simulate_power_and_serialization():
    plan = simulate_power(
        analyze_rows(),
        min_effect_r=0.1,
        target_power=0.9,
        max_instances_cap=500,
    )
    assert isinstance(plan, PowerPlan)
    assert plan.instances_per_generator <= 500
    assert plan.notes

    payload = power_plan_to_dict(plan)
    assert payload["instances_per_generator"] == plan.instances_per_generator
    assert payload["notes"] == plan.notes
