"""Tests for v0.3 gated outcomes and non-vacuous recurrence."""

from __future__ import annotations

from rde.analyze.outcome import GateStatus, assess_outcome
from rde.features.recurrence import estimate_recurrence_order, recurrence_identification_eligible


def test_five_slices_recurrence_not_evaluated():
    assert not recurrence_identification_eligible(5)
    rows = [
        {
            "size": 4,
            "recurrence.estimated_order": 4.0,
            "recurrence.identification_eligible": 0.0,
            "metric.log_slice_rank": 2.0,
        }
        for _ in range(10)
    ]
    result = assess_outcome(rows, "metric.log_slice_rank")
    assert not result.g3_met
    assert result.gates.recurrence in (GateStatus.NOT_EVALUATED, GateStatus.FAIL)


def test_recurrence_with_identification_grid():
    import numpy as np

    slices = [np.ones(4) * i for i in range(14)]
    out = estimate_recurrence_order(list(range(14)), slices)
    assert out["recurrence.identification_eligible"] == 1.0


def test_headline_not_max_grade():
    rows = [{"size": 6, "metric.log_slice_rank": 1.0}]
    result = assess_outcome(rows, "metric.log_slice_rank")
    assert result.headline
    assert "level" not in result.headline.lower() or "=" in result.headline
