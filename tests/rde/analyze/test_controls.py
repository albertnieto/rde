"""Tests for null baselines and positive controls (v0.3)."""

from __future__ import annotations

import numpy as np

from rde.analyze.controls import generator_median_baseline, nr_baseline_r2
from tests.rde.helpers import analyze_rows


def test_generator_median_baseline():
    baseline = generator_median_baseline(analyze_rows(), "metric.log_slice_rank")
    assert baseline["n_groups"] == 2.0
    assert np.isfinite(baseline["mae"])


def test_nr_baseline_r2_edges():
    assert np.isnan(nr_baseline_r2([], "metric.log_slice_rank"))
    constant_rows = [
        {"size": 1, "family_index": 1, "metric.log_slice_rank": 5.0},
        {"size": 2, "family_index": 2, "metric.log_slice_rank": 5.0},
        {"size": 3, "family_index": 3, "metric.log_slice_rank": 5.0},
    ]
    assert nr_baseline_r2(constant_rows, "metric.log_slice_rank") == 0.0
