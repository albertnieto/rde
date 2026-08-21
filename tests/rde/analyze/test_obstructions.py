"""Tests for obstruction harness (v0.2)."""

from __future__ import annotations

from rde.analyze.obstructions import panel_summary, witness_panel
from rde.analyze.scaling import fit_log_scaling
from rde.discovery.promote_lb import build_lower_bound_conjectures
from tests.rde.helpers import obstruction_rows


def test_witness_panel_finds_columns():
    rows = obstruction_rows([4, 6, 8])
    panel = witness_panel(rows, "metric.log_slice_rank")
    assert len(panel) >= 2


def test_scaling_detects_exponential():
    rows = obstruction_rows([4, 6, 8, 10])
    fit = fit_log_scaling(rows, "dynamics.slice_rank_end")
    assert fit is not None
    assert fit.exponential_growth


def test_build_lower_bound_conjectures():
    rows = obstruction_rows([4, 6, 8, 10])
    conjectures = build_lower_bound_conjectures(rows, target="metric.log_slice_rank", run_id="test")
    assert len(conjectures) >= 1
    assert conjectures[0]["claim_type"] == "scoped_lower_bound_conjecture"


def test_panel_summary_negative_outcome():
    rows = obstruction_rows([4, 6, 8, 10])
    summary = panel_summary(witness_panel(rows, "metric.log_slice_rank"))
    assert summary["n_witnesses"] >= 1
