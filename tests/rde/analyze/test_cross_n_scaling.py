"""Tests for cross-N generative rule discovery (discovery/scaling_rule.py)."""

from __future__ import annotations

import json

import numpy as np

from rde.discovery.loop import DiscoveryReport, write_discovery_report
from rde.discovery.report import format_discovery_summary
from rde.discovery.scaling_rule import fit_cross_n_generative_rule


def _synthetic_rows(sizes: list[int], *, rows_per_size: int = 6, noise: float = 1e-4) -> list[dict]:
    rng = np.random.default_rng(0)
    rows: list[dict] = []
    for n in sizes:
        for i in range(rows_per_size):
            f_val = float(i + 1)
            target = (1.0 / n) * f_val + rng.normal(0.0, noise)
            rows.append({"size": n, "seed": i, "F": f_val, "target": target})
    return rows


def test_fit_cross_n_generative_rule_recovers_known_scaling():
    """target = (1/N)*F per row; per-size slope should track 1/N and the
    recursive meta-fit against (N, slope) should score highly."""
    rows = _synthetic_rows([4, 6, 8, 10, 12])
    candidates = [{"source": "metric_dsl", "expression": "F"}]

    results = fit_cross_n_generative_rule(rows, "target", candidates)

    assert len(results) == 1
    result = results[0]
    assert result.status == "ok"
    assert result.stable is True
    assert {f.size for f in result.per_size_fits} == {4, 6, 8, 10, 12}
    for fit in result.per_size_fits:
        assert abs(fit.slope - 1.0 / fit.size) < 1e-2
        assert fit.r_squared > 0.99
    assert result.slope_rule_equation is not None
    assert result.slope_rule_r_squared is not None
    assert result.slope_rule_r_squared > 0.9


def test_fit_cross_n_generative_rule_insufficient_sizes():
    """Only 2 distinct sizes present; default min_sizes=3 should refuse to
    call the meta symbolic-regression step at all."""
    rows = _synthetic_rows([4, 6])
    candidates = [{"source": "metric_dsl", "expression": "F"}]

    results = fit_cross_n_generative_rule(rows, "target", candidates)

    assert len(results) == 1
    result = results[0]
    assert result.status == "insufficient_sizes"
    assert result.slope_rule_equation is None


def test_fit_cross_n_generative_rule_skips_latent_interpretation_candidates():
    """latent_interpretation-source candidates reference latent.* columns
    that never exist in `rows`; they must be filtered out before ever being
    handed to parse_expr, not surfaced as an eval_failed result."""
    rows = _synthetic_rows([4, 6, 8])
    candidates = [{"source": "latent_interpretation", "expression": "latent.pca_0 ≈ 2*x + 1"}]

    results = fit_cross_n_generative_rule(rows, "target", candidates)

    assert results == []


def test_fit_cross_n_generative_rule_eval_failed_for_unparseable_expression():
    rows = _synthetic_rows([4, 6, 8])
    candidates = [{"source": "symbolic", "expression": "sin(F)"}]  # not in the internal DSL grammar

    results = fit_cross_n_generative_rule(rows, "target", candidates)

    assert len(results) == 1
    assert results[0].status == "eval_failed"


def test_fit_cross_n_generative_rule_empty_inputs():
    assert fit_cross_n_generative_rule([], "target", [{"source": "metric_dsl", "expression": "F"}]) == []
    assert fit_cross_n_generative_rule(_synthetic_rows([4, 6, 8]), "target", []) == []


def test_cross_n_rule_is_serialized_and_printed_in_discovery_summary(tmp_path):
    result = fit_cross_n_generative_rule(
        _synthetic_rows([4, 6, 8, 10, 12]),
        "target",
        [{"source": "metric_dsl", "expression": "F"}],
    )[0]
    report = DiscoveryReport(
        run_id="cross-n-smoke",
        target="target",
        n_rows=30,
        cross_n_rule=[result.to_dict()],
    )

    path = write_discovery_report(report, tmp_path / "discovery.json")

    assert json.loads(path.read_text())["cross_n_rule"] == [result.to_dict()]
    summary = format_discovery_summary(report)
    assert "Cross-N rule:" in summary
    assert "slope(F)" in summary
