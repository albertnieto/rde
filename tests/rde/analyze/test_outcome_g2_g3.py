"""Tests for G2/G3 outcome gates."""

from __future__ import annotations

import numpy as np

from rde.analyze.outcome import _assess_g2, _g2_cluster_columns, assess_outcome


def test_g2_cluster_columns_prefers_correlated_over_alphabetical():
    rows = []
    for i in range(40):
        rows.append(
            {
                "size": 8,
                "A_coef": float((i * 17) % 5),
                "compress.zlib_bits": float(i % 3),
                "walsh_log.w_amp.sparsity": float((i % 2) * 10 + i * 0.01),
                "graph.Q.laplacian_entropy": float(i * 0.2),
                "metric.log_slice_rank": float(i % 2) * 0.4 + 0.1,
            }
        )
    cols = _g2_cluster_columns(rows, "metric.log_slice_rank", top_k=3)
    assert cols[0] == "walsh_log.w_amp.sparsity"
    assert "graph.Q.laplacian_entropy" in cols


def test_g2_cluster_profiles_and_subsample():
    rows = []
    for i in range(100):
        rows.append(
            {
                "size": 8,
                "walsh_log.w_amp.sparsity": float((i % 2) * 10),
                "graph.Q.laplacian_entropy": float(i),
                "metric.log_slice_rank": float(i % 2) * 0.4 + 0.1,
            }
        )
    met, triggers, crit = _assess_g2(rows, "metric.log_slice_rank")
    assert crit["cluster_sampled"] is False
    assert len(crit["cluster_profiles"]) == 2
    assert "walsh_log.w_amp.sparsity" in crit["cluster_columns"]

    big = rows * 700  # 70k rows
    _, _, big_crit = _assess_g2(big, "metric.log_slice_rank")
    assert big_crit["cluster_sampled"] is True
    assert big_crit["cluster_rows_used"] == 65536


def test_level_2_hidden_classes():
    rows = []
    for i in range(20):
        g = i % 2
        rows.append(
            {
                "size": 6,
                "generator": f"g{g}",
                "matrix.trace": float(g * 10 + i * 0.1),
                "graph.degree_mean": float(g * 5),
                "metric.representation_complexity": float(1 + g * 3),
            }
        )
    result = assess_outcome(rows, "metric.representation_complexity")
    assert result.g2_met or result.grade >= 1


def test_level_3_recurrence_requires_identification():
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


def test_level_3_recurrence_eligible_data_gate():
    rows = [
        {
            "size": 32,
            "recurrence.estimated_order": 2.0,
            "recurrence.identification_eligible": 1.0,
            "recurrence.withheld_stable": 1.0,
            "metric.log_slice_rank": 2.0,
        }
        for _ in range(10)
    ]
    result = assess_outcome(rows, "metric.log_slice_rank")
    assert result.g3_met


def test_promotion_blocked_from_leak_summary():
    rows = [{"size": 6, "metric.log_slice_rank": 1.0}]
    result = assess_outcome(
        rows,
        "metric.log_slice_rank",
        leak_audit_summary={"promotion_blocked": True, "n_passed": 0},
    )
    assert result.promotion_blocked
