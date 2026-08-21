"""Tests for leak audit (v0.2)."""

from __future__ import annotations

from rde.analyze.leak_audit import audit_conjecture, audit_conjectures, classify_feature


def test_classify_oracle_as_non_predictor_for_hsp():
    cls = classify_feature(
        "hsp_oracle.walsh_max",
        target="metric.structure_strength",
        domain_id="hsp_functions",
    )
    assert cls.value == "enumeration"


def test_rejects_outcome_target_predictor():
    rec = {
        "expression": "0.1 + 0.5*metric.structure_strength",
        "feature_columns": ["metric.structure_strength"],
        "r_squared": 0.99,
    }
    result = audit_conjecture(
        rec,
        target="metric.structure_strength",
        domain_id="hsp_functions",
    )
    assert not result.passed


def test_allows_bounded_query_descriptors():
    rec = {
        "expression": "hsp_sample.f.collision_rate",
        "feature_columns": ["hsp_sample.f.collision_rate"],
        "r_squared": 0.8,
    }
    result = audit_conjecture(
        rec,
        target="metric.structure_strength",
        domain_id="hsp_functions",
    )
    assert result.passed


def test_audit_conjectures_filters_pool():
    records = [
        {"expression": "metric.structure_strength", "feature_columns": ["metric.structure_strength"]},
        {"expression": "hsp_sample.f.collision_rate", "feature_columns": ["hsp_sample.f.collision_rate"]},
    ]
    clean, summary = audit_conjectures(
        records,
        target="metric.structure_strength",
        domain_id="hsp_functions",
    )
    assert summary["n_blocked"] == 1
    assert summary["n_passed"] == 1
    assert len(clean) == 1
