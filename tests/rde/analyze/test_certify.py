"""Tests for certify (v0.2)."""

from __future__ import annotations

from rde.analyze.certify import certify_representation_candidate


def test_certify_fails_on_leaky_conjecture():
    rows = [{"size": 8, "seed": 0}, {"size": 10, "seed": 0}]
    conj = {
        "expression": "metric.structure_strength",
        "feature_columns": ["metric.structure_strength"],
        "cross_n_sign_stability": 0.9,
    }
    result = certify_representation_candidate(
        conj,
        rows=rows,
        target="metric.structure_strength",
        domain_id="hsp_functions",
    )
    assert not result.passed
    assert not result.checks.get("leak_clean", True)


def test_certify_passes_clean_conjecture_with_cross_n():
    rows = [{"size": 8, "seed": 0}, {"size": 10, "seed": 1}, {"size": 12, "seed": 2}]
    conj = {
        "expression": "hsp_sample.f.collision_rate",
        "feature_columns": ["hsp_sample.f.collision_rate"],
        "cross_n_sign_stability": 0.85,
        "extrapolation_r_squared": 0.7,
        "latent_dim": 16,
        "split_fold": "discovery",
    }
    frozen = dict(conj)
    result = certify_representation_candidate(
        conj,
        rows=rows,
        target="metric.structure_strength",
        split_record=conj,
        replication_record={"promotion_eligible": True},
        frozen_candidate=frozen,
        domain_id="hsp_functions",
    )
    assert result.checks["leak_clean"]
    assert result.checks["multi_size"]
