"""Tests for two-seed replication."""

from __future__ import annotations

from rde.analyze.replication import compare_outcomes, replication_to_dict


def test_g1_requires_both_seeds():
    a = {"seed": 0, "grade": 1, "g0_met": False, "g1_met": True}
    b = {"seed": 1, "grade": 0, "g0_met": True, "g1_met": False}
    rep = compare_outcomes(a, b)
    assert rep.exploratory_only
    assert not rep.g1_both


def test_g0_both_seeds():
    a = {"seed": 0, "grade": 0, "g0_met": True, "g1_met": False}
    b = {"seed": 1, "grade": 0, "g0_met": True, "g1_met": False}
    rep = compare_outcomes(a, b)
    assert rep.g0_both
    assert rep.promotion_eligible


def test_replication_to_dict():
    rep = compare_outcomes(
        {"seed": 0, "grade": 1, "g0_met": False, "g1_met": True},
        {"seed": 1, "grade": 1, "g0_met": False, "g1_met": True},
    )
    d = replication_to_dict(rep)
    assert d["g1_both"] is True
    assert d["grade_a"] == 1
    assert d["grade_b"] == 1


def test_frozen_candidate_must_match_for_promotion():
    frozen = {"expression": "x+y", "feature_columns": ["matrix.trace"]}
    seed_a = {
        "seed": 0,
        "grade": 1,
        "g0_met": False,
        "g1_met": True,
        "gates": {"predictor": "pass"},
    }
    seed_b_match = {
        "seed": 1,
        "grade": 1,
        "g0_met": False,
        "g1_met": True,
        "gates": {"predictor": "pass"},
    }
    rep = compare_outcomes(
        seed_a,
        seed_b_match,
        frozen_candidate=frozen,
        seed_b_evaluated_candidate=frozen,
    )
    assert rep.g1_both
    assert rep.frozen_candidate_match
    assert rep.promotion_eligible

    rep_mismatch = compare_outcomes(
        seed_a,
        seed_b_match,
        frozen_candidate=frozen,
        seed_b_evaluated_candidate={"expression": "other", "feature_columns": ["matrix.trace"]},
    )
    assert rep_mismatch.g1_both
    assert not rep_mismatch.frozen_candidate_match
    assert not rep_mismatch.promotion_eligible
    assert any("did not evaluate the frozen" in note for note in rep_mismatch.notes)
