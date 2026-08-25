"""Regression tests for the target-degeneracy guard (2026-08-19).

A TSP cost-landscape campaign found that a `degree2_entropy` target was
mathematically guaranteed constant by the cost function's own algebra
(std ~1.25e-15 across 1200 instances, differing only in floating-point
noise), yet the discovery pipeline reported `outcome_grade_hint=5` --
Phase 6 "rediscovers" a constant trivially on every split, and nothing
anywhere in the gate chain checked whether the target had real variance
before letting that claim through. These tests pin down that a target with
no real variance can never produce a positive outcome level, regardless of
what any upstream stage (Phase 6 rediscovery, symbolic R^2, k-means
separation) claims -- while a genuinely independent Level 3 (recurrence)
finding, which does not depend on this target's variance, must still be
allowed through.
"""

from __future__ import annotations

from rde.analyze.outcome import assess_outcome, assess_target_degeneracy


def _constant_target_rows(n: int = 1200, noise: float = 1e-15) -> list[dict]:
    # Mirrors the real failure: a target that is the same value everywhere
    # except for machine-precision floating-point noise in the last digits.
    rows = []
    for i in range(n):
        rows.append(
            {
                "size": 6,
                "degree2_entropy": 4.094344562222100 + (noise if i % 2 else -noise),
                "matrix.D.trace_power_3": float(i % 7) + 1.0,
                "spectral.l2_norm": float((i * 13) % 11) + 0.5,
            }
        )
    return rows


def test_assess_target_degeneracy_flags_near_constant_target():
    rows = _constant_target_rows()
    result = assess_target_degeneracy(rows, "degree2_entropy")
    assert result["is_degenerate"] is True
    assert result["std"] < 1e-9
    assert result["n"] == len(rows)


def test_assess_target_degeneracy_allows_real_variance():
    rows = [{"metric.log_slice_rank": float(i % 5) + 0.1 * i} for i in range(50)]
    result = assess_target_degeneracy(rows, "metric.log_slice_rank")
    assert result["is_degenerate"] is False


def test_degenerate_target_cannot_produce_positive_outcome_level():
    rows = _constant_target_rows()
    # Even if every downstream signal looks maximally positive, the target
    # itself carries no real variance -- nothing should be promotable.
    result = assess_outcome(
        rows,
        "degree2_entropy",
        metric_candidates=[{"r_squared": 1.0, "extrapolation_r_squared": 1.0}],
        latent={"target_correlations": {"z0": 0.99}, "ridge_r_squared": 1.0},
        phase6={
            "representation": {"g4_met": True, "state_reconstruction_r2": 1.0},
            "rediscovery": {"g5_met": True, "rediscovery_frequency": 1.0},
        },
        certify_result={"passed": True},
        frozen_candidate={"expression": "x"},
        replication_passed=True,
    )
    assert result.target_degenerate is True
    assert result.grade == 0
    assert result.g1_met is False
    assert result.g2_met is False
    assert result.g4_met is False
    assert result.g5_met is False
    assert result.promotion_blocked is True
    assert "TARGET_DEGENERATE" in result.headline


def test_degenerate_target_does_not_block_independent_level_3():
    # Level 3 (recurrence) depends only on recurrence.* columns, never on
    # this target's own variance -- a degenerate target must not suppress
    # a genuinely independent recurrence finding.
    rows = [
        {
            "size": 32,
            "degree2_entropy": 4.0943445622221 + (1e-15 if i % 2 else -1e-15),
            "recurrence.estimated_order": 2.0,
            "recurrence.identification_eligible": 1.0,
            "recurrence.withheld_stable": 1.0,
        }
        for i in range(10)
    ]
    result = assess_outcome(rows, "degree2_entropy")
    assert result.target_degenerate is True
    assert result.g3_met is True
    assert result.grade == 3
