"""Two-seed replication with frozen-candidate identity (v0.3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _grade(report: dict[str, Any]) -> int:
    return int(report.get("grade", 0))


def _g_met(report: dict[str, Any], k: int) -> bool:
    return bool(report.get(f"g{k}_met", False))


@dataclass
class ReplicationReport:
    seed_a: int
    seed_b: int
    grade_a: int
    grade_b: int
    g0_both: bool
    g1_both: bool
    g1_either: bool
    exploratory_only: bool
    promotion_eligible: bool
    frozen_candidate_match: bool
    notes: list[str] = field(default_factory=list)


def _candidate_identity(candidate: dict[str, Any] | None) -> str:
    if not candidate:
        return ""
    return str(candidate.get("expression", "")) + "|" + ",".join(candidate.get("feature_columns") or [])


def compare_outcomes(
    seed_a_report: dict[str, Any],
    seed_b_report: dict[str, Any],
    *,
    frozen_candidate: dict[str, Any] | None = None,
    seed_b_evaluated_candidate: dict[str, Any] | None = None,
) -> ReplicationReport:
    """Compare outcome assessments from two independent seeds."""
    sa = int(seed_a_report.get("seed", 0))
    sb = int(seed_b_report.get("seed", 1))
    la = _grade(seed_a_report)
    lb = _grade(seed_b_report)
    l0_a = _g_met(seed_a_report, 0)
    l0_b = _g_met(seed_b_report, 0)
    l1_a = _g_met(seed_a_report, 1)
    l1_b = _g_met(seed_b_report, 1)

    g0_both = l0_a and l0_b
    g1_both = l1_a and l1_b
    g1_either = l1_a or l1_b
    exploratory_only = g1_either and not g1_both

    frozen_id = _candidate_identity(frozen_candidate)
    seed_b_id = _candidate_identity(seed_b_evaluated_candidate)
    frozen_candidate_match = bool(frozen_id) and frozen_id == seed_b_id

    promotion_eligible = (
        g1_both
        and frozen_candidate is not None
        and frozen_candidate_match
        and bool(seed_a_report.get("gates", {}).get("predictor") == "pass")
        and bool(seed_b_report.get("gates", {}).get("predictor") == "pass")
    ) or (g0_both and la == 0 and lb == 0)

    notes: list[str] = []
    if exploratory_only:
        notes.append("G1 on one seed only — exploratory, not pre-registered pass")
    if g1_both and not frozen_candidate_match:
        notes.append("Seed B did not evaluate the frozen Seed A candidate — not promotion eligible")
    if g1_both and frozen_candidate_match:
        notes.append("Frozen candidate replicated on both seeds")
    if g0_both:
        notes.append("G0 on both seeds — negative evidence supported")

    return ReplicationReport(
        seed_a=sa,
        seed_b=sb,
        grade_a=la,
        grade_b=lb,
        g0_both=g0_both,
        g1_both=g1_both,
        g1_either=g1_either,
        exploratory_only=exploratory_only,
        promotion_eligible=promotion_eligible,
        frozen_candidate_match=frozen_candidate_match,
        notes=notes,
    )


def replication_to_dict(report: ReplicationReport) -> dict[str, Any]:
    return {
        "seed_a": report.seed_a,
        "seed_b": report.seed_b,
        "grade_a": report.grade_a,
        "grade_b": report.grade_b,
        "g0_both": report.g0_both,
        "g1_both": report.g1_both,
        "g1_either": report.g1_either,
        "exploratory_only": report.exploratory_only,
        "promotion_eligible": report.promotion_eligible,
        "frozen_candidate_match": report.frozen_candidate_match,
        "notes": report.notes,
    }
