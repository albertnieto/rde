"""Gated outcome assessment.

Science grades are G0–G5 (`rde/docs/methodology.md` §6).
`to_payload()` writes `grade` / `g{k}_met` only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from rde.analyze.calibration import separation_score
from rde.analyze.controls import nr_baseline_r2
from rde.analyze.query import correlate_with_target, cross_n_report, kmeans_clusters
from rde.analyze.tables import numeric_columns


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    NOT_EVALUATED = "not_evaluated"


@dataclass
class GatedOutcomes:
    null_language: GateStatus = GateStatus.NOT_EVALUATED
    predictor: GateStatus = GateStatus.NOT_EVALUATED
    hidden_class: GateStatus = GateStatus.NOT_EVALUATED
    recurrence: GateStatus = GateStatus.NOT_EVALUATED
    representation: GateStatus = GateStatus.NOT_EVALUATED
    constructive: GateStatus = GateStatus.NOT_EVALUATED
    obstruction: GateStatus = GateStatus.NOT_EVALUATED

    def to_dict(self) -> dict[str, str]:
        return {k: v.value for k, v in self.__dict__.items()}


#: Below this relative std (std / max(|mean|, 1.0)), a target is treated as
#: numerically constant -- floating-point noise, not real signal. Chosen as
#: ~1e6x machine epsilon (float64 eps ~2.2e-16): generous enough not to
#: false-positive on a genuinely small-but-real effect, but far enough above
#: eps to catch the case this guards against (Direction E, 2026-08-19: a
#: `degree2_entropy` target that was mathematically guaranteed constant by
#: the cost function's own algebra -- std ~1.25e-15 across 1200 instances --
#: still produced `outcome_grade_hint=5` because nothing anywhere in the
#: gate chain checked whether the target actually varied before letting
#: Phase 6 rediscovery / symbolic R^2 claims through).
_DEGENERATE_TARGET_RELATIVE_STD = 1e-9
_DEGENERATE_TARGET_ABSOLUTE_STD = 1e-12


def assess_target_degeneracy(rows: list[dict[str, Any]], target: str) -> dict[str, Any]:
    """Check whether `target` has any real variance across `rows`.

    Returns a dict with `is_degenerate`, `std`, `mean`, `relative_std`, `n`.
    A degenerate target means every downstream fit statistic (R^2,
    correlation, "rediscovery") is numerically meaningless -- any expression
    can achieve a perfect-looking fit to something that doesn't vary, and
    that must never be reported as a discovery.
    """
    y = np.array([float(r.get(target, float("nan"))) for r in rows], dtype=float)
    finite = y[np.isfinite(y)]
    if finite.size < 2:
        return {
            "is_degenerate": True,
            "std": float("nan"),
            "mean": float("nan"),
            "relative_std": float("nan"),
            "n": int(finite.size),
            "reason": "fewer than 2 finite target values",
        }
    std = float(np.std(finite))
    mean = float(np.mean(finite))
    relative_std = std / max(abs(mean), 1.0)
    is_degenerate = std < _DEGENERATE_TARGET_ABSOLUTE_STD or relative_std < _DEGENERATE_TARGET_RELATIVE_STD
    reason = ""
    if is_degenerate:
        reason = (
            f"target {target!r} has std={std:.3e} (relative_std={relative_std:.3e}) "
            f"across n={finite.size} finite values -- numerically constant, not a "
            "real regression target. Any R^2/correlation/rediscovery claim against "
            "it is meaningless and must not be promoted."
        )
    return {
        "is_degenerate": is_degenerate,
        "std": std,
        "mean": mean,
        "relative_std": relative_std,
        "n": int(finite.size),
        "reason": reason,
    }


@dataclass
class OutcomeAssessment:
    """Result of G0–G5 gates."""

    grade: int
    g0_met: bool
    g1_met: bool
    g2_met: bool
    g3_met: bool
    g4_met: bool
    g5_met: bool
    target: str
    target_degenerate: bool = False
    criteria: dict[str, Any] = field(default_factory=dict)
    g1_triggers: list[str] = field(default_factory=list)
    g0_triggers: list[str] = field(default_factory=list)
    g2_triggers: list[str] = field(default_factory=list)
    g3_triggers: list[str] = field(default_factory=list)
    g4_triggers: list[str] = field(default_factory=list)
    g5_triggers: list[str] = field(default_factory=list)
    promotion_blocked: bool = False
    negative_outcome: bool = False
    gates: GatedOutcomes = field(default_factory=GatedOutcomes)
    headline: str = ""

    def to_payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "grade": int(self.grade),
            "target": self.target,
            "target_degenerate": self.target_degenerate,
            "promotion_blocked": self.promotion_blocked,
            "negative_outcome": self.negative_outcome,
            "headline": self.headline,
            "gates": self.gates.to_dict(),
            "criteria": dict(self.criteria),
        }
        for k in range(6):
            out[f"g{k}_met"] = bool(getattr(self, f"g{k}_met"))
            out[f"g{k}_triggers"] = list(getattr(self, f"g{k}_triggers"))
        return out


# Criteria the G0 null-language gate reads. A NULL / WEAK_SUBTHRESHOLD
# verdict rests on all of them: each one *not* firing is what "no detectable
# structure" means, so a criterion that never computed is indistinguishable from
# a criterion that computed and found nothing. Note that the thresholds below
# compare with `>=`, which is False for NaN either way — the value is silently
# read as "did not fire". `ExperimentGate.finalize` uses this list to refuse a
# verdict resting on an unevaluated criterion.
NULL_LANGUAGE_CRITERIA: tuple[str, ...] = (
    "best_abs_pearson_r",
    "best_expression_r_squared",
    "max_latent_target_correlation",
    "cross_n_stability_median",
    "generator_separation",
)

# Criteria the G1 predictor gate reads.
PREDICTOR_CRITERIA: tuple[str, ...] = (
    "best_expression_r_squared",
    "best_extrapolation_r_squared",
    "nr_baseline_r2",
    "stable_cross_n_descriptor",
)


def decisive_criteria_for(assessment: "OutcomeAssessment") -> list[str]:
    """The criteria this verdict actually rests on.

    Kept next to the thresholds that define the decision rule so an experiment
    cannot drift from it by hardcoding its own list. A G0 verdict rests on
    the null-language criteria; claiming a predictor additionally rests on the
    G1 criteria.
    """
    names = list(NULL_LANGUAGE_CRITERIA)
    if assessment.grade >= 1:
        names += [c for c in PREDICTOR_CRITERIA if c not in names]
    return names


def _median_cross_n_stability(rows: list[dict[str, Any]], target: str, top_k: int = 20) -> float:
    report = cross_n_report(rows, target, min_abs_r=0.0)
    if not report:
        return float("nan")
    vals = [float(h.get("cross_n_sign_stability", float("nan"))) for h in report[:top_k]]
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.median(finite)) if finite else float("nan")


def _stable_cross_n_descriptor(rows: list[dict[str, Any]], target: str) -> bool:
    report = cross_n_report(rows, target, min_abs_r=0.5)
    if not report:
        return False
    sizes = sorted({int(r["size"]) for r in rows if r.get("size") is not None})
    if len(sizes) < 2:
        return False
    stable_bins = sum(
        1 for hit in report[:20] if float(hit.get("cross_n_sign_stability", 0.0)) >= 0.75
    )
    return stable_bins >= max(1, int(0.75 * min(len(report), 20)))


_LEVEL_2_META_KEYS = frozenset(
    {
        "run_id",
        "instance_id",
        "domain_id",
        "size",
        "seed",
        "family_index",
        "slice_kind",
        "generator",
        "array_size",
    }
)
_LEVEL_2_CLUSTER_TOP_K = 12
_LEVEL_2_MAX_ROWS = 65536
_LEVEL_2_SAMPLE_SEED = 0


def _level_2_sample_rows(
    rows: list[dict[str, Any]],
    *,
    max_rows: int = _LEVEL_2_MAX_ROWS,
    seed: int = _LEVEL_2_SAMPLE_SEED,
) -> tuple[list[dict[str, Any]], int, bool]:
    """Deterministic subsample for k-means when row count exceeds ``max_rows``."""
    total = len(rows)
    if total <= max_rows:
        return rows, total, False
    idx = np.sort(np.random.default_rng(seed).choice(total, size=max_rows, replace=False))
    sampled = [rows[int(i)] for i in idx]
    return sampled, total, True


def _g2_cluster_profiles(
    rows: list[dict[str, Any]],
    labels: list[int] | np.ndarray,
    target: str,
    columns: list[str],
) -> list[dict[str, Any]]:
    """Per-cluster target stats and mean descriptor values used for clustering."""
    lab = np.asarray(labels, dtype=int)
    y = np.array([float(r.get(target, float("nan"))) for r in rows], dtype=float)
    profiles: list[dict[str, Any]] = []
    for cluster_id in sorted(set(int(x) for x in lab.tolist())):
        mask = lab == cluster_id
        n = int(mask.sum())
        profile: dict[str, Any] = {
            "cluster": cluster_id,
            "n": n,
            "fraction": float(n / len(rows)) if rows else float("nan"),
            "target_mean": float(np.nanmean(y[mask])) if n else float("nan"),
            "target_std": float(np.nanstd(y[mask])) if n else float("nan"),
            "descriptor_means": {},
        }
        for col in columns:
            raw = [r.get(col) for r in rows]
            vals = np.array(
                [
                    float(v) if isinstance(v, (int, float)) and np.isfinite(float(v)) else float("nan")
                    for v in raw
                ],
                dtype=float,
            )[mask]
            profile["descriptor_means"][col] = float(np.nanmean(vals)) if n else float("nan")
        profiles.append(profile)
    profiles.sort(key=lambda item: -float(item.get("target_mean", float("nan"))))
    return profiles


def _g2_cluster_columns(
    rows: list[dict[str, Any]],
    target: str,
    *,
    top_k: int = _LEVEL_2_CLUSTER_TOP_K,
    domain_id: str | None = None,
) -> list[str]:
    """Pick k-means features: top |r| vs target among leak-clean descriptors."""
    from rde.analyze.leak_audit import FeatureClass, classify_feature

    hits = correlate_with_target(rows, target, min_abs_r=0.0)
    ranked = sorted(hits, key=lambda hit: -abs(float(hit["pearson_r"])))
    eligible: list[str] = []
    for hit in ranked:
        col = str(hit["column"])
        if col == target or col.startswith("metric.") or col in _LEVEL_2_META_KEYS:
            continue
        feature_class = classify_feature(col, target=target, domain_id=domain_id)
        if feature_class in {
            FeatureClass.OUTCOME_LEAK,
            FeatureClass.TARGET_DERIVED,
            FeatureClass.METADATA,
        }:
            continue
        eligible.append(col)
        if len(eligible) >= top_k:
            break

    if len(eligible) >= 2:
        return eligible

    for col in numeric_columns(rows):
        if col == target or col.startswith("metric.") or col in _LEVEL_2_META_KEYS:
            continue
        feature_class = classify_feature(col, target=target, domain_id=domain_id)
        if feature_class in {
            FeatureClass.OUTCOME_LEAK,
            FeatureClass.TARGET_DERIVED,
            FeatureClass.METADATA,
        }:
            continue
        if col not in eligible:
            eligible.append(col)
        if len(eligible) >= top_k:
            break
    return eligible


def _assess_g2(
    rows: list[dict[str, Any]],
    target: str,
    *,
    domain_id: str | None = None,
) -> tuple[bool, list[str], dict[str, Any]]:
    cols = _g2_cluster_columns(rows, target, domain_id=domain_id)
    if len(cols) < 2:
        return False, [], {"cluster_separation": float("nan"), "cluster_columns": cols}
    cluster_rows, total_rows, sampled = _level_2_sample_rows(rows)
    km = kmeans_clusters(cluster_rows, cols, k=2)
    labels = km.get("labels", [])
    if len(labels) != len(cluster_rows):
        return False, [], {
            "cluster_separation": float("nan"),
            "cluster_columns": cols,
            "cluster_rows_total": total_rows,
            "cluster_rows_used": len(cluster_rows),
            "cluster_sampled": sampled,
        }
    y = np.array([float(r.get(target, float("nan"))) for r in cluster_rows], dtype=float)
    lab = np.array(labels, dtype=int)
    means = []
    for j in range(km["k"]):
        mask = lab == j
        if mask.any():
            means.append(float(np.nanmean(y[mask])))
    sep = abs(means[0] - means[1]) if len(means) >= 2 else 0.0
    std = float(np.nanstd(y)) if np.isfinite(y).any() else 1.0
    norm_sep = sep / max(std, 1e-12)
    triggers = []
    if norm_sep >= 0.75:
        triggers.append("bimodal_target_separation")
    if separation_score(rows, metric=target, label_key="generator") >= 2.0:
        triggers.append("generator_separation")
    met = len(triggers) > 0
    return met, triggers, {
        "cluster_separation": norm_sep,
        "kmeans_k": km["k"],
        "cluster_columns": cols,
        "cluster_column_selection": "top_abs_r_leak_clean",
        "cluster_rows_total": total_rows,
        "cluster_rows_used": len(cluster_rows),
        "cluster_sampled": sampled,
        "cluster_profiles": _g2_cluster_profiles(cluster_rows, labels, target, cols),
    }


def _assess_recurrence(rows: list[dict[str, Any]]) -> tuple[bool, list[str], dict[str, Any]]:
    eligible_vals = [
        float(r.get("recurrence.identification_eligible", 0.0))
        for r in rows
        if np.isfinite(float(r.get("recurrence.identification_eligible", float("nan"))))
    ]
    if not eligible_vals or max(eligible_vals) < 1.0:
        return False, [], {"recurrence_gate": "not_evaluated_insufficient_slices"}

    orders = [
        float(r.get("recurrence.estimated_order", float("nan")))
        for r in rows
        if np.isfinite(float(r.get("recurrence.estimated_order", float("nan"))))
    ]
    if not orders:
        return False, [], {"recurrence_gate": "not_evaluated_no_order"}

    sizes = [int(r["size"]) for r in rows if r.get("size") is not None]
    max_n = max(sizes) if sizes else 6
    krylov_cap = 2**max_n
    median_order = float(np.median(orders))
    ratio = median_order / max(krylov_cap, 1.0)

    withheld_stable = [
        float(r.get("recurrence.withheld_stable", 0.0))
        for r in rows
        if np.isfinite(float(r.get("recurrence.withheld_stable", float("nan"))))
    ]
    triggers: list[str] = []
    eligible = max(eligible_vals) >= 1.0
    if (
        eligible
        and median_order <= 32
        and ratio < 1e-6
        and withheld_stable
        and min(withheld_stable) >= 1.0
    ):
        triggers.append("finite_krylov_closure_withheld_stable")
    met = len(triggers) > 0
    return met, triggers, {
        "krylov_ratio_median": ratio,
        "recurrence_order_median": median_order,
        "recurrence_identification_eligible": True,
    }


def assess_outcome(
    rows: list[dict[str, Any]],
    target: str,
    *,
    discovery: dict[str, Any] | None = None,
    metric_candidates: list[dict[str, Any]] | None = None,
    latent: dict[str, Any] | None = None,
    phase6: dict[str, Any] | None = None,
    leak_audit_summary: dict[str, Any] | None = None,
    certify_result: dict[str, Any] | None = None,
    obstruct_summary: dict[str, Any] | None = None,
    domain_contract: Any | None = None,
    domain_id: str | None = None,
    frozen_candidate: dict[str, Any] | None = None,
    replication_passed: bool | None = None,
) -> OutcomeAssessment:
    """Apply v0.3 gated outcomes with legacy level annotations."""
    disc = discovery or {}
    cands = metric_candidates if metric_candidates is not None else disc.get("metric_candidates", [])
    lat = latent if latent is not None else disc.get("latent", {})
    p6 = phase6 if phase6 is not None else disc.get("phase6", {})

    degeneracy = assess_target_degeneracy(rows, target)
    target_degenerate = bool(degeneracy["is_degenerate"])

    corr_hits = correlate_with_target(rows, target, min_abs_r=0.0)
    best_abs_r = float(max((abs(h["pearson_r"]) for h in corr_hits), default=0.0))
    best_expr_r2 = float(max((c.get("r_squared", 0.0) for c in cands), default=0.0))
    best_extrap_r2 = float(
        max((c.get("extrapolation_r_squared", float("nan")) for c in cands), default=float("nan"))
    )
    if not np.isfinite(best_extrap_r2):
        best_extrap_r2 = float(disc.get("symbolic", {}).get("extrapolation_r_squared", float("nan")))

    latent_corrs = lat.get("target_correlations", {})
    max_latent_corr = float(max((abs(v) for v in latent_corrs.values()), default=0.0))
    ridge_r2 = float(lat.get("ridge_r_squared", float("nan")))
    ridge_extrap_r2 = float(lat.get("ridge_extrapolation_r_squared", float("nan")))
    traj_extrap = float(lat.get("trajectory_predictor", {}).get("extrapolation_r_squared", float("nan")))
    sym_r2 = float(disc.get("symbolic", {}).get("r_squared", 0.0))
    best_expr_r2 = max(best_expr_r2, sym_r2)

    rep = p6.get("representation") or {}
    state_recon_r2 = float(rep.get("state_reconstruction_r2", float("nan")))
    g4_from_rep = bool(rep.get("g4_met"))
    constructive_met = bool(rep.get("constructive_met", False))
    redisc = p6.get("rediscovery") or {}
    rediscovery_freq = float(redisc.get("rediscovery_frequency", 0.0))
    g5_from_redisc = bool(redisc.get("g5_met"))

    cross_n_median = _median_cross_n_stability(rows, target)
    sep = separation_score(rows, metric=target, label_key="generator")
    nr_baseline = nr_baseline_r2(rows, target)

    resolved_domain_id = domain_id
    if resolved_domain_id is None and domain_contract is not None:
        resolved_domain_id = getattr(domain_contract, "domain_id", None)
    g2_met, g2_triggers, l2_crit = _assess_g2(
        rows,
        target,
        domain_id=resolved_domain_id,
    )
    # G2 cluster separation is target_std-normalized (`_assess_g2`
    # divides by std, clamped to a 1e-12 floor) -- a degenerate target makes
    # that ratio numerically unstable/meaningless even though it looks like a
    # normal float. G3 (recurrence) is untouched: it depends only on
    # `recurrence.*` columns, never on this target's own variance.
    g2_met = g2_met and not target_degenerate
    g3_met, g3_triggers, l3_crit = _assess_recurrence(rows)

    promotion_blocked = bool(leak_audit_summary and leak_audit_summary.get("promotion_blocked"))
    promotion_blocked = promotion_blocked or target_degenerate
    certify_passed = bool(certify_result and certify_result.get("passed")) and not target_degenerate
    negative_outcome = bool(obstruct_summary and obstruct_summary.get("negative_outcome"))

    recurrence_applicable = True
    if domain_contract is not None:
        recurrence_applicable = bool(getattr(domain_contract, "recurrence_applicable", True))

    criteria = {
        "target_degenerate": target_degenerate,
        "target_degeneracy": degeneracy,
        "best_abs_pearson_r": best_abs_r,
        "best_expression_r_squared": best_expr_r2,
        "best_extrapolation_r_squared": best_extrap_r2,
        "max_latent_target_correlation": max_latent_corr,
        "ridge_on_pca_r_squared": ridge_r2,
        "ridge_extrapolation_r_squared": ridge_extrap_r2,
        "trajectory_extrapolation_r_squared": traj_extrap,
        "cross_n_stability_median": cross_n_median,
        "generator_separation": sep,
        "nr_baseline_r2": nr_baseline,
        "stable_cross_n_descriptor": _stable_cross_n_descriptor(rows, target),
        "state_reconstruction_r2": state_recon_r2,
        "representation_g4_met": g4_from_rep,
        "constructive_met": constructive_met,
        "rediscovery_frequency": rediscovery_freq,
        "g5_rediscovery_met": g5_from_redisc,
        "certify_passed": certify_passed,
        "promotion_blocked": promotion_blocked,
        "negative_outcome": negative_outcome,
        "frozen_candidate_present": frozen_candidate is not None,
        "replication_passed": replication_passed,
        **l2_crit,
        **l3_crit,
    }

    null_language_triggers = []
    if best_abs_r >= 0.35:
        null_language_triggers.append("best_abs_pearson_r")
    if best_expr_r2 >= 0.40:
        null_language_triggers.append("best_expression_r_squared")
    if max_latent_corr >= 0.50:
        null_language_triggers.append("max_latent_target_correlation")
    if np.isfinite(cross_n_median) and cross_n_median >= 0.60:
        null_language_triggers.append("cross_n_stability_median")
    if np.isfinite(sep) and sep >= 2.0:
        null_language_triggers.append("generator_separation")
    # G0 is the preregistered null-language gate: "met" means no
    # detectable structure. The inversion is explicit so it is not
    # mistaken for a positive discovery.
    g0_met = len(null_language_triggers) == 0

    g1_triggers = []
    if (
        best_expr_r2 >= 0.75
        and best_expr_r2 > nr_baseline + 0.15
        and np.isfinite(best_extrap_r2)
        and best_extrap_r2 >= 0.65
    ):
        g1_triggers.append("best_expression_r_squared")
    if np.isfinite(best_extrap_r2) and best_extrap_r2 >= 0.65:
        g1_triggers.append("best_extrapolation_r_squared")
    if np.isfinite(ridge_extrap_r2) and ridge_extrap_r2 >= 0.80:
        g1_triggers.append("ridge_on_pca_r_squared")
    if np.isfinite(traj_extrap) and traj_extrap >= 0.70:
        g1_triggers.append("trajectory_extrapolation_r_squared")
    if criteria["stable_cross_n_descriptor"]:
        g1_triggers.append("stable_cross_n_descriptor")
    g1_met = len(g1_triggers) > 0 and not promotion_blocked

    g4_triggers = []
    if constructive_met:
        g4_triggers.append("constructive_encoder")
    elif g4_from_rep:
        g4_triggers.append("representation_encoder_decoder_diagnostic")
    if np.isfinite(state_recon_r2) and state_recon_r2 >= 0.70:
        g4_triggers.append("state_reconstruction_r2")
    g4_met = constructive_met and not target_degenerate

    g5_triggers = []
    if g5_from_redisc and frozen_candidate is not None:
        g5_triggers.append("cross_split_rediscovery_frozen")
    if certify_passed and frozen_candidate is not None:
        g5_triggers.append("certify_passed_frozen")
    g5_met = (
        g5_from_redisc
        and certify_passed
        and frozen_candidate is not None
        and replication_passed is True
    )

    gates = GatedOutcomes()
    gates.null_language = GateStatus.PASS if g0_met else GateStatus.FAIL
    gates.predictor = GateStatus.PASS if g1_met and frozen_candidate else GateStatus.FAIL
    if not g1_met and not promotion_blocked:
        gates.predictor = GateStatus.NOT_EVALUATED
    if promotion_blocked:
        gates.predictor = GateStatus.FAIL
    gates.hidden_class = GateStatus.PASS if g2_met else GateStatus.FAIL
    if recurrence_applicable:
        gates.recurrence = GateStatus.PASS if g3_met else (
            GateStatus.NOT_EVALUATED if l3_crit.get("recurrence_gate") else GateStatus.FAIL
        )
    else:
        gates.recurrence = GateStatus.NOT_APPLICABLE
    gates.representation = GateStatus.PASS if g4_from_rep else GateStatus.NOT_EVALUATED
    gates.constructive = GateStatus.PASS if constructive_met else GateStatus.NOT_EVALUATED
    gates.obstruction = GateStatus.PASS if negative_outcome else GateStatus.NOT_EVALUATED

    # g1_met (via promotion_blocked), g2_met, g4_met, and
    # g5_met (via certify_passed) are all already individually gated
    # against target_degenerate above -- each for the specific reason that
    # stage's computation actually depends on the target's variance. G3
    # (recurrence) is deliberately left alone: it's orthogonal to this
    # target's variance, so a degenerate target here must not block a
    # genuinely independent G3 finding.
    if g5_met:
        grade = 5
    elif g4_met:
        grade = 4
    elif g3_met:
        grade = 3
    elif g2_met:
        grade = 2
    elif g1_met:
        grade = 1
    else:
        grade = 0

    headline_parts = [
        f"{k}={v}" for k, v in gates.to_dict().items() if v != GateStatus.NOT_EVALUATED.value
    ]
    headline = "; ".join(headline_parts) if headline_parts else "no gates evaluated"
    if target_degenerate:
        headline = f"TARGET_DEGENERATE: {degeneracy['reason']} | {headline}"

    return OutcomeAssessment(
        grade=grade,
        g0_met=g0_met,
        g1_met=g1_met,
        g2_met=g2_met,
        g3_met=g3_met,
        g4_met=g4_met,
        g5_met=g5_met,
        target=target,
        target_degenerate=target_degenerate,
        criteria=criteria,
        g1_triggers=g1_triggers,
        g0_triggers=null_language_triggers,
        g2_triggers=g2_triggers,
        g3_triggers=g3_triggers,
        g4_triggers=g4_triggers,
        g5_triggers=g5_triggers,
        promotion_blocked=promotion_blocked,
        negative_outcome=negative_outcome,
        gates=gates,
        headline=headline,
    )
