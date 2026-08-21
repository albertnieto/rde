"""Real RDE Metric registrations for tsp_landscape_stats (EXP-063).

Audited 2026-08-19: the targets (near_optimal_fraction etc.) were
originally hand-computed inside `primitive_features()` with keys
hand-prefixed `"metric.<name>"`. That looks like a metric but isn't one --
`write_clean_discovery_run()` (src/rde/experiment/merge.py) only preserves
the target for the discovery loop from `features.jsonl`'s *registered*
`Metric.score()` output (`row["metrics"]`), never from
`instance_features.jsonl`'s `scalars` (which get hard-filtered to
matrix./graph.-prefixed keys only, regardless of the domain contract). The
target silently disappeared from every row reaching the discovery loop --
confirmed via a smoke test (0/600 finite target values in the "clean" run)
that produced a vacuous NULL verdict and then crashed `gate.finalize()` on
unmeasured decisive criteria (`cross_n_stability_median`,
`generator_separation`, both NaN with no non-degenerate target to compute
them from). Registering these as real Metrics, matching the pattern every
other RDE domain uses (see e.g. rde_domains/qubo_encoding/metrics.py), is
the actual fix -- not a workaround.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rde.core.protocols import QueryIntent
from rde.core.registry import Registry

from rde_domains.tsp.landscape_stats import cost_landscape_stats

_CACHE_KEY = "_tsp_landscape_stats_cache"


def _stats(descriptors: dict[str, Any]) -> dict[str, float]:
    """Compute cost_landscape_stats once per scoring pass, cached on `descriptors`.

    `descriptors` (== worker.py's `scoring_context`) is the same dict object
    passed to every metric function in one scoring pass, so caching the
    result on it avoids recomputing the same stats dict up to 6 times per
    instance.
    """
    cached = descriptors.get(_CACHE_KEY)
    if cached is not None:
        return cached
    costs = descriptors.get("costs")
    if costs is None:
        raise KeyError(
            "tsp_landscape_stats metrics require domain-prepared 'costs'; "
            "call primitive_features/prepare_instance first"
        )
    stats = cost_landscape_stats(np.asarray(costs, dtype=float))
    descriptors[_CACHE_KEY] = stats
    return stats


def register_landscape_stats_metrics(registry: Registry) -> None:
    """Register the real (N-1)!/2-brute-force tour-cost-landscape metrics."""

    def near_optimal_fraction(instance, slice_, descriptors: dict[str, Any]) -> float:
        return _stats(descriptors)["near_optimal_fraction"]

    def cost_cv(instance, slice_, descriptors: dict[str, Any]) -> float:
        return _stats(descriptors)["cost_cv"]

    def cost_spectral_gap_ratio(instance, slice_, descriptors: dict[str, Any]) -> float:
        return _stats(descriptors)["cost_spectral_gap_ratio"]

    def cost_min(instance, slice_, descriptors: dict[str, Any]) -> float:
        return _stats(descriptors)["cost_min"]

    def cost_mean(instance, slice_, descriptors: dict[str, Any]) -> float:
        return _stats(descriptors)["cost_mean"]

    def near_optimal_fraction_leak_audit_control(instance, slice_, descriptors: dict[str, Any]) -> float:
        # Deliberate exact duplicate of near_optimal_fraction. Present only
        # so ExperimentGate.check_leak_audit's raw-vs-clean self-test has an
        # unambiguous positive control -- audited 2026-08-19: none of the
        # real statistics or gen.*/spectral.*/stats.* descriptors
        # auto-derived from the tour-cost slice happen to reach the
        # required raw |r| > 0.99 with near_optimal_fraction (each measures
        # something genuinely different, a property of the honest design,
        # not a bug) -- so there was nothing for the self-test to confirm
        # without this. Never used as a scientific finding; marked
        # TARGET_DERIVED/predictor-ineligible in the domain contract.
        return _stats(descriptors)["near_optimal_fraction"]

    registry.register_metric_fn("near_optimal_fraction", QueryIntent.RANK, near_optimal_fraction)
    registry.register_metric_fn("cost_cv", QueryIntent.EVALUATE, cost_cv)
    registry.register_metric_fn("cost_spectral_gap_ratio", QueryIntent.EVALUATE, cost_spectral_gap_ratio)
    registry.register_metric_fn("cost_min", QueryIntent.EVALUATE, cost_min)
    registry.register_metric_fn("cost_mean", QueryIntent.EVALUATE, cost_mean)
    registry.register_metric_fn(
        "near_optimal_fraction_leak_audit_control",
        QueryIntent.RANK,
        near_optimal_fraction_leak_audit_control,
    )


__all__ = ["register_landscape_stats_metrics"]
