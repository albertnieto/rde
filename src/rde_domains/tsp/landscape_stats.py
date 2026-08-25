"""Real tour-cost-distribution statistics domain.

A mechanism-agnostic TSP target, chosen after two failure modes that are
worth documenting since they are easy to reintroduce by accident:

1. `walsh_top_k_energy` (a quadratic-QUBO-penalty cost function's Walsh
   spectrum): provably degree<=2 by the cost function's own algebra --
   guaranteed constant regardless of D, verified empirically to machine
   precision (std ~1e-15 across 1200 instances).
2. A validity-exact edge-indicator landscape (real Hamiltonian-cycle check,
   huge fixed penalty for invalid subsets): the fixed {valid, invalid}
   support pattern is itself purely combinatorial (same 60-out-of-32768
   bitstrings on every N=6 instance, independent of geometry), and the
   penalty's scale dominates the spectrum enough that degree-by-degree
   energy fractions were again constant to 5 decimal places across 15
   instances -- a broader version of the same disease: a fixed
   combinatorial skeleton dominating a padded/inflated domain.

This domain sidesteps both: no penalty, no inflated 2^(N(N-1)/2) domain, no
Walsh transform at all. It works directly with the real distribution of
costs over the (N-1)! genuine closed tours (`all_tour_costs`, exact
brute-force enumeration, same cap discipline as `TspSynthesisDomain`).
Every statistic below is computed only from genuine, D-dependent tour
costs -- there is no fixed skeleton left to dominate.

The question this domain's population + RDE's descriptor/expression search
is built to answer: which structural descriptors of the raw distance
matrix D predict these landscape statistics -- i.e. can a cheap classical
proxy characterize, in advance, what an expensive exact quantity would be.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rde.core.instance import InstanceRecord
from rde.core.limits import validate_bruteforce_size
from rde.core.protocols import SimpleFamilySlice
from rde.representation import rank_representations
from rde_domains.tsp.components import all_tour_costs, distance_matrix, random_euclidean_points
from rde_domains.tsp.representations import upper_triangular_distances

#: Same cap as TspSynthesisDomain, same reason: (N-1)! grows faster than any
#: fixed exponential base. N=9 -> 8! = 40320 tours per instance, still fast
#: at population scale; N=10 -> 362880, the practical ceiling for a
#: multi-day, many-instance, many-size campaign.
DEFAULT_MAX_BRUTEFORCE_N = 9

#: Fixed neighborhood width for the near-optimal-tour-count statistic,
#: matching the standard "within 5% of optimal" convention in the TSP
#: approximation-algorithms literature -- not tuned per instance.
NEAR_OPTIMAL_EPSILON = 0.05

_FAMILIES = ("clustered", "circulant_broken", "uniform")


def _points_on_circle(n: int, radius: float) -> np.ndarray:
    theta = 2.0 * np.pi * np.arange(n) / n
    return np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)


def _clustered_points(rng: np.random.Generator, n: int, gap_ratio: float = 20.0, spread: float = 0.03) -> np.ndarray:
    a = n // 2
    b = n - a
    centers = rng.uniform(0, 1, size=(2, 2))
    while np.linalg.norm(centers[0] - centers[1]) < gap_ratio * spread * 0.5:
        centers = rng.uniform(0, 1, size=(2, 2))
    pts_a = centers[0] + rng.normal(0, spread, size=(a, 2))
    pts_b = centers[1] + rng.normal(0, spread, size=(b, 2))
    pts = np.vstack([pts_a, pts_b])
    return pts[rng.permutation(n)]


def cost_landscape_stats(costs: np.ndarray, *, epsilon: float = NEAR_OPTIMAL_EPSILON) -> dict[str, float]:
    """Real, D-dependent statistics of a tour-cost distribution.

    No free parameters beyond the fixed, stated `epsilon`. Every value here
    is computed only from genuine tour costs -- nothing padded, nothing
    penalized, no fixed combinatorial skeleton to dominate the result.
    """
    c = np.asarray(costs, dtype=float)
    c_min = float(c.min())
    mean = float(c.mean())
    std = float(c.std())
    sorted_c = np.sort(c)
    second_best = float(sorted_c[1]) if sorted_c.size > 1 else c_min
    near_optimal = c <= c_min * (1.0 + epsilon)
    return {
        "cost_min": c_min,
        "cost_mean": mean,
        "cost_cv": std / mean if mean > 1e-15 else 0.0,
        "cost_spectral_gap_ratio": (second_best - c_min) / c_min if c_min > 1e-15 else 0.0,
        "near_optimal_fraction": float(near_optimal.sum()) / float(c.size),
        "n_tours": float(c.size),
    }


class TspLandscapeStatsDomain:
    """Mixed-family Observe-layer domain over real (not padded/penalized) tour-cost statistics."""

    def __init__(
        self,
        *,
        domain_id: str = "tsp_landscape_stats",
        max_symmetry_break: float = 0.6,
        max_bruteforce_n: int | None = DEFAULT_MAX_BRUTEFORCE_N,
    ) -> None:
        self.domain_id = domain_id
        self.max_symmetry_break = max_symmetry_break
        self.max_bruteforce_n = max_bruteforce_n

    def generate(self, n: int, size: int, seed: int) -> list[InstanceRecord]:
        validate_bruteforce_size(size, self.max_bruteforce_n)
        instances: list[InstanceRecord] = []
        for i in range(n):
            inst_seed = seed + i
            rng = np.random.default_rng(inst_seed)
            family = _FAMILIES[i % len(_FAMILIES)]
            if family == "clustered":
                points = _clustered_points(rng, size)
            elif family == "circulant_broken":
                symmetry_break = float(rng.uniform(0.0, self.max_symmetry_break))
                points = _points_on_circle(size, radius=1.0)
                if symmetry_break > 0:
                    points = points + rng.normal(0.0, symmetry_break, size=points.shape)
            else:
                points = random_euclidean_points(rng, size)
            d = distance_matrix(points)
            instances.append(
                InstanceRecord(
                    domain_id=self.domain_id,
                    size=size,
                    seed=inst_seed,
                    params={
                        "points": points.tolist(),
                        "D": d.tolist(),
                        #: Read by rde.runtime.worker for the `generator` row
                        #: column -- the held-out-family mechanism assess_outcome
                        #: / separation_score / DomainContract.held_out_generator_groups
                        #: key off of.
                        "generator": family,
                    },
                )
            )
        return instances

    def prepare_instance(
        self, instance: InstanceRecord, *, indices: list[int] | None = None
    ) -> dict[str, Any]:
        """Compute the (expensive, factorial) brute-force tour costs once per instance.

        `materialize()` and `primitive_features()` both need it; without this
        cache each was independently re-running the full (N-1)!/2-permutation
        enumeration, doubling the dominant cost of a multi-day campaign for
        no reason (caught during the pre-execution audit, 2026-08-19).
        """
        d = np.asarray(instance.params["D"], dtype=float)
        costs = all_tour_costs(d, list(range(d.shape[0])))
        return {"D": d, "costs": costs}

    def materialize(
        self, instance: InstanceRecord, index: int, *, cache: dict[str, Any] | None = None
    ) -> SimpleFamilySlice:
        # Exposes the real tour-cost array as the family slice -- legitimate
        # and intended (Phase 6/Walsh exploration should be able to see it,
        # and it's what makes ExperimentGate.check_leak_audit's raw-leak
        # self-test meaningful: descriptors auto-derived from it, e.g.
        # `gen.*`/`spectral.*`/`stats.*`, correlate strongly with the
        # metric.* targets computed from the same array -- by design, and
        # explicitly marked ENUMERATED_ORACLE/predictor-ineligible in the
        # domain contract, same pattern as DEFAULT_TRAJECTORY_SPECS's
        # `landscape.costs.*`/`spectral.*`/etc. entries). What is NOT
        # legitimate is treating any of those as a predictor -- the contract
        # enforces that, not this method.
        c = cache if cache is not None else self.prepare_instance(instance)
        costs = c["costs"]
        return SimpleFamilySlice(values=np.sort(costs), index=index, kind="tour_costs")

    def primitive_features(
        self, instance: InstanceRecord, *, cache: dict[str, Any] | None = None
    ) -> dict[str, float | np.ndarray]:
        # Audited 2026-08-19: this used to hand-prefix stat keys with
        # "metric." directly in this dict. That does NOT make them real RDE
        # metrics -- write_clean_discovery_run() only preserves the target
        # from features.jsonl's registered-Metric `row["metrics"]` output,
        # and instance_features.jsonl's `scalars` (where these hand-prefixed
        # keys actually land) get hard-filtered to matrix./graph.-prefixed
        # keys only. Net effect: the target silently vanished from every
        # "clean" discovery row (verified: 0/600 finite values in a smoke
        # test), producing a vacuous NULL verdict and crashing
        # gate.finalize() on unmeasured decisive criteria. Real targets are
        # now registered as proper Metrics in landscape_stats_metrics.py;
        # "costs" is exposed here as a named array primitive so those metric
        # functions (and the anticipated `landscape.costs.*` auto-descriptor
        # family, already excluded in the domain contract) can read it.
        c = cache if cache is not None else self.prepare_instance(instance)
        d, costs = c["D"], c["costs"]
        out: dict[str, float | np.ndarray] = {
            "n_cities": float(d.shape[0]),
            "D": d,
            "costs": costs,
        }
        #: `rde.representation`'s best-achievable roundtrip complexity for this
        #: instance's own upper-triangular distance profile -- computed purely
        #: from D (never touches tour costs or the metric.* targets derived
        #: from them), so it is leak-free by the same argument as `matrix.D.*`
        #: itself. Same pattern as hsp_functions/domain.py's `repr.best_complexity`.
        profile = upper_triangular_distances(d)
        ranked = rank_representations(profile[None, :], n=profile.shape[0])
        verified_complexities = [cand.complexity for cand in ranked if cand.certificate.status == "verified"]
        if verified_complexities:
            out["repr.best_complexity"] = float(min(verified_complexities))
        return out


def landscape_stats_domain(**kwargs) -> TspLandscapeStatsDomain:
    return TspLandscapeStatsDomain(**kwargs)
