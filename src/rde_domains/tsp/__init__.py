"""Backward-synthesis (Mode 2) domain over Euclidean TSP instances.

Treatment/control pair: `tsp_clustered` (planted, well-separated spatial
clusters) and `tsp_uniform_control` (no planted structure). See `domain.py`
for why exact TSP decomposition is a harder problem than block-separable
optimization, and why the control's expected "nothing accepted" outcome is
part of the result.
"""

from rde_domains.tsp.circulant import (
    TspCirculantSymmetryDomain,
    circulant_deviation,
    circulant_symmetry_domain,
)
from rde_domains.tsp.landscape import (
    TspCostLandscapeDomain,
    cost_landscape_domain,
    full_landscape,
    walsh_top_k_energy,
)
from rde_domains.tsp.landscape_stats import (
    NEAR_OPTIMAL_EPSILON,
    TspLandscapeStatsDomain,
    cost_landscape_stats,
    landscape_stats_domain,
)
from rde_domains.tsp.components import (
    all_pairs_open_path_optimal,
    closed_tour_optimal,
    distance_matrix,
    mst_edges,
    mst_gap_groups,
    open_path_optimal,
    path_cost,
    random_euclidean_points,
    tour_cost,
)
from rde_domains.tsp.domain import (
    DEFAULT_MAX_BRUTEFORCE_N,
    DEFAULT_MAX_LEAF_WIDTH,
    TspSynthesisDomain,
)

__all__ = [
    "DEFAULT_MAX_BRUTEFORCE_N",
    "DEFAULT_MAX_LEAF_WIDTH",
    "NEAR_OPTIMAL_EPSILON",
    "TspCirculantSymmetryDomain",
    "TspCostLandscapeDomain",
    "TspLandscapeStatsDomain",
    "TspSynthesisDomain",
    "all_pairs_open_path_optimal",
    "circulant_deviation",
    "circulant_symmetry_domain",
    "closed_tour_optimal",
    "cost_landscape_domain",
    "cost_landscape_stats",
    "distance_matrix",
    "full_landscape",
    "landscape_stats_domain",
    "mst_edges",
    "mst_gap_groups",
    "open_path_optimal",
    "path_cost",
    "random_euclidean_points",
    "tour_cost",
    "walsh_top_k_energy",
]


def clustered_domain(**kwargs) -> TspSynthesisDomain:
    """Planted-structure registration (the treatment)."""
    kwargs.setdefault("domain_id", "tsp_clustered")
    kwargs.setdefault("planted", True)
    return TspSynthesisDomain(**kwargs)


def uniform_control_domain(**kwargs) -> TspSynthesisDomain:
    """No-planted-structure registration (the control)."""
    kwargs.setdefault("domain_id", "tsp_uniform_control")
    kwargs.setdefault("planted", False)
    return TspSynthesisDomain(**kwargs)
