"""Edge-indicator TSP cost-landscape domain.

Unlike `circulant.py` (which presupposes the mechanism -- abelian/circulant
symmetry -- and only characterizes detectability of that one hypothesis),
this domain exposes the full 2^(N(N-1)/2) edge-indicator cost landscape
itself as a family-slice array, without presupposing which structural
feature (if any) makes it exploitable. The point is to let RDE's actual
search machinery -- Phase 2b/3 descriptor+expression ranking, and Phase 6
Walsh generating-function search -- look broadly for exploitable structure
(Fourier/Walsh sparsity being the concrete, checkable signature of a hidden
abelian/(Z_2)^n structure a Simon-style algorithm could exploit) across a
population that mixes multiple instance families (clustered, circulant with
varying symmetry-breaking, uniform control) rather than one hand-picked
hypothesis.

Cost function: standard edge-indicator TSP-QUBO reduction (not itself
claimed novel -- a standard permutation-QUBO / edge-indexed encoding).
`cost(x) = sum of selected-edge weights + penalty * sum_v (degree(v)-2)^2`
for x in {0,1}^(N(N-1)/2). What's actually being searched here -- whether
this landscape has exploitable Walsh-sparse structure, and for which
instance families -- is the open question, not the encoding.

Kept to small N (default N=6, 15 edges, 2^15=32768-entry landscape) so the
full landscape is exactly enumerable and Walsh-transformable; this does not
scale to realistic N and is not intended to.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rde.core.instance import InstanceRecord
from rde.core.protocols import SimpleFamilySlice
from rde.features.fourier import walsh_hadamard
from rde_domains.tsp.circulant import _points_on_circle
from rde_domains.tsp.components import distance_matrix, random_euclidean_points

_FAMILIES = ("clustered", "circulant_broken", "uniform")


def _edges_for(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def full_landscape(D: np.ndarray, edges: list[tuple[int, int]], penalty_weight: float) -> np.ndarray:
    """Exact cost of all 2^len(edges) edge-indicator bitstrings, vectorized."""
    n = D.shape[0]
    n_edges = len(edges)
    all_x = np.arange(2**n_edges, dtype=np.uint32)
    shifts = np.arange(n_edges, dtype=np.uint32)
    bits = ((all_x[:, None] >> shifts[None, :]) & 1).astype(np.float64)
    weights = np.array([D[i, j] for i, j in edges])
    edge_cost = bits @ weights
    incidence = np.zeros((n, n_edges))
    for k, (i, j) in enumerate(edges):
        incidence[i, k] = 1.0
        incidence[j, k] = 1.0
    degrees = bits @ incidence.T
    penalty = ((degrees - 2.0) ** 2).sum(axis=1)
    return edge_cost + penalty_weight * penalty


def walsh_top_k_energy(costs: np.ndarray, k: int) -> float:
    """Fraction of Walsh-spectrum energy (excluding the DC term) in the top k coefficients.

    NOTE: for a cost function that is an exact degree-2 polynomial in the
    bits (as `full_landscape` is by construction: linear edge weights +
    quadratic degree penalty), this is trivially ~constant across every
    instance regardless of D or family -- verified empirically (2026-08-19):
    52.8%/47.2% degree-1/degree-2 split, degree>=3 exactly 0.0, essentially
    invariant across clustered/circulant/uniform families (std ~0.001-0.003
    around the same mean). Kept only as a sanity-check column, not a real
    target -- see `degree2_entropy` for the actual D-dependent quantity.
    """
    coeffs = walsh_hadamard(costs)
    ac = coeffs[1:]  # drop the constant/DC component
    e2 = np.sort(ac**2)[::-1]
    denom = e2.sum()
    if denom < 1e-15:
        return 0.0
    k = max(1, min(k, e2.size))
    return float(e2[:k].sum() / denom)


def _subset_degree(n_edges: int) -> np.ndarray:
    idx = np.arange(2**n_edges, dtype=np.int64)
    deg = np.zeros_like(idx)
    x = idx.copy()
    while x.any():
        deg += (x & 1)
        x >>= 1
    return deg


def degree2_entropy(costs: np.ndarray, n_edges: int) -> float:
    """Entropy of the normalized degree-2 Walsh coefficient magnitude distribution.

    Unlike `walsh_top_k_energy`, this is NOT trivially guaranteed by the
    cost function's polynomial degree: which pairwise (degree-2) edge
    interactions carry the most weight genuinely depends on the instance's
    own geometry (D), since the *support* (which vertex pairs share an
    edge) is fixed by the complete-graph structure but the *magnitudes*
    are not. Low entropy = a few dominant pairwise interactions (a
    concentrated, potentially exploitable landscape); high entropy = energy
    spread evenly across all degree-2 terms (generic, unstructured).
    """
    coeffs = walsh_hadamard(costs)
    deg = _subset_degree(n_edges)
    mask = deg == 2
    e2 = coeffs[mask] ** 2
    total = e2.sum()
    if total < 1e-15:
        return float("nan")
    p = e2 / total
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log(p)))


def _clustered_points(rng: np.random.Generator, n: int, gap_ratio: float = 20.0, spread: float = 0.03) -> np.ndarray:
    a = n // 2
    b = n - a
    centers = rng.uniform(0, 1, size=(2, 2))
    while np.linalg.norm(centers[0] - centers[1]) < gap_ratio * spread * 0.5:
        centers = rng.uniform(0, 1, size=(2, 2))
    pts_a = centers[0] + rng.normal(0, spread, size=(a, 2))
    pts_b = centers[1] + rng.normal(0, spread, size=(b, 2))
    pts = np.vstack([pts_a, pts_b])
    perm = rng.permutation(n)
    return pts[perm]


class TspCostLandscapeDomain:
    """Mixed-family Observe-layer domain exposing the full edge-indicator cost landscape."""

    def __init__(
        self,
        *,
        domain_id: str = "tsp_cost_landscape",
        max_symmetry_break: float = 0.6,
        penalty_scale: float = 5.0,
    ) -> None:
        self.domain_id = domain_id
        self.max_symmetry_break = max_symmetry_break
        self.penalty_scale = penalty_scale

    def generate(self, n: int, size: int, seed: int) -> list[InstanceRecord]:
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
                        "family": family,
                    },
                )
            )
        return instances

    def prepare_instance(
        self, instance: InstanceRecord, *, indices: list[int] | None = None
    ) -> dict[str, Any]:
        """Enumerate the 2^{n_edges} edge-indicator landscape once per instance.

        `materialize()` and `primitive_features()` both need `costs`; without
        this cache the worker independently re-ran `full_landscape` in each.
        """
        del indices
        d = np.asarray(instance.params["D"], dtype=float)
        edges = _edges_for(d.shape[0])
        costs = full_landscape(d, edges, penalty_weight=self.penalty_scale * float(d.max()))
        return {"D": d, "edges": edges, "costs": costs, "n_edges": len(edges)}

    def materialize(
        self, instance: InstanceRecord, index: int, *, cache: dict[str, Any] | None = None
    ) -> SimpleFamilySlice:
        c = cache if cache is not None and "costs" in cache else self.prepare_instance(instance)
        return SimpleFamilySlice(values=c["costs"], index=index, kind="edge_cost_landscape")

    def primitive_features(
        self, instance: InstanceRecord, *, cache: dict[str, Any] | None = None
    ) -> dict[str, float | np.ndarray]:
        c = cache if cache is not None and "costs" in cache else self.prepare_instance(instance)
        d, costs, n_edges = c["D"], c["costs"], c["n_edges"]
        return {
            "n_cities": float(d.shape[0]),
            "family_is_clustered": 1.0 if instance.params["family"] == "clustered" else 0.0,
            "family_is_circulant_broken": 1.0 if instance.params["family"] == "circulant_broken" else 0.0,
            #: Sanity-check column only -- trivially ~constant by construction,
            #: see the docstring on `walsh_top_k_energy`. Not a real target.
            "walsh_top_k_energy": walsh_top_k_energy(costs, k=n_edges),
            #: The real, D-dependent target: see `degree2_entropy` docstring.
            "degree2_entropy": degree2_entropy(costs, n_edges),
            "D": d,
        }


def cost_landscape_domain(**kwargs) -> TspCostLandscapeDomain:
    return TspCostLandscapeDomain(**kwargs)
