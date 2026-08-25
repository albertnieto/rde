"""`SynthesisDomain` over Euclidean TSP instances (Mode 2).

Decomposition moves must be *derived* from the raw distance-matrix data
itself, never handed in as a prior -- this module never reads or is told
the "correct" tour; the only structural signal it uses is
`components.mst_gap_groups`, which looks solely at the instance's own
distance matrix.

**Why this is a harder exactness problem than block-separable optimization.**
A block-diagonal cost function's components are *provably* non-interacting
-- zero off-diagonal coupling means the cost is exactly additive across
components, so independently-optimal sub-solutions combine into the exact
global optimum by construction (see `rde.testing.block_separable`). TSP has
no such free lunch: a single Hamiltonian cycle must connect every city, so
even maximally-separated spatial clusters interact through the (small
number of) edges that cross between them. There is no "truly independent
sub-problem" analogue here.

**What `combine` actually does about that.** For exactly two well-separated
clusters (this version only supports the pairwise case -- see
`decompose_divide`), the classical "clustered TSP" argument is: for a large
enough gap between clusters, the global optimum crosses between them exactly
twice (entering once, leaving once), so it decomposes into (a) an optimal
*open* Hamiltonian path through each cluster between two boundary cities,
plus (b) the two crossing edges, minimized jointly over every choice of
boundary-city pair in each cluster. `combine` performs that exact joint
search directly against the instance's own distance matrix -- it does not
trust or reuse the closed-tour solutions the harness independently computed
for each leaf, only the *city membership* of each cluster.

That joint search is exact **conditional on** the "exactly two crossings"
structural assumption holding for the specific generated instance. Whether it
actually holds -- and how the match rate depends on the generation gap ratio
-- is not asserted here; it is exactly the empirical question a campaign
against `brute_force` answers, the same way `tsp_uniform_control`'s "nothing
is accepted" is part of that domain's result rather than a bug to fix.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rde.core.instance import InstanceRecord
from rde.core.limits import validate_bruteforce_size
from rde.core.protocols import SimpleFamilySlice, SynthesisSolution
from rde_domains.tsp.components import (
    all_pairs_open_path_optimal,
    closed_tour_optimal,
    distance_matrix,
    mst_gap_groups,
    random_euclidean_points,
    tour_cost,
)

#: Brute force here is (n-1)!/2 permutations, not 2^n -- factorial beats any
#: fixed exponential base, so this cap must be much smaller than the QUBO
#: domain's. 9 cities -> 8!/2 = 20160 tours, still fast; 10 -> 181440.
DEFAULT_MAX_BRUTEFORCE_N = 9

#: Leaves passed to the exact joint-boundary combine pay (k-2)! per candidate
#: endpoint pair via `all_pairs_open_path_optimal`'s O(k^2) pairs -- keep
#: small independently of the top-level brute-force cap.
DEFAULT_MAX_LEAF_WIDTH = 6


def _instance_view(instance: Any) -> tuple[np.ndarray, tuple[int, ...]]:
    """Normalize a top-level record or a derived sub-instance to (D, variables)."""
    if isinstance(instance, InstanceRecord):
        d = np.asarray(instance.params["D"], dtype=float)
        return d, tuple(range(d.shape[0]))
    d = np.asarray(instance["D"], dtype=float)
    variables = instance.get("vars")
    if variables is None:
        return d, tuple(range(d.shape[0]))
    return d, tuple(int(v) for v in variables)


class TspSynthesisDomain:
    """`SynthesisDomain` (and minimal `Domain`) over Euclidean TSP instances."""

    def __init__(
        self,
        *,
        domain_id: str = "tsp_clustered",
        planted: bool = True,
        cluster_gap_ratio: float = 20.0,
        min_gap_ratio: float = 3.0,
        cluster_spread: float = 0.03,
        max_leaf_width: int = DEFAULT_MAX_LEAF_WIDTH,
        max_bruteforce_n: int | None = DEFAULT_MAX_BRUTEFORCE_N,
    ) -> None:
        self.domain_id = domain_id
        self.planted = planted
        #: Actual center-to-center separation used at *generation* time,
        #: relative to `cluster_spread` -- deliberately kept well above
        #: `min_gap_ratio` so the planted gap is comfortably detectable
        #: rather than borderline. 20.0 is not arbitrary: swept empirically
        #: (200 draws, N=6..9, seeds 0-49) against `min_gap_ratio=3.0` --
        #: 8.0 only detects the planted gap 7/20 times (Gaussian tails
        #: occasionally inflate intra-cluster spread enough to close the
        #: margin), 20.0 detects it 200/200 with observed MST-edge ratio
        #: never below ~4.2, a real safety margin above the 3.0 threshold,
        #: not just a passing average.
        self.cluster_gap_ratio = cluster_gap_ratio
        #: Threshold `components.mst_gap_groups` uses at *decomposition*
        #: time. Kept as a separate, independent parameter from
        #: `cluster_gap_ratio` on purpose: detection must work from the
        #: distance matrix alone, not from knowing what was planted.
        self.min_gap_ratio = min_gap_ratio
        self.cluster_spread = cluster_spread
        self.max_leaf_width = max_leaf_width
        self.max_bruteforce_n = max_bruteforce_n

    # ---- Domain protocol (generation / registry compatibility) ----

    def generate(self, n: int, size: int, seed: int) -> list[InstanceRecord]:
        validate_bruteforce_size(size, self.max_bruteforce_n)
        instances: list[InstanceRecord] = []
        for i in range(n):
            inst_seed = seed + i
            rng = np.random.default_rng(inst_seed)
            if self.planted:
                points, groups = self._clustered_instance(rng, size)
                params: dict[str, Any] = {
                    "points": points.tolist(),
                    "D": distance_matrix(points).tolist(),
                    "structure": "clustered",
                    "planted_groups": [list(g) for g in groups],
                }
            else:
                points = random_euclidean_points(rng, size)
                params = {
                    "points": points.tolist(),
                    "D": distance_matrix(points).tolist(),
                    "structure": "uniform",
                }
            instances.append(
                InstanceRecord(
                    domain_id=self.domain_id,
                    size=size,
                    seed=inst_seed,
                    params=params,
                )
            )
        return instances

    def _clustered_instance(
        self, rng: np.random.Generator, size: int
    ) -> tuple[np.ndarray, list[tuple[int, ...]]]:
        """Two spatially well-separated point clusters, scattered over a random index permutation.

        Cluster centers are placed `cluster_gap_ratio * cluster_spread`
        apart; each cluster's points are drawn from a small Gaussian blob of
        std `cluster_spread` around its center. The permutation step matters:
        without it, cluster A always occupies indices `[0, size_a)` and
        cluster B `[size_a, size)`, so a decomposition that blindly cuts the
        city list in half by index would accidentally coincide with the real
        clusters and "pass" without doing any real detection -- exactly the
        contiguous-block trap this scattering guards against.
        Detection here (`mst_gap_groups`) never reads index order in
        the first place, so this permutation cannot help it -- it only
        removes an accidental shortcut a *different*, wrong decomposition
        could otherwise exploit by luck.

        The returned groups are provenance only -- `decompose_divide` never
        reads them, it re-derives the split from `D` via `mst_gap_groups`
        exactly as it would for an instance nobody planted.
        """
        if size < 2:
            raise ValueError("clustered TSP instances need at least 2 cities")
        size_a = size // 2 if size // 2 >= 1 else 1
        size_b = size - size_a
        if size_b < 1:
            size_a, size_b = size - 1, 1
        angle = rng.uniform(0, 2 * np.pi)
        center_a = np.array([0.3, 0.3])
        offset = self.cluster_gap_ratio * self.cluster_spread
        center_b = center_a + offset * np.array([np.cos(angle), np.sin(angle)])
        pts_a = center_a + rng.normal(scale=self.cluster_spread, size=(size_a, 2))
        pts_b = center_b + rng.normal(scale=self.cluster_spread, size=(size_b, 2))
        ordered_points = np.vstack([pts_a, pts_b])
        ordered_groups = [tuple(range(size_a)), tuple(range(size_a, size_a + size_b))]
        perm = rng.permutation(size)
        points = ordered_points[perm]
        inverse = np.empty(size, dtype=int)
        inverse[perm] = np.arange(size)
        groups = [tuple(sorted(int(inverse[i]) for i in g)) for g in ordered_groups]
        return points, groups

    def prepare_instance(
        self, instance: InstanceRecord, *, indices: list[int] | None = None
    ) -> dict[str, Any]:
        """Cache D and MST-gap groups once for materialize + primitive_features.

        Synthesis `decompose_divide` still re-derives groups from D (it is
        not on the Observe-layer worker path and must not depend on a
        pipeline cache).
        """
        del indices
        d, variables = _instance_view(instance)
        groups = mst_gap_groups(d, min_gap_ratio=self.min_gap_ratio)
        return {"D": d, "variables": variables, "groups": groups}

    def materialize(
        self, instance: InstanceRecord, index: int, *, cache: dict[str, Any] | None = None
    ) -> SimpleFamilySlice:
        c = cache if cache is not None and "groups" in cache else self.prepare_instance(instance)
        sizes = np.array([len(g) for g in c["groups"]], dtype=float)
        return SimpleFamilySlice(values=sizes, index=index, kind="detected_group_sizes")

    def primitive_features(
        self, instance: InstanceRecord, *, cache: dict[str, Any] | None = None
    ) -> dict[str, float | np.ndarray]:
        c = cache if cache is not None and "groups" in cache else self.prepare_instance(instance)
        d, groups = c["D"], c["groups"]
        sizes = [len(g) for g in groups]
        return {
            "n_cities": float(d.shape[0]),
            "n_detected_groups": float(len(groups)),
            "largest_detected_group": float(max(sizes)) if sizes else float("nan"),
            # Array-valued primitive: RDE's generic matrix/graph descriptor
            # sweep (src/rde/runtime/worker.py) auto-derives matrix.D.*/
            # graph.D.* stats from this the same way it does for QUBO's `Q`.
            # Previously only the MST-gap-derived scalars above were exposed,
            # so the full descriptor catalog never saw raw TSP structure.
            "D": d,
        }

    # ---- SynthesisDomain protocol (Mode 2) ----

    def base_case_cost_exponent(self) -> float:
        """Deliberately conservative label for the trivial "no decomposition" skeleton.

        True whole-instance brute force is `(n-1)!/2` closed tours, which
        grows strictly faster than any fixed `2^(c*n)` -- there is no
        constant `c` for which `2^(c*n)` matches factorial growth exactly.
        This value exists only to make `rde.synthesis`'s symbolic pruning
        correctly classify the undecomposed skeleton as non-polynomial (any
        positive constant does that, since factorial growth is a fortiori
        super-polynomial); it is not, and must not be read as, an accurate
        asymptotic label. `brute_force` below is the real, independently
        guarded factorial-time reference.
        """
        return 1.0

    def size_of(self, instance: Any) -> int:
        d, _ = _instance_view(instance)
        return int(d.shape[0])

    def brute_force(self, instance: Any) -> SynthesisSolution:
        """Exact optimum by full permutation enumeration -- reference oracle.

        Deliberately ignores any cluster structure the instance happens to
        have: it is independent ground truth for `verify_skeleton`, not a
        faster version of the thing being verified.
        """
        d, variables = _instance_view(instance)
        n = d.shape[0]
        if n == 0:
            return SynthesisSolution(assignment=(), cost=0.0)
        validate_bruteforce_size(n, self.max_bruteforce_n)
        local_tour, cost = closed_tour_optimal(d, tuple(range(n)))
        assignment = tuple(variables[i] for i in local_tour)
        return SynthesisSolution(assignment=assignment, cost=float(cost))

    def decompose_flat(self, instance: Any) -> list[Any] | None:
        """Not supported in this version.

        A single-step k-way split (k > 2) needs a k-way generalization of
        `combine`'s exact joint-boundary search (which cluster visiting
        order? how many crossings between non-adjacent clusters?) that this
        version does not implement -- returning `None` here is an honest
        "this skeleton does not apply yet," not an oversight. See
        `decompose_divide` for the supported pairwise case.
        """
        return None

    def decompose_divide(self, instance: Any, branches: int) -> list[Any] | None:
        """Split into exactly two well-separated clusters, or `None`.

        Only `branches == 2` is supported (see `decompose_flat`'s docstring
        for why k > 2 is future work, not implemented here). Returns `None`
        -- "this skeleton does not apply here" -- when `mst_gap_groups`
        does not find exactly two groups (no real gap, or more than two
        gaps) or when either group exceeds `max_leaf_width`.
        """
        if branches != 2:
            return None
        d, variables = _instance_view(instance)
        groups = mst_gap_groups(d, min_gap_ratio=self.min_gap_ratio)
        if len(groups) != 2:
            return None
        if max(len(g) for g in groups) > self.max_leaf_width:
            return None
        return [self._sub(d, variables, g) for g in groups]

    def _sub(self, d: np.ndarray, variables: tuple[int, ...], local: tuple[int, ...]) -> dict[str, Any]:
        local_sorted = tuple(sorted(local))
        idx = np.asarray(local_sorted, dtype=int)
        return {
            "D": d[np.ix_(idx, idx)],
            "vars": tuple(variables[i] for i in local_sorted),
        }

    def combine(self, instance: Any, sub_solutions: list[SynthesisSolution]) -> SynthesisSolution:
        """Exact joint two-crossing merge of two cluster solutions.

        Uses the *city membership* of each `sub_solutions[i].assignment`
        only -- their specific closed-tour order/cost from the harness's
        independent leaf brute force is discarded, since the exact merge
        requires *open*-path optima between every candidate boundary-city
        pair, which a closed-tour solve does not give. Re-derives those open
        paths directly from `instance`'s own distance matrix
        (`all_pairs_open_path_optimal`), which is legitimate structural work
        on the instance, not a peek at any answer.

        Raises if `sub_solutions` is not exactly a 2-way partition -- this
        domain never calls `combine` any other way itself, so that would
        indicate a harness/skeleton mismatch, not a normal "unsupported"
        case (those are signaled by `decompose_divide` returning `None`
        before `combine` is ever invoked).
        """
        if len(sub_solutions) != 2:
            raise ValueError(f"expected exactly 2 sub-solutions, got {len(sub_solutions)}")
        d, variables = _instance_view(instance)
        group_a = tuple(int(v) for v in sub_solutions[0].assignment)
        group_b = tuple(int(v) for v in sub_solutions[1].assignment)

        def boundary_options(group: tuple[int, ...]) -> dict[tuple[int, int], tuple[tuple[int, ...], float]]:
            if len(group) == 1:
                return {(group[0], group[0]): ((group[0],), 0.0)}
            return all_pairs_open_path_optimal(d, group)

        options_a = boundary_options(group_a)
        options_b = boundary_options(group_b)

        best_cost = float("inf")
        best_tour: tuple[int, ...] = group_a + group_b
        for (e1, x1), (path_a, cost_a) in options_a.items():
            for (e2, x2), (path_b, cost_b) in options_b.items():
                total = cost_a + cost_b + float(d[x1, e2]) + float(d[x2, e1])
                if total < best_cost:
                    best_cost = total
                    best_tour = tuple(path_a) + tuple(path_b)

        return SynthesisSolution(assignment=best_tour, cost=float(best_cost))

    def cost(self, instance: Any, solution: SynthesisSolution) -> float:
        """Independently re-evaluate `solution.assignment` as a closed-tour length.

        Never reads `solution.cost` -- a candidate skeleton's reported cost
        is exactly what is under test.
        """
        d, variables = _instance_view(instance)
        assignment = tuple(int(v) for v in solution.assignment)
        if sorted(assignment) != sorted(variables):
            raise ValueError(
                f"assignment does not cover the instance's cities "
                f"(expected {sorted(variables)}, got {sorted(assignment)})"
            )
        local_index = {v: i for i, v in enumerate(variables)}
        local_tour = tuple(local_index[v] for v in assignment)
        return tour_cost(d, local_tour)
