"""TSP SynthesisDomain (Direction E) -- genuine distance matrix, MST-gap detection.

Treatment ``tsp_clustered`` plants two well-separated spatial clusters so
only genuine MST-gap detection (not a hand-fed grouping) recovers them.
Control ``tsp_uniform_control`` has no planted structure. Unlike
``qubo_synthesis``'s block-separable case, TSP clusters are not provably
independent (a single Hamiltonian cycle must still cross between them), so
the load-bearing check here is not "combine is obviously exact by
construction" but "the joint two-crossing combine actually matches
brute-force ground truth on every generated treatment instance" -- see
``domain.py``'s module docstring.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rde.core.plugins import build_registry
from rde.core.protocols import SynthesisSolution
from rde.synthesis import synthesize
from rde.synthesis.search import execute_skeleton, verify_skeleton
from rde.synthesis.skeleton import divide_skeleton
from rde_domains.tsp import mst_gap_groups
from rde_domains.tsp.components import (
    closed_tour_optimal,
    distance_matrix,
    open_path_optimal,
    tour_cost,
)

_DIVIDE = divide_skeleton(2, 2.0, 1.0)


def _treatment(n_instances=6, size=8, seed=0, **kwargs):
    reg = build_registry("tsp_clustered")
    domain = reg.get_domain("tsp_clustered")
    for k, v in kwargs.items():
        setattr(domain, k, v)
    return domain, domain.generate(n_instances, size, seed)


def _control(n_instances=6, size=8, seed=0):
    reg = build_registry("tsp_uniform_control")
    domain = reg.get_domain("tsp_uniform_control")
    return domain, domain.generate(n_instances, size, seed)


def test_domains_registered():
    treat, t_inst = _treatment(n_instances=2, size=6)
    ctrl, c_inst = _control(n_instances=2, size=6)
    assert treat.domain_id == "tsp_clustered"
    assert ctrl.domain_id == "tsp_uniform_control"
    assert treat.size_of(t_inst[0]) == 6
    assert ctrl.size_of(c_inst[0]) == 6


def test_planted_clusters_are_detected_from_distance_matrix_alone():
    domain, instances = _treatment(n_instances=6, size=8, seed=1)
    for inst in instances:
        planted = sorted(tuple(g) for g in inst.params["planted_groups"])
        d = np.asarray(inst.params["D"], dtype=float)
        detected = sorted(mst_gap_groups(d, min_gap_ratio=domain.min_gap_ratio))
        assert planted == detected


def test_uniform_control_rarely_shows_a_spurious_gap():
    """Random uniform points essentially never trip the >=3x MST-gap threshold.

    "Essentially never," not "never" by a proof -- this is an empirical claim
    about the random instance family, checked directly rather than assumed.
    A single false positive out of many draws would not indicate a domain
    bug (the detector is honestly doing its job on whatever gap happens to
    exist); a *systematic* one would mean the control isn't really a control.
    """
    domain, instances = _control(n_instances=30, size=8, seed=0)
    n_split = 0
    for inst in instances:
        d = np.asarray(inst.params["D"], dtype=float)
        groups = mst_gap_groups(d, min_gap_ratio=domain.min_gap_ratio)
        if len(groups) > 1:
            n_split += 1
        if len(groups) == 1:
            assert domain.decompose_divide(inst, 2) is None
    assert n_split <= 3, f"control split on {n_split}/30 instances -- threshold too weak"


def test_divide_skeleton_matches_brute_force_on_treatment():
    """The load-bearing correctness check: exact joint combine vs real ground truth."""
    domain, instances = _treatment(n_instances=8, size=8, seed=2)
    for inst in instances:
        truth = domain.brute_force(inst)
        # base_threshold must reach max_leaf_width: below that, `_execute`
        # tries to decompose each leaf cluster *again* (there is no further
        # gap inside a single planted cluster), decompose_divide correctly
        # returns None, and the whole run reports unsupported -- not because
        # combine is wrong, but because the leaf was never allowed to be
        # solved directly. This is a real, load-bearing campaign-config
        # requirement (see the Step 4a audit), not a test-only workaround.
        cand = execute_skeleton(domain, inst, _DIVIDE, base_threshold=domain.max_leaf_width)
        assert cand is not None, "expected the planted gap to be detected"
        true_cost = domain.cost(inst, truth)
        cand_cost = domain.cost(inst, cand)
        assert cand_cost <= true_cost + 1e-6, "candidate beat the exact optimum -- a real bug"
        assert abs(cand_cost - true_cost) < 1e-6 * max(1.0, true_cost)


def test_divide_skeleton_matches_brute_force_across_sizes_and_seeds():
    """Same check, swept over several small sizes/seeds -- one instance is not enough."""
    for size in (6, 7, 8, 9):
        domain, instances = _treatment(n_instances=4, size=size, seed=100 + size)
        for inst in instances:
            truth = domain.brute_force(inst)
            cand = execute_skeleton(domain, inst, _DIVIDE, base_threshold=domain.max_leaf_width)
            assert cand is not None
            true_cost = domain.cost(inst, truth)
            cand_cost = domain.cost(inst, cand)
            assert abs(cand_cost - true_cost) < 1e-6 * max(1.0, true_cost), (
                f"mismatch at size={size} seed={inst.seed}: "
                f"cand={cand_cost} truth={true_cost}"
            )


def test_synthesize_accepts_divide_on_treatment():
    domain, instances = _treatment(n_instances=6, size=8, seed=3)
    report = synthesize(
        domain, instances, target_degree=None, catalog=[_DIVIDE], base_threshold=domain.max_leaf_width
    )
    assert report.accepted, [c.detail for c in report.candidates]
    accepted = report.accepted[0]
    assert accepted.n_matched == accepted.n_checked == len(instances)


def test_synthesize_accepts_nothing_on_uniform_control():
    domain, instances = _control(n_instances=6, size=8, seed=4)
    report = synthesize(
        domain, instances, target_degree=None, catalog=[_DIVIDE], base_threshold=domain.max_leaf_width
    )
    assert report.accepted == []


def test_decompose_flat_not_supported():
    domain, instances = _treatment(n_instances=1, size=8, seed=5)
    assert domain.decompose_flat(instances[0]) is None


def test_decompose_divide_rejects_branches_other_than_two():
    domain, instances = _treatment(n_instances=1, size=8, seed=6)
    assert domain.decompose_divide(instances[0], 3) is None
    assert domain.decompose_divide(instances[0], 4) is None


def test_decompose_divide_refuses_when_leaf_too_wide():
    domain, instances = _treatment(n_instances=3, size=8, seed=7, max_leaf_width=2)
    for inst in instances:
        assert domain.decompose_divide(inst, 2) is None


def test_cost_ignores_carried_solution_cost():
    domain, instances = _treatment(n_instances=1, size=6, seed=8)
    inst = instances[0]
    truth = domain.brute_force(inst)
    lying = SynthesisSolution(assignment=truth.assignment, cost=-1e9)
    assert abs(domain.cost(inst, lying) - domain.cost(inst, truth)) < 1e-9


def test_cost_rejects_assignment_not_covering_instance():
    domain, instances = _treatment(n_instances=1, size=6, seed=9)
    inst = instances[0]
    bad = SynthesisSolution(assignment=(0, 1, 2), cost=0.0)
    with pytest.raises(ValueError):
        domain.cost(inst, bad)


def test_combine_does_not_read_planted_groups(monkeypatch):
    """Safety: decompose_divide must not consult planted_groups provenance."""
    domain, instances = _treatment(n_instances=1, size=8, seed=10)
    inst = instances[0]
    inst.params["planted_groups"] = [[0, 1, 2, 3, 4, 5, 6, 7]]
    parts = domain.decompose_divide(inst, 2)
    assert parts is not None
    d = np.asarray(inst.params["D"], dtype=float)
    detected = sorted(mst_gap_groups(d, min_gap_ratio=domain.min_gap_ratio))
    assert sorted(tuple(p["vars"]) for p in parts) == detected


class _BlindContiguousSplitter:
    """Wraps a real domain but splits by index halves, ignoring the distance matrix.

    Used only to prove verification rejects a decomposition that is
    structurally wrong for the instance -- must not be registered as a
    science domain.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.domain_id = "blind_contiguous_splitter"

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def decompose_divide(self, instance: Any, branches: int) -> list[Any] | None:
        if branches != 2:
            return None
        if isinstance(instance, dict):
            d = np.asarray(instance["D"], dtype=float)
            variables = tuple(int(v) for v in instance["vars"])
        else:
            d = np.asarray(instance.params["D"], dtype=float)
            variables = tuple(range(d.shape[0]))
        n = d.shape[0]
        if n < 4:
            return None
        mid = n // 2
        left, right = tuple(range(mid)), tuple(range(mid, n))
        idx_l, idx_r = np.asarray(left), np.asarray(right)
        return [
            {"D": d[np.ix_(idx_l, idx_l)], "vars": tuple(variables[i] for i in left)},
            {"D": d[np.ix_(idx_r, idx_r)], "vars": tuple(variables[i] for i in right)},
        ]


def test_blind_contiguous_split_is_rejected_incorrect_on_treatment():
    """City index order carries no spatial meaning here (unlike the QUBO
    domain's variable-permutation discipline, cluster membership is not tied
    to index order at all -- see components.py's module docstring), so an
    index-halves split has no reason to align with the planted spatial
    clusters and should fail verification."""
    inner, instances = _treatment(n_instances=6, size=8, seed=11)
    blind = _BlindContiguousSplitter(inner)
    result = verify_skeleton(blind, instances, _DIVIDE, base_threshold=4)
    assert result.n_checked >= 1
    assert result.n_matched < result.n_checked


def test_mst_gap_groups_no_gap_returns_single_group():
    rng = np.random.default_rng(0)
    points = rng.random((6, 2))
    d = distance_matrix(points)
    groups = mst_gap_groups(d, min_gap_ratio=1000.0)
    assert groups == [tuple(range(6))]


def test_mst_gap_groups_detects_planted_two_way_split():
    a = np.array([0.0, 0.0]) + 0.01 * np.random.default_rng(1).normal(size=(3, 2))
    b = np.array([5.0, 5.0]) + 0.01 * np.random.default_rng(2).normal(size=(3, 2))
    points = np.vstack([a, b])
    d = distance_matrix(points)
    groups = mst_gap_groups(d, min_gap_ratio=3.0)
    assert sorted(groups) == [(0, 1, 2), (3, 4, 5)]


def test_open_path_optimal_matches_closed_tour_minus_best_edge_on_small_case():
    rng = np.random.default_rng(3)
    points = rng.random((5, 2))
    d = distance_matrix(points)
    cities = tuple(range(5))
    tour, tour_c = closed_tour_optimal(d, cities)
    # Removing the tour's most expensive edge gives a valid open path; the
    # true open-path optimum between those same two endpoints can only be
    # cheaper or equal.
    edge_costs = [d[tour[i], tour[(i + 1) % 5]] for i in range(5)]
    worst = int(np.argmax(edge_costs))
    start, end = tour[(worst + 1) % 5], tour[worst]
    _, open_c = open_path_optimal(d, cities, start, end)
    assert open_c <= tour_c - edge_costs[worst] + 1e-9


def test_open_path_optimal_rejects_equal_endpoints_for_multi_city_set():
    rng = np.random.default_rng(4)
    points = rng.random((4, 2))
    d = distance_matrix(points)
    with pytest.raises(ValueError):
        open_path_optimal(d, (0, 1, 2, 3), 0, 0)


def test_tour_cost_matches_manual_sum():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    d = distance_matrix(points)
    assert abs(tour_cost(d, (0, 1, 2, 3)) - 4.0) < 1e-9


def test_clustered_prepare_instance_runs_mst_gap_groups_once(monkeypatch):
    import rde_domains.tsp.domain as tsp_domain

    calls = {"n": 0}
    real = tsp_domain.mst_gap_groups

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(tsp_domain, "mst_gap_groups", wrapped)
    domain, instances = _treatment(n_instances=1, size=6, seed=5)
    inst = instances[0]
    cache = domain.prepare_instance(inst)
    assert calls["n"] == 1
    sl = domain.materialize(inst, 0, cache=cache)
    feats = domain.primitive_features(inst, cache=cache)
    assert calls["n"] == 1
    assert sl.values.tolist() == [len(g) for g in cache["groups"]]
    assert feats["n_detected_groups"] == float(len(cache["groups"]))
    np.testing.assert_allclose(feats["D"], cache["D"])
