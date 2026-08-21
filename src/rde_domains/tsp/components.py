"""Geometric structure detection and exact small-instance TSP primitives.

Mirrors the discipline of `rde_domains.qubo_synthesis.components`: nothing
here looks at a brute-force answer, an optimum, or any planted-generation
provenance. The only input to structure detection (`mst_gap_groups`) is the
instance's own distance matrix, exactly as `interaction_components` only
looks at a QUBO's own coupling pattern.

Unlike a QUBO's coupling graph, a Euclidean distance matrix is dense (every
pair of cities has a finite distance), so thresholding entries directly does
not split anything. The dense-graph analogue of "which pairs are coupled" is
a minimum-spanning-tree gap: cut the tree's longest edge(s) and see whether a
large, unambiguous drop separates "within-cluster" edges from "across-the-gap"
edges. This is the standard single-linkage-clustering-by-MST-edge-cut
construction, not a QPFA invention -- what is specific to this module is
using it as a *SynthesisDomain decomposition move* rather than a clustering
end in itself.
"""

from __future__ import annotations

import math
from itertools import permutations
from typing import Sequence

import numpy as np


def random_euclidean_points(rng: np.random.Generator, n: int) -> np.ndarray:
    """`n` points in the unit square."""
    return rng.random((n, 2))


def distance_matrix(points: np.ndarray) -> np.ndarray:
    """Full symmetric Euclidean distance matrix, zero diagonal."""
    pts = np.asarray(points, dtype=float)
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def mst_edges(D: np.ndarray) -> list[tuple[int, int, float]]:
    """Minimum spanning tree of the complete graph with weights `D`.

    Prim's algorithm, O(n^2) -- appropriate here since every instance in this
    domain is small by construction (`max_leaf_width`-bounded leaves, or a
    top-level size already capped by `max_bruteforce_n`). Returns `n-1` edges
    as `(i, j, weight)`, unordered otherwise.
    """
    d = np.asarray(D, dtype=float)
    n = d.shape[0]
    if n < 2:
        return []
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    best_dist = d[0].copy()
    best_from = np.zeros(n, dtype=int)
    edges: list[tuple[int, int, float]] = []
    for _ in range(n - 1):
        candidates = np.where(~in_tree, best_dist, np.inf)
        j = int(np.argmin(candidates))
        edges.append((int(best_from[j]), j, float(best_dist[j])))
        in_tree[j] = True
        better = (~in_tree) & (d[j] < best_dist)
        best_dist = np.where(better, d[j], best_dist)
        best_from = np.where(better, j, best_from)
    return edges


def mst_gap_groups(D: np.ndarray, *, min_gap_ratio: float) -> list[tuple[int, ...]]:
    """Split indices into groups by cutting a large gap in the MST edge weights.

    Sorts MST edge weights descending and cuts the largest prefix `c` for
    which weight[c-1] / weight[c] >= `min_gap_ratio` (a genuine "big drop"
    in the sorted weight sequence -- an elbow/gap-statistic detector, not a
    QPFA-specific idea). Cutting `c` edges from a spanning tree yields
    exactly `c+1` connected components.

    Returns a single group `[tuple(range(n))]` when no such gap exists --
    "no cluster structure detected," the honest analogue of
    `interaction_components` returning one component for a dense QUBO.
    """
    n = D.shape[0]
    all_idx = tuple(range(n))
    if n < 2:
        return [all_idx]
    edges = mst_edges(D)
    weights = sorted((w for _, _, w in edges), reverse=True)
    cut = 0
    for i in range(len(weights) - 1):
        denom = weights[i + 1] if weights[i + 1] > 1e-12 else 1e-12
        if weights[i] / denom >= min_gap_ratio:
            cut = i + 1
            break
    if cut == 0:
        return [all_idx]
    # Remove the `cut` largest edges from the tree, then take connected
    # components of what remains. Matching by weight against a shrinking
    # pool (rather than a value check) is safe here because `weights` and
    # `edges` are two views of the same MST edge set.
    weight_pool = list(weights[:cut])
    kept_edges: list[tuple[int, int]] = []
    for i, j, w in edges:
        match = next((ww for ww in weight_pool if abs(w - ww) < 1e-12), None)
        if match is not None:
            weight_pool.remove(match)
            continue
        kept_edges.append((i, j))
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, j in kept_edges:
        adj[i].append(j)
        adj[j].append(i)
    seen = [False] * n
    groups: list[tuple[int, ...]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        groups.append(tuple(sorted(comp)))
    return sorted(groups, key=lambda g: g[0])


def path_cost(D: np.ndarray, path: Sequence[int]) -> float:
    """Length of an open path (no wraparound)."""
    if len(path) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        total += float(D[a, b])
    return total


def tour_cost(D: np.ndarray, tour: Sequence[int]) -> float:
    """Length of a closed cyclic tour (wraparound included)."""
    if len(tour) < 2:
        return 0.0
    return path_cost(D, list(tour) + [tour[0]])


def all_tour_costs(D: np.ndarray, cities: Sequence[int]) -> np.ndarray:
    """Exact cost of every distinct closed tour over `cities`, by full enumeration.

    Fixes `cities[0]` as the rotation anchor, same discipline as
    `closed_tour_optimal` -- removes the redundant len(cities)-fold rotational
    duplication. Also deduplicates direction (keeps only `perm[0] < perm[-1]`):
    a tour and its exact reverse traverse the same edges and always have
    identical cost, so leaving both in means the minimum always appears at
    least twice -- caught empirically (2026-08-19, EXP-063 design check):
    `cost_spectral_gap_ratio` was exactly 0.0 on every one of 60 real
    instances because "second-best" was trivially the optimal tour's own
    reverse, not a genuinely different tour. `(N-1)!/2` distinct tours, not
    `(N-1)!`. Exact, exponential/factorial by design -- callers must guard
    `len(cities)` (see `validate_bruteforce_size`).
    """
    cities = tuple(cities)
    n = len(cities)
    if n <= 2:
        return np.array([tour_cost(D, cities)], dtype=float)
    anchor, rest = cities[0], cities[1:]
    costs = np.empty(math.factorial(n - 1) // 2, dtype=float)
    i = 0
    for perm in permutations(rest):
        if perm[0] > perm[-1]:
            continue  # skip the reverse-direction duplicate
        costs[i] = tour_cost(D, (anchor,) + perm)
        i += 1
    return costs


def closed_tour_optimal(D: np.ndarray, cities: Sequence[int]) -> tuple[tuple[int, ...], float]:
    """Exact optimal closed tour over `cities` by full permutation enumeration.

    Fixes `cities[0]` as the rotation anchor (a cyclic tour has no distinguished
    start, so this removes the redundant `len(cities)`-fold rotational
    duplication without changing the optimum). Exponential/factorial by
    design -- callers must guard `len(cities)`.
    """
    cities = tuple(cities)
    n = len(cities)
    if n <= 1:
        return cities, 0.0
    if n == 2:
        return cities, 2.0 * float(D[cities[0], cities[1]])
    anchor, rest = cities[0], cities[1:]
    best_tour: tuple[int, ...] = cities
    best_cost = float("inf")
    for perm in permutations(rest):
        candidate = (anchor,) + perm
        c = tour_cost(D, candidate)
        if c < best_cost:
            best_cost = c
            best_tour = candidate
    return best_tour, best_cost


def open_path_optimal(
    D: np.ndarray, cities: Sequence[int], start: int, end: int
) -> tuple[tuple[int, ...], float]:
    """Exact optimal open Hamiltonian path over `cities` from `start` to `end`.

    `start` and `end` must both be members of `cities`; if they are equal and
    `cities` has more than one member, there is no valid Hamiltonian path
    (a path needs distinct endpoints once there is more than one city to
    order), so this raises -- callers are expected to skip that combination,
    not treat it as zero-cost.
    """
    cities = tuple(cities)
    if start not in cities or end not in cities:
        raise ValueError("start/end must be members of cities")
    if len(cities) == 1:
        return cities, 0.0
    if start == end:
        raise ValueError("start == end is not a valid open path for len(cities) > 1")
    middle = tuple(c for c in cities if c not in (start, end))
    if not middle:
        return (start, end), float(D[start, end])
    best_path: tuple[int, ...] = (start,) + middle + (end,)
    best_cost = float("inf")
    for perm in permutations(middle):
        candidate = (start,) + perm + (end,)
        c = path_cost(D, candidate)
        if c < best_cost:
            best_cost = c
            best_path = candidate
    return best_path, best_cost


def all_pairs_open_path_optimal(
    D: np.ndarray, cities: Sequence[int]
) -> dict[tuple[int, int], tuple[tuple[int, ...], float]]:
    """`open_path_optimal` for every ordered pair of distinct endpoints in `cities`.

    Computed once per cluster so `combine`'s O(|A|^2 * |B|^2) crossing search
    is O(1) per crossing choice instead of recomputing an O((k-2)!) brute
    force inside the innermost loop.
    """
    cities = tuple(cities)
    out: dict[tuple[int, int], tuple[tuple[int, ...], float]] = {}
    for start in cities:
        for end in cities:
            if start == end:
                continue
            out[(start, end)] = open_path_optimal(D, cities, start, end)
    return out
