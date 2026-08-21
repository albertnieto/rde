"""Graph descriptors from a weighted adjacency/coupling matrix."""

from __future__ import annotations

import numpy as np
import networkx as nx

from rde.core.instance import InstanceRecord
from rde.core.protocols import SimpleFamilySlice
from rde.features.numeric import safe_pearson_r


def adjacency_from_matrix(matrix: np.ndarray, *, absolute: bool = True) -> np.ndarray:
    """Build unweighted adjacency from off-diagonal entries."""
    mat = np.asarray(matrix, dtype=float)
    n = mat.shape[0]
    w = np.abs(mat) if absolute else mat
    triu = np.triu(np.ones((n, n), dtype=bool), k=1)
    mask = (w != 0) & triu
    adj = np.zeros((n, n), dtype=float)
    adj[mask] = 1.0
    adj = adj + adj.T
    return adj


def _degree_assortativity(g: nx.Graph) -> float:
    """Undirected degree assortativity via Pearson r on edge endpoints.

    Equivalent to NetworkX's coefficient, but returns NaN without RuntimeWarning
    when endpoint-degree variance is zero (regular / disconnected equal-degree cases).
    """
    if g.number_of_edges() < 1 or g.number_of_nodes() < 2:
        return float("nan")
    deg = dict(g.degree())
    xs: list[float] = []
    ys: list[float] = []
    for u, v in g.edges():
        du = float(deg[u])
        dv = float(deg[v])
        xs.extend((du, dv))
        ys.extend((dv, du))
    return safe_pearson_r(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), min_count=2)


def descriptor_for_matrix(
    matrix: np.ndarray,
    name: str = "array",
) -> dict[str, float]:
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] == 0:
        return {}
    p = f"graph.{name}"
    adj = adjacency_from_matrix(mat)
    g = nx.from_numpy_array(adj)
    n = g.number_of_nodes()
    m = g.number_of_edges()
    out: dict[str, float] = {
        f"{p}.n_nodes": float(n),
        f"{p}.n_edges": float(m),
        f"{p}.density": float(nx.density(g)),
    }
    if n == 0:
        return out
    degrees = np.array([d for _, d in g.degree()], dtype=float)
    out[f"{p}.avg_degree"] = float(degrees.mean()) if degrees.size else 0.0
    out[f"{p}.degree_variance"] = float(np.var(degrees)) if degrees.size else 0.0
    if nx.is_connected(g):
        out[f"{p}.diameter"] = float(nx.diameter(g))
        out[f"{p}.avg_shortest_path"] = float(nx.average_shortest_path_length(g))
    else:
        out[f"{p}.diameter"] = float("nan")
        out[f"{p}.avg_shortest_path"] = float("nan")
    try:
        out[f"{p}.clustering"] = float(nx.average_clustering(g))
    except Exception:
        out[f"{p}.clustering"] = float("nan")
    # Laplacian spectrum proxy
    lap = np.diag(degrees) - adj
    lap_eigs = np.linalg.eigvalsh(lap)
    lap_eigs = np.sort(lap_eigs)
    if lap_eigs.size >= 2:
        out[f"{p}.laplacian_gap"] = float(lap_eigs[1] - lap_eigs[0])
        out[f"{p}.laplacian_max"] = float(lap_eigs[-1])
        out[f"{p}.laplacian_entropy"] = _laplacian_entropy(lap_eigs)
    try:
        out[f"{p}.assortativity"] = _degree_assortativity(g)
    except Exception:
        out[f"{p}.assortativity"] = float("nan")
    try:
        cycles = nx.cycle_basis(g)
        out[f"{p}.cycle_count"] = float(len(cycles))
        out[f"{p}.cycle_length_sum"] = float(sum(len(c) for c in cycles))
    except Exception:
        out[f"{p}.cycle_count"] = float("nan")
        out[f"{p}.cycle_length_sum"] = float("nan")
    try:
        out[f"{p}.degeneracy"] = float(nx.algorithms.core.degeneracy(g))
    except Exception:
        out[f"{p}.degeneracy"] = float("nan")
    out[f"{p}.modularity_proxy"] = float(m / max(n * (n - 1) / 2, 1))
    w = np.abs(mat)
    w_off = w.copy()
    np.fill_diagonal(w_off, 0.0)
    edge_weights = w_off[w_off > 0]
    if edge_weights.size:
        out[f"{p}.weighted_edge_sum"] = float(edge_weights.sum())
        out[f"{p}.weighted_edge_mean"] = float(edge_weights.mean())
        out[f"{p}.weighted_edge_entropy"] = _laplacian_entropy(edge_weights)
    return out


def _laplacian_entropy(eigs: np.ndarray) -> float:
    v = np.maximum(eigs, 0.0)
    total = float(v.sum())
    if total <= 0:
        return 0.0
    p = v[v > 0] / total
    return float(-np.sum(p * np.log(p)))


def descriptor(
    _instance: InstanceRecord,
    _slice_: SimpleFamilySlice | None,
    array: np.ndarray | None,
) -> dict[str, float]:
    if array is None:
        return {}
    arr = np.asarray(array, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        return descriptor_for_matrix(arr)
    return {}
