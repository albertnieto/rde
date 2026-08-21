"""Unit tests for matrix, graph, and landscape descriptors."""

from __future__ import annotations

import numpy as np

from rde.features.graph import descriptor_for_matrix as graph_desc
from rde.features.landscape import count_local_minima, descriptor_for_vector
from rde.features.matrix import descriptor_for_matrix as matrix_desc


def test_matrix_descriptor_on_qubo_like_matrix():
    q = np.array([[1.0, -2.0, 0.0], [0.0, 2.0, 1.0], [0.0, 0.0, -1.0]])
    out = matrix_desc(q, name="Q")
    assert out["matrix.Q.size"] == 3.0
    assert out["matrix.Q.trace"] == 2.0
    assert "matrix.Q.effective_rank" in out


def test_graph_descriptor_connected_chain():
    q = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    out = graph_desc(q, name="Q")
    assert out["graph.Q.n_nodes"] == 3.0
    assert out["graph.Q.n_edges"] == 2.0
    assert out["graph.Q.diameter"] == 2.0


def test_landscape_local_minima():
    costs = np.array([3.0, 1.0, 2.0, 1.0, 4.0])
    assert count_local_minima(costs) == 2
    out = descriptor_for_vector(costs, name="costs")
    assert out["landscape.costs.local_minima"] == 2.0
