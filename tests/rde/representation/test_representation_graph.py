"""Tests for RepresentationGraph: nodes, edges, path search, comparison."""

from __future__ import annotations

import numpy as np
import pytest

from rde.representation import Representation, RepresentationGraph, Transformation

VECTOR = Representation(
    representation_id="vector",
    object_type="numeric_4",
    carrier="R^4",
    encode=lambda x: np.asarray(x, dtype=float),
    decode=lambda x: np.asarray(x, dtype=float),
)

MATRIX = Representation(
    representation_id="matrix_2x2",
    object_type="numeric_4",
    carrier="R^{2x2}",
    encode=lambda x: np.asarray(x, dtype=float).reshape(2, 2),
    decode=lambda m: np.asarray(m, dtype=float).reshape(4),
)

FLAT_STRING = Representation(
    representation_id="flat_string",
    object_type="numeric_4",
    carrier="comma-separated string",
    encode=lambda x: ",".join(str(v) for v in np.asarray(x, dtype=float)),
    decode=lambda s: np.array([float(v) for v in s.split(",")]),
)

OTHER_TYPE = Representation(
    representation_id="other",
    object_type="numeric_9",
    carrier="R^9",
    encode=lambda x: x,
    decode=lambda x: x,
)


def _graph_with_two_hops() -> RepresentationGraph:
    graph = RepresentationGraph()
    graph.add_transformation(
        Transformation(transformation_id="vector_to_matrix", source=VECTOR, target=MATRIX, cost=1.0),
        bidirectional=True,
    )
    graph.add_transformation(
        Transformation(transformation_id="matrix_to_string", source=MATRIX, target=FLAT_STRING, cost=1.0),
        bidirectional=True,
    )
    return graph


def test_add_transformation_registers_both_endpoint_nodes():
    graph = RepresentationGraph()
    graph.add_transformation(Transformation(transformation_id="v2m", source=VECTOR, target=MATRIX))
    assert graph.representation("vector") is VECTOR
    assert graph.representation("matrix_2x2") is MATRIX


def test_add_representation_rejects_conflicting_id():
    graph = RepresentationGraph()
    graph.add_representation(VECTOR)
    conflicting = Representation(
        representation_id="vector",
        object_type="numeric_4",
        carrier="different carrier",
        encode=lambda x: x,
        decode=lambda x: x,
    )
    with pytest.raises(ValueError):
        graph.add_representation(conflicting)


def test_bidirectional_transformation_registers_inverse_edge():
    graph = RepresentationGraph()
    graph.add_transformation(
        Transformation(transformation_id="v2m", source=VECTOR, target=MATRIX), bidirectional=True
    )
    neighbor_ids = {edge.target.representation_id for edge in graph.neighbors("matrix_2x2")}
    assert "vector" in neighbor_ids


def test_find_path_direct_edge():
    graph = _graph_with_two_hops()
    path = graph.find_path("vector", "matrix_2x2")
    assert path is not None
    assert path.source.representation_id == "vector"
    assert path.target.representation_id == "matrix_2x2"
    assert len(path.edges) == 1


def test_find_path_multi_hop_and_apply():
    graph = _graph_with_two_hops()
    path = graph.find_path("vector", "flat_string")
    assert path is not None
    assert len(path.edges) == 2
    assert path.cost == pytest.approx(2.0)

    value = np.array([1.0, 2.0, 3.0, 4.0])
    encoded = VECTOR.encode(value)
    result = path.apply(encoded)
    assert result == "1.0,2.0,3.0,4.0"


def test_find_path_prefers_cheaper_route():
    graph = RepresentationGraph()
    graph.add_transformation(
        Transformation(transformation_id="direct", source=VECTOR, target=FLAT_STRING, cost=10.0)
    )
    graph.add_transformation(
        Transformation(transformation_id="v2m", source=VECTOR, target=MATRIX, cost=1.0)
    )
    graph.add_transformation(
        Transformation(transformation_id="m2s", source=MATRIX, target=FLAT_STRING, cost=1.0)
    )
    path = graph.find_path("vector", "flat_string")
    assert path is not None
    assert path.cost == pytest.approx(2.0)
    assert [edge.transformation_id for edge in path.edges] == ["v2m", "m2s"]


def test_find_path_returns_none_when_unreachable():
    graph = RepresentationGraph()
    graph.add_representation(VECTOR)
    graph.add_representation(FLAT_STRING)
    assert graph.find_path("vector", "flat_string") is None


def test_find_path_same_source_and_target_returns_none():
    graph = _graph_with_two_hops()
    assert graph.find_path("vector", "vector") is None


def test_find_path_unknown_source_raises():
    graph = _graph_with_two_hops()
    with pytest.raises(KeyError):
        graph.find_path("nonexistent", "vector")


def test_compare_certifies_both_representations():
    graph = _graph_with_two_hops()
    value = np.array([1.0, 2.0, 3.0, 4.0])
    result = graph.compare("vector", "matrix_2x2", value)
    assert result["vector"].status == "verified"
    assert result["matrix_2x2"].status == "verified"


def test_transformation_path_rejects_non_chaining_edges():
    from rde.representation.graph import TransformationPath

    v2m = Transformation(transformation_id="v2m", source=VECTOR, target=MATRIX)
    v2m_again = Transformation(transformation_id="v2m_2", source=VECTOR, target=MATRIX)
    with pytest.raises(ValueError):
        TransformationPath(edges=(v2m, v2m_again))
