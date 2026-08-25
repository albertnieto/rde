"""Tests for typed equivalence notions (ExactEquality, LinearIsomorphism, Isometry,
UnitaryEquivalence, StructurePreservingMap).

Every `holds=True`/`holds=False` asserted here was independently verified
numerically (Gram-matrix off-diagonal energy, roundtrip probing, or direct
edge-set comparison for the graph checks) before being written into a test
— see `equivalence_types.py`'s module docstring.
"""

from __future__ import annotations

import numpy as np
import pytest

from rde.representation import (
    RepresentationGraph,
    Transformation,
    build_primitive_representations,
    check_exact_equality,
    check_isometry,
    check_linear_isomorphism,
    check_structure_preserving_map,
    check_unitary_equivalence,
)
from rde.representation.array_backend import NumpySearchBackend

BACKEND = NumpySearchBackend()
N = 8
GRAMMAR = build_primitive_representations(N, backend=BACKEND)


def test_exact_equality_holds_for_identity():
    value = np.arange(N, dtype=float)[None, :]
    claim = check_exact_equality(GRAMMAR["identity"], value)
    assert claim.holds
    assert claim.equivalence_type == "exact_equality"


def test_exact_equality_fails_at_zero_tolerance_for_floating_point_primitive():
    # A generic (non-integer) value leaves ~1e-16 floating-point residue
    # through dft_full's roundtrip — arange(N) happens to cancel exactly,
    # so a random value is used here instead. ExactEquality's default
    # tolerance=0.0 must catch that residue, unlike check_roundtrip's 1e-9.
    rng = np.random.default_rng(3)
    value = rng.normal(size=N)[None, :]
    claim = check_exact_equality(GRAMMAR["dft_full"], value, tolerance=0.0)
    assert not claim.holds


def test_exact_equality_passes_with_explicit_floating_point_tolerance():
    rng = np.random.default_rng(3)
    value = rng.normal(size=N)[None, :]
    claim = check_exact_equality(GRAMMAR["dft_full"], value, tolerance=1e-9)
    assert claim.holds


def test_linear_isomorphism_holds_for_identity_and_dft_full():
    assert check_linear_isomorphism(GRAMMAR["identity"], N).holds
    assert check_linear_isomorphism(GRAMMAR["dft_full"], N).holds


def test_linear_isomorphism_holds_for_difference_and_polynomial():
    # Both are square, exactly invertible linear maps, even though neither
    # is an isometry (see test_isometry_fails_for_difference_and_polynomial).
    assert check_linear_isomorphism(GRAMMAR["difference"], N).holds
    assert check_linear_isomorphism(GRAMMAR["polynomial_vandermonde"], N).holds


def test_linear_isomorphism_fails_for_compact_dft():
    # dft (rfft) is non-square: C^{n//2+1} != R^n.
    claim = check_linear_isomorphism(GRAMMAR["dft"], N)
    assert not claim.holds


def test_linear_isomorphism_fails_for_nonlinear_sorted_permutation():
    claim = check_linear_isomorphism(GRAMMAR["sorted_permutation"], N)
    assert not claim.holds


def test_isometry_holds_for_identity():
    claim = check_isometry(GRAMMAR["identity"], N)
    assert claim.holds
    assert "scale c=1" in claim.detail or "c=1.000e+00" in claim.detail


def test_isometry_holds_for_dft_full_with_scale_n():
    claim = check_isometry(GRAMMAR["dft_full"], N)
    assert claim.holds
    assert f"c={float(N):.3e}" in claim.detail


def test_isometry_fails_for_difference_and_polynomial():
    assert not check_isometry(GRAMMAR["difference"], N).holds
    assert not check_isometry(GRAMMAR["polynomial_vandermonde"], N).holds


def test_isometry_gives_real_answer_for_non_square_compact_dft():
    # Non-square must not make this vacuously True or raise — encode-only
    # probing is sound regardless of squareness (see probe_encode_matrix).
    claim = check_isometry(GRAMMAR["dft"], N)
    assert claim.holds is False
    assert "off-diagonal max=2.4" in claim.detail


def test_unitary_equivalence_holds_only_for_complex_domain_dft_full():
    assert check_unitary_equivalence(GRAMMAR["dft_full"], N).holds
    assert not check_unitary_equivalence(GRAMMAR["identity"], N).holds
    assert not check_unitary_equivalence(GRAMMAR["dft"], N).holds


def test_isometry_holds_for_dct_with_unit_scale():
    # dct's basis matrix is orthonormal by construction (unlike dft_full's
    # scale-n Gram matrix) -- a real, distinct isometry example.
    claim = check_isometry(GRAMMAR["dct"], N)
    assert claim.holds
    assert "c=1.000e+00" in claim.detail


def test_linear_isomorphism_holds_for_dct():
    assert check_linear_isomorphism(GRAMMAR["dct"], N).holds


def test_unitary_equivalence_fails_for_real_valued_dct():
    # dct is real (no "complex_domain" tag) -- unitary equivalence is
    # specifically a complex-domain notion, same reasoning as identity/dft.
    claim = check_unitary_equivalence(GRAMMAR["dct"], N)
    assert not claim.holds
    assert "complex_domain" in claim.detail


def test_unitary_equivalence_reports_reason_for_non_complex_domain():
    claim = check_unitary_equivalence(GRAMMAR["identity"], N)
    assert "complex_domain" in claim.detail


@pytest.mark.parametrize(
    "representation_id",
    ["identity", "dft", "dft_full", "difference", "polynomial_vandermonde", "dct"],
)
def test_every_linear_primitive_produces_a_claim_without_raising(representation_id):
    rep = GRAMMAR[representation_id]
    for check in (check_linear_isomorphism, check_isometry, check_unitary_equivalence):
        claim = check(rep, N)
        assert isinstance(claim.holds, bool)
        assert claim.representation_id == representation_id


def _two_edge_graph(edge_ids: tuple[str, str] = ("e1", "e2")) -> RepresentationGraph:
    g = RepresentationGraph()
    g.add_transformation(Transformation(edge_ids[0], GRAMMAR["identity"], GRAMMAR["dft_full"]))
    g.add_transformation(Transformation(edge_ids[1], GRAMMAR["dft_full"], GRAMMAR["difference"]))
    return g


def test_structure_preserving_map_is_isomorphism_for_the_identity_map_onto_an_identical_graph():
    # Distinct Transformation objects (different transformation_id/cost),
    # same node/edge structure -- the map's edge-preservation must come
    # from the graph's node/edge structure, not from object identity.
    source = _two_edge_graph(("a", "b"))
    target = _two_edge_graph(("x", "y"))
    node_map = {"identity": "identity", "dft_full": "dft_full", "difference": "difference"}
    claim = check_structure_preserving_map(source, target, node_map)
    assert claim.holds
    assert claim.is_isomorphism


def test_structure_preserving_map_fails_for_a_map_that_breaks_an_edge():
    source = _two_edge_graph()
    target = _two_edge_graph()
    # Swap the two non-identity nodes' images: identity->dft_full is fine,
    # but dft_full->difference no longer has an image edge.
    broken_map = {"identity": "identity", "dft_full": "difference", "difference": "dft_full"}
    claim = check_structure_preserving_map(source, target, broken_map)
    assert not claim.holds
    assert not claim.is_isomorphism
    assert "not edge-preserving" in claim.detail


def test_structure_preserving_map_is_homomorphism_but_not_isomorphism_when_collapsing_nodes():
    # A genuine graph homomorphism that maps two source nodes onto the same
    # target node -- edge-preserving (identity->dft_full and dft_full->dft_full
    # both exist in target), but not injective, so not an isomorphism.
    source = _two_edge_graph()
    target = RepresentationGraph()
    target.add_transformation(Transformation("a", GRAMMAR["identity"], GRAMMAR["dft_full"]))
    target.add_transformation(Transformation("self_loop", GRAMMAR["dft_full"], GRAMMAR["dft_full"]))
    collapsing_map = {"identity": "identity", "dft_full": "dft_full", "difference": "dft_full"}
    claim = check_structure_preserving_map(source, target, collapsing_map)
    assert claim.holds
    assert not claim.is_isomorphism
    assert "bijective onto target's 2 node(s): False" in claim.detail


def test_structure_preserving_map_rejects_a_partial_node_map():
    source = _two_edge_graph()
    target = _two_edge_graph()
    partial_map = {"identity": "identity", "dft_full": "dft_full"}  # "difference" missing
    claim = check_structure_preserving_map(source, target, partial_map)
    assert not claim.holds
    assert not claim.is_isomorphism
    assert "not total on source_graph" in claim.detail


def test_structure_preserving_map_rejects_a_map_to_an_unregistered_target_node():
    source = _two_edge_graph()
    target = _two_edge_graph()
    bad_map = {"identity": "identity", "dft_full": "dft_full", "difference": "not_a_real_node"}
    claim = check_structure_preserving_map(source, target, bad_map)
    assert not claim.holds
    assert "not registered in target_graph" in claim.detail


def test_structure_preserving_map_bijection_check_ignores_irrelevant_extra_map_keys():
    # Regression: an audit stress test found that len(node_map) alone (not
    # restricted to source_graph's own nodes) could be inflated by an
    # irrelevant extra key, risking a spuriously "bijective" count even
    # when the map isn't actually a bijection restricted to source_nodes.
    # The edge-set-exact-match condition happens to catch this particular
    # case too, but is_bijection itself must be correct on its own.
    source = RepresentationGraph()
    source.add_transformation(Transformation("a", GRAMMAR["identity"], GRAMMAR["dft_full"]))

    target = RepresentationGraph()
    target.add_transformation(Transformation("b", GRAMMAR["identity"], GRAMMAR["dft_full"]))
    target.add_transformation(Transformation("c", GRAMMAR["dft_full"], GRAMMAR["difference"]))

    node_map = {
        "identity": "identity",
        "dft_full": "dft_full",
        "difference": "difference",  # "difference" is not a node of source_graph
    }
    claim = check_structure_preserving_map(source, target, node_map)
    assert claim.holds  # source's one edge is still preserved
    assert not claim.is_isomorphism
    assert "bijective onto target's 3 node(s): False" in claim.detail


def test_representation_graph_node_ids_and_edge_pairs():
    g = _two_edge_graph()
    assert set(g.node_ids()) == {"identity", "dft_full", "difference"}
    assert g.edge_pairs() == {("identity", "dft_full"), ("dft_full", "difference")}


def test_representation_graph_edge_pairs_collapses_parallel_edges():
    g = RepresentationGraph()
    g.add_transformation(Transformation("a", GRAMMAR["identity"], GRAMMAR["dft_full"], cost=1.0))
    g.add_transformation(Transformation("b", GRAMMAR["identity"], GRAMMAR["dft_full"], cost=2.0))
    assert g.edge_pairs() == {("identity", "dft_full")}
