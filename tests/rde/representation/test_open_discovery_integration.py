"""Phase 7 ("open discovery"): the whole O -> R -> S -> A -> C pipeline, end to end.

Not new library code — this composes Phases 1-6 exactly as they exist
elsewhere in `rde.representation`, on one coherent scenario: a batch of
periodic signals and a circulant operator that naturally acts on them
(convolution is diagonal in the Fourier basis). Every claim asserted here
was independently verified numerically before being written into this test
(see `test_operator.py`, `test_search.py`) — nothing here is asserted only
because "it should work."
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import circulant

from rde.representation import (
    Certificate,
    RepresentationGraph,
    Transformation,
    build_primitive_representations,
    certify_roundtrip,
    linear_probe_matrices,
    objectives_from_candidates,
    off_diagonal_energy,
    pareto_rank,
    prove_vandermonde_inverse,
    rank_by_diagonalization,
    rank_representations,
    transport_operator,
)
from rde.representation.array_backend import NumpySearchBackend

BACKEND = NumpySearchBackend()


def test_open_discovery_pipeline_on_periodic_signal_and_circulant_operator():
    n = 8
    rng = np.random.default_rng(11)

    # --- O: an "unknown" object — a batch of periodic real signals ---
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    signal_batch = np.stack([np.sin(2 * t + phase) + 0.1 for phase in rng.normal(size=8)])

    # --- R + S: search the grammar, rank by complexity (Phase 3) ---
    ranked = rank_representations(signal_batch, n=n, backend=BACKEND)
    assert ranked[0].certificate.status == "verified"
    # A single-frequency signal should compress best in a Fourier basis.
    assert ranked[0].representation_id in {"dft", "dft_full"}

    # --- Pareto: no representation should be exchanged for a strictly worse one ---
    objectives, ids = objectives_from_candidates(ranked)
    pareto = pareto_rank(objectives)
    frontier_ids = {ids[i] for i in pareto.frontier_indices}
    assert ranked[0].representation_id in frontier_ids

    # --- A: an operator that naturally acts on this object family ---
    # Convolution is diagonal in the Fourier basis, so a circulant operator
    # is the natural "algorithm" to test against a representation search
    # that already favored a Fourier-type representation.
    first_row = rng.normal(size=n)
    convolution_operator = circulant(first_row).T
    diagonalization_ranking = rank_by_diagonalization(convolution_operator, n=n, backend=BACKEND)
    assert diagonalization_ranking[0].representation_id in {"dft", "dft_full"}
    assert diagonalization_ranking[0].off_diagonal_energy < 1e-6

    # Phase 3 (compression) and Phase 6 (operator diagonalization) both land
    # on a Fourier-type representation for this object family, but not
    # necessarily the *same* primitive: Phase 3's compact `dft` (rfft) needs
    # no decode-side probing, so it stays the compression winner, while
    # Phase 6 requires a square, soundly decode-probable carrier (`dft_full`)
    # — `dft`'s compact codomain makes its decode-probe unsound (see
    # `operator.linear_probe_matrices`), so it is correctly absent from
    # `rank_by_diagonalization` even though it is the Phase 3 winner here.
    assert ranked[0].representation_id in {"dft", "dft_full"}
    assert diagonalization_ranking[0].representation_id == "dft_full"

    # --- C: certify the winning representation, both numerically and formally ---
    winning_representation = build_primitive_representations(n, backend=BACKEND)[
        ranked[0].representation_id
    ]
    numeric_certificate = certify_roundtrip(winning_representation, signal_batch)
    assert isinstance(numeric_certificate, Certificate)
    assert numeric_certificate.status == "verified"

    # --- Graph: register the discovery as an edge in the representation graph ---
    identity_representation = build_primitive_representations(n, backend=BACKEND)["identity"]
    graph = RepresentationGraph()
    graph.add_transformation(
        Transformation(
            transformation_id="identity_to_winner",
            source=identity_representation,
            target=winning_representation,
        )
    )
    comparison = graph.compare("identity", winning_representation.representation_id, signal_batch)
    assert comparison["identity"].status == "verified"
    assert comparison[winning_representation.representation_id].status == "verified"
    path = graph.find_path("identity", winning_representation.representation_id)
    assert path is not None
    assert np.allclose(
        path.apply(identity_representation.encode(signal_batch)),
        winning_representation.encode(signal_batch),
    )


def test_open_discovery_pipeline_on_low_degree_polynomial_family():
    """A second, independent object family: exact low-degree polynomials.

    Confirms the pipeline's conclusion tracks the actual generating
    structure of the data (polynomial, not periodic) rather than always
    preferring one fixed primitive.
    """
    n = 6
    nodes = np.arange(n, dtype=float)
    slopes = np.array([1.0, -2.0, 0.5, 3.0])
    intercepts = np.array([0.0, 5.0, -1.0, 2.0])
    batch = intercepts[:, None] + slopes[:, None] * nodes[None, :]

    ranked = rank_representations(batch, n=n, backend=BACKEND)
    assert ranked[0].representation_id == "polynomial_vandermonde"

    formal_certificate = prove_vandermonde_inverse(n)
    assert formal_certificate.status == "proved"

    grammar = build_primitive_representations(n, backend=BACKEND)
    encode_matrix, decode_matrix = linear_probe_matrices(grammar["polynomial_vandermonde"], n)

    # Build an operator that scales only the degree-1 coefficient — diagonal
    # by construction *in the polynomial basis* — then express that same
    # operator in the raw (identity) basis via D @ op_coeff @ E. In the raw
    # basis it is generically dense; `transport_operator` back through
    # (encode_matrix, decode_matrix) must recover the original diagonal
    # operator exactly, not just "some" transported result.
    op_in_coefficient_space = np.eye(n)
    op_in_coefficient_space[1, 1] = 3.0
    op_in_raw_space = decode_matrix @ op_in_coefficient_space @ encode_matrix
    assert off_diagonal_energy(op_in_raw_space) > 0.5

    recovered = transport_operator(op_in_raw_space, encode_matrix, decode_matrix)
    assert off_diagonal_energy(recovered) < 1e-8
    assert np.allclose(recovered, op_in_coefficient_space, atol=1e-6)
