"""Tests for the fixed representation grammar (Phase 3 primitives)."""

from __future__ import annotations

import numpy as np
import pytest

from rde.representation import build_primitive_representations, certify_roundtrip, primitive_names
from rde.representation.array_backend import NumpySearchBackend

BACKEND = NumpySearchBackend()


def test_primitive_names_lists_all_builders():
    names = primitive_names()
    assert set(names) == {
        "identity",
        "matrix_reshape",
        "dft",
        "dft_full",
        "difference",
        "sorted_permutation",
        "polynomial_vandermonde",
        "dct",
    }


def test_build_primitive_representations_omits_matrix_reshape_for_non_square_n():
    grammar = build_primitive_representations(7, backend=BACKEND)
    assert "matrix_reshape" not in grammar
    assert set(grammar) == set(primitive_names()) - {"matrix_reshape"}


def test_build_primitive_representations_includes_matrix_reshape_for_square_n():
    grammar = build_primitive_representations(9, backend=BACKEND)
    assert "matrix_reshape" in grammar


@pytest.mark.parametrize("n", [4, 6, 8])
def test_every_primitive_roundtrips_within_tolerance(n):
    rng = np.random.default_rng(42)
    batch = rng.normal(size=(6, n))
    grammar = build_primitive_representations(n, backend=BACKEND)
    for representation_id, representation in grammar.items():
        certificate = certify_roundtrip(representation, batch, tolerance=1e-6)
        assert certificate.status == "verified", (
            f"{representation_id} failed roundtrip at n={n}: error={certificate.error}"
        )


def test_matrix_reshape_encode_produces_square_matrices():
    grammar = build_primitive_representations(9, backend=BACKEND)
    rep = grammar["matrix_reshape"]
    batch = np.arange(9 * 2, dtype=float).reshape(2, 9)
    encoded = rep.encode(batch)
    assert encoded.shape == (2, 3, 3)
    decoded = rep.decode(encoded)
    assert np.array_equal(decoded, batch)


def test_dft_is_compact_and_dft_full_is_square():
    n = 8
    grammar = build_primitive_representations(n, backend=BACKEND)
    batch = np.zeros((1, n))
    assert grammar["dft"].encode(batch).shape == (1, n // 2 + 1)
    assert grammar["dft_full"].encode(batch).shape == (1, n)


def test_sorted_permutation_recovers_original_order():
    grammar = build_primitive_representations(5, backend=BACKEND)
    rep = grammar["sorted_permutation"]
    batch = np.array([[3.0, 1.0, 4.0, 1.5, 9.0]])
    values, perm = rep.encode(batch)
    assert np.all(np.diff(values[0]) >= 0)
    decoded = rep.decode((values, perm))
    assert np.allclose(decoded, batch)


def test_polynomial_vandermonde_recovers_exact_linear_signal():
    n = 6
    grammar = build_primitive_representations(n, backend=BACKEND)
    rep = grammar["polynomial_vandermonde"]
    nodes = np.arange(n, dtype=float)
    batch = np.stack([2.0 + 3.0 * nodes, -1.0 + 0.5 * nodes])
    coeffs = rep.encode(batch)
    # A linear signal has zero coefficients beyond degree 1.
    assert np.allclose(coeffs[:, 2:], 0.0, atol=1e-8)
    decoded = rep.decode(coeffs)
    assert np.allclose(decoded, batch, atol=1e-6)


def test_grammar_primitives_share_object_type():
    grammar = build_primitive_representations(6, backend=BACKEND)
    object_types = {rep.object_type for rep in grammar.values()}
    assert object_types == {"numeric_batch_6"}


def test_custom_object_type_is_propagated():
    grammar = build_primitive_representations(4, object_type="my_object", backend=BACKEND)
    assert all(rep.object_type == "my_object" for rep in grammar.values())


def test_primitive_subset_restricts_to_exactly_the_named_primitives():
    grammar = build_primitive_representations(
        8, backend=BACKEND, primitive_subset=["identity", "dft"]
    )
    assert set(grammar.keys()) == {"identity", "dft"}


def test_primitive_subset_still_omits_matrix_reshape_for_non_square_n():
    grammar = build_primitive_representations(
        7, backend=BACKEND, primitive_subset=["identity", "matrix_reshape"]
    )
    assert set(grammar.keys()) == {"identity"}


def test_primitive_subset_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_primitive_representations(8, backend=BACKEND, primitive_subset=["not_a_primitive"])


def test_dct_matches_scipy_dct_ii_ortho():
    from scipy.fft import dct as scipy_dct

    n = 8
    grammar = build_primitive_representations(n, backend=BACKEND)
    rng = np.random.default_rng(0)
    batch = rng.normal(size=(4, n))
    ours = grammar["dct"].encode(batch)
    theirs = np.stack([scipy_dct(row, type=2, norm="ortho") for row in batch])
    assert np.allclose(ours, theirs, atol=1e-10)


def test_dct_matrix_is_orthogonal():
    from rde.representation.grammar import _dct_matrix

    n = 8
    C = _dct_matrix(n)
    assert np.allclose(C @ C.T, np.eye(n), atol=1e-10)


def test_dct_beats_dft_and_identity_on_dct_sparse_data():
    # Verified numerically before being written here (see grammar.py's
    # _dct_representation docstring for the full story, including a
    # disproven earlier claim on a different data shape): built as an
    # exact 2-of-16-sparse combination of DCT basis vectors, dct reaches
    # a strictly lower complexity than identity, dft (rfft), and dft_full
    # -- a real structural difference (DCT's half-integer-frequency basis
    # vs DFT's integer-frequency one), not a tuning artifact.
    from rde.representation.grammar import _dct_matrix

    n = 16
    C = _dct_matrix(n)
    rng = np.random.default_rng(1)
    batch = []
    for _ in range(6):
        coeffs = np.zeros(n)
        idx = rng.choice(n, size=2, replace=False)
        coeffs[idx] = rng.uniform(1, 5, size=2)
        batch.append(coeffs @ C)
    batch = np.stack(batch)

    grammar = build_primitive_representations(n, backend=BACKEND)
    complexities = {}
    for name in ("identity", "dft", "dft_full", "dct"):
        rep = grammar[name]
        cert = certify_roundtrip(rep, batch, tolerance=1e-6)
        assert cert.status == "verified"
        complexities[name] = rep.complexity(rep.encode(batch))

    assert complexities["dct"] == pytest.approx(2.0)
    assert complexities["dct"] < complexities["dft"]
    assert complexities["dct"] < complexities["dft_full"]
    assert complexities["dct"] < complexities["identity"]


def test_dct_does_not_need_complex_domain_tag():
    grammar = build_primitive_representations(8, backend=BACKEND)
    assert "complex_domain" not in grammar["dct"].invariants
    assert "linear" in grammar["dct"].invariants


def test_dct_inverse_is_well_conditioned_unlike_polynomial_vandermonde():
    # dct's matrix is orthogonal (cond == 1 by construction); unlike
    # polynomial_vandermonde's famously ill-conditioned inverse, this
    # primitive's roundtrip error must not grow with n.
    from rde.representation.grammar import _dct_matrix

    for n in (8, 16, 32):
        C = _dct_matrix(n)
        cond = np.linalg.cond(C)
        assert cond == pytest.approx(1.0, abs=1e-6)
