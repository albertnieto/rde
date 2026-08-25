"""Tests for the Representation Core: Object/Representation/Transformation/Certificate.

These exercise the roundtrip law `decode(encode(x)) == x` on reference
representations (vector, reshaped matrix, polynomial-coefficient) that are
generic numerical fixtures, not any research domain's objects — core RDE
ships no domain-specific representations.
"""

from __future__ import annotations

import numpy as np
import pytest

from rde.representation import (
    Certificate,
    Object,
    Representation,
    Transformation,
    certify_roundtrip,
    check_roundtrip,
    serialized_size_complexity,
)

VECTOR = Representation(
    representation_id="vector",
    object_type="numeric_4",
    carrier="R^4",
    encode=lambda x: np.asarray(x, dtype=float),
    decode=lambda x: np.asarray(x, dtype=float),
    complexity=serialized_size_complexity,
)

MATRIX = Representation(
    representation_id="matrix_2x2",
    object_type="numeric_4",
    carrier="R^{2x2}",
    encode=lambda x: np.asarray(x, dtype=float).reshape(2, 2),
    decode=lambda m: np.asarray(m, dtype=float).reshape(4),
    complexity=serialized_size_complexity,
)


def _polyfit_nodes() -> np.ndarray:
    return np.arange(4, dtype=float)


def _poly_encode(y: np.ndarray) -> np.ndarray:
    return np.polyfit(_polyfit_nodes(), np.asarray(y, dtype=float), deg=3)


def _poly_decode(coeffs: np.ndarray) -> np.ndarray:
    return np.polyval(coeffs, _polyfit_nodes())


POLYNOMIAL = Representation(
    representation_id="polynomial_coeffs",
    object_type="numeric_4",
    carrier="R[x]_{<=3} evaluated at fixed nodes",
    encode=_poly_encode,
    decode=_poly_decode,
    complexity=serialized_size_complexity,
)


def test_object_is_identity_and_metadata_only():
    obj = Object(object_id="v1", object_type="numeric_4", metadata={"source": "test"})
    assert obj.object_id == "v1"
    assert obj.metadata["source"] == "test"


def test_vector_representation_roundtrips_exactly():
    value = np.array([1.0, 2.0, 3.0, 4.0])
    result = check_roundtrip(VECTOR, value)
    assert result.equivalent
    assert result.error == 0.0


def test_matrix_representation_roundtrips_exactly():
    value = np.array([1.0, 2.0, 3.0, 4.0])
    result = check_roundtrip(MATRIX, value)
    assert result.equivalent
    assert result.error == 0.0


def test_polynomial_representation_roundtrips_within_tolerance():
    value = np.array([1.0, 2.0, 3.0, 4.0])
    result = check_roundtrip(POLYNOMIAL, value, tolerance=1e-6)
    assert result.equivalent


def test_transformation_composes_source_decode_and_target_encode():
    value = np.array([1.0, 2.0, 3.0, 4.0])
    to_matrix = Transformation(transformation_id="vector_to_matrix", source=VECTOR, target=MATRIX)
    encoded_matrix = to_matrix.apply(VECTOR.encode(value))
    assert np.array_equal(encoded_matrix, value.reshape(2, 2))

    back = to_matrix.invert()
    recovered = back.apply(encoded_matrix)
    assert np.array_equal(recovered, value)


def test_transformation_rejects_mismatched_object_types():
    other = Representation(
        representation_id="other",
        object_type="numeric_9",
        carrier="R^9",
        encode=lambda x: x,
        decode=lambda x: x,
    )
    with pytest.raises(ValueError):
        Transformation(transformation_id="bad", source=VECTOR, target=other)


def test_certify_roundtrip_verifies_exact_representation():
    value = np.array([1.0, 2.0, 3.0, 4.0])
    cert = certify_roundtrip(MATRIX, value)
    assert isinstance(cert, Certificate)
    assert cert.status == "verified"
    assert cert.claim == "exact_roundtrip"
    assert cert.to_payload()["representation_id"] == "matrix_2x2"


def test_certify_roundtrip_refutes_broken_representation():
    broken = Representation(
        representation_id="broken",
        object_type="numeric_4",
        carrier="R^4 (lossy)",
        encode=lambda x: np.asarray(x, dtype=float),
        decode=lambda x: np.zeros_like(np.asarray(x, dtype=float)),
    )
    value = np.array([1.0, 2.0, 3.0, 4.0])
    cert = certify_roundtrip(broken, value)
    assert cert.status == "refuted"


def test_serialized_size_complexity_matches_nbytes():
    value = np.array([1.0, 2.0, 3.0, 4.0])
    assert serialized_size_complexity(value) == value.nbytes
