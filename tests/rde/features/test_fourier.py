"""Unit tests for Walsh/Fourier descriptors."""

from __future__ import annotations

import numpy as np
import pytest

from rde.features.fourier import descriptor, walsh_hadamard, walsh_hadamard_batch
from rde.core.instance import InstanceRecord


def _inst() -> InstanceRecord:
    return InstanceRecord(domain_id="test", size=4, seed=0, params={})


def test_walsh_hadamard_inverse_up_to_scale():
    x = np.array([1.0, -1.0, 1.0, 1.0])
    fwd = walsh_hadamard(x)
    back = walsh_hadamard(fwd)
    assert np.allclose(back, x, atol=1e-10)


def test_walsh_hadamard_batch_matches_single_rows():
    rng = np.random.default_rng(123)
    matrix = rng.normal(size=(5, 32))
    batch = walsh_hadamard_batch(matrix)
    expected = np.stack([walsh_hadamard(row) for row in matrix])
    assert np.allclose(batch, expected, atol=1e-12)


def test_walsh_hadamard_large_vector_roundtrip():
    rng = np.random.default_rng(456)
    x = rng.normal(size=2**10)
    assert np.allclose(walsh_hadamard(walsh_hadamard(x)), x, atol=1e-10)


def test_walsh_invalid_length_raises():
    with pytest.raises(ValueError, match="2\\^k"):
        walsh_hadamard(np.array([1.0, 2.0, 3.0]))


def test_descriptor_invalid_length():
    out = descriptor(_inst(), None, np.array([1.0, 2.0, 3.0]))
    assert out["fourier.valid_length"] == 0.0


def test_descriptor_delta_function_sparsity():
    delta = np.zeros(8)
    delta[0] = 1.0
    out = descriptor(_inst(), None, delta)
    assert out["fourier.valid_length"] == 1.0
    assert out["fourier.sparsity"] >= 0.0
