"""Tests for the dual-backend (NumPy/MLX) batched array kernels.

NumPy is exercised fully here (every machine). MLX kernels are exercised
only when `rde.backends.mlx_usable()` is true, matching the existing
`tests/rde/runtime/test_performance.py` convention — skipped, not faked,
on hardware without Metal GPU access.
"""

from __future__ import annotations

import numpy as np
import pytest

from rde.backends import mlx_usable
from rde.representation.array_backend import MlxSearchBackend, NumpySearchBackend, get_array_backend

NP = NumpySearchBackend()


def test_rfft_irfft_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 8))
    coeffs = NP.rfft(x)
    assert coeffs.shape == (5, 5)
    recon = NP.irfft(coeffs, 8)
    assert np.allclose(recon, x, atol=1e-10)


def test_fft_ifft_roundtrip_is_square():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 6))
    coeffs = NP.fft(x)
    assert coeffs.shape == (4, 6)
    assert np.iscomplexobj(coeffs)
    recon = NP.ifft(coeffs)
    assert np.allclose(recon.real, x, atol=1e-10)
    assert np.allclose(recon.imag, 0, atol=1e-10)


def test_diff_with_first_and_cumsum_are_inverses():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 7))
    diffs = NP.diff_with_first(x)
    assert diffs.shape == x.shape
    recon = NP.cumsum(diffs)
    assert np.allclose(recon, x)


def test_diff_with_first_complex_and_cumsum_complex_are_inverses():
    rng = np.random.default_rng(1)
    real = rng.normal(size=(3, 7))
    imag = rng.normal(size=(3, 7))
    x = real + 1j * imag
    diffs = NP.diff_with_first_complex(x)
    assert diffs.shape == x.shape
    assert np.iscomplexobj(diffs)
    recon = NP.cumsum_complex(diffs)
    assert np.allclose(recon, x)


def test_diff_with_first_complex_preserves_imaginary_part_unlike_diff_with_first():
    # The whole reason diff_with_first_complex/cumsum_complex exist: the
    # real-only pair silently discards the imaginary part instead of
    # failing loudly, which would corrupt a genuinely complex carrier
    # (sort_by_magnitude's "sorted_complex_pair") rather than just fail to
    # compress it.
    x = np.array([[1.0 + 2.0j, 3.0 - 1.0j, 0.5 + 0.5j]])
    complex_safe = NP.diff_with_first_complex(x)
    with pytest.warns(np.exceptions.ComplexWarning):
        # Expected: diff_with_first's dtype=float cast on genuinely complex
        # input is exactly the truncation this test demonstrates is wrong
        # for this carrier -- not a mistake to silence.
        real_only = NP.diff_with_first(x)
    assert np.any(np.imag(complex_safe) != 0)
    assert np.all(np.imag(real_only) == 0)
    recon_complex_safe = NP.cumsum_complex(complex_safe)
    assert np.allclose(recon_complex_safe, x)


def test_diff_with_first_first_column_is_original_first_value():
    x = np.array([[10.0, 12.0, 9.0]])
    diffs = NP.diff_with_first(x)
    assert diffs[0, 0] == 10.0
    assert np.allclose(diffs[0, 1:], [2.0, -3.0])


def test_sort_with_permutation_and_apply_inverse_permutation_roundtrip():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(4, 6))
    sorted_values, perm = NP.sort_with_permutation(x)
    assert np.all(np.diff(sorted_values, axis=-1) >= 0)
    recon = NP.apply_inverse_permutation(sorted_values, perm)
    assert np.allclose(recon, x)


def test_matmul_shared_matches_manual_matmul():
    rng = np.random.default_rng(3)
    batch = rng.normal(size=(5, 4))
    matrix = rng.normal(size=(4, 4))
    result = NP.matmul_shared(batch, matrix)
    assert np.allclose(result, batch @ matrix.T)


def test_sparsity_fraction_counts_significant_entries():
    x = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]])
    assert NP.sparsity_fraction(x, eps=1e-8) == pytest.approx(5 / 8)


def test_permutation_complexity_identity_permutation_costs_zero():
    perm = np.array([[0, 1, 2, 3, 4, 5]])
    assert NP.permutation_complexity(perm) == pytest.approx(0.0)


def test_permutation_complexity_block_swap_costs_one_break():
    # Two contiguous ascending blocks swapped -- one break, not n-1.
    perm = np.array([[3, 4, 5, 0, 1, 2]])
    assert NP.permutation_complexity(perm) == pytest.approx(1.0)


def test_permutation_complexity_full_reversal_is_maximally_broken():
    n = 6
    perm = np.array([np.arange(n - 1, -1, -1)])
    assert NP.permutation_complexity(perm) == pytest.approx(n - 1)


def test_permutation_complexity_matches_manual_break_count():
    rng = np.random.default_rng(5)
    perm = np.stack([rng.permutation(10) for _ in range(4)])
    breaks = [int(np.sum(row[1:] != row[:-1] + 1)) for row in perm]
    assert NP.permutation_complexity(perm) == pytest.approx(np.mean(breaks))


def test_permutation_complexity_no_longer_flatly_charges_every_permutation_the_same():
    # Regression test for the bug this replaced: `sparsity_fraction(perm,
    # eps=0.5)` counted every permutation entry except value 0 as
    # "significant," so it charged ~n-1 for *every* permutation, identity
    # included -- it measured the value range, not the actual structure.
    identity = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    scrambled = np.array([[5, 2, 7, 0, 4, 1, 6, 3]])
    assert NP.permutation_complexity(identity) < NP.permutation_complexity(scrambled)


def test_max_abs_diff():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.5, 2.0])
    assert NP.max_abs_diff(a, b) == pytest.approx(1.0)


def test_get_array_backend_defaults_to_numpy_when_mlx_unusable():
    if mlx_usable():
        pytest.skip("MLX is usable on this machine; default may resolve to mlx")
    backend = get_array_backend()
    assert backend.name == "numpy"


def test_get_array_backend_explicit_numpy():
    assert get_array_backend("numpy").name == "numpy"


def test_get_array_backend_rejects_unknown_name():
    with pytest.raises(ValueError):
        get_array_backend("not_a_backend")


def test_get_array_backend_mlx_without_hardware_raises_import_error():
    if mlx_usable():
        pytest.skip("mlx is usable on this machine")
    with pytest.raises((ImportError, Exception)):
        get_array_backend("mlx")


class TestMlxSearchBackendParity:
    """Every kernel, checked against the NumPy reference. Requires Apple Silicon + MLX."""

    def setup_method(self):
        if not mlx_usable():
            pytest.skip("mlx not available")

    def test_rfft_irfft_matches_numpy(self):
        mlx = MlxSearchBackend()
        rng = np.random.default_rng(0)
        x = rng.normal(size=(5, 8)).astype(np.float32)
        np_coeffs = NP.rfft(x)
        mx_coeffs = mlx.to_numpy(mlx.rfft(x))
        assert np.allclose(mx_coeffs, np_coeffs, atol=1e-4)
        np_recon = NP.irfft(np_coeffs, 8)
        mx_recon = mlx.to_numpy(mlx.irfft(mx_coeffs, 8))
        assert np.allclose(mx_recon, np_recon, atol=1e-4)

    def test_fft_ifft_matches_numpy(self):
        mlx = MlxSearchBackend()
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4, 6)).astype(np.float32)
        np_coeffs = NP.fft(x)
        mx_coeffs = mlx.to_numpy(mlx.fft(x))
        assert np.allclose(mx_coeffs, np_coeffs, atol=1e-4)

    def test_diff_cumsum_matches_numpy(self):
        mlx = MlxSearchBackend()
        rng = np.random.default_rng(1)
        x = rng.normal(size=(3, 7)).astype(np.float32)
        np_diff = NP.diff_with_first(x)
        mx_diff = mlx.to_numpy(mlx.diff_with_first(x))
        assert np.allclose(mx_diff, np_diff, atol=1e-4)
        mx_recon = mlx.to_numpy(mlx.cumsum(mx_diff))
        assert np.allclose(mx_recon, x, atol=1e-4)

    def test_sort_with_permutation_matches_numpy(self):
        mlx = MlxSearchBackend()
        rng = np.random.default_rng(2)
        x = rng.normal(size=(4, 6)).astype(np.float32)
        np_sorted, np_perm = NP.sort_with_permutation(x)
        mx_sorted, mx_perm = mlx.sort_with_permutation(x)
        assert np.allclose(mlx.to_numpy(mx_sorted), np_sorted, atol=1e-4)
        mx_recon = mlx.to_numpy(mlx.apply_inverse_permutation(mx_sorted, mx_perm))
        assert np.allclose(mx_recon, x, atol=1e-4)

    def test_permutation_complexity_matches_numpy(self):
        mlx = MlxSearchBackend()
        rng = np.random.default_rng(5)
        perm = np.stack([rng.permutation(10) for _ in range(4)]).astype(np.int32)
        np_cost = NP.permutation_complexity(perm)
        mx_cost = mlx.permutation_complexity(perm)
        assert mx_cost == pytest.approx(np_cost)

    def test_sort_by_magnitude_with_permutation_matches_numpy(self):
        mlx = MlxSearchBackend()
        rng = np.random.default_rng(2)
        real = rng.normal(size=(4, 6)).astype(np.float32)
        imag = rng.normal(size=(4, 6)).astype(np.float32)
        x = (real + 1j * imag).astype(np.complex64)
        np_sorted, np_perm = NP.sort_by_magnitude_with_permutation(x)
        mx_sorted, mx_perm = mlx.sort_by_magnitude_with_permutation(x)
        assert np.allclose(mlx.to_numpy(mx_sorted), np_sorted, atol=1e-4)
        mx_recon = mlx.to_numpy(mlx.apply_inverse_permutation(mx_sorted, mx_perm))
        assert np.allclose(mx_recon, x, atol=1e-4)

    def test_diff_cumsum_complex_matches_numpy(self):
        mlx = MlxSearchBackend()
        rng = np.random.default_rng(4)
        real = rng.normal(size=(3, 7)).astype(np.float32)
        imag = rng.normal(size=(3, 7)).astype(np.float32)
        x = (real + 1j * imag).astype(np.complex64)
        np_diff = NP.diff_with_first_complex(x)
        mx_diff = mlx.to_numpy(mlx.diff_with_first_complex(x))
        assert np.allclose(mx_diff, np_diff, atol=1e-4)
        mx_recon = mlx.to_numpy(mlx.cumsum_complex(mx_diff))
        assert np.allclose(mx_recon, x, atol=1e-4)

    def test_matmul_shared_matches_numpy(self):
        mlx = MlxSearchBackend()
        rng = np.random.default_rng(3)
        batch = rng.normal(size=(5, 4)).astype(np.float32)
        matrix = rng.normal(size=(4, 4)).astype(np.float32)
        np_result = NP.matmul_shared(batch, matrix)
        mx_result = mlx.to_numpy(mlx.matmul_shared(batch, matrix))
        assert np.allclose(mx_result, np_result, atol=1e-3)

    def test_get_array_backend_auto_prefers_mlx(self):
        assert get_array_backend().name == "mlx"
