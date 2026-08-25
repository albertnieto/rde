"""Dual-backend (NumPy / MLX) batched array kernels for representation search.

Every kernel here operates on a whole batch `(B, N)` in one call — a single
vectorized/BLAS/GPU op, never a Python loop over `B` or `N`. This is the
numeric floor that `grammar.py`, `search.py`, `pareto.py`, and `operator.py`
build on, mirroring the existing dual-backend split in `rde.backends`
(`NumpyBackend` / `MlxBackend`) rather than inventing a second convention.

`NumpySearchBackend` is exercised by the full test suite on every machine.
`MlxSearchBackend` targets Apple Silicon GPU and is exercised only when
`rde.backends.mlx_usable()` is true (see `dev-machine-profile`); it is
implemented against MLX's documented, numpy-compatible core ops
(`mx.fft.rfft/irfft`, `mx.cumsum`, `mx.argsort`, `mx.take_along_axis`,
matmul, reductions) so the two backends compute identical results, but it
has not been run on Apple Silicon hardware in this environment — treat it
as implemented-and-reviewed, not hardware-verified, until it is.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from rde.backends import mlx_usable


@runtime_checkable
class ArraySearchBackend(Protocol):
    """Batched numeric kernels shared by every representation primitive."""

    name: str

    def rfft(self, x: Any) -> Any:
        """Real FFT along the last axis of `(B, N)` -> `(B, N//2+1)` complex."""
        ...

    def irfft(self, x: Any, n: int) -> Any:
        """Inverse of `rfft`: `(B, N//2+1)` complex -> `(B, N)` real."""
        ...

    def fft(self, x: Any) -> Any:
        """Full complex FFT along the last axis of `(B, N)` -> `(B, N)` complex.

        Unlike `rfft` (which packs a real signal's Hermitian-symmetric
        spectrum into `N//2+1` coefficients — compact, but not a plain
        square matrix), this is a genuine square, exactly invertible linear
        map on `C^N`. `operator.py`'s transport needs that square structure;
        `grammar.py`'s `dft` primitive needs `rfft`'s compactness instead —
        they are kept as two separate primitives for that reason.
        """
        ...

    def ifft(self, x: Any) -> Any:
        """Inverse of `fft`."""
        ...

    def diff_with_first(self, x: Any) -> Any:
        """`[x[:, :1], x[:, 1:] - x[:, :-1]]` concatenated along the last axis."""
        ...

    def cumsum(self, x: Any) -> Any:
        """Cumulative sum along the last axis — inverse of `diff_with_first`."""
        ...

    def diff_with_first_complex(self, x: Any) -> Any:
        """`diff_with_first`, but dtype-preserving (no real-part coercion).

        `diff_with_first`/`cumsum` deliberately discard the imaginary part
        (`_to_real`) because every `real_vector`-typed grammar stage that
        uses them (`difference`, `polynomial_vandermonde`) is only ever
        exact on real data — a complex-dtyped value reaching that boundary
        is `dft_full.decode`'s known artifact (see `_to_real`'s docstring),
        not genuine signal. `sort_by_magnitude`'s `"sorted_complex_pair"`
        carrier is the opposite case: its values are *genuinely* complex
        (a DFT coefficient's phase, not a decode artifact), so a
        `"sorted_complex_pair" -> "sorted_complex_pair"` continuation needs
        a difference/cumsum pair that preserves both parts exactly.
        """
        ...

    def cumsum_complex(self, x: Any) -> Any:
        """Inverse of `diff_with_first_complex` — cumulative sum without real coercion."""
        ...

    def sort_with_permutation(self, x: Any) -> tuple[Any, Any]:
        """`(sorted_values, permutation)` along the last axis."""
        ...

    def apply_inverse_permutation(self, values: Any, perm: Any) -> Any:
        """Undo `sort_with_permutation`: scatter `values` back to `perm`'s order."""
        ...

    def sort_by_magnitude_with_permutation(self, x: Any) -> tuple[Any, Any]:
        """`(values sorted by descending |value|, permutation)` along the last axis.

        Distinct from `sort_with_permutation`: `np.argsort`/`mx.argsort` on
        complex input sorts lexicographically (real part, then imaginary
        part), not by magnitude — wrong for ranking e.g. `dft_full`
        coefficients by how much they contribute. `apply_inverse_permutation`
        still undoes this (it only ever needs the permutation, not the sort
        key), so no separate inverse is needed.
        """
        ...

    def matmul_shared(self, batch: Any, matrix: np.ndarray) -> Any:
        """`(B, N) @ matrix.T` — `matrix` is a small fixed `(N, N)` set once."""
        ...

    def sparsity_fraction(self, x: Any, *, eps: float) -> float:
        """Fraction of entries with `abs(value) > eps`, averaged over the batch."""
        ...

    def permutation_complexity(self, perm: Any) -> float:
        """Structural cost of describing an `argsort`-style permutation, averaged over the batch.

        Counts the positions where consecutive entries are *not* a
        contiguous ascending run (`perm[i+1] != perm[i] + 1`) — the number
        of contiguous ascending blocks a permutation decomposes into, minus
        one. The identity permutation (`[0, 1, ..., n-1]`, one run) costs
        `0`; a permutation made of `k` contiguous ascending blocks costs
        `k - 1`; a maximally scrambled permutation costs up to `n - 1`.

        Replaces charging every permutation as maximally complex regardless
        of structure: since permutation entries are the distinct integers
        `0..n-1`, the previous charge (`sparsity_fraction(perm, eps=0.5)`)
        counted every entry except value `0` as "significant," giving
        essentially the same ~`n-1` cost whether the permutation was the
        identity or a full scramble — it measured the *value range*, not
        the permutation's actual structure.
        """
        ...

    def max_abs_diff(self, a: Any, b: Any) -> float:
        """Scalar `max(abs(a - b))` roundtrip-error reduction."""
        ...

    def to_numpy(self, x: Any) -> np.ndarray:
        """Materialize a backend array as a NumPy array (host sync for MLX)."""
        ...


def _to_real(x: np.ndarray) -> np.ndarray:
    """Explicitly drop the imaginary part of complex input, real input passed through.

    A chain stage typed `carrier_kind="real_vector"` (e.g. `difference`,
    `polynomial_vandermonde`) may still receive complex-*dtyped* data at its
    decode boundary — `dft_full`'s decode (`ifft`) always returns complex
    dtype even when the underlying values are mathematically real, because
    `operator.py`'s diagonalization proof needs that decode to stay the
    genuine, unrestricted complex inverse-DFT matrix (see `dft_full`'s
    docstring). Rather than let `np.asarray(x, dtype=float)` implicitly warn
    and truncate at that boundary, do it explicitly here: any genuine
    roundtrip failure this would mask is still caught downstream by
    `certify_roundtrip`'s distance check against the *original* object, not
    hidden by it.
    """
    arr = np.asarray(x)
    return arr.real if np.iscomplexobj(arr) else arr


class NumpySearchBackend:
    name = "numpy"

    def rfft(self, x: np.ndarray) -> np.ndarray:
        return np.fft.rfft(np.asarray(x, dtype=float), axis=-1)

    def irfft(self, x: np.ndarray, n: int) -> np.ndarray:
        return np.fft.irfft(np.asarray(x), n=n, axis=-1)

    def fft(self, x: np.ndarray) -> np.ndarray:
        return np.fft.fft(np.asarray(x, dtype=complex), axis=-1)

    def ifft(self, x: np.ndarray) -> np.ndarray:
        return np.fft.ifft(np.asarray(x, dtype=complex), axis=-1)

    def diff_with_first(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.concatenate([x[..., :1], x[..., 1:] - x[..., :-1]], axis=-1)

    def cumsum(self, x: np.ndarray) -> np.ndarray:
        return np.cumsum(_to_real(x), axis=-1)

    def diff_with_first_complex(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return np.concatenate([x[..., :1], x[..., 1:] - x[..., :-1]], axis=-1)

    def cumsum_complex(self, x: np.ndarray) -> np.ndarray:
        return np.cumsum(np.asarray(x), axis=-1)

    def sort_with_permutation(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        perm = np.argsort(x, axis=-1)
        return np.take_along_axis(x, perm, axis=-1), perm

    def apply_inverse_permutation(self, values: np.ndarray, perm: np.ndarray) -> np.ndarray:
        inverse = np.argsort(perm, axis=-1)
        return np.take_along_axis(np.asarray(values), inverse, axis=-1)

    def sort_by_magnitude_with_permutation(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x)
        perm = np.argsort(-np.abs(x), axis=-1)
        return np.take_along_axis(x, perm, axis=-1), perm

    def matmul_shared(self, batch: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        return _to_real(batch) @ np.asarray(matrix, dtype=float).T

    def sparsity_fraction(self, x: np.ndarray, *, eps: float) -> float:
        arr = np.asarray(x)
        significant = np.abs(arr) > eps
        return float(np.mean(significant))

    def permutation_complexity(self, perm: np.ndarray) -> float:
        arr = np.asarray(perm)
        breaks = np.sum(arr[..., 1:] != arr[..., :-1] + 1, axis=-1)
        return float(np.mean(breaks))

    def max_abs_diff(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

    def to_numpy(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x)


class MlxSearchBackend:
    """Apple Silicon GPU path. Requires `rde.backends.mlx_usable()`."""

    name = "mlx"

    def __init__(self) -> None:
        if not mlx_usable():
            raise ImportError(
                "MLX search backend requires MLX with Metal GPU access "
                "(Apple Silicon, `pip install mlx`)."
            )
        import mlx.core as mx

        self._mx = mx

    def _as_mx(self, x: Any, *, dtype: Any = None) -> Any:
        """Coerce `x` to an `mx.array`, passing an already-mx array through unchanged.

        `isinstance(x, mx.array)`, not `hasattr(x, "dtype")` — a plain NumPy
        array also has `.dtype`, so that check would silently hand raw
        NumPy data to `mx.*` ops instead of converting it.
        """
        mx = self._mx
        if isinstance(x, mx.array):
            return x
        return mx.array(np.asarray(x, dtype=dtype))

    def rfft(self, x: Any) -> Any:
        mx = self._mx
        out = mx.fft.rfft(self._as_mx(x, dtype=np.float32), axis=-1)
        mx.eval(out)
        return out

    def irfft(self, x: Any, n: int) -> Any:
        mx = self._mx
        out = mx.fft.irfft(self._as_mx(x, dtype=np.complex64), n=n, axis=-1)
        mx.eval(out)
        return out

    def fft(self, x: Any) -> Any:
        mx = self._mx
        out = mx.fft.fft(self._as_mx(x, dtype=np.complex64), axis=-1)
        mx.eval(out)
        return out

    def ifft(self, x: Any) -> Any:
        mx = self._mx
        out = mx.fft.ifft(self._as_mx(x, dtype=np.complex64), axis=-1)
        mx.eval(out)
        return out

    def diff_with_first(self, x: Any) -> Any:
        mx = self._mx
        arr = self._as_mx(x, dtype=np.float32)
        out = mx.concatenate([arr[..., :1], arr[..., 1:] - arr[..., :-1]], axis=-1)
        mx.eval(out)
        return out

    def cumsum(self, x: Any) -> Any:
        mx = self._mx
        real_input = _to_real(self.to_numpy(x)) if isinstance(x, mx.array) else _to_real(x)
        out = mx.cumsum(self._as_mx(real_input, dtype=np.float32), axis=-1)
        mx.eval(out)
        return out

    def diff_with_first_complex(self, x: Any) -> Any:
        mx = self._mx
        dtype = np.complex64 if np.iscomplexobj(np.asarray(x)) else np.float32
        arr = self._as_mx(x, dtype=dtype)
        out = mx.concatenate([arr[..., :1], arr[..., 1:] - arr[..., :-1]], axis=-1)
        mx.eval(out)
        return out

    def cumsum_complex(self, x: Any) -> Any:
        mx = self._mx
        dtype = np.complex64 if np.iscomplexobj(np.asarray(x)) else np.float32
        out = mx.cumsum(self._as_mx(x, dtype=dtype), axis=-1)
        mx.eval(out)
        return out

    def sort_with_permutation(self, x: Any) -> tuple[Any, Any]:
        mx = self._mx
        arr = self._as_mx(x, dtype=np.float32)
        perm = mx.argsort(arr, axis=-1)
        sorted_values = mx.take_along_axis(arr, perm, axis=-1)
        mx.eval(sorted_values, perm)
        return sorted_values, perm

    def apply_inverse_permutation(self, values: Any, perm: Any) -> Any:
        mx = self._mx
        dtype = np.complex64 if np.iscomplexobj(np.asarray(values)) else np.float32
        values_mx = self._as_mx(values, dtype=dtype)
        perm_mx = self._as_mx(perm, dtype=np.int32)
        inverse = mx.argsort(perm_mx, axis=-1)
        out = mx.take_along_axis(values_mx, inverse, axis=-1)
        mx.eval(out)
        return out

    def sort_by_magnitude_with_permutation(self, x: Any) -> tuple[Any, Any]:
        mx = self._mx
        dtype = np.complex64 if np.iscomplexobj(np.asarray(x)) else np.float32
        arr = self._as_mx(x, dtype=dtype)
        perm = mx.argsort(-mx.abs(arr), axis=-1)
        sorted_values = mx.take_along_axis(arr, perm, axis=-1)
        mx.eval(sorted_values, perm)
        return sorted_values, perm

    def matmul_shared(self, batch: Any, matrix: np.ndarray) -> Any:
        mx = self._mx
        real_batch = _to_real(self.to_numpy(batch)) if isinstance(batch, mx.array) else _to_real(batch)
        arr = self._as_mx(real_batch, dtype=np.float32)
        mat = mx.array(np.asarray(matrix, dtype=np.float32))
        out = arr @ mat.T
        mx.eval(out)
        return out

    def sparsity_fraction(self, x: Any, *, eps: float) -> float:
        mx = self._mx
        dtype = np.complex64 if np.iscomplexobj(np.asarray(x)) else np.float32
        arr = self._as_mx(x, dtype=dtype)
        significant = (mx.abs(arr) > eps).astype(mx.float32)
        out = mx.mean(significant)
        mx.eval(out)
        return float(out)

    def permutation_complexity(self, perm: Any) -> float:
        mx = self._mx
        arr = self._as_mx(perm, dtype=np.int32)
        breaks = (arr[..., 1:] != arr[..., :-1] + 1).astype(mx.float32)
        out = mx.mean(mx.sum(breaks, axis=-1))
        mx.eval(out)
        return float(out)

    def max_abs_diff(self, a: Any, b: Any) -> float:
        mx = self._mx
        dtype = np.complex64 if np.iscomplexobj(np.asarray(a)) or np.iscomplexobj(np.asarray(b)) else np.float32
        arr_a = self._as_mx(a, dtype=dtype)
        arr_b = self._as_mx(b, dtype=dtype)
        out = mx.max(mx.abs(arr_a - arr_b))
        mx.eval(out)
        return float(out)

    def to_numpy(self, x: Any) -> np.ndarray:
        self._mx.eval(x)
        return np.asarray(x)


def get_array_backend(name: str | None = None) -> ArraySearchBackend:
    """Select a search backend: "numpy" (default), "mlx", or `None`/"auto".

    "auto" prefers MLX when `rde.backends.mlx_usable()` is true, else NumPy —
    the same policy `rde.backends.default_compute_backend()` uses elsewhere.
    """
    key = name or ("mlx" if mlx_usable() else "numpy")
    if key == "numpy":
        return NumpySearchBackend()
    if key == "mlx":
        return MlxSearchBackend()
    raise ValueError(f"Unknown array search backend: {key!r}")
