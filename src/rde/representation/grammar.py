"""Representation grammar: a fixed set of composable batched primitives.

Phase 3 of the representation-discovery roadmap. Each primitive is a
`Representation` whose `encode`/`decode` operate on a whole batch `(B, N)`
in one backend call (see `array_backend.py`) — building a grammar entry
never loops over samples or coefficients.

The grammar is deliberately small and generic (identity, reshape, DFT, DCT,
sort+permutation, first-difference, polynomial-in-a-fixed-basis) — these are
textbook representations, not domain-specific features, matching the
core/domain boundary the rest of `rde` enforces. `search.py` composes and
ranks them; it does not invent new primitives at search time.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from rde.representation.array_backend import ArraySearchBackend, get_array_backend
from rde.representation.representation import Representation


def _elements_per_sample(shape: tuple[int, ...]) -> int:
    return int(np.prod(shape[1:])) if len(shape) > 1 else int(shape[0])


def _sparsity_complexity(backend: ArraySearchBackend, encoded: Any, *, eps: float) -> float:
    return backend.sparsity_fraction(encoded, eps=eps) * _elements_per_sample(tuple(encoded.shape))


def _identity_representation(n: int, object_type: str, backend: ArraySearchBackend) -> Representation:
    return Representation(
        representation_id="identity",
        object_type=object_type,
        carrier=f"R^{n}",
        encode=lambda x: x,
        decode=lambda x: x,
        invariants=("linear", "elementwise"),
        distance=backend.max_abs_diff,
        complexity=lambda enc: _sparsity_complexity(backend, enc, eps=1e-6),
    )


def _matrix_reshape_representation(n: int, object_type: str, backend: ArraySearchBackend) -> Representation:
    side = int(round(math.sqrt(n)))
    if side * side != n:
        raise ValueError(f"matrix_reshape requires a perfect-square N; got N={n}")
    return Representation(
        representation_id="matrix_reshape",
        object_type=object_type,
        carrier=f"R^{{{side}x{side}}}",
        encode=lambda x: x.reshape(x.shape[0], side, side),
        decode=lambda m: m.reshape(m.shape[0], n),
        invariants=("linear", "elementwise"),
        distance=backend.max_abs_diff,
        complexity=lambda enc: _sparsity_complexity(backend, enc, eps=1e-6),
        carrier_kind="matrix",
    )


def _dft_representation(n: int, object_type: str, backend: ArraySearchBackend) -> Representation:
    return Representation(
        representation_id="dft",
        object_type=object_type,
        carrier=f"C^{n // 2 + 1} (real FFT)",
        encode=backend.rfft,
        decode=lambda coeffs: backend.irfft(coeffs, n),
        exact=True,
        invariants=("linear", "shift_to_phase"),
        distance=backend.max_abs_diff,
        complexity=lambda enc: _sparsity_complexity(backend, enc, eps=1e-6),
        carrier_kind="complex_vector",
    )


def _dft_full_representation(n: int, object_type: str, backend: ArraySearchBackend) -> Representation:
    """Full complex FFT/IFFT — a square, exactly invertible linear map on `C^n`.

    Distinct from `dft` (`rfft`/`irfft`): that primitive exploits real-input
    Hermitian symmetry for compact storage but is not a plain square matrix,
    so `operator.py`'s transport (which needs `decode_matrix @ encode_matrix
    == I` as literal matrices) uses this primitive instead. See
    `operator.py` module docstring for the circulant-diagonalization result
    this primitive makes possible.
    """
    return Representation(
        representation_id="dft_full",
        object_type=object_type,
        carrier=f"C^{n} (full complex FFT)",
        encode=backend.fft,
        decode=backend.ifft,
        exact=True,
        invariants=("linear", "shift_to_phase", "diagonalizes_circulant", "complex_domain"),
        distance=backend.max_abs_diff,
        complexity=lambda enc: _sparsity_complexity(backend, enc, eps=1e-6),
        carrier_kind="complex_vector",
    )


def _difference_representation(n: int, object_type: str, backend: ArraySearchBackend) -> Representation:
    return Representation(
        representation_id="difference",
        object_type=object_type,
        carrier=f"R^{n} (first value + successive differences)",
        encode=backend.diff_with_first,
        decode=backend.cumsum,
        invariants=("linear",),
        distance=backend.max_abs_diff,
        complexity=lambda enc: _sparsity_complexity(backend, enc, eps=1e-6),
    )


def _sorted_permutation_representation(n: int, object_type: str, backend: ArraySearchBackend) -> Representation:
    def _distance(original: Any, reconstructed: Any) -> float:
        return backend.max_abs_diff(original, reconstructed)

    def _complexity(encoded: tuple[Any, Any]) -> float:
        values, perm = encoded
        return _sparsity_complexity(backend, values, eps=1e-6) + backend.permutation_complexity(perm)

    return Representation(
        representation_id="sorted_permutation",
        object_type=object_type,
        carrier=f"R^{n} sorted + S_{n} permutation",
        encode=backend.sort_with_permutation,
        decode=lambda encoded: backend.apply_inverse_permutation(encoded[0], encoded[1]),
        invariants=("order_preserving",),
        distance=_distance,
        complexity=_complexity,
        carrier_kind="sorted_pair",
    )


def _vandermonde_matrix(n: int) -> np.ndarray:
    nodes = np.arange(n, dtype=float)
    return np.vander(nodes, N=n, increasing=True)


def _polynomial_representation(n: int, object_type: str, backend: ArraySearchBackend) -> Representation:
    """Coefficients of the degree-`(n-1)` polynomial through fixed nodes `0..n-1`.

    `matrix` (Vandermonde) and its inverse are a tiny `(n, n)` one-time setup
    computed once here with NumPy/LAPACK — not part of the per-batch hot
    path, which is a single `matmul_shared` call regardless of backend.

    Equally-spaced-integer-node Vandermonde matrices are famously
    ill-conditioned (`cond(V) ~ 1e8` already at `n=8`, `~1e14` at `n=12`);
    this is a real numerical limit, not something this representation hides
    — `exact=True` is a claim `certify_roundtrip` checks, and it will
    correctly report `status="refuted"` once conditioning outgrows the
    caller's tolerance, rather than this primitive silently degrading.
    """
    matrix = _vandermonde_matrix(n)
    inverse = np.linalg.inv(matrix)
    return Representation(
        representation_id="polynomial_vandermonde",
        object_type=object_type,
        carrier=f"R[x]_{{<{n}}} evaluated at nodes 0..{n - 1}",
        encode=lambda x: backend.matmul_shared(x, inverse),
        decode=lambda coeffs: backend.matmul_shared(coeffs, matrix),
        invariants=("linear",),
        distance=backend.max_abs_diff,
        complexity=lambda enc: _sparsity_complexity(backend, enc, eps=1e-6),
    )


def _dct_matrix(n: int) -> np.ndarray:
    """The orthonormal DCT-II basis matrix (`scipy.fft.dct(..., type=2, norm="ortho")`'s matrix form).

    `C[k, m] = alpha_k * cos(pi/n * (m + 0.5) * k)`, `alpha_0 = sqrt(1/n)`,
    `alpha_k = sqrt(2/n)` for `k > 0` — verified to match `scipy.fft.dct`
    exactly before this primitive was written. `C` is orthogonal
    (`C @ C.T == I` to `~1e-15`), so `C.T` *is* the exact inverse — no
    `np.linalg.inv` needed, unlike `polynomial_vandermonde`'s famously
    ill-conditioned Vandermonde inverse (see that primitive's docstring).
    """
    k = np.arange(n, dtype=float)[:, None]
    m = np.arange(n, dtype=float)[None, :]
    basis = np.cos(np.pi / n * (m + 0.5) * k)
    alpha = np.full(n, np.sqrt(2.0 / n))
    alpha[0] = np.sqrt(1.0 / n)
    return basis * alpha[:, None]


def _dct_representation(n: int, object_type: str, backend: ArraySearchBackend) -> Representation:
    """The orthonormal type-II Discrete Cosine Transform — real-valued, exactly invertible.

    The concrete gap this closes: `dft`/`dft_full`'s basis functions are
    integer-frequency sinusoids, exactly periodic in the `n`-window; the
    DCT's basis functions are *half*-integer-frequency cosines, which are
    not. A signal built as a sparse combination of a few DCT basis vectors
    is therefore not sparse in the DFT domain (its energy leaks across many
    Fourier bins), and vice versa — the two bases expose genuinely
    different structure, not the same structure through two windows.

    An earlier draft of this docstring claimed a win on a smooth,
    non-periodic *ramp* using a loose `eps=1e-2` sparsity threshold — that
    claim quietly disappeared once checked against this grammar's actual
    `eps=1e-6` (decaying-but-nonzero ramp coefficients never cross that
    tighter bar for either basis), the same class of mistake §4 of
    `docs/representation-synthesis-theory.md` documents on purpose rather
    than quietly fixing. The real, `eps=1e-6`-honest result, checked before
    being written here: for `n=16` data built as an exact 2-of-16-sparse
    combination of DCT basis vectors, `dct` reaches complexity `2.0` versus
    `identity`'s `16.0`, `dft`'s (`rfft`) `7.0`, and `dft_full`'s `13.17`
    — an exact roundtrip, and a real, decisive, structural compression
    difference, not a tuning artifact
    (`tests/rde/representation/test_grammar.py::
    test_dct_beats_dft_and_identity_on_dct_sparse_data`). Unlike `dft_full`,
    stays entirely real-valued (no `"complex_domain"` tag needed) and, unlike
    `polynomial_vandermonde`, its inverse is exact and well-conditioned by
    construction (orthogonal matrix, `cond == 1`), not merely exact in
    principle.
    """
    matrix = _dct_matrix(n)
    return Representation(
        representation_id="dct",
        object_type=object_type,
        carrier=f"R^{n} (type-II DCT, orthonormal)",
        encode=lambda x: backend.matmul_shared(x, matrix),
        decode=lambda coeffs: backend.matmul_shared(coeffs, matrix.T),
        exact=True,
        invariants=("linear", "orthogonal"),
        distance=backend.max_abs_diff,
        complexity=lambda enc: _sparsity_complexity(backend, enc, eps=1e-6),
    )


_PRIMITIVE_BUILDERS = {
    "identity": _identity_representation,
    "matrix_reshape": _matrix_reshape_representation,
    "dft": _dft_representation,
    "dft_full": _dft_full_representation,
    "difference": _difference_representation,
    "sorted_permutation": _sorted_permutation_representation,
    "polynomial_vandermonde": _polynomial_representation,
    "dct": _dct_representation,
}


def primitive_names() -> tuple[str, ...]:
    return tuple(_PRIMITIVE_BUILDERS.keys())


def build_primitive_representations(
    n: int,
    *,
    object_type: str | None = None,
    backend: ArraySearchBackend | None = None,
    primitive_subset: Sequence[str] | None = None,
) -> dict[str, Representation]:
    """Build grammar primitives valid for batches of `N=n` vectors.

    `matrix_reshape` is omitted when `n` is not a perfect square rather than
    raising — callers get whatever of the grammar actually applies to `n`.

    `primitive_subset`, when given, restricts the result to exactly those
    names (raising on an unknown one) — this is what `holdout.py`'s
    visible/held-out ablation is built on; it is not a performance
    shortcut, it exists so a caller can genuinely withhold primitives.
    """
    backend = backend or get_array_backend()
    object_type = object_type or f"numeric_batch_{n}"
    names = _PRIMITIVE_BUILDERS.keys()
    if primitive_subset is not None:
        unknown = set(primitive_subset) - set(_PRIMITIVE_BUILDERS.keys())
        if unknown:
            raise ValueError(f"Unknown primitive_subset entries: {sorted(unknown)}")
        names = [name for name in _PRIMITIVE_BUILDERS if name in set(primitive_subset)]
    representations: dict[str, Representation] = {}
    for name in names:
        if name == "matrix_reshape":
            side = int(round(math.sqrt(n)))
            if side * side != n:
                continue
        representations[name] = _PRIMITIVE_BUILDERS[name](n, object_type, backend)
    return representations
