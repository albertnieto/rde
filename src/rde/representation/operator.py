"""Operator transport across representations (Phase 6).

Given a linear operator `U` (an `(n, n)` matrix acting on the object's raw
vector value) and a representation `R` whose encode/decode are themselves
linear, transport `U` into `R`'s carrier:

    U_R = E_R @ U @ D_R

where `E_R`/`D_R` are `R`'s encode/decode recovered as literal matrices
(`linear_probe_matrices`). `off_diagonal_energy(U_R)` then makes "this
representation diagonalizes this operator" a checkable number, not a claim.

The concrete, verifiable result this module reproduces: the `dft_full`
grammar primitive (full complex FFT) diagonalizes any circulant operator —
a textbook fact (circulant matrices share the DFT eigenbasis). Verified
numerically before this module was written (`off_diagonal_energy` ~1e-16
for `dft_full` vs. ~0.9+ for `identity`/`difference` on a random circulant
operator, `tests/rde/representation/test_operator.py`), not asserted.
`rank_by_diagonalization` reuses `grammar.py`'s actual primitives to find
this, rather than special-casing DFT — the same search machinery
(`grammar.build_primitive_representations`) `search.py` uses for roundtrip
ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rde.representation.array_backend import ArraySearchBackend, get_array_backend
from rde.representation.grammar import build_primitive_representations
from rde.representation.representation import Representation


def probe_encode_matrix(representation: Representation, n: int) -> np.ndarray:
    """Recover `representation.encode` as a literal `(m, n)` matrix.

    Requires `"linear"` in `representation.invariants` and a flat `(B, M)`
    carrier (rules out e.g. `matrix_reshape`, whose carrier is `(B, s, s)`,
    and `sorted_permutation`, which is nonlinear/value-dependent). `m` need
    not equal `n` — probing only `encode` (never `decode`) is sound for any
    flat linear map, square or not: `E @ x == encode(x)` follows directly
    from linearity applied to the standard basis, regardless of codomain
    dimension. This is what makes `equivalence_types.check_isometry` able to
    give a real (not vacuous) answer for `dft` (compact `rfft`, `m < n`),
    verified directly: `E @ x` was checked to equal `encode(x)` for random
    `x` before this function was written this way.

    `"complex_domain"` in `invariants` selects a complex identity probe
    (needed for `dft_full`, whose input is already complex); everything
    else probes with a real identity.
    """
    if "linear" not in representation.invariants:
        raise ValueError(
            f"{representation.representation_id!r} is not tagged 'linear'; "
            "encode/decode must be a literal matrix for this probe."
        )
    domain_probe_dtype = complex if "complex_domain" in representation.invariants else float
    domain_identity = np.eye(n, dtype=domain_probe_dtype)
    encoded_probe = np.asarray(representation.encode(domain_identity))
    if encoded_probe.ndim != 2:
        raise ValueError(
            f"{representation.representation_id!r} encodes to a non-flat carrier "
            f"(per-sample shape {encoded_probe.shape[1:]}); probe_encode_matrix "
            "only supports flat (B, M) carriers."
        )
    return encoded_probe.T


def linear_probe_matrices(representation: Representation, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Recover `representation`'s encode/decode as literal `(n, n)`/`(n, n)` matrices.

    Builds on `probe_encode_matrix`, then additionally requires a *square*
    carrier (`m == n`) before probing `decode` — decode-side probing is
    only sound when `decode`'s stated input space is exactly the encode
    matrix's codomain, which fails for `dft` (compact `rfft`/`irfft`):
    its codomain (`C^{n//2+1}`) is smaller than its domain (`R^n`), and
    `irfft`'s Hermitian-symmetry packing makes even a dtype-correct
    complex-identity probe of its decode side unsound in general
    (confirmed while building this module — probing `irfft` with a plain
    complex identity does not reproduce `decode` exactly; see `dft_full`'s
    docstring for why operator transport uses that primitive instead).
    `dft` therefore correctly raises here rather than silently returning a
    `decode_matrix` that looks shape-valid but is not mathematically sound
    — use `probe_encode_matrix` alone for `dft` where only `encode`'s
    linear structure is needed (e.g. isometry checking).

    The *codomain* probe dtype (for the decode side only) is read off the
    actual dtype `encode` returned, not off `"complex_domain"` — domain and
    codomain complexity can differ in principle, so deriving both from one
    tag would risk probing the codomain with the wrong dtype. No current
    grammar primitive is square with a domain/codomain dtype mismatch, but
    this keeps the derivation correct rather than merely untested.

    Probing is `encode`/`decode` applied once to an `n x n` identity batch
    each — an `O(n^2)` one-time setup per representation, not a per-sample
    loop in any hot path.
    """
    encode_matrix = probe_encode_matrix(representation, n)
    m = encode_matrix.shape[0]
    if m != n:
        raise ValueError(
            f"{representation.representation_id!r} carrier dimension {m} != domain "
            f"dimension {n}; linear_probe_matrices only probes the decode side of "
            "square (bijective-candidate) carriers — a non-square carrier cannot be "
            "a linear isomorphism, and probing its decode side is not generally sound."
        )

    codomain_probe_dtype = complex if np.iscomplexobj(encode_matrix) else float
    codomain_identity = np.eye(m, dtype=codomain_probe_dtype)
    decoded_probe = np.asarray(representation.decode(codomain_identity))
    if decoded_probe.shape != (m, n):
        raise ValueError(
            f"{representation.representation_id!r} decode did not return shape "
            f"({m}, {n}) for an ({m}, {m}) probe; not a literal (n, m) matrix."
        )
    decode_matrix = decoded_probe.T

    return encode_matrix, decode_matrix


def transport_operator(
    operator: np.ndarray, encode_matrix: np.ndarray, decode_matrix: np.ndarray
) -> np.ndarray:
    """`U_R = encode_matrix @ operator @ decode_matrix`.

    `operator` may be `(n, n)` or a batch `(B, n, n)` — NumPy's `@` matmul
    broadcasts the shared `(m, n)`/`(n, m)` matrices over the leading batch
    axis in one call, never a Python loop over the batch.
    """
    return encode_matrix @ operator @ decode_matrix


def off_diagonal_energy(matrix: np.ndarray) -> Any:
    """Relative Frobenius energy off the main diagonal; `0.0` == exactly diagonal.

    Supports an optional leading batch dimension (`(B, n, n)` -> `(B,)`)
    via axis-reduced `np.linalg.norm` calls — no loop over rows, columns,
    or batch elements.
    """
    m = np.asarray(matrix)
    size = m.shape[-1]
    eye_mask = np.eye(size, dtype=bool)
    off = np.where(eye_mask, 0, m)
    off_norm = np.linalg.norm(off, axis=(-2, -1))
    total_norm = np.linalg.norm(m, axis=(-2, -1))
    total_norm = np.where(total_norm == 0, 1.0, total_norm)
    return off_norm / total_norm


@dataclass(frozen=True)
class DiagonalizationCandidate:
    """One grammar primitive's transport of an operator, ranked by diagonality."""

    representation_id: str
    off_diagonal_energy: float
    transported_operator: np.ndarray


def rank_by_diagonalization(
    operator: np.ndarray,
    *,
    n: int,
    backend: ArraySearchBackend | None = None,
    object_type: str | None = None,
) -> list[DiagonalizationCandidate]:
    """Rank every linear, flat-carrier grammar primitive by how well it

    diagonalizes `operator` (most diagonal first). Iterates the grammar's
    (small, fixed) primitive set — control flow over algorithm choices, not
    a data loop; each candidate's transport is one vectorized matmul chain.
    """
    backend = backend or get_array_backend()
    grammar = build_primitive_representations(n, object_type=object_type, backend=backend)

    candidates: list[DiagonalizationCandidate] = []
    for representation_id, representation in grammar.items():
        if "linear" not in representation.invariants:
            continue
        try:
            encode_matrix, decode_matrix = linear_probe_matrices(representation, n)
        except ValueError:
            continue
        transported = transport_operator(operator, encode_matrix, decode_matrix)
        energy = float(off_diagonal_energy(transported))
        candidates.append(
            DiagonalizationCandidate(
                representation_id=representation_id,
                off_diagonal_energy=energy,
                transported_operator=transported,
            )
        )

    candidates.sort(key=lambda c: c.off_diagonal_energy)
    return candidates
