"""A small, checkable structure vocabulary (gap closure toward "Structure Language").

The original proposal's Structure Language names a long generic vocabulary:
symmetry, sparsity, locality, separability, periodicity, low_rank,
factorization, conservation, duality, compositionality, degeneracy,
topology, spectral_structure, correlation, entanglement. Before this
module, only two of those existed at all in this package — informal
string tags on `Representation.invariants`, and one hardcoded parity check
(`symbolic.discover_parity_claim`) — with no generic, reusable checker.

This module makes four of those notions genuinely checkable against real
batch data, each independently verified numerically before being written
here (not asserted):

- `check_sparsity` — reuses `array_backend.sparsity_fraction`.
- `check_periodicity` — spectral energy concentration in the (real) DFT
  magnitude spectrum; a pure sinusoid concentrates ~100% of its non-DC
  energy in one frequency, a random signal spreads it out (verified: 1.0
  vs. ~0.37 for `top_k=1` on `n=16`).
- `check_low_rank` / `check_separability` — singular-value decay of a
  `matrix_reshape`-encoded batch; an outer-product (`g(x)h(y)`, exactly
  separable) matrix has one nonzero normalized singular value, a random
  matrix's decay slowly (verified: `[1, ~0, ~0, ~0, ~0]` vs.
  `[1, 0.72, 0.54, 0.27, 0.03]` on random `5x5` examples).

Two more are now checkable, each against a real conserved/dual pair rather
than a fabricated one — see each function's docstring:

- `check_conservation` — is a matrix batch invariant under simultaneous
  row/column permutation by every element of a group action? Generalizes
  `rde_domains.tsp.circulant.circulant_deviation`'s cyclic-shift projection
  (the domain-side function this was built to be verified against, kept out
  of core per the domain-plugin boundary) to an arbitrary permutation group,
  defaulting to the cyclic group `Z_n` — the concrete case a genuinely
  circulant TSP distance matrix instantiates: `D` is exactly conserved
  (unchanged) under simultaneous cyclic row/column shift.
- `check_duality` — does a named grammar representation diagonalize an
  operator, i.e. is its carrier the operator's eigenbasis? Reuses
  `operator.py`'s already-verified machinery (`linear_probe_matrices`,
  `transport_operator`, `off_diagonal_energy`) for one specific claim
  ("is `dual_representation_id` dual to this operator") instead of
  `rank_by_diagonalization`'s full-grammar ranking. The concrete instance:
  `dft_full` is dual to any circulant operator (textbook fact, already
  verified in `operator.py` against a random circulant test matrix); this
  makes the same check reusable against real data, e.g. a genuinely
  circulant TSP distance matrix.

The remaining vocabulary (compositionality, degeneracy, topology,
correlation, entanglement) is still NOT implemented — this package has no
compositional, topological, or entangled representation to check any of
those against. Declaring them here with nothing to instantiate would be
exactly the unaudited placeholder this project's methodology forbids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from rde.representation.array_backend import ArraySearchBackend, get_array_backend
from rde.representation.grammar import build_primitive_representations
from rde.representation.operator import linear_probe_matrices, off_diagonal_energy, transport_operator


@dataclass(frozen=True)
class StructureClaim:
    """One structural claim, checked against real data — not a taxonomy entry."""

    structure_type: str
    holds: bool
    score: float
    detail: str


def check_sparsity(
    encoded: Any, backend: ArraySearchBackend, *, eps: float = 1e-6, max_fraction: float = 0.5
) -> StructureClaim:
    """Does fewer than `max_fraction` of `encoded`'s entries exceed `eps` in magnitude?"""
    fraction_significant = backend.sparsity_fraction(encoded, eps=eps)
    return StructureClaim(
        structure_type="sparsity",
        holds=fraction_significant <= max_fraction,
        score=float(fraction_significant),
        detail=(
            f"fraction of entries with |value| > {eps:.1e}: {fraction_significant:.3f} "
            f"(threshold {max_fraction:.3f})"
        ),
    )


def check_periodicity(
    values_batch: np.ndarray, *, top_k: int = 1, energy_threshold: float = 0.9
) -> StructureClaim:
    """Do the top `top_k` non-DC Fourier magnitudes capture `>= energy_threshold` of AC energy?

    Computed with a plain `np.fft.rfft` (not `array_backend`/`grammar`'s
    primitives) — this is a diagnostic over raw sample values, independent
    of which representation a caller might separately choose to encode
    them with.
    """
    values = np.asarray(values_batch, dtype=float)
    coefficients = np.fft.rfft(values, axis=-1)
    ac_energy = np.abs(coefficients[..., 1:]) ** 2  # exclude the DC term
    total_energy = np.sum(ac_energy, axis=-1)
    total_energy = np.where(total_energy == 0, 1.0, total_energy)
    top_energy = np.sort(ac_energy, axis=-1)[..., ::-1][..., :top_k].sum(axis=-1)
    concentration = float(np.mean(top_energy / total_energy))
    return StructureClaim(
        structure_type="periodicity",
        holds=concentration >= energy_threshold,
        score=concentration,
        detail=(
            f"top-{top_k} AC-frequency energy fraction (batch mean): {concentration:.3f} "
            f"(threshold {energy_threshold:.3f})"
        ),
    )


def _normalized_singular_value_profile(matrix_batch: np.ndarray) -> np.ndarray:
    """Batch-mean singular values, each normalized by its own matrix's largest."""
    matrices = np.asarray(matrix_batch, dtype=float)
    singular_values = np.linalg.svd(matrices, compute_uv=False)  # (B, k)
    largest = singular_values[..., :1]
    normalized = singular_values / np.where(largest == 0, 1.0, largest)
    return np.mean(normalized, axis=0)


def check_low_rank(
    matrix_batch: np.ndarray, *, rank_threshold: float = 0.1, max_effective_rank: int | None = None
) -> StructureClaim:
    """Effective rank: is the count of normalized singular values `> rank_threshold`

    at most `max_effective_rank` (default `full_rank // 2`, floored at 1)?

    `effective_rank < full_rank` alone is *not* a meaningful "low rank"
    claim — a matrix with 4 of 5 singular values significant still
    trivially satisfies that (found while testing this function: a random
    `5x5` matrix's profile `[1, 0.72, 0.54, 0.27, 0.03]` gives
    `effective_rank=4`, which is not low rank in any useful sense). Requiring
    the effective rank to be at most half the full rank is the actual bar.

    `matrix_batch` is expected to already be 2-D-per-sample (e.g.
    `grammar.matrix_reshape`'s encode output), not a flat vector.
    """
    profile = _normalized_singular_value_profile(matrix_batch)
    effective_rank = int(np.sum(profile > rank_threshold))
    full_rank = profile.shape[0]
    limit = max_effective_rank if max_effective_rank is not None else max(1, full_rank // 2)
    return StructureClaim(
        structure_type="low_rank",
        holds=effective_rank <= limit,
        score=float(effective_rank),
        detail=(
            f"effective rank {effective_rank} / full rank {full_rank}, "
            f"significant if <= {limit} (singular values > {rank_threshold:.2f}x the largest)"
        ),
    )


def check_separability(matrix_batch: np.ndarray, *, rank_threshold: float = 0.1) -> StructureClaim:
    """`F(x, y) = g(x) h(y)`: is the matrix carrier (approximately) rank 1?

    The `rank == 1` special case of `check_low_rank`, named separately
    because "separable" is the structurally meaningful claim (a product of
    two independent factors), not merely "compressible".
    """
    profile = _normalized_singular_value_profile(matrix_batch)
    second_largest = float(profile[1]) if profile.shape[0] > 1 else 0.0
    return StructureClaim(
        structure_type="separability",
        holds=second_largest <= rank_threshold,
        score=second_largest,
        detail=f"second singular value / largest = {second_largest:.3e} (threshold {rank_threshold:.2f})",
    )


def _cyclic_group_permutations(n: int) -> list[np.ndarray]:
    """The full cyclic group `Z_n`'s action on `{0, ..., n-1}`: all `n` rotations."""
    idx = np.arange(n)
    return [np.roll(idx, -s) for s in range(n)]


def _group_orbit_average(matrix_batch: np.ndarray, permutations: Sequence[np.ndarray]) -> np.ndarray:
    """Average of `M[p][:, p]` over every `p` in `permutations` — the projection onto the
    subspace of matrices exactly invariant under this group action (generalizes
    `rde_domains.tsp.circulant._circulant_projection`'s cyclic-shift-only version to an
    arbitrary permutation group).
    """
    acc = np.zeros_like(matrix_batch, dtype=float)
    for p in permutations:
        acc += matrix_batch[..., p, :][..., :, p]
    return acc / len(permutations)


def check_conservation(
    matrix_batch: np.ndarray,
    *,
    permutations: Sequence[np.ndarray] | None = None,
    deviation_threshold: float = 0.05,
) -> StructureClaim:
    """Is `matrix_batch` (approximately) invariant under simultaneous row/column
    permutation by every element of a group action -- a conserved quantity in the
    same sense a physical quantity is conserved under a symmetry's action, applied
    to "this matrix, under the group generating its structure" rather than to a
    scalar.

    `permutations` is a sequence of length-`n` index arrays; each `p` is applied as
    `M[p][:, p]` (simultaneous row/column permutation). Defaults to the full cyclic
    group `Z_n` -- the concrete case this was built to be checkable against: a
    genuinely circulant matrix is *exactly* unchanged by simultaneous cyclic
    row/column shift (`M[p][:, p] == M` for every rotation `p`), the same invariant
    `rde_domains.tsp.circulant.circulant_deviation` measures for real TSP distance
    matrices (`deviation_threshold=0.0` and the default cyclic-group permutations
    reduce this function to exactly that formula -- verified in
    `tests/rde_domains/test_tsp_circulant_structure.py` against real domain-generated
    `D` matrices, both the exactly-circulant case (`holds=True`, score `~0`) and the
    symmetry-broken case (`holds=False`, score growing with the perturbation), not
    fabricated data). Left `None`, works for *any* supplied permutation group, not
    only cyclic shifts, since "conserved under a group action" is the actual
    structural claim, not shift-invariance specifically.
    """
    matrices = np.asarray(matrix_batch, dtype=float)
    n = matrices.shape[-1]
    perms = list(permutations) if permutations is not None else _cyclic_group_permutations(n)
    projection = _group_orbit_average(matrices, perms)
    residual_norm = np.linalg.norm(matrices - projection, axis=(-2, -1))
    total_norm = np.linalg.norm(matrices, axis=(-2, -1))
    total_norm = np.where(total_norm == 0, 1.0, total_norm)
    deviation = float(np.mean(residual_norm / total_norm))
    return StructureClaim(
        structure_type="conservation",
        holds=deviation <= deviation_threshold,
        score=deviation,
        detail=(
            f"relative deviation from the {len(perms)}-element group's invariant "
            f"subspace: {deviation:.3e} (threshold {deviation_threshold:.3e})"
        ),
    )


def check_duality(
    operator_batch: np.ndarray,
    *,
    n: int,
    dual_representation_id: str = "dft_full",
    backend: ArraySearchBackend | None = None,
    object_type: str | None = None,
    energy_threshold: float = 0.05,
) -> StructureClaim:
    """Does `dual_representation_id` (default `dft_full`, the full complex FFT)
    diagonalize `operator_batch` -- i.e. is its carrier the operator's eigenbasis?

    The textbook duality this package can check: circulant matrices are exactly
    diagonalized by the Fourier basis, already verified numerically in
    `operator.py` against a random circulant test matrix
    (`off_diagonal_energy` ~1e-16 for `dft_full`). This function makes that same
    check reusable against any operator batch — including real near-circulant
    data, e.g. `rde_domains.tsp.circulant`'s genuinely circulant TSP distance
    matrices (verified in `tests/rde_domains/test_tsp_circulant_structure.py`,
    not fabricated).

    Reuses `operator.py`'s probing/transport machinery for one specific,
    named claim ("is `dual_representation_id` dual to this operator") rather
    than `rank_by_diagonalization`'s full-grammar ranking — this function
    answers "does the claimed dual pair actually hold", not "search for one".
    """
    backend = backend or get_array_backend()
    grammar = build_primitive_representations(
        n, object_type=object_type, backend=backend, primitive_subset=[dual_representation_id]
    )
    representation = grammar[dual_representation_id]
    encode_matrix, decode_matrix = linear_probe_matrices(representation, n)
    transported = transport_operator(operator_batch, encode_matrix, decode_matrix)
    energy = float(np.mean(off_diagonal_energy(transported)))
    return StructureClaim(
        structure_type="duality",
        holds=energy <= energy_threshold,
        score=energy,
        detail=(
            f"off-diagonal energy of {dual_representation_id!r}'s transport of the "
            f"operator: {energy:.3e} (threshold {energy_threshold:.3e})"
        ),
    )
