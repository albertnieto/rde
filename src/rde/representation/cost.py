"""Conversion cost: an honest, asymptotic operation-count estimate per primitive.

Gap closure: `Transformation.cost` (Phase 2, the graph) and the
`(complexity, error)` Pareto objectives (Phase 5) used to live in
disconnected spaces — nothing connected "how simple does this
representation look" to "how expensive is it to get there." This module
estimates the latter for every grammar primitive from its actual
algorithm's operation count (FFT is `O(n log n)`, a dense matmul is
`O(n^2)`, a view/no-op is `O(1)`) — it is an asymptotic estimate with a
stated constant, not a measured wall-clock benchmark. Read
`computational_cost(...)` as "this many arithmetic operations, order of
magnitude," never as "this many nanoseconds."
"""

from __future__ import annotations

import math

_KNOWN_PRIMITIVES = (
    "identity",
    "matrix_reshape",
    "difference",
    "sorted_permutation",
    "dft",
    "dft_full",
    "polynomial_vandermonde",
    "dct",
    # layered.py stage-2 primitives (see docstring for each estimate's basis)
    "sort_by_magnitude",
    "sorted_then_difference",
    "sorted_complex_then_difference",
    "row_dft",
)


def _stage_cost(representation_id: str, n: int) -> float:
    """Cost of one atomic (non-composed) primitive at size `n`.

    - `identity`, `matrix_reshape`: `0` — pure views, no arithmetic.
    - `difference`, `sorted_then_difference`: `~2n` — one subtract per
      element (encode), one add per element via `cumsum` (decode).
    - `sorted_complex_then_difference`: `~4n` — same shape as `difference`,
      but each add/subtract is a complex operation (2 real flops instead
      of 1), so double the real-valued estimate.
    - `sorted_permutation`, `sort_by_magnitude`: `~n*log2(n) + 2n` — a
      comparison/magnitude sort (encode) plus `O(n)` gather/scatter each way.
    - `dft`: `~2 * 5*n*log2(n)` — the standard "5 real flops per real-FFT
      butterfly" estimate (Van Loan), once for `rfft` (encode), once for
      `irfft` (decode).
    - `dft_full`: `~4x` `dft`'s estimate — a complex multiply costs ~4 real
      multiplies + 2 real adds vs. a real-only butterfly, so the full
      complex FFT costs roughly 4x the real FFT for the same `n`.
    - `row_dft`: `side` independent length-`side` full complex FFTs
      (`side = sqrt(n)`), each way — `side` times `dft_full`'s per-row
      estimate at length `side`, i.e. `dft_full`'s formula with `n`
      replaced by `side` and the whole thing scaled by `side` rows.
    - `polynomial_vandermonde`, `dct`: `~2*n^2` — one dense `(n, n)` matvec
      each way (this implementation's actual algorithm: a literal
      orthonormal-basis matmul via `matmul_shared`, not the `O(n log n)`
      fast-DCT algorithm textbooks describe — the honest cost of the code
      that actually runs, not of the theoretically-best algorithm for the
      same transform).
    """
    if representation_id in ("identity", "matrix_reshape"):
        return 0.0
    if representation_id in ("difference", "sorted_then_difference"):
        return 2.0 * n
    if representation_id == "sorted_complex_then_difference":
        return 4.0 * n
    log2n = math.log2(n) if n > 1 else 1.0
    if representation_id in ("sorted_permutation", "sort_by_magnitude"):
        return n * log2n + 2.0 * n
    if representation_id == "dft":
        return 2.0 * (5.0 * n * log2n)
    if representation_id == "dft_full":
        return 4.0 * (2.0 * (5.0 * n * log2n))
    if representation_id == "row_dft":
        side = int(round(math.sqrt(n)))
        log2side = math.log2(side) if side > 1 else 1.0
        return side * (4.0 * (2.0 * (5.0 * side * log2side)))
    return 2.0 * n * n  # polynomial_vandermonde, dct


def computational_cost(representation_id: str, n: int) -> float:
    """Total (encode + decode) arithmetic operation count for one representation at size `n`.

    A composed chain id (`"stage1+stage2+..."`, see `layered.py`/
    `program_search.py`) is the sum of each stage's own cost — each stage's
    encode/decode genuinely runs in sequence, so the total operation count
    genuinely is additive, not a new estimate needing its own model.

    Raises `KeyError` for any stage name outside `grammar.py`'s or
    `layered.py`'s primitive set — there is no cost model for a
    representation this package didn't build, and guessing one would be
    exactly the kind of unaudited placeholder this module exists to avoid.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    stages = representation_id.split("+")
    unknown = [s for s in stages if s not in _KNOWN_PRIMITIVES]
    if unknown:
        raise KeyError(
            f"No conversion-cost model for stage(s) {unknown!r} of representation_id="
            f"{representation_id!r}; computational_cost only estimates cost for "
            "rde.representation.grammar's and layered.py's own primitives."
        )
    return sum(_stage_cost(stage, n) for stage in stages)
