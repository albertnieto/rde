"""Layered representation synthesis: compose two typed stages into one new `Representation`.

Implements the theory in `docs/representation-synthesis-theory.md` (Phase 1
scope: depth-2 composition only). `search.py` documents why composing two
`grammar.py` primitives is mathematically vacuous — every primitive there
shares `object_type` as *both* its input and output domain (they all
round-trip through the same raw vector), so chaining two of them collapses
to using the second one directly. That stops being true the moment a
primitive's *carrier* differs from its input (`dft_full`'s carrier is
`C^n`, not `R^n`; `sorted_permutation`'s carrier is a `(values, permutation)`
pair, not a plain vector) — a second stage operating on *that* carrier sees
structure the flat, single-stage view cannot: verified numerically before
this module was written (not asserted) — `sorted_permutation+sorted_then_
difference` beats plain `sorted_permutation` in every tested scenario, with
an exact (`~1e-17`) roundtrip. It does **not** beat the permutation-free
`identity` baseline under the standard complexity metric (permutation
storage dominates) — see `docs/representation-synthesis-theory.md` §4 for
the honest full result and why that is a real, scoped limitation of
today's complexity metric, not of this composition mechanism.

Typing contract every `Representation` used here follows (see
`representation.py`): `encode: input_carrier_kind -> carrier_kind`,
`decode: carrier_kind -> input_carrier_kind`. A second stage only composes
onto a first stage when `stage2.input_carrier_kind == stage1.carrier_kind`
— `compose_layers` checks this and raises otherwise, rather than silently
composing incompatible carriers.

Four stage-2 primitives now cover every carrier kind `grammar.py` produces
that isn't already terminal: `"complex_vector"` (`sort_by_magnitude`),
`"sorted_pair"` (`sorted_then_difference`), `"sorted_complex_pair"`
(`sorted_complex_then_difference`), and `"matrix"` (`row_dft`) — not a
general "any two primitives" composer, still one hand-written stage-2 per
carrier kind. `dft`'s compact, non-square `"complex_vector"`-shaped-
differently carrier is unaffected — it shares the plain `"complex_vector"`
kind with `dft_full` and already composes with `sort_by_magnitude` (see
`docs/representation-synthesis-theory.md` §3).
"""

from __future__ import annotations

import math
from typing import Any, Callable

from rde.representation.array_backend import ArraySearchBackend, get_array_backend
from rde.representation.grammar import _sparsity_complexity, build_primitive_representations
from rde.representation.representation import Representation

StageBuilder = Callable[[int, str, ArraySearchBackend], Representation]


def _sort_by_magnitude_representation(n: int, object_type: str, backend: ArraySearchBackend) -> Representation:
    """`complex_vector -> sorted_complex_pair`: sort coefficients by `|value|` descending.

    Meaningful stage 2 on top of `dft_full` (or any `complex_vector`
    carrier): separates *which* components dominate (the permutation) from
    *how much* (the sorted magnitude sequence) — see the theory doc for why
    that split matters for signals whose dominant frequencies are not the
    low ones a plain `dft`/`dft_full` ranking would favor.
    """

    def _distance(a: Any, b: Any) -> float:
        return backend.max_abs_diff(a, b)

    def _complexity(encoded: tuple[Any, Any]) -> float:
        values, perm = encoded
        return _sparsity_complexity(backend, values, eps=1e-6) + backend.permutation_complexity(perm)

    return Representation(
        representation_id="sort_by_magnitude",
        object_type=object_type,
        carrier=f"C^{n} sorted by |value| + S_{n} permutation",
        encode=backend.sort_by_magnitude_with_permutation,
        decode=lambda encoded: backend.apply_inverse_permutation(encoded[0], encoded[1]),
        invariants=("order_preserving",),
        distance=_distance,
        complexity=_complexity,
        input_carrier_kind="complex_vector",
        carrier_kind="sorted_complex_pair",
    )


def _sorted_then_difference_representation(
    n: int, object_type: str, backend: ArraySearchBackend
) -> Representation:
    """`sorted_pair -> sorted_pair`: delta-encode already-sorted values, carrying the permutation through.

    Meaningful stage 2 on top of `sorted_permutation`: sorted data is
    monotonic, so consecutive differences are small within a cluster and
    large only at cluster boundaries — a classic delta-encoding win that a
    flat `difference` (on *unsorted* data) cannot see, since it has no
    reason for consecutive elements to be close in value.

    Decode reconstructs the *sorted* values (`cumsum`) but deliberately does
    **not** undo the permutation — that is `sorted_permutation.decode`'s
    job. Returning to this primitive's own `input_carrier_kind`
    (`"sorted_pair"`), not all the way to the raw object, is the contract
    `compose_layers` relies on; undoing the permutation here too would
    double-invert it once `stage1.decode` runs.
    """

    def _encode(pair: tuple[Any, Any]) -> tuple[Any, Any]:
        values, perm = pair
        return backend.diff_with_first(values), perm

    def _decode(pair: tuple[Any, Any]) -> tuple[Any, Any]:
        diffs, perm = pair
        return backend.cumsum(diffs), perm

    def _distance(a: Any, b: Any) -> float:
        return backend.max_abs_diff(a, b)

    def _complexity(encoded: tuple[Any, Any]) -> float:
        diffs, perm = encoded
        return _sparsity_complexity(backend, diffs, eps=1e-6) + backend.permutation_complexity(perm)

    return Representation(
        representation_id="sorted_then_difference",
        object_type=object_type,
        carrier=f"R^{n} sorted+differenced + S_{n} permutation",
        encode=_encode,
        decode=_decode,
        invariants=("order_preserving",),
        distance=_distance,
        complexity=_complexity,
        input_carrier_kind="sorted_pair",
        carrier_kind="sorted_pair",
    )


def _sorted_complex_then_difference_representation(
    n: int, object_type: str, backend: ArraySearchBackend
) -> Representation:
    """`sorted_complex_pair -> sorted_complex_pair`: delta-encode the magnitude-sorted complex values.

    Same mechanism as `sorted_then_difference`, one layer up: `sort_by_
    magnitude`'s values are genuinely complex (a DFT coefficient's phase,
    not a decode artifact), so this uses `diff_with_first_complex`/
    `cumsum_complex` rather than `diff_with_first`/`cumsum` — those discard
    the imaginary part by design (see their docstrings), which would
    silently corrupt this carrier's roundtrip instead of just failing to
    compress it.
    """

    def _encode(pair: tuple[Any, Any]) -> tuple[Any, Any]:
        values, perm = pair
        return backend.diff_with_first_complex(values), perm

    def _decode(pair: tuple[Any, Any]) -> tuple[Any, Any]:
        diffs, perm = pair
        return backend.cumsum_complex(diffs), perm

    def _distance(a: Any, b: Any) -> float:
        return backend.max_abs_diff(a, b)

    def _complexity(encoded: tuple[Any, Any]) -> float:
        diffs, perm = encoded
        return _sparsity_complexity(backend, diffs, eps=1e-6) + backend.permutation_complexity(perm)

    return Representation(
        representation_id="sorted_complex_then_difference",
        object_type=object_type,
        carrier=f"C^{n} sorted-by-magnitude+differenced + S_{n} permutation",
        encode=_encode,
        decode=_decode,
        invariants=("order_preserving",),
        distance=_distance,
        complexity=_complexity,
        input_carrier_kind="sorted_complex_pair",
        carrier_kind="sorted_complex_pair",
    )


def _row_dft_representation(n: int, object_type: str, backend: ArraySearchBackend) -> Representation:
    """`matrix -> complex_matrix`: full complex FFT along each row of `matrix_reshape`'s carrier.

    The natural stage 2 on `matrix_reshape`'s `(side, side)` carrier the
    theory doc's roadmap flagged as missing: a per-row DFT can expose
    row-periodic structure a flat 1-D view of the same `n` values cannot.
    `fft`/`ifft` already operate on the last axis regardless of an array's
    rank, so applying them to a `(B, side, side)` batch is a genuine
    per-row transform with no new backend kernel needed — only `matmul_
    shared`/`cumsum`/`diff_with_first` (the `real_vector`-typed stages)
    assume a flat last axis of length `n`, and none of those are used here.
    """
    side = int(round(math.sqrt(n)))

    return Representation(
        representation_id="row_dft",
        object_type=object_type,
        carrier=f"C^{{{side}x{side}}} (row-wise full FFT)",
        encode=backend.fft,
        decode=backend.ifft,
        exact=True,
        invariants=("linear",),
        distance=backend.max_abs_diff,
        complexity=lambda enc: _sparsity_complexity(backend, enc, eps=1e-6),
        input_carrier_kind="matrix",
        carrier_kind="complex_matrix",
    )


#: Stage-2 primitives, keyed by the `carrier_kind` they accept as input.
_STAGE2_BUILDERS: dict[str, dict[str, StageBuilder]] = {
    "complex_vector": {"sort_by_magnitude": _sort_by_magnitude_representation},
    "sorted_pair": {"sorted_then_difference": _sorted_then_difference_representation},
    "sorted_complex_pair": {"sorted_complex_then_difference": _sorted_complex_then_difference_representation},
    "matrix": {"row_dft": _row_dft_representation},
}


def stage2_primitive_names(carrier_kind: str) -> tuple[str, ...]:
    """Stage-2 primitive names available for a given `carrier_kind` (possibly empty)."""
    return tuple(_STAGE2_BUILDERS.get(carrier_kind, {}).keys())


def compose_layers(stage1: Representation, stage2: Representation) -> Representation:
    """Compose `stage2` onto `stage1`'s carrier into one `Representation` over `stage1.object_type`.

    Requires `stage2.input_carrier_kind == stage1.carrier_kind` — raises
    `ValueError` otherwise rather than composing incompatible carriers
    silently. The composed `distance` is `stage1.distance` (roundtrip is
    always checked against the original object, in `stage1`'s domain);
    the composed `complexity` is `stage2.complexity` (measured on the final,
    innermost carrier — where the compression claim actually lives).
    """
    if stage2.input_carrier_kind != stage1.carrier_kind:
        raise ValueError(
            f"cannot compose {stage2.representation_id!r} (expects "
            f"input_carrier_kind={stage2.input_carrier_kind!r}) onto "
            f"{stage1.representation_id!r} (carrier_kind={stage1.carrier_kind!r})"
        )

    def encode(x: Any) -> Any:
        return stage2.encode(stage1.encode(x))

    def decode(y: Any) -> Any:
        return stage1.decode(stage2.decode(y))

    return Representation(
        representation_id=f"{stage1.representation_id}+{stage2.representation_id}",
        object_type=stage1.object_type,
        carrier=f"{stage2.carrier} (via {stage1.representation_id})",
        encode=encode,
        decode=decode,
        exact=stage1.exact and stage2.exact,
        invariants=tuple(sorted(set(stage1.invariants) | set(stage2.invariants))),
        distance=stage1.distance,
        complexity=stage2.complexity,
        input_carrier_kind=stage1.input_carrier_kind,
        carrier_kind=stage2.carrier_kind,
    )


def build_layered_representations(
    n: int,
    *,
    object_type: str | None = None,
    backend: ArraySearchBackend | None = None,
) -> dict[str, Representation]:
    """Every valid depth-2 composition over `grammar.py`'s primitives, by composed id.

    Exhaustive, not searched: `grammar.py` has 7 primitives and (currently)
    2 stage-2 primitives, each accepting exactly one carrier kind, so there
    are at most a handful of valid pairs — enumerating all of them is
    already optimal, the same reasoning `search.py` gives for staying
    exhaustive at depth 1.
    """
    backend = backend or get_array_backend()
    object_type = object_type or f"numeric_batch_{n}"
    stage1_reps = build_primitive_representations(n, object_type=object_type, backend=backend)

    composed: dict[str, Representation] = {}
    for stage1 in stage1_reps.values():
        for stage2_name in stage2_primitive_names(stage1.carrier_kind):
            stage2 = _STAGE2_BUILDERS[stage1.carrier_kind][stage2_name](n, object_type, backend)
            layered = compose_layers(stage1, stage2)
            composed[layered.representation_id] = layered
    return composed
