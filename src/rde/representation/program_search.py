"""Exhaustive typed-chain program search: depth-K composition with mandatory generalization checking.

Gap closure toward "novel primitive invention" (see
`docs/representation-synthesis-theory.md`). The honest scope, agreed before
writing this: not "search for primitives nobody wrote" (open-ended
research, no finish line), but "search *compositions* of the primitives
`grammar.py` and `layered.py` already ship, deeper than `layered.py`'s
fixed depth-2" — `layered.py`'s `compose_layers` generalizes cleanly to a
chain by repeated application, since a composed `Representation` is a
plain `Representation` (same `input_carrier_kind`/`carrier_kind` typing),
so nothing new has to be invented to go from depth 2 to depth K.

**Exhaustive DFS, not genetic programming.** Checked before building this:
the registry's carrier-kind compatibility graph is small and sparse
(`"sorted_pair"` and `"sorted_complex_pair"` each have exactly one
continuation, and both happen to accept their own output, so each can chain
onto itself; `"matrix"` has exactly one, `row_dft`, whose own output carrier
`"complex_matrix"` is a dead end in turn). Exhaustive enumeration of that
graph up to `max_depth` is already optimal — the same reasoning `search.py`
gives for staying exhaustive at depth 1. A genetic algorithm (population,
mutation, crossover) over a search space this constrained would be
complexity for its own sake, not a real capability; add one if the registry
grows enough that exhaustive stops being cheap, not before.

**This already finds something depth-2 can't.** Verified numerically before
being written into a test: `sorted_permutation` -> `sorted_then_difference`
-> `sorted_then_difference` (second-order differencing of sorted,
piecewise-linear data) reaches complexity `18.0`, beating the depth-2
chain's `29.0` — a real, depth-3-only result, not a reproduction of what
`layered.py` already knew (`tests/rde/representation/test_program_search.py`).
`"matrix"` and `"sorted_complex_pair"` are no longer dead ends —
`layered.py`'s `row_dft` and `sorted_complex_then_difference` give both a
registered continuation (see the theory doc's §9 for the verified numeric
wins each finds).

**Mandatory holdout generalization checking.** Unconstrained search over
compositions is exactly the setup that finds spurious overfit programs — a
chain that looks compressive on the one batch it was measured against by
coincidence, not structure. `search_chains` requires an independent
`holdout_batch` (not an optional keyword) and ranks by *holdout* complexity,
never train complexity — the same design choice `holdout.py`'s
`leakage_ratio` already made once: report the number, do not silently trust
the training-set score.

`search_chains`'s verify-on-train/drop/verify-on-holdout/drop/rank-by-holdout
shape is implemented via `rde.search.holdout_search.search_with_holdout` —
`rde.recovery.search_space.search_recovery_chains` independently converged
on the identical shape for a completely different candidate type
(`RecoveryProtocol` chains, not `Representation` chains), which is why that
shape now lives in `rde.search` rather than being reimplemented per module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from rde.representation.array_backend import ArraySearchBackend, get_array_backend
from rde.representation.certificate import Certificate, certify_roundtrip
from rde.representation.grammar import build_primitive_representations
from rde.representation.grammar import primitive_names as _grammar_primitive_names
from rde.representation.layered import _STAGE2_BUILDERS, compose_layers
from rde.representation.representation import Representation
from rde.search import VerifyResult, search_with_holdout


def atomic_registry(
    n: int,
    *,
    object_type: str | None = None,
    backend: ArraySearchBackend | None = None,
    primitive_subset: Sequence[str] | None = None,
) -> dict[str, Representation]:
    """Every single-stage typed primitive chain search can use: `grammar.py`'s plus `layered.py`'s.

    Each stage-2 builder is instantiated standalone here (not yet composed
    onto anything) — its `input_carrier_kind` still restricts where it can
    attach, `enumerate_chains` is what actually respects that.

    `primitive_subset`, when given, restricts both `grammar.py`'s stage-1
    primitives (via `build_primitive_representations`) and `layered.py`'s
    stage-2 ones (filtered by name here) to exactly those named — the same
    genuine-withholding semantics `search.py`'s `rank_representations` gives
    `holdout.py`, extended to chains: a stage-2 primitive named in
    `primitive_subset` is still withheld even though it can only ever
    appear past a chain's first stage.
    """
    backend = backend or get_array_backend()
    object_type = object_type or f"numeric_batch_{n}"
    stage2_names = {name for builders in _STAGE2_BUILDERS.values() for name in builders}
    if primitive_subset is not None:
        known = set(_grammar_primitive_names()) | stage2_names
        unknown = set(primitive_subset) - known
        if unknown:
            raise ValueError(f"Unknown primitive_subset entries: {sorted(unknown)}")
    grammar_subset = (
        [name for name in primitive_subset if name in set(_grammar_primitive_names())]
        if primitive_subset is not None
        else None
    )
    registry = dict(
        build_primitive_representations(
            n, object_type=object_type, backend=backend, primitive_subset=grammar_subset
        )
    )
    allowed = set(primitive_subset) if primitive_subset is not None else None
    for builders in _STAGE2_BUILDERS.values():
        for name, builder in builders.items():
            if allowed is not None and name not in allowed:
                continue
            registry[name] = builder(n, object_type, backend)
    return registry


def enumerate_chains(
    n: int,
    *,
    max_depth: int = 4,
    object_type: str | None = None,
    backend: ArraySearchBackend | None = None,
    primitive_subset: Sequence[str] | None = None,
) -> dict[str, Representation]:
    """Every valid typed chain up to `max_depth`, keyed by its composed `representation_id`.

    A chain may reuse the same atomic primitive more than once (e.g.
    `sorted_then_difference` chained onto itself for second-order
    differencing) — `max_depth` alone bounds recursion, so there is no
    separate "no repeats" guard to accidentally rule out a legitimate
    deeper composition. `starts` (primitives usable as a chain's first
    stage) are exactly those with `input_carrier_kind == "real_vector"` —
    every `grammar.py` primitive, never a `layered.py` stage-2 one, since
    those only accept an already-transformed carrier.

    `identity` is excluded from `continuations` (never appended as a
    non-first stage) — its `encode`/`decode` are both the identity
    function, so composing it anywhere past the first stage can never
    change a chain's observable behavior, only pad its `representation_id`
    with a structurally distinct duplicate of a shallower chain already in
    this dict (e.g. `identity+identity+identity+matrix_reshape` next to
    plain `matrix_reshape`). It remains a valid `start` on its own — the
    one legitimate `real_vector -> real_vector` no-op chain.

    `primitive_subset`, when given, is forwarded to `atomic_registry` —
    withholding a primitive here means no chain, at any depth, can reach it
    as any stage, not just as a `start` (the same genuine-withholding
    guarantee `holdout.py` relies on for the flat grammar, extended to
    chains).
    """
    backend = backend or get_array_backend()
    object_type = object_type or f"numeric_batch_{n}"
    registry = atomic_registry(
        n, object_type=object_type, backend=backend, primitive_subset=primitive_subset
    )

    starts = [rep for rep in registry.values() if rep.input_carrier_kind == "real_vector"]
    continuations = [rep for rep in registry.values() if rep.representation_id != "identity"]
    by_input_kind: dict[str, list[Representation]] = {}
    for rep in continuations:
        by_input_kind.setdefault(rep.input_carrier_kind, []).append(rep)

    chains: dict[str, Representation] = {}

    def _extend(current: Representation, depth: int) -> None:
        chains[current.representation_id] = current
        if depth >= max_depth:
            return
        for continuation in by_input_kind.get(current.carrier_kind, ()):
            composed = compose_layers(current, continuation)
            _extend(composed, depth + 1)

    for start in starts:
        _extend(start, 1)

    return chains


@dataclass(frozen=True)
class ChainSearchResult:
    """One enumerated chain, certified and scored on both `train` and `holdout` data."""

    representation_id: str
    representation: Representation
    depth: int
    train_certificate: Certificate
    holdout_certificate: Certificate
    train_complexity: float
    holdout_complexity: float
    generalization_ratio: float


def search_chains(
    train_batch,
    holdout_batch,
    *,
    n: int,
    max_depth: int = 4,
    object_type: str | None = None,
    backend: ArraySearchBackend | None = None,
    tolerance: float = 1e-6,
    primitive_subset: Sequence[str] | None = None,
) -> list[ChainSearchResult]:
    """Enumerate chains, then certify and score each on *independent* train/holdout batches.

    `holdout_batch` is required — pass a second, independently drawn batch
    from the same distribution as `train_batch`, not a copy or a subset of
    it (that would silently defeat the check this function exists for).

    A chain is dropped, not just penalized, if it fails to verify (exact
    roundtrip) on *either* batch — a chain that only round-trips on the
    data it happened to be checked against is not a valid representation,
    holdout complexity aside. Surviving chains are ranked by
    `holdout_complexity`, never `train_complexity` — `generalization_ratio`
    (`holdout / train`) is reported alongside so a caller can see *how much*
    a chain's apparent compression held up, not just a pass/fail cutoff
    this function does not get to invent.
    """
    backend = backend or get_array_backend()
    chains = enumerate_chains(
        n,
        max_depth=max_depth,
        object_type=object_type,
        backend=backend,
        primitive_subset=primitive_subset,
    )

    def _verify(
        candidates: Sequence[Representation], _domain: Any, batch: Any
    ) -> dict[str, VerifyResult]:
        out: dict[str, VerifyResult] = {}
        for representation in candidates:
            certificate = certify_roundtrip(representation, batch, tolerance=tolerance)
            ok = certificate.status == "verified"
            objective = _complexity_or_inf(representation, batch) if ok else float("inf")
            out[representation.representation_id] = VerifyResult(
                ok=ok, objective=objective, detail=certificate
            )
        return out

    search_results = search_with_holdout(
        list(chains.values()),
        train_batch,
        holdout_batch,
        verify=_verify,
        candidate_id=lambda r: r.representation_id,
    )

    results: list[ChainSearchResult] = []
    for item in search_results:
        train_complexity = item.train.objective
        holdout_complexity = item.holdout.objective
        if train_complexity > 0:
            ratio = holdout_complexity / train_complexity
        else:
            ratio = 1.0 if holdout_complexity == 0 else float("inf")

        results.append(
            ChainSearchResult(
                representation_id=item.candidate_id,
                representation=item.candidate,
                depth=item.candidate_id.count("+") + 1,
                train_certificate=item.train.detail,
                holdout_certificate=item.holdout.detail,
                train_complexity=train_complexity,
                holdout_complexity=holdout_complexity,
                generalization_ratio=ratio,
            )
        )
    return results


def _complexity_or_inf(representation: Representation, batch) -> float:
    if representation.complexity is None:
        return float("inf")
    return float(representation.complexity(representation.encode(batch)))
