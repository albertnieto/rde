"""Toy end-to-end demonstration: rediscover `f(x) = x + offset` via
`rde.substrate`'s universal execution substrate, plugged into
`rde.search.holdout_search`.

Core-only (no `rde_domains` import), following the same pattern as
`synthetic_poly.py`/`block_separable.py`: a minimal target a reader can
verify by inspection, used to prove the substrate + generic search engine
actually compose end to end -- not a claim about program synthesis at
scale.
"""

from __future__ import annotations

from typing import Sequence

from rde.search import VerifyResult, search_with_holdout
from rde.substrate.enumeration import enumerate_programs
from rde.substrate.program import Program
from rde.substrate.vm import ResourceExceeded, execute

Batch = tuple[tuple[int, int], ...]


def make_batches(
    offset: int, xs_train: Sequence[int], xs_holdout: Sequence[int]
) -> tuple[Batch, Batch]:
    """`f(x) = x + offset` (input, expected-output) batches for train/holdout verification."""
    train = tuple((x, x + offset) for x in xs_train)
    holdout = tuple((x, x + offset) for x in xs_holdout)
    return train, holdout


def _verify_programs(
    candidates: Sequence[Program],
    _domain: object,
    batch: Batch,
    *,
    max_steps: int,
) -> dict[str, VerifyResult]:
    results: dict[str, VerifyResult] = {}
    for program in candidates:
        ok = True
        for x, expected in batch:
            try:
                trace = execute(program, x, max_steps=max_steps)
            except ResourceExceeded:
                ok = False
                break
            if trace.output != expected:
                ok = False
                break
        # Objective is program length -- among every program that verifies,
        # prefer the shortest (search_with_holdout's default is
        # lower-is-better, matching "prefer simple explanations" the same
        # way rde.representation's complexity objective does).
        results[program.program_id] = VerifyResult(ok=ok, objective=float(len(program)))
    return results


def rediscover_offset(
    offset: int,
    *,
    xs_train: Sequence[int] = (0, 1, 2, 3),
    xs_holdout: Sequence[int] = (10, 11, 12),
    max_length: int = 3,
    operand_range: Sequence[int] = range(0, 4),
    max_steps: int = 64,
) -> Program | None:
    """Search `rde.substrate`'s brute-force program space for `f(x) = x + offset`.

    Returns the shortest verified program, or `None` if the bounded search
    space (`max_length`, `operand_range`) doesn't contain one -- e.g.
    `offset` outside `operand_range` is expected to fail, not a bug.
    """
    train_batch, holdout_batch = make_batches(offset, xs_train, xs_holdout)
    candidates = list(enumerate_programs(max_length, operand_range=operand_range))

    def _verify(cands: Sequence[Program], domain: object, batch: Batch) -> dict[str, VerifyResult]:
        return _verify_programs(cands, domain, batch, max_steps=max_steps)

    results = search_with_holdout(
        candidates,
        train_batch,
        holdout_batch,
        verify=_verify,
        candidate_id=lambda p: p.program_id,
    )
    return results[0].candidate if results else None
