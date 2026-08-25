"""Generic enumerate -> verify -> holdout-rank search engine.

See `rde.search.protocol` for the shapes this operates on. This module has
no opinion about what a candidate *is* -- a `rde.representation.Representation`,
a `rde.core.protocols.RecoveryProtocol`, or a `rde.substrate.program.Program`
all fit, as long as a caller supplies a `Verifier` and a `candidate_id`
function for it. It also has no opinion about what "better" means --
`higher_is_better` picks the sort direction; the caller (which knows whether
its objective is a complexity to minimize or a recall to maximize) decides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from rde.search.protocol import CandidateId, CandidateT, Verifier, VerifyResult

_FAILED = VerifyResult(ok=False, objective=float("inf"))


@dataclass(frozen=True)
class HoldoutSearchResult:
    """One candidate that verified on both `train_batch` and `holdout_batch`."""

    candidate: Any
    candidate_id: str
    train: VerifyResult
    holdout: VerifyResult


def search_with_holdout(
    candidates: Sequence[CandidateT],
    train_batch: Any,
    holdout_batch: Any,
    *,
    verify: Verifier,
    candidate_id: CandidateId,
    domain: Any = None,
    higher_is_better: bool = False,
) -> list[HoldoutSearchResult]:
    """Verify on `train_batch`, drop failures, re-verify survivors on `holdout_batch`, rank.

    `holdout_batch` is required and must be independently drawn from
    `train_batch` -- a candidate that only verifies on the data it was
    checked against is not a valid discovery, whatever its holdout objective
    would have been. Survivors are ranked by *holdout* objective only, never
    train -- callers needing a generalization ratio compute it themselves
    from `result.train.objective` / `result.holdout.objective`, since the
    right zero-handling convention differs by objective (a complexity of 0
    and a recall of 0 do not mean the same thing).

    `verify` is called once with the full `candidates` list against
    `train_batch`, then once more with only the train-survivors against
    `holdout_batch` -- candidates that never verify on train are never
    scored on holdout at all, matching the short-circuit discipline both
    original callers already used.
    """
    train_results = verify(candidates, domain, train_batch)
    survivors = [c for c in candidates if train_results.get(candidate_id(c), _FAILED).ok]
    if not survivors:
        return []

    holdout_results = verify(survivors, domain, holdout_batch)

    out: list[HoldoutSearchResult] = []
    for candidate in survivors:
        cid = candidate_id(candidate)
        holdout_result = holdout_results.get(cid)
        if holdout_result is None or not holdout_result.ok:
            continue
        out.append(
            HoldoutSearchResult(
                candidate=candidate,
                candidate_id=cid,
                train=train_results[cid],
                holdout=holdout_result,
            )
        )

    out.sort(key=lambda r: r.holdout.objective, reverse=higher_is_better)
    return out
