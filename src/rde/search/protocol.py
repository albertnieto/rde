"""Generic verify -> holdout-rank search protocol.

Formalizes the shape `rde.representation.program_search.search_chains` and
`rde.recovery.search_space.search_recovery_chains` each independently
implemented: enumerate candidates, verify against a domain-supplied ground
truth on a train batch, drop failures, re-verify survivors on an
*independent* holdout batch, drop failures again, and only then compare
survivors by holdout objective. Neither of those two callers is a "science
ontology" -- this is engineering plumbing (like `rde.backends`), so it lives
outside both `representation/` and `recovery/` rather than picking one to
depend on the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, TypeVar

CandidateT = TypeVar("CandidateT")


@dataclass(frozen=True)
class VerifyResult:
    """One candidate's outcome against one batch.

    `ok` is the hard pass/fail gate (a candidate that fails is dropped, not
    penalized); `objective` is the score used for ranking survivors (lower
    or higher is better depending on the caller -- see
    `rde.search.holdout_search.search_with_holdout`'s `higher_is_better`);
    `detail` carries whatever caller-specific evidence object it verified
    against (a `Certificate`, a recall count, ...) so callers can
    reconstruct their own richer result type without this module needing to
    know its shape.
    """

    ok: bool
    objective: float
    detail: Any = None


# Verify every candidate in `candidates` against one `batch` (with optional
# domain-specific context), keyed by whatever id `CandidateId` extracts.
Verifier = Callable[[Sequence[CandidateT], Any, Any], Mapping[str, VerifyResult]]

CandidateId = Callable[[CandidateT], str]
