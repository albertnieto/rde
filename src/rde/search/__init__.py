"""Generic candidate search engine, shared across subsystems.

`rde.representation.program_search` and `rde.recovery.search_space`
independently converged on the same shape (enumerate -> verify on train ->
drop failures -> verify survivors on an independent holdout -> drop
failures -> rank by holdout objective only). This package names that shape
once so future candidate spaces (see `rde.substrate` for the first new one)
don't have to re-derive it. Cross-cutting engineering, like `rde.backends`
-- not a new science letter; see `docs/ARCHITECTURE.md`.
"""

from __future__ import annotations

from rde.search.holdout_search import HoldoutSearchResult, search_with_holdout
from rde.search.protocol import CandidateId, Verifier, VerifyResult

__all__ = [
    "CandidateId",
    "HoldoutSearchResult",
    "Verifier",
    "VerifyResult",
    "search_with_holdout",
]
