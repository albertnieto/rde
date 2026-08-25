"""Mode 2 — backward (target-first) algorithm synthesis.

Public surface for reverse-engineering an algorithm skeleton from a declared
resource budget, as opposed to forward discovery which fits representations
to already-generated data. See `rde.synthesis.search` for the two-stage
(symbolic prune, then domain-verified) search loop and
`src/rde/docs/hierarchical-synthesis.md` for the design note.
"""

from rde.synthesis.recurrence import CostClass, Recurrence, meets_target, solve_recurrence
from rde.synthesis.search import (
    SynthesisCandidate,
    SynthesisReport,
    execute_skeleton,
    synthesize,
    verify_skeleton,
    write_synthesis_conjectures_jsonl,
)
from rde.synthesis.skeleton import VERBS, AlgorithmSkeleton, default_skeleton_catalog

__all__ = [
    "VERBS",
    "AlgorithmSkeleton",
    "CostClass",
    "Recurrence",
    "SynthesisCandidate",
    "SynthesisReport",
    "default_skeleton_catalog",
    "execute_skeleton",
    "meets_target",
    "solve_recurrence",
    "synthesize",
    "verify_skeleton",
    "write_synthesis_conjectures_jsonl",
]
