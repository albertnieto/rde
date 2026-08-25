"""Compositional search over recovery-extraction atoms.

Ports `rde.representation.program_search`'s discipline -- enumerate typed
compositions, require an *independent* holdout, drop anything that fails on
it, rank survivors by holdout performance, report a generalization ratio --
into `rde.recovery`. `programs.py` is a flat, hand-enumerated catalog scored
once by `campaign.py`; nothing there searches or composes. This module does:
`enumerate_recovery_chains` builds depth-1 candidates (the existing flat
grammar plus the newer confidence-gated and closure-based atoms) and depth-2
`PairCombine` compositions of them, and `search_recovery_chains` scores every
one against an independent discovery/confirmatory split and keeps only what
survives both.

Atoms are family-agnostic (see `programs.py`'s `ConfidentCollisionProgram`,
`GroupClosureProgram`, `PairCombine` docstrings). Which family a given chain
ends up solving is discovered by the evaluation below, not designed in by
picking a family and hand-fitting a program to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from rde.core.protocols import RecoveryDomain
from rde.recovery.programs import (
    BAGS,
    POSTS,
    ConfidentCollisionProgram,
    GroupClosureProgram,
    PairCombine,
    enumerate_recovery_programs,
)
from rde.recovery.search import evaluate_protocols


def _encoder_key(protocol: Any) -> tuple[str, Any]:
    """Coarse "what does this pipeline read" tag, used to avoid pairing near-duplicates.

    Two pipelines built off the same bag (e.g. `xor_mode_id` and
    `xor_mode_confident_high_half`) both fail or succeed together far more
    often than not, since they're reading the same pairwise quantity; pairing
    them adds search-space size (and multiple-comparisons risk) without much
    real expressive gain over pairing across different encoders.
    """
    bag = getattr(protocol, "bag", None)
    if bag is not None:
        return ("bag", bag)
    if isinstance(protocol, GroupClosureProgram):
        return ("closure", protocol.op, protocol.mask_bits)
    return ("other", protocol.protocol_id)


def enumerate_recovery_chains(
    *,
    max_depth: int = 2,
    mask_bits_choices: Sequence[int] = (2, 3, 4),
) -> tuple[Any, ...]:
    """Every depth-1 recovery program plus, at `max_depth >= 2`, every valid depth-2 pairing.

    Depth-1 = `programs.enumerate_recovery_programs()`'s existing flat
    grammar, plus `ConfidentCollisionProgram` (every bag x post) and
    `GroupClosureProgram` (every `mask_bits_choices` entry) -- the new,
    general atoms. Depth-2 = `PairCombine` over every depth-1 pair whose
    `_encoder_key` differs, so a family whose planted object needs two
    independently-recovered pieces (not just one) has a real candidate
    without an exhaustive, mostly-redundant cross product.
    """
    depth1: list[Any] = list(enumerate_recovery_programs())
    for bag in BAGS:
        for post in POSTS:
            depth1.append(ConfidentCollisionProgram(bag, post))
    for mask_bits in mask_bits_choices:
        depth1.append(GroupClosureProgram(mask_bits, op="xor"))
    # mask_bits is meaningless for op="mult" (the raw domain values already
    # are the group's own coordinates), so it is enumerated once, not swept.
    depth1.append(GroupClosureProgram(mask_bits=0, op="mult"))
    depth1_tuple = tuple(depth1)
    if max_depth < 2:
        return depth1_tuple

    depth2: list[Any] = []
    for i, protocol_a in enumerate(depth1_tuple):
        key_a = _encoder_key(protocol_a)
        for protocol_b in depth1_tuple[i + 1 :]:
            if _encoder_key(protocol_b) == key_a:
                continue
            depth2.append(PairCombine(protocol_a, protocol_b))
    return depth1_tuple + tuple(depth2)


@dataclass(frozen=True)
class RecoveryChainResult:
    """One enumerated chain that cleared `min_recall` on *both* independent splits."""

    protocol_id: str
    protocol: Any
    depth: int
    discovery_recall: float
    confirmatory_recall: float
    recall_ratio: float


def search_recovery_chains(
    domain: RecoveryDomain,
    discovery_instances: Sequence[Any],
    confirmatory_instances: Sequence[Any],
    *,
    family: str,
    max_depth: int = 2,
    min_recall: float = 0.80,
    mask_bits_choices: Sequence[int] = (2, 3, 4),
    rng: np.random.Generator | None = None,
) -> list[RecoveryChainResult]:
    """Search recovery chains for `family`, keeping only what survives an independent holdout.

    `discovery_instances`/`confirmatory_instances` must be independently
    drawn (e.g. even vs. odd seeds, matching `campaign.py`'s existing split)
    -- a chain is dropped, not merely penalized, if it fails to clear
    `min_recall` on *either* split. Survivors are ranked by confirmatory
    recall, never discovery recall, with `recall_ratio`
    (confirmatory / discovery) reported so a chain that only "worked" on the
    discovery split is visibly penalized rather than silently absent.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    chains = enumerate_recovery_chains(max_depth=max_depth, mask_bits_choices=mask_bits_choices)

    discovery_report = evaluate_protocols(domain, discovery_instances, chains, rng=rng)
    confirmatory_report = evaluate_protocols(domain, confirmatory_instances, chains, rng=rng)

    results: list[RecoveryChainResult] = []
    for protocol in chains:
        discovery_recall = discovery_report.rate(protocol.protocol_id, family)
        if discovery_recall != discovery_recall or discovery_recall < min_recall:
            continue
        confirmatory_recall = confirmatory_report.rate(protocol.protocol_id, family)
        if confirmatory_recall != confirmatory_recall or confirmatory_recall < min_recall:
            continue
        ratio = confirmatory_recall / discovery_recall if discovery_recall > 0 else float("inf")
        depth = 2 if isinstance(protocol, PairCombine) else 1
        results.append(
            RecoveryChainResult(
                protocol_id=protocol.protocol_id,
                protocol=protocol,
                depth=depth,
                discovery_recall=discovery_recall,
                confirmatory_recall=confirmatory_recall,
                recall_ratio=ratio,
            )
        )

    results.sort(key=lambda r: -r.confirmatory_recall)
    return results
