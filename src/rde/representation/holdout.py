"""Representation holdout: an honest audit that ranking doesn't overclaim.

Gap closure: nothing previously verified that ranking against a restricted
grammar subset actually reports *lack* of structure when the true
explaining primitive is withheld, rather than silently latching onto an
unrelated primitive and claiming false compression. This is a controlled
ablation over `grammar.py`'s fixed, known primitive set — not general
representation *synthesis* (`search.py`'s docstring explains why that
isn't built here); it audits the integrity of the existing ranking, not a
new search capability. Matches the original proposal's "representation
holdout" idea (hide the primitive that would explain the data, see whether
the system honestly reports it doesn't know) at the scope this package
actually supports.

Verified numerically before this module was written: a periodic signal
ranked with `{dft, dft_full, polynomial_vandermonde}` withheld gets no
better than `identity`'s complexity (ratio `1.0` — no false claim), while
the full grammar correctly finds `dft` (`8x` lower complexity). A
low-degree-polynomial batch ranked with `polynomial_vandermonde` withheld
but `dft`/`dft_full` visible shows *partial* leakage (`dft`'s complexity is
`~0.68x` identity's, not `1.0x`) — a real mathematical fact (low-degree
polynomials do have some spectral concentration too), not a bug, and
exactly the kind of nuance a binary "did it cheat" flag would hide; that is
why `HoldoutAudit.leakage_ratio` is reported as a number, not folded away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from rde.representation.array_backend import ArraySearchBackend, get_array_backend
from rde.representation.search import SearchCandidate, rank_representations


@dataclass(frozen=True)
class HoldoutAudit:
    """Compares a visible-primitive-only ranking against the full grammar's."""

    visible_best: SearchCandidate
    full_best: SearchCandidate
    visible_primitives: tuple[str, ...]
    held_out_primitives: tuple[str, ...]
    leakage_ratio: float
    honestly_uncertain: bool
    discovered_held_out_structure: bool


def audit_holdout(
    values: Any,
    *,
    n: int,
    visible_primitives: Sequence[str],
    held_out_primitives: Sequence[str],
    backend: ArraySearchBackend | None = None,
    object_type: str | None = None,
    tolerance: float = 1e-6,
    uncertainty_threshold: float = 0.5,
    chain_max_depth: int | None = None,
) -> HoldoutAudit:
    """Rank with only `visible_primitives`, then compare against the full grammar.

    `leakage_ratio` is `visible_best.complexity / identity.complexity` —
    `1.0` means the visible subset found no compression at all (fully
    honest), values near `0` mean it found compression comparable to using
    the full grammar (either a visible primitive genuinely also explains
    the data, or `held_out_primitives` wasn't actually the thing making
    this data compressible — this function cannot distinguish those two;
    it only reports the number).

    `honestly_uncertain` is `leakage_ratio >= uncertainty_threshold` — the
    visible-only search did not claim strong compression.
    `discovered_held_out_structure` is true when any of `full_best`'s
    `"+"`-split stages names a held-out primitive — confirming the true
    structure really was reachable through the withheld primitives, not
    merely absent from the grammar entirely. For a depth-1 (flat grammar)
    `full_best`, this is exactly the old "id equals a held-out name" check
    (a chain id has no `"+"` to split); it generalizes to a composed chain
    using a held-out primitive as *any* of its stages, not only as the
    chain's entire identity.

    `chain_max_depth`, when given, forwards to `rank_representations` so
    both the visible-only and full rankings consider `program_search`'s
    composed chains, not just the flat grammar — withholding a stage-2
    primitive (e.g. `sort_by_magnitude`) genuinely removes every chain that
    would have used it, not just its depth-1 entry (see `program_search
    .atomic_registry`'s `primitive_subset` docstring for why that holds).
    """
    backend = backend or get_array_backend()
    visible_ranked = rank_representations(
        values,
        n=n,
        backend=backend,
        object_type=object_type,
        tolerance=tolerance,
        primitive_subset=visible_primitives,
        chain_max_depth=chain_max_depth,
    )
    full_ranked = rank_representations(
        values,
        n=n,
        backend=backend,
        object_type=object_type,
        tolerance=tolerance,
        chain_max_depth=chain_max_depth,
    )
    if not visible_ranked or not full_ranked:
        raise ValueError("empty ranking; visible_primitives or the grammar produced no candidates")

    identity_candidate = next(
        (c for c in full_ranked if c.representation_id == "identity"), full_ranked[0]
    )
    identity_complexity = identity_candidate.complexity
    visible_best = visible_ranked[0]
    leakage_ratio = (
        float(visible_best.complexity / identity_complexity) if identity_complexity > 0 else 1.0
    )
    full_best = full_ranked[0]

    return HoldoutAudit(
        visible_best=visible_best,
        full_best=full_best,
        visible_primitives=tuple(visible_primitives),
        held_out_primitives=tuple(held_out_primitives),
        leakage_ratio=leakage_ratio,
        honestly_uncertain=leakage_ratio >= uncertainty_threshold,
        discovered_held_out_structure=bool(
            set(full_best.representation_id.split("+")) & set(held_out_primitives)
        ),
    )
