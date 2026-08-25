"""`rde.representation` applied to real hsp_functions bounded-query data.

Domain-integration gap closure: `rde.representation`'s grammar/search/Pareto
machinery previously only ran on synthetic NumPy vectors built for its own
tests. This module runs it on the actual difference-profile vector
`sampling.sample_difference_estimates` produces (also this domain's
`materialize()` signature array) — real bounded-query HSP data, not a
surrogate built to make the demo work. Core (`rde.representation`) still
never imports `rde_domains`; this module lives on the domain side and
imports core, which is the direction the architecture already allows.

Every family's difference-profile has exactly `n_bits` entries regardless
of family (`sample_difference_estimates`'s `candidates` list is always
`n_bits` long for both `"gf2"` and `"cyclic"` domain kinds) — that fixed
length is what makes stacking instances from *different* families into one
`rde.representation` batch valid at all.

Exploratory tooling, not a `DomainContract`-integrated experiment: this does
not add predictor columns to `hsp_functions`' feature contract, does not run
the leak-audit / held-out-family / preregistered-decision-rule discovery
loop `docs/experiment-playbook.md` requires of a "real experiment", and
makes no claim about `structure_strength` or algorithm class. It answers a
narrower, honest question — "does representation search find anything on
this domain's actual data" — not "here is a validated discovery".
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from rde.representation import SearchCandidate, rank_representations
from rde_domains.hsp_functions.functions import ALL_FAMILIES, make_instance
from rde_domains.hsp_functions.sampling import sample_difference_estimates


def diff_profile_batch(
    *,
    n_bits: int,
    families: Sequence[str] = ALL_FAMILIES,
    instances_per_family: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """Stack difference-profile vectors across several instances at fixed `n_bits`.

    Returns `(batch, family_labels)`: `batch` has shape
    `(len(families) * instances_per_family, n_bits)`; `family_labels[i]`
    names which family produced `batch[i]`. Building this list via a Python
    loop over `(family, instance)` pairs is control flow over a small,
    caller-bounded set (typically a handful of families times a handful of
    instances), not a per-sample data loop — each individual instance's
    `sample_difference_estimates` call is itself the vectorized-per-oracle
    query batch `sampling.py` documents.
    """
    rows: list[np.ndarray] = []
    row_families: list[str] = []
    cursor = 0
    for family in families:
        for _ in range(instances_per_family):
            inst_seed = seed + cursor
            cursor += 1
            instance = make_instance(family, n_bits=n_bits, seed=inst_seed)
            rng = np.random.default_rng(inst_seed ^ 0x1234_5678)
            diff_estimates = sample_difference_estimates(instance, rng)
            rows.append(np.array(list(diff_estimates.values()), dtype=float))
            row_families.append(family)
    return np.stack(rows, axis=0), row_families


def rank_diff_profile_representations(
    *,
    n_bits: int,
    families: Sequence[str] = ALL_FAMILIES,
    instances_per_family: int = 4,
    seed: int = 0,
    tolerance: float = 1e-6,
) -> list[SearchCandidate]:
    """Rank `rde.representation`'s grammar against real hsp_functions difference profiles."""
    batch, _ = diff_profile_batch(
        n_bits=n_bits, families=families, instances_per_family=instances_per_family, seed=seed
    )
    return rank_representations(batch, n=n_bits, tolerance=tolerance)
