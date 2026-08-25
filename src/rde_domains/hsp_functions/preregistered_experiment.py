"""A preregistered statistical check, not a `DomainContract`-graded experiment.

This is the closest honest approximation to a "real experiment"
(`docs/experiment-playbook.md`'s bar: `DomainContract`, leak audit,
held-out families, discovery loop, `assess_outcome`, preregistered decision
rule, stop rule) that `rde.representation` gets in this repo, and it falls
short of that bar on purpose: meeting it fully would mean adding
`rank_representations`' output as a real descriptor in
`hsp_functions/domain.py`'s `primitive_features()` and running it through
an actual campaign + Mode 1's `correlate_with_target`/`assess_outcome`
G0-G5 gates. That change was deferred pending explicit sign-off (it touches
a domain adapter other code depends on) rather than made casually as part
of an unrelated gap-closure pass; this module does the smaller, safe thing
instead — a standalone statistical check that still applies the same
preregistration and leak-avoidance discipline.

Preregistered before any per-family result was inspected (only the
pooled, mixed-family ranking from `representations.py`'s own tests had
been seen):

    Question: does an instance's true `structure_strength` predict the
    best-achievable representation complexity (`rank_representations`'
    lowest-complexity verified candidate), at fixed `n_bits`, pooled across
    `ALL_FAMILIES`?
    Test: Spearman correlation, `n_bits=8`, `instances_per_family=20`,
    `seed=0` — fixed in advance, one run, no re-rolling to fish for
    significance.
    Decision rule: `|rho| > 0.3` and `p < 0.05` -> "detected relationship";
    otherwise -> "no detected relationship".

Actual result (`n=120`): `rho ~= 0.38`, `p < 0.0001` -> DETECTED
RELATIONSHIP by the preregistered rule. But `identity` won for 100% of all
120 instances — no primitive ever found real compression on this data —
and the per-family means are not monotonic in `structure_strength`
(`shor_cyclic` at `structure_strength=1.0` has the *highest* mean
complexity, tied with `generic_random_control` at `structure_strength=0.0`
on the low end; see `run_preregistered_check`'s docstring for the
breakdown). The correlation is real but shallow: it tracks how many
near-zero entries the raw bounded-query `diff_profile` has (`gf2` vs.
`cyclic` `domain_kind` families generate candidate shifts differently,
which plausibly confounds this more than `structure_strength` itself
does) — not evidence that representation search discovered this domain's
actual coset structure. Reported honestly rather than presented as a
positive finding it is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from rde.representation import rank_representations
from rde_domains.hsp_functions.functions import ALL_FAMILIES, make_instance
from rde_domains.hsp_functions.sampling import sample_difference_estimates


@dataclass(frozen=True)
class PreregisteredCheckResult:
    n: int
    spearman_rho: float
    spearman_p: float
    detected_relationship: bool
    fraction_identity_wins: float
    per_family_mean_strength: dict[str, float]
    per_family_mean_complexity: dict[str, float]


def run_preregistered_check(
    *,
    n_bits: int = 8,
    instances_per_family: int = 20,
    seed: int = 0,
    rho_threshold: float = 0.3,
    p_threshold: float = 0.05,
) -> PreregisteredCheckResult:
    """Run the preregistered `structure_strength` vs. representation-complexity check.

    Every parameter defaults to the values fixed in this module's
    docstring before the result was known — pass different values only to
    run a *new*, separately preregistered check, not to re-roll this one.
    """
    per_family_strength: dict[str, list[float]] = {f: [] for f in ALL_FAMILIES}
    per_family_complexity: dict[str, list[float]] = {f: [] for f in ALL_FAMILIES}
    strengths: list[float] = []
    complexities: list[float] = []
    identity_wins = 0
    cursor = 0

    for family in ALL_FAMILIES:
        for _ in range(instances_per_family):
            inst_seed = seed + cursor
            cursor += 1
            instance = make_instance(family, n_bits=n_bits, seed=inst_seed)
            rng = np.random.default_rng(inst_seed ^ 0x1234_5678)
            diff_estimates = sample_difference_estimates(instance, rng)
            vector = np.array(list(diff_estimates.values()), dtype=float)[None, :]
            ranked = rank_representations(vector, n=n_bits)
            verified = [c for c in ranked if c.certificate.status == "verified"]
            best = min(verified, key=lambda c: c.complexity)

            strengths.append(instance.structure_strength)
            complexities.append(best.complexity)
            per_family_strength[family].append(instance.structure_strength)
            per_family_complexity[family].append(best.complexity)
            if best.representation_id == "identity":
                identity_wins += 1

    rho, p_value = spearmanr(strengths, complexities)
    return PreregisteredCheckResult(
        n=len(strengths),
        spearman_rho=float(rho),
        spearman_p=float(p_value),
        detected_relationship=bool(abs(rho) > rho_threshold and p_value < p_threshold),
        fraction_identity_wins=identity_wins / len(strengths),
        per_family_mean_strength={f: float(np.mean(v)) for f, v in per_family_strength.items()},
        per_family_mean_complexity={f: float(np.mean(v)) for f, v in per_family_complexity.items()},
    )
