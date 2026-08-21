"""Bounded-query descriptor computation for hsp_functions instances.

Implements the query-only access model the charter's fixed contract
(S4.1) requires: every predictor-eligible descriptor below is computed
from a poly(n_bits) random sample of oracle calls
(query_budget_for: O(n^2) with confirmatory-horizon leading constant)
-- never from the full 2^n_bits table. A separate, explicitly-gated
oracle-only exact tier exists for small-n audit/calibration use only (Gate
0 mechanism validation), matching this project's `max_bruteforce_n`
convention.

See docs/research/hidden-subgroup-function-discovery-charter.md S4.1 and
docs/algorithms/ (the ALGO card for this subroutine, registered before any
experiment relies on it).
"""

from __future__ import annotations

import numpy as np

from rde.features import boolean as boolean_features
from rde_domains.hsp_functions.functions import FunctionInstance

MAX_ORACLE_N_BITS = 14  # mirrors this project's existing brute-force enumeration cap

#: Leading constant of the O(n^2) budget. For a 2-to-1 hidden-shift function
#: (Simon / period / dihedral coset), two uniform samples collide on the
#: hidden pair with probability ~2^{-n}, so a B-sample has expected pair
#: count λ = B² / 2^{n+1}. EXP-065's first confirmatory run used c=8
#: (B=8n² ⇒ λ≈0.64 at n=24, held-out recall ≈0.51). c=20 gives λ≈3.95 at
#: n=24 (P(miss)≈e^{-λ}≈0.02) while remaining poly(n) and B ≪ 2^n
#: (11520 vs 2^24). This is a confirmatory-horizon calibration, not an
#: asymptotic classical Simon solver: at any fixed c, λ → 0 for n ≫ 24.
QUERY_BUDGET_COEFF = 20
CONFIRMATORY_HORIZON_N_BITS = 24
MIN_SIMON_PAIR_EXPECTATION = 3.0


def query_budget_for(n_bits: int) -> int:
    """poly(n_bits) query budget -- never scales with x_size=2**n_bits."""
    n = int(n_bits)
    return max(64, QUERY_BUDGET_COEFF * n * n)


def simon_pair_expectation(n_bits: int, budget: int | None = None) -> float:
    """Expected hidden-shift pairs in a uniform B-sample of a 2-to-1 function."""
    b = query_budget_for(n_bits) if budget is None else int(budget)
    return float(b * b) / float(2 ** (int(n_bits) + 1))


def _difference(inst: FunctionInstance, a: int, b: int) -> int:
    if inst.domain_kind == "gf2":
        return a ^ b
    return (a - b) % inst.x_size


def sample_difference_estimates(inst: FunctionInstance, rng: np.random.Generator) -> dict[int, float]:
    """Bounded-query estimate g(d) = Pr_x[f(x) == f(x (+/^) d)] at O(n_bits) candidate shifts d.

    O(n_bits) candidates x O(n_bits) sub-sample each = O(n_bits^2) total
    queries -- poly(n_bits), independent of x_size. Used both as the
    domain's `materialize()` signature array and as raw input to the
    entropy/concentration descriptors in `bounded_query_descriptors`. Each
    candidate's sub-sample is evaluated via `evaluate_batch` (vectorized,
    ALGO-062) rather than a Python per-query loop; the remaining Python
    loop is only over the O(n_bits) candidates themselves.

    `sub_budget` is deliberately kept at the same O(n_bits) order as
    `query_budget_for` (not a small fixed constant): each g(d) estimate is
    a `hits/sub_budget` fraction, so a too-small denominator quantizes
    every instance's derived statistics onto the same few coarse values
    regardless of its actual random secret -- confirmed empirically as the
    root cause of EXP-064's Gate-0 population-distinctness failures (a
    small denominator made genuinely different instances collide onto
    identical rounded structural feature vectors, tripping
    `rde.experiment.gate.distinct_structural_instances`, which is exact-
    float-tuple equality with no rounding tolerance).
    """
    sub_budget = max(64, 8 * inst.n_bits)
    candidates = (
        [1 << i for i in range(inst.n_bits)]
        if inst.domain_kind == "gf2"
        else [max(1, inst.x_size // (2**k)) for k in range(1, inst.n_bits + 1)]
    )
    diff_estimates: dict[int, float] = {}
    for d in candidates:
        probes = rng.integers(0, inst.x_size, size=sub_budget, dtype=np.int64)
        others = (probes ^ d) if inst.domain_kind == "gf2" else ((probes + d) % inst.x_size)
        hits = np.count_nonzero(inst.evaluate_batch(probes) == inst.evaluate_batch(others))
        diff_estimates[d] = float(hits) / float(sub_budget)
    return diff_estimates


def bounded_query_descriptors(
    inst: FunctionInstance,
    rng: np.random.Generator,
    name: str = "f",
    *,
    diff_estimates: dict[int, float] | None = None,
) -> dict[str, float]:
    """Predictor-eligible descriptors of `inst` from a bounded random query sample.

    Pass a precomputed `diff_estimates` (from `sample_difference_estimates`,
    also used as the domain's materialize signature) to avoid a second
    independent O(n_bits^2)-query draw of the same g(d) profile.
    """
    budget = query_budget_for(inst.n_bits)
    xs = rng.integers(0, inst.x_size, size=budget, dtype=np.int64)
    ys = inst.evaluate_batch(xs)  # single vectorized oracle-evaluation batch, not a Python loop

    labels: dict[int, list[int]] = {}
    for x, y in zip(xs.tolist(), ys.tolist()):
        labels.setdefault(y, []).append(x)

    diffs: list[int] = []
    n_collision_pairs = 0
    for members in labels.values():
        # Sampling is with replacement: the same x can be drawn twice,
        # which is a trivial self-collision (difference 0), not real
        # signal. Dedupe distinct x-values before counting.
        unique_members = sorted(set(members))
        if len(unique_members) < 2:
            continue
        n_collision_pairs += len(unique_members) - 1
        base = unique_members[0]
        for other in unique_members[1:]:
            diffs.append(_difference(inst, base, other))

    p = f"hsp_sample.{name}"
    unique_labels = len({int(y) for y in ys.tolist()})
    ys_f = ys.astype(np.float64)
    label_scale = float((1 << 63) - 1)
    out: dict[str, float] = {
        f"{p}.query_budget": float(budget),
        f"{p}.collision_rate": float(n_collision_pairs) / float(budget),
        f"{p}.n_collisions_found": float(n_collision_pairs),
        #: Label-side statistics from the same bounded query draw. For
        #: structureless families (e.g. `generic_random_control`) collision-
        #: based probes often collapse to all-zero difference profiles, so
        #: every independent random instance would otherwise share an identical
        #: structural fingerprint and fail `ExperimentGate.check_population`.
        #: These stats vary per seed without full-table access.
        f"{p}.unique_label_fraction": float(unique_labels) / float(budget),
        f"{p}.label_mean_normalized": float(np.mean(ys_f)) / label_scale,
        f"{p}.label_std_normalized": float(np.std(ys_f)) / label_scale,
    }

    if diffs:
        if inst.domain_kind == "gf2":
            bits = np.array([[(d >> i) & 1 for i in range(inst.n_bits)] for d in diffs], dtype=np.uint8)
            rank = boolean_features.gf2_rank(bits)
            out[f"{p}.difference_span_dim_fraction"] = float(rank) / float(inst.n_bits)
        else:
            g = diffs[0]
            for d in diffs[1:]:
                g = np.gcd(g, d)
            g = max(1, int(g))
            out[f"{p}.detected_period_divisor_fraction"] = float(np.log2(g)) / float(max(1, inst.n_bits))
    else:
        out[f"{p}.difference_span_dim_fraction"] = 0.0
        out[f"{p}.detected_period_divisor_fraction"] = 0.0

    if diff_estimates is None:
        diff_estimates = sample_difference_estimates(inst, rng)
    out.update(boolean_features.sample_difference_profile(diff_estimates, name=name))
    out[f"{p}.difference_profile_query_cost"] = float(len(diff_estimates) * max(8, 2 * inst.n_bits))

    # The summary stats above (entropy/concentration/mean/max) are
    # deliberately invariant to *which* candidate shift carries the
    # collision signal -- that is the point for the discovery question,
    # but it also means every instance in a family with a fixed structural
    # regime (e.g. exact Simon) collapses toward the same descriptor
    # vector, since only the *location* of the spike differs by secret,
    # not its shape. Also expose the raw per-candidate g(d_i) values under
    # stable, size-consistent positional keys (candidates are generated in
    # a fixed, deterministic order for a given n_bits/domain_kind -- see
    # `sample_difference_estimates`) so distinct random secrets remain
    # distinguishable row-to-row without leaking the secret itself.
    for idx, (_d, g_val) in enumerate(diff_estimates.items()):
        out[f"{p}.diff_g_{idx}"] = g_val

    return out


def exact_oracle_audit_descriptors(inst: FunctionInstance, name: str = "f") -> dict[str, float]:
    """Exact, full-table oracle-only descriptors -- audit/Gate-0 use only.

    Requires materializing the full length-2**n_bits difference table;
    gated to `MAX_ORACLE_N_BITS`. Never used as a predictor -- see the
    domain contract's `hsp_oracle.*` FeatureSpec (ENUMERATED_ORACLE).
    """
    if inst.n_bits > MAX_ORACLE_N_BITS or inst.domain_kind != "gf2":
        return {}
    x_size = inst.x_size
    labels = inst.evaluate_batch(np.arange(x_size)).astype(np.int64)
    diff_table = np.zeros(x_size, dtype=np.float64)
    for d in range(x_size):
        shifted = labels[np.arange(x_size) ^ d]
        diff_table[d] = float(np.mean(labels == shifted))
    return boolean_features.exact_spectral_descriptors(diff_table, name=name)
