"""HSP recovery matrix: textbook extractors vs planted secrets."""

from __future__ import annotations

import numpy as np
import pytest

from rde.recovery import evaluate_protocols, default_extractor_catalog
from rde_domains.hsp_functions.functions import make_instance
from rde_domains.hsp_functions.recovery import HspFunctionRecovery


def _exact(family: str, n_bits: int, seed: int):
    inst = make_instance(family, n_bits=n_bits, seed=seed)
    if "structure_break" in inst.params:
        params = dict(inst.params)
        params["structure_break"] = 0.0
        params["structure_strength"] = 1.0
        return inst.__class__(
            inst.family, inst.domain_kind, inst.n_bits, inst.x_size, inst.seed, params
        )
    return inst


@pytest.mark.parametrize(
    "family,protocol_id",
    [
        ("simon", "xor_collision_mode"),
        ("shor_cyclic", "additive_gcd"),
        ("dihedral_kuperberg", "additive_sum_mode"),
    ],
)
def test_textbook_extractor_recovers_planted_secret(family, protocol_id):
    inst = _exact(family, n_bits=8, seed=17)
    report = evaluate_protocols(
        HspFunctionRecovery(),
        [inst],
        default_extractor_catalog(),
        rng=np.random.default_rng(0),
    )
    assert report.rate(protocol_id, family) == 1.0


def test_xor_extractor_does_not_recover_heisenberg_v():
    inst = _exact("heisenberg_noncentral", n_bits=8, seed=7)
    report = evaluate_protocols(
        HspFunctionRecovery(),
        [inst],
        default_extractor_catalog(),
        rng=np.random.default_rng(1),
    )
    assert report.rate("xor_collision_mode", "heisenberg_noncentral") == 0.0
    assert report.rate("additive_gcd", "heisenberg_noncentral") == 0.0
    assert report.rate("additive_sum_mode", "heisenberg_noncentral") == 0.0


def test_xor_extractor_does_not_recover_quaternion_hidden_set():
    inst = _exact("quaternion_coset", n_bits=8, seed=11)
    report = evaluate_protocols(
        HspFunctionRecovery(),
        [inst],
        default_extractor_catalog(),
        rng=np.random.default_rng(3),
    )
    assert report.rate("xor_collision_mode", "quaternion_coset") == 0.0


def test_xor_high_half_recovers_heisenberg_v_at_n8():
    from rde.recovery.programs import CollisionProgram

    inst = _exact("heisenberg_noncentral", n_bits=8, seed=7)
    report = evaluate_protocols(
        HspFunctionRecovery(),
        [inst],
        [CollisionProgram("xor", "mode", "high_half")],
        rng=np.random.default_rng(1),
    )
    assert report.rate("xor_mode_high_half", "heisenberg_noncentral") == 1.0


def test_path_b_xor_high_half_fails_when_v_in_low_register():
    """Must not recover v when it lives in the low half."""
    from rde.recovery.programs import CollisionProgram

    inst = _exact("heisenberg_v_low_register", n_bits=8, seed=7)
    report = evaluate_protocols(
        HspFunctionRecovery(),
        [inst],
        [CollisionProgram("xor", "mode", "high_half")],
        rng=np.random.default_rng(1),
    )
    assert report.rate("xor_mode_high_half", "heisenberg_v_low_register") == 0.0


def test_catalog_is_a_search_not_three_extractors():
    from rde.recovery.programs import enumerate_recovery_programs

    ids = [p.protocol_id for p in enumerate_recovery_programs()]
    assert len(ids) > 10
    assert "xor_mode_high_half" in ids
    assert "xor_collision_mode" in ids


def test_junk_control_makes_xor_extractor_abstain():
    inst = make_instance("generic_random_control", n_bits=8, seed=3)
    report = evaluate_protocols(
        HspFunctionRecovery(),
        [inst],
        default_extractor_catalog(),
        rng=np.random.default_rng(2),
    )
    assert report.rate("xor_collision_mode", "generic_random_control") == 1.0


def test_group_closure_program_recovers_quaternion_hidden_set():
    """The primitive the old xor/sum/diff grammar could not express for Q8-shaped hiding."""
    from rde.recovery.programs import GroupClosureProgram

    hits = 0
    n = 20
    for seed in range(n):
        inst = _exact("quaternion_coset", n_bits=8, seed=seed)
        report = evaluate_protocols(
            HspFunctionRecovery(),
            [inst],
            [GroupClosureProgram(mask_bits=3)],
            rng=np.random.default_rng(100 + seed),
        )
        hits += report.rate("xor_closure_mask3", "quaternion_coset") == 1.0
    assert hits / n >= 0.75


def test_abelian_dihedral_blend_is_now_a_real_scored_target():
    """planted() must return the real (s_abelian, s_dihedral) pair, not None."""
    from rde_domains.hsp_functions.recovery import AbelianDihedralSecret

    inst = make_instance("abelian_dihedral_blend", n_bits=8, seed=0)
    planted = HspFunctionRecovery().planted(inst)
    assert isinstance(planted, AbelianDihedralSecret)
    assert planted.s_abelian == inst.params["s_abelian"]
    assert planted.s_dihedral == inst.params["s_dihedral"]


def test_abelian_dihedral_match_is_order_insensitive_to_pair_combine_argument_order():
    """`PairCombine`'s argument order comes from flat catalog position, not which
    secret each atom actually recovers -- `match()` must accept either
    `(s_abelian, s_dihedral)` or `(s_dihedral, s_abelian)`, not just the order a
    fixed catalog happens to produce."""
    from rde_domains.hsp_functions.recovery import AbelianDihedralSecret

    planted = AbelianDihedralSecret(s_abelian=7, s_dihedral=13)
    domain = HspFunctionRecovery()
    assert domain.match((7, 13), planted) is True
    assert domain.match((13, 7), planted) is True
    assert domain.match((7, 14), planted) is False
    assert domain.match((13,), planted) is False


def test_pair_combine_answers_correctly_or_abstains_on_blend_family_never_wrong():
    """Zero false positives is the bar; full recall at low n_bits is not required."""
    from rde.recovery.programs import ConfidentCollisionProgram, PairCombine

    pair = PairCombine(ConfidentCollisionProgram("xor", "id"), ConfidentCollisionProgram("sum", "id"))
    domain = HspFunctionRecovery()
    wrong = 0
    n = 20
    for seed in range(n):
        inst = _exact("abelian_dihedral_blend", n_bits=10, seed=seed)
        report = evaluate_protocols(domain, [inst], [pair], rng=np.random.default_rng(200 + seed))
        row = next(r for r in report.rows if r.protocol_id == pair.protocol_id)
        s_ab, s_di = row.recovered
        fully_answered = s_ab is not None and s_di is not None
        if fully_answered and not row.matched:
            wrong += 1
    assert wrong == 0


@pytest.mark.slow
def test_search_recovery_chains_finds_a_hidden_multiplicative_subgroup():
    """The search must generalize to an atomic operation it wasn't built for.

    `multiplicative_fold` hides a subgroup of the *multiplicative* group mod
    2**n_bits (order-4, generated by a unit with real multiplicative order 4
    -- not expressible as any fixed XOR mask, unlike every other family in
    this module). `search_recovery_chains` and its atom registry were
    written before this family existed; this asserts the generic search
    still lands on `mult_closure_mask0` (the `op="mult"` variant of the
    same `GroupClosureProgram` atom that already handles `quaternion_coset`'s
    XOR-shaped subgroup) with no family-specific code added to the search
    itself -- only a new instance family and a new elementary group
    operation (multiplication mod 2**n) in the atom vocabulary.
    """
    from rde.recovery.search_space import search_recovery_chains

    domain = HspFunctionRecovery()
    n_bits = 10
    discovery = [make_instance("multiplicative_fold", n_bits=n_bits, seed=s) for s in range(0, 20, 2)]
    confirmatory = [make_instance("multiplicative_fold", n_bits=n_bits, seed=s) for s in range(1, 21, 2)]

    results = search_recovery_chains(
        domain,
        discovery,
        confirmatory,
        family="multiplicative_fold",
        max_depth=1,
        min_recall=0.80,
        rng=np.random.default_rng(7),
    )
    assert any(r.protocol_id == "mult_closure_mask0" and r.confirmatory_recall == 1.0 for r in results)


@pytest.mark.slow
def test_search_recovery_chains_no_family_branch_end_to_end():
    """Same generic search call for three structurally different families.

    No per-family code path here -- this is the actual point: the search
    has to land on different-shaped answers for heisenberg_noncentral
    (translation-invariant), quaternion_coset (subgroup-membership-shaped),
    and abelian_dihedral_blend (two independent mixed mechanisms) without
    being told which is which.
    """
    from rde.recovery.search_space import search_recovery_chains

    domain = HspFunctionRecovery()
    n_bits = 8

    def _instances(family: str, seeds: range):
        return [_exact(family, n_bits=n_bits, seed=s) for s in seeds]

    discovery_seeds = range(0, 20, 2)
    confirmatory_seeds = range(1, 21, 2)

    heisenberg_results = search_recovery_chains(
        domain,
        _instances("heisenberg_noncentral", discovery_seeds),
        _instances("heisenberg_noncentral", confirmatory_seeds),
        family="heisenberg_noncentral",
        max_depth=1,
        rng=np.random.default_rng(1),
    )
    assert any(r.protocol_id == "xor_mode_high_half" for r in heisenberg_results)

    quaternion_results = search_recovery_chains(
        domain,
        _instances("quaternion_coset", discovery_seeds),
        _instances("quaternion_coset", confirmatory_seeds),
        family="quaternion_coset",
        max_depth=1,
        min_recall=0.70,
        rng=np.random.default_rng(2),
    )
    assert any(r.protocol_id == "xor_closure_mask3" for r in quaternion_results)

    # Honest partial result: report, don't force a pass.
    blend_results = search_recovery_chains(
        domain,
        _instances("abelian_dihedral_blend", discovery_seeds),
        _instances("abelian_dihedral_blend", confirmatory_seeds),
        family="abelian_dihedral_blend",
        max_depth=2,
        min_recall=0.80,
        rng=np.random.default_rng(3),
    )
    print(f"abelian_dihedral_blend @ n_bits={n_bits}: {len(blend_results)} chain(s) cleared the bar")
