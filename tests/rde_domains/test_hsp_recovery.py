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
