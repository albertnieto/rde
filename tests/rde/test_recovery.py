"""Core recovery loop: extractors never see the planted secret."""

from __future__ import annotations

import numpy as np

from rde.core.instance import InstanceRecord
from rde.core.protocols import QueryTape
from rde.recovery import (
    AdditiveGcdExtractor,
    XorCollisionExtractor,
    default_extractor_catalog,
    evaluate_protocols,
)


class _XorHideRecovery:
    domain_id = "xor_hide_test"

    def size_of(self, instance: InstanceRecord) -> int:
        return int(instance.size)

    def family_of(self, instance: InstanceRecord) -> str:
        return str(instance.params["family"])

    def draw_tape(self, instance: InstanceRecord, rng: np.random.Generator) -> QueryTape:
        n_bits = instance.size
        modulus = 1 << n_bits
        budget = max(64, 20 * n_bits * n_bits)
        xs = rng.integers(0, modulus, size=budget, dtype=np.int64)
        family = instance.params["family"]
        if family == "simon":
            s = int(instance.params["s"])
            labels = np.minimum(xs, xs ^ s)
        else:
            labels = xs
        return QueryTape(xs=xs, ys=labels, budget=budget, modulus=modulus, n_bits=n_bits)

    def planted(self, instance: InstanceRecord) -> int | None:
        if instance.params["family"] == "simon":
            return int(instance.params["s"])
        return None

    def match(self, recovered, planted) -> bool:
        if planted is None:
            return recovered is None
        return recovered == planted


def _instances() -> list[InstanceRecord]:
    return [
        InstanceRecord("xor_hide_test", 8, 1, {"family": "simon", "s": 0b10110011}),
        InstanceRecord("xor_hide_test", 8, 2, {"family": "junk"}),
    ]


def test_xor_extractor_recovers_hidden_string_and_abstains_on_junk():
    rng = np.random.default_rng(0)
    report = evaluate_protocols(
        _XorHideRecovery(),
        _instances(),
        default_extractor_catalog(),
        rng=rng,
    )
    assert report.rate("xor_collision_mode", "simon") == 1.0
    assert report.rate("xor_collision_mode", "junk") == 1.0
    gcd_on_simon = report.rate("additive_gcd", "simon")
    assert gcd_on_simon < 1.0


def test_extractor_does_not_receive_planted_secret():
    seen = {}

    class Probe:
        protocol_id = "probe"

        def extract(self, tape: QueryTape):
            seen["has_s_attr"] = hasattr(tape, "s")
            seen["n"] = tape.n_bits
            return XorCollisionExtractor().extract(tape)

    inst = InstanceRecord("xor_hide_test", 8, 3, {"family": "simon", "s": 7})
    evaluate_protocols(_XorHideRecovery(), [inst], [Probe()], rng=np.random.default_rng(1))
    assert seen["has_s_attr"] is False
    assert seen["n"] == 8


def test_additive_gcd_recovers_even_period():
    class PeriodHide:
        domain_id = "period_hide_test"

        def size_of(self, instance: InstanceRecord) -> int:
            return int(instance.size)

        def family_of(self, instance: InstanceRecord) -> str:
            return "cyclic"

        def draw_tape(self, instance: InstanceRecord, rng: np.random.Generator) -> QueryTape:
            n_bits = instance.size
            modulus = 1 << n_bits
            r = int(instance.params["r"])
            budget = max(64, 20 * n_bits * n_bits)
            xs = rng.integers(0, modulus, size=budget, dtype=np.int64)
            return QueryTape(xs=xs, ys=xs % r, budget=budget, modulus=modulus, n_bits=n_bits)

        def planted(self, instance: InstanceRecord) -> int:
            return int(instance.params["r"])

        def match(self, recovered, planted) -> bool:
            return recovered == planted

    inst = InstanceRecord("period_hide_test", 8, 0, {"r": 128})
    report = evaluate_protocols(
        PeriodHide(),
        [inst],
        [AdditiveGcdExtractor()],
        rng=np.random.default_rng(4),
    )
    assert report.rate("additive_gcd", "cyclic") == 1.0


def test_enumerate_recovery_programs_is_larger_than_pipeline():
    from rde.recovery.programs import enumerate_recovery_programs

    ids = [p.protocol_id for p in enumerate_recovery_programs()]
    assert len(set(ids)) == len(ids)
    assert len(ids) > 10


def test_assess_protocol_search_null_without_discovery_lead():
    from rde.recovery.campaign import CONFIRMATORY_SIZES, assess_protocol_search
    from rde.recovery.search import RecoveryRow

    rows: list[RecoveryRow] = []
    for size in CONFIRMATORY_SIZES:
        for family, proto in (
            ("simon", "xor_collision_mode"),
            ("shor_cyclic", "additive_gcd"),
            ("dihedral_kuperberg", "additive_sum_mode"),
        ):
            for seed in (0, 1):
                rows.append(
                    RecoveryRow(proto, family, size, seed, True, 1, 1, 64)
                )
        rows.append(
            RecoveryRow("xor_collision_mode", "heisenberg_noncentral", size, 1, False, None, 2, 64)
        )
    verdict = assess_protocol_search(rows)
    assert verdict.pipeline_ok is True
    assert verdict.verdict == "NULL"
    assert verdict.grade == 0


def test_recovery_receipt_validator_accepts_extractor_blindness():
    from rde.experiment.gate import RECEIPT_VERSION, RECOVERY_REQUIRED_PHASES, validate_receipt

    phases = {p: {"work_units": 1, "work_kind": "x"} for p in RECOVERY_REQUIRED_PHASES}
    payload = {
        "receipt_version": RECEIPT_VERSION,
        "gate_kind": "recovery",
        "domain_id": "hsp_functions",
        "target": "recovery.exact_k",
        "preregistration_sha256": "abc",
        "verdict": "NULL",
        "phases": phases,
        "population": {
            "per_size": {"8": {"distinct_structural_instances": 100, "required_distinct": 50}},
        },
        "leak_audit": {
            "kind": "extractor_blindness",
            "extract_sees_planted": False,
            "extract_sees_family": False,
        },
        "criteria_audit": {"decisive": ["pipeline_ok", "n_confirmed"], "vacuous": []},
    }
    assert validate_receipt(payload) == []
