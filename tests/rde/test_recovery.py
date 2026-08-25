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


def test_group_closure_program_recovers_synthetic_subgroup():
    """A 4-element XOR-closed set {0,1,4,5} collapsing to one label, per 'rest' block."""
    from rde.recovery.programs import GroupClosureProgram

    subgroup = (0, 1, 4, 5)
    xs: list[int] = []
    ys: list[int] = []
    for rest in range(6):
        for k in subgroup:
            xs.append((rest << 3) | k)
            ys.append(rest)
    tape = QueryTape(
        xs=np.array(xs, dtype=np.int64),
        ys=np.array(ys, dtype=np.int64),
        budget=len(xs),
        modulus=1 << 6,
        n_bits=6,
    )
    recovered = GroupClosureProgram(mask_bits=3).extract(tape)
    assert recovered == subgroup


def test_group_closure_program_recovers_multiplicative_subgroup():
    """`op="mult"` reads a hidden subgroup of the *multiplicative* group, not XOR.

    Synthetic, no `hsp_functions` dependency: an order-4 unit `s` mod 64
    (5 generates the cyclic factor of (Z/64Z)*, so 5**4 has order 4) hides
    the subgroup H={1,s,s^2,s^3} the same way `test_group_closure_program_
    recovers_synthetic_subgroup` hides an XOR subgroup -- same atom class,
    different `op`, no family-specific code.
    """
    from rde.recovery.programs import GroupClosureProgram

    modulus = 64
    s = pow(5, 1 << (6 - 4), modulus)
    hidden = tuple(sorted({1, s, pow(s, 2, modulus), pow(s, 3, modulus)}))
    xs: list[int] = []
    ys: list[int] = []
    for label, x0 in enumerate((3, 7, 9, 11, 13, 15, 19, 21)):
        for x in sorted({(x0 * h) % modulus for h in hidden}):
            xs.append(x)
            ys.append(label)
    tape = QueryTape(
        xs=np.array(xs, dtype=np.int64),
        ys=np.array(ys, dtype=np.int64),
        budget=len(xs),
        modulus=modulus,
        n_bits=6,
    )
    assert GroupClosureProgram(mask_bits=0, op="mult").extract(tape) == hidden


def test_group_closure_program_abstains_without_a_closed_group():
    from rde.recovery.programs import GroupClosureProgram

    rng = np.random.default_rng(0)
    xs = rng.integers(0, 64, size=200, dtype=np.int64)
    ys = xs  # bijective: no collisions at all, so no candidate group exists
    tape = QueryTape(xs=xs, ys=ys, budget=200, modulus=64, n_bits=6)
    assert GroupClosureProgram(mask_bits=3).extract(tape) is None


def test_confident_collision_program_abstains_on_weak_plurality_but_plain_mode_guesses():
    from rde.recovery.programs import CollisionProgram, ConfidentCollisionProgram

    # Four distinct xor values, no clear winner (plurality 1/4).
    groups_xor = [17, 17, 33, 49]
    xs: list[int] = []
    ys: list[int] = []
    for i, v in enumerate(groups_xor):
        base = 100 + i * 10
        xs.extend([base, base ^ v])
        ys.extend([i, i])
    tape = QueryTape(
        xs=np.array(xs, dtype=np.int64),
        ys=np.array(ys, dtype=np.int64),
        budget=len(xs),
        modulus=1 << 8,
        n_bits=8,
    )
    assert ConfidentCollisionProgram("xor", "id").extract(tape) is None
    assert CollisionProgram("xor", "mode", "id").extract(tape) is not None


def test_confident_collision_program_answers_on_clear_majority():
    from rde.recovery.programs import ConfidentCollisionProgram

    xs: list[int] = []
    ys: list[int] = []
    for i in range(10):
        base = 100 + i * 10
        xs.extend([base, base ^ 42])
        ys.extend([i, i])
    tape = QueryTape(
        xs=np.array(xs, dtype=np.int64),
        ys=np.array(ys, dtype=np.int64),
        budget=len(xs),
        modulus=1 << 8,
        n_bits=8,
    )
    assert ConfidentCollisionProgram("xor", "id").extract(tape) == 42


def test_pair_combine_extracts_both_pipelines_independently():
    from rde.recovery.programs import ConfidentCollisionProgram, PairCombine

    mod = 1 << 8
    xs: list[int] = []
    ys: list[int] = []
    label = 0
    # xor-branch collisions: constant xor = 11. Bases chosen so the sums
    # they incidentally contribute to the *other* pool (below) are 8
    # distinct values, none equal to 77 -- pure dispersed noise there, not
    # an accidental second signal.
    for base in (1, 4, 16, 20, 32, 36, 48, 52):
        xs.extend([base, base ^ 11])
        ys.extend([label, label])
        label += 1
    # sum-branch collisions: constant sum (mod) = 77. Bases chosen so the
    # xors they incidentally contribute to the xor pool are 8 distinct
    # values, none equal to 11.
    for base in (2, 5, 6, 15, 17, 18, 22, 33):
        xs.extend([base, (77 - base) % mod])
        ys.extend([label, label])
        label += 1
    tape = QueryTape(
        xs=np.array(xs, dtype=np.int64),
        ys=np.array(ys, dtype=np.int64),
        budget=len(xs),
        modulus=mod,
        n_bits=8,
    )
    pair = PairCombine(ConfidentCollisionProgram("xor", "id"), ConfidentCollisionProgram("sum", "id"))
    recovered = pair.extract(tape)
    assert recovered == (11, 77)
    assert pair.protocol_id == "pair[xor_mode_confident_id|sum_mode_confident_id]"


def test_enumerate_recovery_chains_depth1_ids_unique_and_grow_with_depth2():
    from rde.recovery.search_space import enumerate_recovery_chains

    depth1 = enumerate_recovery_chains(max_depth=1)
    depth2 = enumerate_recovery_chains(max_depth=2)
    ids1 = [p.protocol_id for p in depth1]
    ids2 = [p.protocol_id for p in depth2]
    assert len(set(ids1)) == len(ids1)
    assert len(set(ids2)) == len(ids2)
    assert len(depth2) > len(depth1)
    assert set(ids1).issubset(set(ids2))


def test_search_recovery_chains_drops_a_chain_that_fails_on_holdout():
    """A chain that only clears the bar by discovery-split luck must not survive confirmation."""
    from rde.recovery.programs import PairCombine
    from rde.recovery.search_space import search_recovery_chains

    class _FlukyDomain:
        domain_id = "fluky_test"

        def size_of(self, instance) -> int:
            return int(instance["n_bits"])

        def family_of(self, instance) -> str:
            return "target"

        def draw_tape(self, instance, rng: np.random.Generator) -> QueryTape:
            n_bits = instance["n_bits"]
            mod = 1 << n_bits
            xs: list[int] = []
            ys: list[int] = []
            if instance["seed"] % 2 == 0:
                # Discovery (even seed): looks like a clean xor=42 signal.
                for i in range(20):
                    base = int(rng.integers(0, mod))
                    xs.extend([base, base ^ 42])
                    ys.extend([i, i])
            else:
                # Confirmatory (odd seed): pure noise, no real xor=42 structure.
                xs = list(rng.integers(0, mod, size=40))
                ys = list(range(40))
            return QueryTape(
                xs=np.array(xs, dtype=np.int64),
                ys=np.array(ys, dtype=np.int64),
                budget=len(xs),
                modulus=mod,
                n_bits=n_bits,
            )

        def planted(self, instance):
            # Always 42: the discovery split's apparent signal is a fluke,
            # not a real property that changes between splits -- the
            # confirmatory tapes below deliberately carry no such structure,
            # so a chain that only "worked" on discovery must fail here.
            return 42

        def match(self, recovered, planted) -> bool:
            return recovered == planted

    discovery = [{"n_bits": 8, "seed": s} for s in (0, 2, 4, 6, 8, 10)]
    confirmatory = [{"n_bits": 8, "seed": s} for s in (1, 3, 5, 7, 9, 11)]
    results = search_recovery_chains(
        _FlukyDomain(),
        discovery,
        confirmatory,
        family="target",
        max_depth=1,
        min_recall=0.80,
        rng=np.random.default_rng(0),
    )
    protocol_ids = {r.protocol_id for r in results}
    assert "xor_mode_id" not in protocol_ids


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
