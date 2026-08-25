"""Tests for the hsp_functions RDE domain."""

from __future__ import annotations

import numpy as np
import pytest

from rde.core.domain_contract import domain_contract
from rde.core.feature_contract import catalog_for_domain
from rde.core.plugins import build_registry
from rde.features import boolean as boolean_features
from rde_domains.hsp_functions import functions, sampling
from rde_domains.hsp_functions.domain import HspFunctionDomain


# ---------------------------------------------------------------------------
# Boolean-feature primitives (ALGO-061)
# ---------------------------------------------------------------------------


def test_mobius_transform_recovers_known_anf():
    # f(x0,x1) = x0 XOR x1 -- degree 1, ANF = {x0: 1, x1: 1}, no constant/product term.
    table = np.array([0, 1, 1, 0], dtype=np.uint8)  # index bit0=x0, bit1=x1
    anf = boolean_features.mobius_transform_gf2(table)
    assert anf.tolist() == [0, 1, 1, 0]
    assert boolean_features.exact_algebraic_degree(table) == 1


def test_algebraic_degree_of_and_is_two():
    # f(x0,x1) = x0 AND x1 -- degree 2 (the x0*x1 monomial fires).
    table = np.array([0, 0, 0, 1], dtype=np.uint8)
    assert boolean_features.exact_algebraic_degree(table) == 2


def test_gf2_rank_known_cases():
    assert boolean_features.gf2_rank(np.zeros((3, 4), dtype=np.uint8)) == 0
    identity_rows = np.eye(4, dtype=np.uint8)
    assert boolean_features.gf2_rank(identity_rows) == 4
    dependent = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.uint8)
    assert boolean_features.gf2_rank(dependent) == 2


# ---------------------------------------------------------------------------
# Function-family correctness (ALGO-062)
# ---------------------------------------------------------------------------


def test_simon_exact_coset_invariance():
    inst = functions.make_instance("simon", n_bits=6, seed=7)
    s = inst.params["s"]
    assert s != 0
    for x in range(inst.x_size):
        assert inst.evaluate(x) == inst.evaluate(x ^ s)
    # Injective on cosets: distinct {x, x^s} pairs get distinct labels.
    labels = {inst.evaluate(x) for x in range(inst.x_size)}
    assert len(labels) == inst.x_size // 2
    assert inst.structure_strength == 1.0


def test_shor_cyclic_exact_period_invariance():
    inst = functions.make_instance("shor_cyclic", n_bits=6, seed=11)
    r = inst.params["r"]
    for x in range(inst.x_size):
        assert inst.evaluate(x) == inst.evaluate((x + r) % inst.x_size)
    assert inst.structure_strength == 1.0


def test_dihedral_exact_reflection_invariance():
    inst = functions.make_instance("dihedral_kuperberg", n_bits=6, seed=13)
    s = inst.params["s"]
    for x in range(inst.x_size):
        assert inst.evaluate(x) == inst.evaluate((s - x) % inst.x_size)
    assert inst.structure_strength == 1.0


def test_generic_random_control_has_zero_structure_strength():
    inst = functions.make_instance("generic_random_control", n_bits=6, seed=3)
    assert inst.structure_strength == 0.0


def test_structure_break_abelian_strength_matches_break_param():
    inst = functions.make_instance("structure_break_abelian", n_bits=6, seed=5)
    break_frac = inst.params["structure_break"]
    assert inst.structure_strength == pytest.approx(1.0 - break_frac)
    assert 0.0 <= break_frac < 0.85


def test_abelian_dihedral_blend_strength_is_coherence():
    inst = functions.make_instance("abelian_dihedral_blend", n_bits=6, seed=9)
    w = inst.params["blend_weight"]
    expected = 2.0 * max(w, 1.0 - w) - 1.0
    assert inst.structure_strength == pytest.approx(expected)
    assert 0.0 <= inst.structure_strength <= 1.0


def test_evaluate_batch_matches_scalar_evaluate():
    for family in functions.KNOWN_FAMILIES:
        if family == functions.RECIPE_FAMILY:
            continue
        inst = functions.make_instance(family, n_bits=9 if family != "quaternion_coset" else 8, seed=55)
        xs = np.array([0, 1, 2, 3, 17, 100, 255, inst.x_size - 1], dtype=np.int64)
        scalar = np.array([inst.evaluate(int(x)) for x in xs], dtype=np.uint64)
        batch = inst.evaluate_batch(xs)
        assert np.array_equal(scalar, batch), f"batch/scalar mismatch for family={family!r}"


def _exact_phase3(family: str, n_bits: int, seed: int) -> functions.FunctionInstance:
    inst = functions.make_instance(family, n_bits=n_bits, seed=seed)
    params = dict(inst.params)
    params["structure_break"] = 0.0
    params["structure_strength"] = 1.0
    return functions.FunctionInstance(
        inst.family, inst.domain_kind, inst.n_bits, inst.x_size, inst.seed, params
    )


def _collision_pair_stats(inst: functions.FunctionInstance) -> tuple[set[int], set[int], set[int]]:
    labels: dict[int, list[int]] = {}
    for x in range(inst.x_size):
        labels.setdefault(inst.evaluate(x), []).append(x)
    xor_diffs: set[int] = set()
    add_diffs: set[int] = set()
    sums: set[int] = set()
    for members in labels.values():
        base = members[0]
        for other in members[1:]:
            xor_diffs.add(base ^ other)
            add_diffs.add((other - base) % inst.x_size)
            sums.add((base + other) % inst.x_size)
    return xor_diffs, add_diffs, sums


def test_heisenberg_noncentral_invariance_is_not_constant_xor():
    inst = _exact_phase3("heisenberg_noncentral", n_bits=8, seed=7)
    v = inst.params["v"]
    assert v != 0
    assert (v & 1) == 0 and (v & 2) == 2
    for x in range(inst.x_size):
        partner = functions.heisenberg_right_mul_k(x, inst.n_bits, v)
        assert inst.evaluate(x) == inst.evaluate(partner)
        assert partner != x
        twice = functions.heisenberg_right_mul_k(partner, inst.n_bits, v)
        assert twice == x
    xor_diffs, add_diffs, sums = _collision_pair_stats(inst)
    assert 0 not in xor_diffs
    assert len(xor_diffs) > 1, "Heisenberg pairing must not be XOR with a fixed string"
    assert len(add_diffs) > 1
    assert len(sums) > 1
    labels = {inst.evaluate(x) for x in range(inst.x_size)}
    assert len(labels) == inst.x_size // 2


def test_quaternion_coset_invariance_is_not_constant_xor():
    inst = _exact_phase3("quaternion_coset", n_bits=8, seed=11)
    for x in range(inst.x_size):
        for k_q in functions._Q8_HIDDEN:
            assert inst.evaluate(x) == inst.evaluate(functions.quaternion_right_mul(x, k_q))
    xor_diffs, add_diffs, _sums = _collision_pair_stats(inst)
    assert len(xor_diffs) > 1, "Q8 coset pairing must not be XOR with a fixed string"
    assert len(add_diffs) > 1
    labels = {inst.evaluate(x) for x in range(inst.x_size)}
    assert len(labels) == inst.x_size // 4


def test_domain_families_override_round_robins_phase3_roster():
    roster = functions.PHASE3_POPULATION
    dom = HspFunctionDomain(families=roster)
    instances = dom.generate(n=len(roster) * 2, size=8, seed=100)
    assert {inst.params["family"] for inst in instances} == set(roster)
    assert set(functions.ALL_FAMILIES) != set(roster)


def test_deterministic_given_seed():
    a = functions.make_instance("simon", n_bits=6, seed=42)
    b = functions.make_instance("simon", n_bits=6, seed=42)
    assert a.params == b.params
    assert all(a.evaluate(x) == b.evaluate(x) for x in range(a.x_size))


def test_query_budget_never_scales_with_x_size():
    # poly(n_bits) can exceed x_size at small n (expected -- the budget is a
    # function of n_bits alone); the asymptotic separation only has to hold
    # once n_bits is large enough that poly(n) << 2**n_bits.
    assert sampling.query_budget_for(20) < (1 << 20)
    assert sampling.query_budget_for(24) < (1 << 24)
    assert sampling.query_budget_for(20) == sampling.query_budget_for(20)  # pure function of n_bits
    assert sampling.query_budget_for(24) == 20 * 24 * 24
    assert sampling.simon_pair_expectation(24) >= sampling.MIN_SIMON_PAIR_EXPECTATION
    # c=8 (an earlier, smaller calibration budget) is below the confirmatory-horizon bar.
    assert sampling.simon_pair_expectation(24, budget=8 * 24 * 24) < 1.0


def test_n24_simon_collision_rate_detects_with_calibrated_budget():
    """Hidden-shift collisions must fire at n=24 under the calibrated O(n^2) budget."""
    hits = 0
    n_trials = 16
    for i in range(n_trials):
        inst = functions.make_instance("simon", n_bits=24, seed=2000 + i)
        rng = np.random.default_rng(2000 + i)
        desc = sampling.bounded_query_descriptors(inst, rng, name="f")
        if desc["hsp_sample.f.collision_rate"] > 0.0:
            hits += 1
    # λ≈3.95 ⇒ P(miss)≈0.02; 16 independent misses is not plausible.
    assert hits >= 13, f"simon n=24 detections={hits}/{n_trials}"


# ---------------------------------------------------------------------------
# Bounded-query descriptors detect planted structure (sanity, not a claim)
# ---------------------------------------------------------------------------


def test_bounded_query_descriptors_detect_exact_simon_structure():
    inst = functions.make_instance("simon", n_bits=10, seed=21)
    rng = np.random.default_rng(0)
    desc = sampling.bounded_query_descriptors(inst, rng, name="f")
    # |K_true|=2 -> a correctly-recovered difference span has rank 1.
    assert desc["hsp_sample.f.difference_span_dim_fraction"] == pytest.approx(1.0 / inst.n_bits)


def test_bounded_query_descriptors_find_no_structure_in_control():
    inst = functions.make_instance("generic_random_control", n_bits=10, seed=23)
    rng = np.random.default_rng(1)
    desc = sampling.bounded_query_descriptors(inst, rng, name="f")
    # With a 63-bit label space and a poly(n) query budget, a genuine
    # collision among independently-hashed labels is astronomically
    # unlikely -- the control should show no detected structure.
    assert desc["hsp_sample.f.n_collisions_found"] == 0.0
    assert desc["hsp_sample.f.difference_span_dim_fraction"] == 0.0


def test_exact_oracle_audit_matches_bounded_signal_direction():
    inst = functions.make_instance("simon", n_bits=8, seed=27)
    exact = sampling.exact_oracle_audit_descriptors(inst, name="f")
    control = functions.make_instance("generic_random_control", n_bits=8, seed=27)
    exact_control = sampling.exact_oracle_audit_descriptors(control, name="f")
    # An exact hidden-shift structure is far sparser in the Walsh domain
    # than a structureless control (Gate-0 mechanism check).
    assert exact["hsp_oracle.f.sparsity"] > exact_control["hsp_oracle.f.sparsity"]


def test_oracle_audit_gated_off_above_max_n_bits():
    inst = functions.make_instance("simon", n_bits=sampling.MAX_ORACLE_N_BITS + 2, seed=1)
    assert sampling.exact_oracle_audit_descriptors(inst) == {}


# ---------------------------------------------------------------------------
# Domain protocol + contract + registration
# ---------------------------------------------------------------------------


def test_domain_generate_round_robins_families_and_is_distinct():
    dom = HspFunctionDomain()
    instances = dom.generate(n=len(functions.ALL_FAMILIES) * 3, size=8, seed=100)
    families_seen = {inst.params["family"] for inst in instances}
    assert families_seen == set(functions.ALL_FAMILIES)
    # Distinct instances -> distinct structural feature vectors, not one
    # landscape reused (the anti-pattern this project's playbook forbids).
    strengths = {inst.params["structure_strength"] for inst in instances}
    assert len(strengths) > 1


def test_domain_materialize_and_primitive_features():
    dom = HspFunctionDomain()
    [inst] = dom.generate(n=1, size=8, seed=200)
    cache: dict = {}
    sl = dom.materialize(inst, 0, cache=cache)
    assert sl.values.ndim == 1
    assert sl.values.size == inst.params["n_bits"]
    feats = dom.primitive_features(inst, cache=cache)
    assert feats["structure_strength"] == inst.params["structure_strength"]
    assert any(k.startswith("hsp_sample.") for k in feats)


def test_prepare_instance_runs_difference_estimates_once(monkeypatch):
    calls = {"n": 0}
    real = sampling.sample_difference_estimates

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(sampling, "sample_difference_estimates", wrapped)
    dom = HspFunctionDomain()
    [inst] = dom.generate(n=1, size=8, seed=201)
    cache = dom.prepare_instance(inst)
    assert calls["n"] == 1
    sl = dom.materialize(inst, 0, cache=cache)
    feats = dom.primitive_features(inst, cache=cache)
    assert calls["n"] == 1
    np.testing.assert_allclose(sl.values, feats["diff_profile"])
    assert "sample_desc" in cache
    assert cache["sample_desc"]["hsp_sample.f.query_budget"] == feats["hsp_sample.f.query_budget"]


def test_worker_process_instance_samples_difference_estimates_once(monkeypatch):
    from rde.runtime.worker import process_instance

    calls = {"n": 0}
    real = sampling.sample_difference_estimates

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(sampling, "sample_difference_estimates", wrapped)
    reg = build_registry("hsp_functions")
    dom = reg.get_domain("hsp_functions")
    [inst] = dom.generate(n=1, size=6, seed=7)
    process_instance(
        dom,
        reg,
        inst,
        indices=[0],
        pending_indices=[0],
        descriptor_names=[],
        metric_names=[],
        run_id="hsp-cache-contract",
        write_instance=False,
        write_instance_features=True,
        save_arrays=False,
    )
    assert calls["n"] == 1


def test_bounded_query_descriptors_reuse_cached_difference_estimates(monkeypatch):
    inst = functions.make_instance("simon", n_bits=8, seed=31)
    rng = np.random.default_rng(0)
    estimates = sampling.sample_difference_estimates(inst, rng)

    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("sample_difference_estimates must not rerun when cached")

    monkeypatch.setattr(sampling, "sample_difference_estimates", _should_not_run)
    desc = sampling.bounded_query_descriptors(
        inst, np.random.default_rng(1), name="f", diff_estimates=estimates
    )
    assert "hsp_sample.f.collision_rate" in desc
    for idx, g_val in enumerate(estimates.values()):
        assert desc[f"hsp_sample.f.diff_g_{idx}"] == g_val


def test_domain_registered_via_plugin_loader():
    reg = build_registry("hsp_functions")
    dom = reg.get_domain("hsp_functions")
    assert dom.domain_id == "hsp_functions"
    metric = reg.get_metric("structure_strength")
    row = metric.score(None, None, {"structure_strength": 0.42})
    assert row == pytest.approx(0.42)


def test_hsp_functions_contract_shape():
    c = domain_contract("hsp_functions")
    assert c.primary_target == "metric.structure_strength"
    assert set(c.held_out_generator_groups) == set(functions.FAMILIES_HELD_OUT)
    cat = catalog_for_domain("hsp_functions")
    assert cat.is_predictor_eligible("hsp_sample.f.collision_rate")
    assert not cat.is_predictor_eligible("hsp_oracle.f.sparsity")
    assert not cat.is_predictor_eligible("metric.structure_strength")
    assert not cat.is_predictor_eligible("structure_strength")
    assert not cat.is_predictor_eligible("metric.algorithm_class")
    assert "metric.algorithm_class" in c.secondary_targets


def test_hsp_population_gate_distinctness_at_confirmatory_scale():
    """1500 independent instances at N=8 must pass the 50% distinct bar."""
    from rde.experiment.gate import distinct_structural_instances, structural_columns

    dom = HspFunctionDomain()
    instances = dom.generate(n=1500, size=8, seed=8)
    rows: list[dict] = []
    for inst in instances:
        feats = dom.primitive_features(inst)
        row = {"size": inst.size, "generator": inst.params["family"]}
        for key, val in feats.items():
            if isinstance(val, (int, float)):
                row[key] = float(val)
        rows.append(row)
    cols = structural_columns(rows)
    distinct = distinct_structural_instances(rows, cols)
    needed = max(2, int(0.5 * len(rows)))
    assert distinct >= needed, f"distinct={distinct} needed={needed}"
    generic = [r for r in rows if r.get("generator") == "generic_random_control"]
    assert distinct_structural_instances(generic, cols) >= len(generic) // 2


def test_bounded_query_label_stats_vary_for_structureless_control():
    inst_a = functions.make_instance("generic_random_control", n_bits=8, seed=13)
    inst_b = functions.make_instance("generic_random_control", n_bits=8, seed=19)
    rng = np.random.default_rng(0)
    desc_a = sampling.bounded_query_descriptors(inst_a, rng, name="f")
    desc_b = sampling.bounded_query_descriptors(inst_b, rng, name="f")
    assert desc_a["hsp_sample.f.unique_label_fraction"] != desc_b["hsp_sample.f.unique_label_fraction"] or (
        desc_a["hsp_sample.f.label_mean_normalized"] != desc_b["hsp_sample.f.label_mean_normalized"]
    )


def test_hsp_pipeline_allows_n_bits_above_default_bruteforce_cap(tmp_path):
    """n_bits=16 must not hit the generic max_bruteforce_n=14 guard."""
    from rde.runtime.pipeline import RunConfig, run_pipeline

    reg = build_registry("hsp_functions")
    dom = reg.get_domain("hsp_functions")
    assert getattr(dom, "bruteforce_enumeration", True) is False

    result = run_pipeline(
        RunConfig(
            domain_id="hsp_functions",
            n_instances=2,
            size=16,
            seed=0,
            indices=[0],
            store_root=tmp_path,
            run_id="hsp_n16",
        ),
        registry=reg,
    )
    assert result.n_feature_rows == 2


def _screen_row(family: str, size: int, rate: float, strength: float, idx: int) -> dict:
    return {
        "instance_id": f"{family}_{size}_{idx}",
        "size": size,
        "generator": family,
        "hsp_sample.f.collision_rate": rate,
        "metric.structure_strength": strength,
    }


def test_heldout_screen_passes_on_size_normalized_separation():
    from rde_domains.hsp_functions.heldout_screen import evaluate_heldout_screen

    rows: list[dict] = []
    for size in (8, 12):
        for i in range(12):
            rows.append(_screen_row("generic_random_control", size, 0.0, 0.0, i))
            rows.append(_screen_row("structure_break_abelian", size, 0.25, 1.0, i))
            rows.append(_screen_row("abelian_dihedral_blend", size, 0.20, 0.7, i))
            rows.append(_screen_row("simon", size, 0.30, 1.0, i))
            rows.append(_screen_row("shor_cyclic", size, 0.28, 1.0, i))
            rows.append(_screen_row("dihedral_kuperberg", size, 0.22, 1.0, i))
    result = evaluate_heldout_screen(rows)
    assert result["passed"] is True
    assert result["discovery_abs_r"] >= 0.35
    for stats in result["per_n"].values():
        assert stats["recall"] >= 0.80
        assert stats["fpr"] <= 0.05


def test_heldout_screen_fails_when_heldout_matches_random():
    from rde_domains.hsp_functions.heldout_screen import evaluate_heldout_screen

    rows: list[dict] = []
    for size in (8, 12):
        for i in range(12):
            rows.append(_screen_row("generic_random_control", size, 0.0, 0.0, i))
            rows.append(_screen_row("structure_break_abelian", size, 0.25, 1.0, i))
            rows.append(_screen_row("abelian_dihedral_blend", size, 0.20, 0.7, i))
            rows.append(_screen_row("simon", size, 0.0, 1.0, i))
            rows.append(_screen_row("shor_cyclic", size, 0.0, 1.0, i))
            rows.append(_screen_row("dihedral_kuperberg", size, 0.0, 1.0, i))
    result = evaluate_heldout_screen(rows)
    assert result["passed"] is False
    assert all(stats["recall"] == 0.0 for stats in result["per_n"].values())


def test_heldout_screen_accepts_phase3_discovery_roster():
    from rde_domains.hsp_functions.heldout_screen import evaluate_heldout_screen
    from rde_domains.hsp_functions.functions import FAMILIES_PHASE3_DISCOVERY

    rows: list[dict] = []
    for size in (8, 12):
        for i in range(12):
            rows.append(_screen_row("generic_random_control", size, 0.0, 0.0, i))
            rows.append(_screen_row("heisenberg_noncentral", size, 0.24, 1.0, i))
            rows.append(_screen_row("quaternion_coset", size, 0.22, 0.8, i))
            rows.append(_screen_row("simon", size, 0.30, 1.0, i))
            rows.append(_screen_row("shor_cyclic", size, 0.28, 1.0, i))
            rows.append(_screen_row("dihedral_kuperberg", size, 0.22, 1.0, i))
    result = evaluate_heldout_screen(rows, discovery_families=FAMILIES_PHASE3_DISCOVERY)
    assert result["passed"] is True
    assert result["discovery_families"] == list(FAMILIES_PHASE3_DISCOVERY)


def test_recipe_catalog_is_10k_and_biased_toward_structure():
    from rde_domains.hsp_functions.recipes import (
        N_BLEND,
        N_RANDOM,
        N_STRUCTURED,
        decode_recipe,
        draw_recipe_ids,
    )
    from rde_domains.hsp_functions.functions import N_RECIPES_DEFAULT

    assert N_RECIPES_DEFAULT == 10_000
    assert N_STRUCTURED + N_BLEND + N_RANDOM == 10_000
    tiers = [decode_recipe(i)["usefulness_tier"] for i in range(10_000)]
    assert tiers.count("hsp_gap") == 8_000
    assert tiers.count("blend") == 1_000
    assert tiers.count("control") == 1_000
    exact = sum(1 for i in range(N_STRUCTURED) if decode_recipe(i)["structure_break"] == 0.0)
    assert exact / N_STRUCTURED >= 0.75
    ids = draw_recipe_ids(51, seed=8)
    assert len(ids) == 51
    bands = {
        "struct": sum(i < N_STRUCTURED for i in ids),
        "blend": sum(N_STRUCTURED <= i < N_STRUCTURED + N_BLEND for i in ids),
        "random": sum(i >= N_STRUCTURED + N_BLEND for i in ids),
    }
    assert bands["struct"] >= 35
    assert bands["blend"] >= 4
    assert bands["random"] >= 4


def test_recipe_xor_rank2_is_not_single_xor_and_batch_matches():
    from rde_domains.hsp_functions.recipes import cyclic_period, make_recipe_instance

    # recipe_id=5: xor, hidden_rank=2, structure_break=0 (see decode_recipe).
    inst = make_recipe_instance(8, seed=7, recipe_id=5)
    assert inst.params["pairing"] == "xor"
    assert inst.params["hidden_rank"] == 2
    assert inst.params["structure_break"] == pytest.approx(0.0)
    gens = tuple(inst.params["gens"])
    assert len(gens) == 2
    labels = {inst.evaluate(x) for x in range(inst.x_size)}
    assert len(labels) == inst.x_size // 4
    xs = np.array([0, 1, 17, 100, 255], dtype=np.int64)
    assert np.array_equal(
        np.array([inst.evaluate(int(x)) for x in xs], dtype=np.uint64),
        inst.evaluate_batch(xs),
    )
    rank3 = make_recipe_instance(8, seed=7, recipe_id=10)
    assert rank3.params["pairing"] == "xor"
    assert rank3.params["hidden_rank"] == 3
    labels3 = {rank3.evaluate(x) for x in range(rank3.x_size)}
    assert len(labels3) == rank3.x_size // 8
    periods = {cyclic_period(8, 1, klass) for klass in range(4)}
    assert len(periods) >= 3
    assert 128 in periods


def test_domain_recipe_catalog_roundtrip_and_heldout_exam():
    from rde_domains.hsp_functions.functions import FAMILIES_HELD_OUT, RECIPE_FAMILY
    from rde_domains.hsp_functions.recipes import make_recipe_instance, required_recipe_generators

    dom = HspFunctionDomain(recipe_catalog_size=10_000, exam_fraction=0.15)
    instances = dom.generate(n=60, size=8, seed=8)
    exam = [inst for inst in instances if inst.params["family"] in FAMILIES_HELD_OUT]
    recs = [inst for inst in instances if inst.params["family"] == RECIPE_FAMILY]
    assert exam and recs
    gens = {str(inst.params["generator"]) for inst in instances}
    assert set(required_recipe_generators()).issubset(gens)
    rec = recs[0]
    fi = make_recipe_instance(rec.params["n_bits"], rec.seed, int(rec.params["recipe_id"]))
    assert dom._function_instance(rec).evaluate(3) == fi.evaluate(3)
    cache = dom.prepare_instance(rec)
    assert "hsp_recipe.hidden_rank" in dom.primitive_features(rec, cache=cache)


def test_pairing_shape_summary_is_diagnostic_not_a_gate():
    from rde_domains.hsp_functions.heldout_screen import summarize_pairing_shape

    rows = [
        {
            "size": 8,
            "generator": "simon",
            "hsp_sample.f.collision_rate": 0.1,
            "hsp_sample.f.difference_span_dim_fraction": 0.125,
        },
        {
            "size": 8,
            "generator": "heisenberg_noncentral",
            "hsp_sample.f.collision_rate": 0.04,
            "hsp_sample.f.difference_span_dim_fraction": 0.25,
        },
        {
            "size": 8,
            "generator": "shor_cyclic",
            "hsp_sample.f.collision_rate": 0.1,
        },
    ]
    result = summarize_pairing_shape(rows, families=("simon", "heisenberg_noncentral", "shor_cyclic"))
    assert result["gated"] is False
    assert result["per_n"]["8"]["simon"]["mean_span_fraction"] == pytest.approx(0.125)
    assert result["per_n"]["8"]["heisenberg_noncentral"]["mean_span_fraction"] == pytest.approx(0.25)
    assert np.isnan(result["per_n"]["8"]["shor_cyclic"]["mean_span_fraction"])
    assert "per_n_by_rank" in result


def test_algorithm_class_maps_known_families_and_nan_for_unknown_gap():
    from rde_domains.hsp_functions.kind_screen import (
        CLASS_ABELIAN,
        CLASS_DIHEDRAL,
        CLASS_JUNK,
        algorithm_class_for_generator,
        evaluate_algorithm_class_screen,
    )

    assert algorithm_class_for_generator("hsp_recipe.random") == CLASS_JUNK
    assert algorithm_class_for_generator("hsp_recipe.dihedral") == CLASS_DIHEDRAL
    assert algorithm_class_for_generator("hsp_recipe.xor") == CLASS_ABELIAN
    assert algorithm_class_for_generator("simon") == CLASS_ABELIAN
    assert np.isnan(algorithm_class_for_generator("hsp_recipe.heisenberg"))
    assert np.isnan(algorithm_class_for_generator("hsp_recipe.blend"))
    rows = []
    for i, (gen, klass, span, period) in enumerate(
        (
            ("hsp_recipe.random", 0.0, 0.0, 0.0),
            ("hsp_recipe.dihedral", 1.0, 0.0, 0.3),
            ("hsp_recipe.xor", 2.0, 0.25, 0.0),
            ("hsp_recipe.cyclic", 2.0, 0.0, 0.8),
            ("simon", 2.0, 0.25, 0.0),
            ("shor_cyclic", 2.0, 0.0, 0.8),
            ("dihedral_kuperberg", 1.0, 0.0, 0.3),
        )
    ):
        rows.append(
            {
                "size": 8,
                "instance_id": str(i),
                "generator": gen,
                "metric.algorithm_class": klass,
                "hsp_sample.f.difference_span_dim_fraction": span,
                "hsp_sample.f.detected_period_divisor_fraction": period,
                "hsp_sample.f.collision_rate": 0.0 if klass == 0.0 else 0.1,
            }
        )
    result = evaluate_algorithm_class_screen(
        rows, held_out_families=("simon", "shor_cyclic", "dihedral_kuperberg")
    )
    assert result["discovery_abs_r"] >= 0.35
    assert result["classifier"] == "span_period_2d_nearest_kind"
    assert result["per_n"]["8"]["recall"] == pytest.approx(1.0)


def test_discover_latent_from_run_excludes_contract_leaked_outcome_column(tmp_path):
    """`structure_strength` (raw OUTCOME, only `metric.structure_strength` is a legal target)
    must never enter PCA's own feature set -- otherwise PC1 trivially recovers the target from
    its own leaked ground truth, the exact tautological-R^2=1.0 failure mode
    `contract_excluded_columns` was built to close for `metric_variable_columns`/
    `descriptor_variable_columns` (see `rde.analyze.tables.contract_excluded_columns`), just not
    previously wired into `discover_latent_from_run`.
    """
    from rde.discovery.checkpoint import load_latent_checkpoint
    from rde.discovery.latent import discover_latent_from_run
    from rde.runtime.pipeline import RunConfig, run_pipeline

    reg = build_registry("hsp_functions")
    run_pipeline(
        RunConfig(
            domain_id="hsp_functions",
            n_instances=6,
            size=8,
            seed=11,
            indices=[0],
            store_root=tmp_path,
            run_id="hsp_latent_leak",
        ),
        registry=reg,
    )

    from rde.analyze.query import flatten_features

    rows = flatten_features("hsp_latent_leak", tmp_path)
    assert any("structure_strength" in row for row in rows), (
        "fixture must actually contain the raw leak column for this test to be meaningful"
    )

    discover_latent_from_run(
        "hsp_latent_leak", tmp_path, target_column="metric.structure_strength", checkpoint=True
    )
    _, meta = load_latent_checkpoint(tmp_path, "hsp_latent_leak", "pca")
    assert "structure_strength" not in meta["feature_columns"]
