"""Tests for the preregistered structure_strength vs. representation-complexity check.

These lock in the actual result found when this check was first run
(`rho ~= 0.381`, `p ~= 1.8e-5`, `identity` winning 100% of the time) as a
reproducibility regression — not as an assertion that this is what a
"correct" or "good" result should look like. See
`preregistered_experiment.py`'s module docstring for the honest
interpretation (a real but shallow, likely `domain_kind`-confounded
correlation, not evidence of coset-structure discovery).
"""

from __future__ import annotations

import pytest

from rde_domains.hsp_functions.functions import ALL_FAMILIES
from rde_domains.hsp_functions.preregistered_experiment import run_preregistered_check


def test_preregistered_check_is_reproducible_for_fixed_defaults():
    result_a = run_preregistered_check()
    result_b = run_preregistered_check()
    assert result_a == result_b


def test_preregistered_check_reproduces_the_actual_first_run_result():
    result = run_preregistered_check()
    assert result.n == 120
    assert result.spearman_rho == pytest.approx(0.3810752638960436, abs=1e-9)
    assert result.spearman_p == pytest.approx(1.755201356985661e-05, abs=1e-9)
    assert result.detected_relationship is True
    assert result.fraction_identity_wins == pytest.approx(1.0)


def test_preregistered_check_covers_every_family():
    result = run_preregistered_check(instances_per_family=2)
    assert set(result.per_family_mean_strength) == set(ALL_FAMILIES)
    assert set(result.per_family_mean_complexity) == set(ALL_FAMILIES)


def test_preregistered_check_per_family_strength_matches_known_construction():
    # simon/shor_cyclic/dihedral_kuperberg are exact (structure_strength == 1.0
    # by construction); generic_random_control is exactly 0.0.
    result = run_preregistered_check(instances_per_family=3)
    assert result.per_family_mean_strength["simon"] == pytest.approx(1.0)
    assert result.per_family_mean_strength["shor_cyclic"] == pytest.approx(1.0)
    assert result.per_family_mean_strength["dihedral_kuperberg"] == pytest.approx(1.0)
    assert result.per_family_mean_strength["generic_random_control"] == pytest.approx(0.0)


def test_preregistered_check_decision_rule_respects_thresholds():
    # An impossibly strict threshold must flip the decision even though the
    # underlying correlation is unchanged.
    lenient = run_preregistered_check(instances_per_family=5, rho_threshold=0.1, p_threshold=0.5)
    strict = run_preregistered_check(instances_per_family=5, rho_threshold=0.99, p_threshold=1e-30)
    assert lenient.spearman_rho == strict.spearman_rho
    assert strict.detected_relationship is False
