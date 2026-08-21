"""Tests for v0.3 feature catalogs, domain contracts, and splits."""

from __future__ import annotations

from rde.analyze.leak_audit import audit_features
from rde.analyze.splits import SplitFold, SplitPolicy, assign_instance_fold
from rde.core.domain_contract import domain_contract
from rde.core.feature_contract import catalog_for_domain
from rde.core.instance import InstanceRecord


def test_unknown_feature_blocked():
    cat = catalog_for_domain("hsp_functions")
    audit = audit_features(
        ["unknown.feature.xyz"],
        target="metric.structure_strength",
        catalog=cat,
        domain_id="hsp_functions",
    )
    assert not audit.passed


def test_matrix_allowed_oracle_blocked():
    cat = catalog_for_domain("hsp_functions")
    assert cat.is_predictor_eligible("hsp_sample.f.collision_rate")
    assert not cat.is_predictor_eligible("hsp_oracle.walsh_max")
    assert not cat.is_predictor_eligible("metric.structure_strength")


def test_instance_grouped_splits():
    inst = InstanceRecord(
        domain_id="hsp_functions",
        size=8,
        seed=0,
        params={"generator": "structure_break_abelian"},
    )
    policy = SplitPolicy(
        calibration_sizes=(6,),
        confirmatory_sizes=(20,),
        discovery_fraction=1.0,
        validation_fraction=0.0,
    )
    fold = assign_instance_fold(inst, policy)
    assert fold == SplitFold.DISCOVERY


def test_all_domains_have_contracts():
    for domain_id in ("hsp_functions", "tsp_landscape_stats"):
        c = domain_contract(domain_id)
        assert c.primary_target.startswith("metric.")
