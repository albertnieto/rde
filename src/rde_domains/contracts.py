"""Per-domain science contracts for HSP and TSP adapters."""

from __future__ import annotations

from rde.core.domain_contract import (
    DomainContract,
    StageSizePolicy,
    register_domain_contract,
)
from rde.core.feature_contract import (
    FeatureCatalog,
    FeatureOrigin,
    FeatureSpec,
)


def hsp_functions_contract() -> DomainContract:
    hsp_specs = (
        FeatureSpec(
            "hsp_sample.*",
            FeatureOrigin.POLYNOMIAL_INPUT,
            True,
            source_primitive="query_sample",
            asymptotic_cost="poly",
            notes="bounded poly(n_bits) random-query sample; never touches the full 2^n_bits table",
        ),
        FeatureSpec(
            "landscape.diff_profile.*",
            FeatureOrigin.POLYNOMIAL_INPUT,
            True,
            source_primitive="diff_profile",
            asymptotic_cost="poly",
            notes="auto-derived stats of the bounded poly(n_bits)-query difference profile",
        ),
        FeatureSpec(
            "landscape.array.*",
            FeatureOrigin.POLYNOMIAL_INPUT,
            True,
            source_primitive="diff_profile",
            asymptotic_cost="poly",
            notes="RDE's generic materialize()-slice descriptor sweep over the same difference-profile array",
        ),
        FeatureSpec(
            "landscape.collision_rate",
            FeatureOrigin.POLYNOMIAL_INPUT,
            True,
            source_primitive="query_sample",
            asymptotic_cost="poly",
            notes="re-exposure of hsp_sample.f.collision_rate under a gate-recognized prefix; real per-instance variance",
        ),
        FeatureSpec(
            "landscape.n_collisions_found",
            FeatureOrigin.POLYNOMIAL_INPUT,
            True,
            source_primitive="query_sample",
            asymptotic_cost="poly",
            notes="re-exposure of hsp_sample.f.n_collisions_found under a gate-recognized prefix; real per-instance variance",
        ),
        FeatureSpec(
            "landscape.difference_span_dim_fraction",
            FeatureOrigin.POLYNOMIAL_INPUT,
            True,
            source_primitive="query_sample",
            asymptotic_cost="poly",
            notes="re-exposure of hsp_sample.f.difference_span_dim_fraction under a gate-recognized prefix",
        ),
        FeatureSpec(
            "landscape.detected_period_divisor_fraction",
            FeatureOrigin.POLYNOMIAL_INPUT,
            True,
            source_primitive="query_sample",
            asymptotic_cost="poly",
            notes="re-exposure of hsp_sample.f.detected_period_divisor_fraction under a gate-recognized prefix",
        ),
        FeatureSpec(
            "repr.best_complexity",
            FeatureOrigin.POLYNOMIAL_INPUT,
            True,
            source_primitive="diff_profile",
            asymptotic_cost="poly",
            notes=(
                "rde.representation.rank_representations' lowest verified "
                "complexity over diff_profile (the fixed 7-primitive grammar); "
                "leak-free by the same argument as diff_profile itself -- never "
                "reads structure_strength"
            ),
        ),
        FeatureSpec(
            "hsp_oracle.*",
            FeatureOrigin.ENUMERATED_ORACLE,
            False,
            source_primitive="full_table",
            asymptotic_cost="exp",
            notes="exact full-table spectrum; audit/Gate-0 use only, small n_bits",
        ),
        FeatureSpec(
            "structure_strength",
            FeatureOrigin.OUTCOME,
            False,
            notes="generation-time planted-structure ground truth (raw primitive); see metric.structure_strength",
        ),
        FeatureSpec(
            "metric.structure_strength",
            FeatureOrigin.OUTCOME,
            False,
            notes="the primary target -- registered via register_hsp_functions_metrics, not a predictor",
        ),
        FeatureSpec(
            "algorithm_class",
            FeatureOrigin.OUTCOME,
            False,
            notes="literature query-algorithm class: 0 junk, 1 Kuperberg-dihedral, 2 abelian HSP; NaN if Q_quantum unknown",
        ),
        FeatureSpec(
            "metric.algorithm_class",
            FeatureOrigin.OUTCOME,
            False,
            notes="catalog-kind target; not guessed for Heisenberg/Q8/blend",
        ),
        FeatureSpec("n_bits", FeatureOrigin.METADATA, False, notes="size covariate; see size/n_bits confound check"),
    )
    return DomainContract(
        domain_id="hsp_functions",
        primary_target="metric.structure_strength",
        secondary_targets=("metric.algorithm_class",),
        feature_specs=hsp_specs,
        stage_sizes=StageSizePolicy(
            calibration_sizes=(6,),
            discovery_sizes=(8, 10, 12),
            validation_sizes=(16,),
            confirmatory_sizes=(20,),
            max_science_size=24,
        ),
        indices=(0,),
        generator_id=None,
        held_out_generator_groups=("simon", "shor_cyclic", "dihedral_kuperberg"),
        recurrence_applicable=False,
        representation_applicable=True,
    )


def tsp_landscape_stats_contract() -> DomainContract:
    specs = (
        FeatureSpec("matrix.D.*", FeatureOrigin.POLYNOMIAL_INPUT, True, source_primitive="D"),
        FeatureSpec("graph.D.*", FeatureOrigin.POLYNOMIAL_INPUT, True, source_primitive="D"),
        FeatureSpec(
            "repr.best_complexity",
            FeatureOrigin.POLYNOMIAL_INPUT,
            True,
            source_primitive="D",
            asymptotic_cost="poly",
            notes=(
                "rde.representation.rank_representations' lowest verified "
                "complexity over the upper-triangular distance profile of D "
                "(the fixed 7-primitive grammar); leak-free by the same "
                "argument as matrix.D.* itself -- never reads tour costs"
            ),
        ),
        FeatureSpec("landscape.costs.*", FeatureOrigin.ENUMERATED_ORACLE, False, source_primitive="costs", asymptotic_cost="exp"),
        FeatureSpec("spectral.*", FeatureOrigin.ENUMERATED_ORACLE, False, source_primitive="costs"),
        FeatureSpec("stats.*", FeatureOrigin.ENUMERATED_ORACLE, False, source_primitive="costs"),
        FeatureSpec("compress.*", FeatureOrigin.ENUMERATED_ORACLE, False, source_primitive="costs"),
        FeatureSpec("fourier.*", FeatureOrigin.ENUMERATED_ORACLE, False, source_primitive="costs"),
        FeatureSpec("gen.*", FeatureOrigin.ENUMERATED_ORACLE, False, source_primitive="costs"),
        FeatureSpec("dynamics.*", FeatureOrigin.TARGET_DERIVED, False),
        FeatureSpec("metric.*", FeatureOrigin.OUTCOME, False),
        FeatureSpec("size", FeatureOrigin.METADATA, False, notes="baseline covariate only"),
        FeatureSpec("seed", FeatureOrigin.METADATA, False),
        FeatureSpec("generator", FeatureOrigin.METADATA, False, notes="stratification only"),
        FeatureSpec("array_size", FeatureOrigin.METADATA, False),
    )
    return DomainContract(
        domain_id="tsp_landscape_stats",
        primary_target="metric.near_optimal_fraction",
        secondary_targets=("metric.cost_cv", "metric.cost_spectral_gap_ratio"),
        feature_specs=specs,
        stage_sizes=StageSizePolicy(
            calibration_sizes=(4,),
            discovery_sizes=(5, 6),
            validation_sizes=(7,),
            confirmatory_sizes=(8, 9),
            max_science_size=9,
        ),
        indices=(0,),
        generator_id=None,
        held_out_generator_groups=("circulant_broken",),
        recurrence_applicable=False,
        representation_applicable=True,
    )


_CONTRACTS: dict[str, DomainContract] = {
    "hsp_functions": hsp_functions_contract(),
    "tsp_landscape_stats": tsp_landscape_stats_contract(),
}

for _contract in _CONTRACTS.values():
    register_domain_contract(_contract)


def domain_contract(domain_id: str) -> DomainContract:
    if domain_id not in _CONTRACTS:
        raise KeyError(f"No domain contract for {domain_id!r}")
    return _CONTRACTS[domain_id]


def feature_catalog_for_domain(domain_id: str) -> FeatureCatalog:
    c = domain_contract(domain_id)
    return FeatureCatalog(domain_id=domain_id, specs=c.feature_specs)


def n_per_size_v03(domain_id: str, size: int) -> int:
    """Staged v0.3 instance counts — conservative until power-plan locks."""
    c = domain_contract(domain_id)
    if size not in (
        *c.stage_sizes.calibration_sizes,
        *c.stage_sizes.discovery_sizes,
        *c.stage_sizes.validation_sizes,
        *c.stage_sizes.confirmatory_sizes,
    ):
        return 0
    if size <= 4:
        return 200
    if size <= 6:
        return 300
    return 200
