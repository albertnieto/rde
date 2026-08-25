"""Regression tests: an OUTCOME-marked column must never be selectable as a predictor.

Found via a real, live bug: `rde discover` against a real hsp_functions
campaign produced `outcome_grade_hint=5`, `R^2=1.0` from the literal
expression `structure_strength` predicting `metric.structure_strength` (the
ground truth predicting itself) and from `algorithm_class / log(algorithm_class)`
— both explicitly marked `predictor_eligible=False` in the domain's
`DomainContract`, yet reachable by `metric_variable_columns` /
`descriptor_variable_columns` / `discover_equation` /
`run_symbolic_discovery`'s default candidate selection, none of which
consulted the contract before this fix. These tests pin the fix down with a
synthetic contract so they do not depend on `rde_domains` being installed
(core tests must not import domain plugins).
"""

from __future__ import annotations

from rde.core.domain_contract import DomainContract, StageSizePolicy, register_domain_contract
from rde.core.feature_contract import FeatureOrigin, FeatureSpec
from rde.descriptor_gen.rank import descriptor_variable_columns
from rde.discovery.symbolic import discover_equation
from rde.expression.generators import metric_variable_columns

_DOMAIN_ID = "test_leak_audit_domain"

_ROWS = [
    {
        "size": 8,
        "seed": i,
        "metric.target": float(i % 2),
        # A raw OUTCOME scalar re-exposed for bookkeeping, exactly like
        # hsp_functions' `structure_strength` -- correlates with the target
        # by construction and must never be offered as a predictor.
        "leaked_outcome": float(i % 2),
        # A genuine predictor-eligible column.
        "real_predictor": float(i % 3),
    }
    for i in range(12)
]


def _register_test_contract() -> None:
    contract = DomainContract(
        domain_id=_DOMAIN_ID,
        primary_target="metric.target",
        secondary_targets=(),
        feature_specs=(
            FeatureSpec("leaked_outcome", FeatureOrigin.OUTCOME, False),
            FeatureSpec("real_predictor", FeatureOrigin.POLYNOMIAL_INPUT, True),
            FeatureSpec("metric.*", FeatureOrigin.OUTCOME, False),
        ),
        stage_sizes=StageSizePolicy(),
        indices=(0,),
        generator_id=None,
    )
    try:
        register_domain_contract(contract)
    except ValueError:
        pass  # already registered by an earlier test in this process


def test_metric_variable_columns_excludes_outcome_column_when_domain_id_given():
    _register_test_contract()
    cols = metric_variable_columns(_ROWS, "metric.target", domain_id=_DOMAIN_ID)
    assert "leaked_outcome" not in cols
    assert "real_predictor" in cols


def test_metric_variable_columns_includes_outcome_column_without_domain_id():
    # Documents the fail-open default: no domain_id means no contract
    # filtering, unchanged from before this fix existed.
    cols = metric_variable_columns(_ROWS, "metric.target")
    assert "leaked_outcome" in cols


def test_descriptor_variable_columns_excludes_outcome_column_when_domain_id_given():
    _register_test_contract()
    cols = descriptor_variable_columns(_ROWS, target="metric.target", domain_id=_DOMAIN_ID)
    assert "leaked_outcome" not in cols
    assert "real_predictor" in cols


def test_descriptor_variable_columns_includes_outcome_column_without_domain_id():
    cols = descriptor_variable_columns(_ROWS, target="metric.target")
    assert "leaked_outcome" in cols


def test_discover_equation_excludes_outcome_column_when_domain_id_given():
    _register_test_contract()
    # Only the polynomial fallback runs -- deterministic, no heavy backends.
    result = discover_equation(
        _ROWS,
        "metric.target",
        use_pysr=False,
        use_operon=False,
        use_physics_templates=False,
        use_native_gp=False,
        domain_id=_DOMAIN_ID,
    )
    assert "leaked_outcome" not in result.equation


def test_discover_equation_can_use_outcome_column_without_domain_id():
    # Documents the fail-open default: without domain_id, the raw OUTCOME
    # column is a candidate feature like any other, exactly as before this
    # fix -- and a perfectly-correlated feature dominates the polynomial fit.
    result = discover_equation(
        _ROWS,
        "metric.target",
        use_pysr=False,
        use_operon=False,
        use_physics_templates=False,
        use_native_gp=False,
    )
    assert "leaked_outcome" in result.equation


def test_unknown_domain_id_fails_open_not_closed():
    # A domain_id with no registered contract must not silently exclude
    # everything (fail-closed) -- that would be a different, also-wrong
    # failure mode. It should behave exactly like domain_id=None.
    cols = metric_variable_columns(_ROWS, "metric.target", domain_id="no_such_domain_registered")
    assert "leaked_outcome" in cols
    assert "real_predictor" in cols
