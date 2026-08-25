"""The experiment gate must raise on every shortcut it exists to prevent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rde.experiment import (
    ExperimentGate,
    ExperimentPreflightError,
    distinct_structural_instances,
    structural_columns,
    validate_receipt,
)

DOMAIN = "hsp_functions"
TARGET = "metric.structure_strength"


def _exp_dir(tmp_path: Path, *, with_prereg: bool = True) -> Path:
    d = tmp_path / "EXP-999_probe"
    d.mkdir(parents=True, exist_ok=True)
    if with_prereg:
        (d / "PREREGISTRATION.md").write_text("# fixed decision rule\n")
    return d


def _gate(tmp_path: Path, **kw) -> ExperimentGate:
    return ExperimentGate(
        experiment_dir=_exp_dir(tmp_path), domain_id=DOMAIN, target=TARGET, **kw
    )


def _ready_gate(tmp_path: Path) -> ExperimentGate:
    """A gate that has passed every check up to finalize()."""
    gate = _gate(tmp_path)
    gate.check_plan(sizes=[6, 8, 10], n_per_size=60)
    gate.check_population(_rows())
    gate.check_leak_audit(raw_best_abs_r=1.0, clean_best_abs_r=0.2)
    for phase in ("expression_ranker", "latent", "symbolic", "phase6"):
        gate.record_phase(phase, work_units=3, work_kind="fields")
    return gate


def _rows(*, sizes=(6, 8, 10), n_per_size=60, distinct=True, generators=None):
    # The second family must be one the contract actually holds out, otherwise
    # check_population rejects the fixture before the test reaches its subject.
    generators = generators or ["simon", "structure_break_abelian"]
    out = []
    for size in sizes:
        for i in range(n_per_size):
            val = float(i) if distinct else 1.0
            out.append(
                {
                    "instance_id": f"i{size}_{i}",
                    "size": size,
                    "generator": generators[i % len(generators)],
                    "landscape.diff_profile.mean": val,
                    "hsp_sample.f.collision_rate": val + 0.5,
                    TARGET: float(i % 7),
                }
            )
    return out


def test_missing_contract_raises(tmp_path: Path):
    gate = ExperimentGate(
        experiment_dir=_exp_dir(tmp_path), domain_id="not_a_domain", target=TARGET
    )
    with pytest.raises(ExperimentPreflightError, match="no DomainContract"):
        gate.check_plan(sizes=[6, 8, 10], n_per_size=100)


def test_missing_preregistration_raises(tmp_path: Path):
    d = _exp_dir(tmp_path, with_prereg=False)
    gate = ExperimentGate(experiment_dir=d, domain_id=DOMAIN, target=TARGET)
    with pytest.raises(ExperimentPreflightError, match="PREREGISTRATION"):
        gate.check_plan(sizes=[6, 8, 10], n_per_size=100)


def test_too_few_sizes_and_instances_raise(tmp_path: Path):
    gate = _gate(tmp_path)
    with pytest.raises(ExperimentPreflightError, match="distinct sizes"):
        gate.check_plan(sizes=[6, 8], n_per_size=100)
    with pytest.raises(ExperimentPreflightError, match="instances per size"):
        gate.check_plan(sizes=[6, 8, 10], n_per_size=5)


def test_reused_instance_population_raises(tmp_path: Path):
    """The signature failure: one landscape reused across a categorical grid."""
    gate = _gate(tmp_path)
    gate.check_plan(sizes=[6, 8, 10], n_per_size=60)
    with pytest.raises(ExperimentPreflightError, match="distinct structural instances"):
        gate.check_population(_rows(distinct=False))


def test_missing_structural_columns_raises(tmp_path: Path):
    gate = _gate(tmp_path)
    gate.check_plan(sizes=[6, 8, 10], n_per_size=60)
    rows = [
        {"instance_id": f"i{i}", "size": 6 + 2 * (i % 3), "generator": "g", TARGET: 1.0}
        for i in range(60)
    ]
    with pytest.raises(ExperimentPreflightError, match="structural predictor columns"):
        gate.check_population(rows)


def test_no_held_out_rows_raises(tmp_path: Path):
    gate = _gate(tmp_path)
    gate.check_plan(sizes=[6, 8, 10], n_per_size=60)
    rows = _rows(generators=["structure_break_abelian", "generic_random_control"])
    with pytest.raises(ExperimentPreflightError, match="held-out families"):
        gate.check_population(rows)


def test_leak_audit_requires_raw_leak_and_clean_non_leak(tmp_path: Path):
    gate = _gate(tmp_path)
    with pytest.raises(ExperimentPreflightError, match="did not confirm a raw leak"):
        gate.check_leak_audit(raw_best_abs_r=0.4, clean_best_abs_r=0.2)
    with pytest.raises(ExperimentPreflightError, match="still reconstruct the target"):
        gate.check_leak_audit(raw_best_abs_r=1.0, clean_best_abs_r=1.0)
    gate.check_leak_audit(raw_best_abs_r=1.0, clean_best_abs_r=0.27)


def test_finalize_requires_full_discovery_phases(tmp_path: Path):
    gate = _gate(tmp_path)
    gate.check_plan(sizes=[6, 8, 10], n_per_size=60)
    gate.check_population(_rows())
    gate.check_leak_audit(raw_best_abs_r=1.0, clean_best_abs_r=0.2)
    # Pipeline-only run: the discovery loop never ran.
    with pytest.raises(ExperimentPreflightError, match="required discovery phases"):
        gate.finalize(verdict="NULL", grade=0)


def test_finalize_writes_valid_receipt(tmp_path: Path):
    gate = _gate(tmp_path)
    gate.check_plan(sizes=[6, 8, 10], n_per_size=60)
    gate.check_population(_rows())
    gate.check_leak_audit(raw_best_abs_r=1.0, clean_best_abs_r=0.2)
    for phase in ("expression_ranker", "latent", "symbolic", "phase6"):
        gate.record_phase(phase, work_units=3, work_kind="fields")

    receipt = gate.finalize(
        verdict="NULL",
        grade=0,
        criteria={"best_abs_pearson_r": 0.2},
        decisive_criteria=["best_abs_pearson_r"],
    )
    assert validate_receipt(receipt) == []

    written = json.loads((gate.experiment_dir / "receipt.json").read_text())
    assert written["verdict"] == "NULL"
    assert written["preregistration_sha256"]
    assert written["population"]["per_size"]["6"]["distinct_structural_instances"] == 60
    # Receipts account for work done, never wall-clock.
    assert "elapsed" not in json.dumps(written).lower()


def test_finalize_requires_declaring_the_decisive_criteria(tmp_path: Path):
    """A verdict must name the criteria its decision rule reads."""
    gate = _ready_gate(tmp_path)
    with pytest.raises(ExperimentPreflightError, match="requires decisive_criteria"):
        gate.finalize(verdict="NULL", grade=0, criteria={"best_abs_pearson_r": 0.2})


def test_finalize_rejects_a_verdict_resting_on_a_nan(tmp_path: Path):
    """A real failure mode: a NaN criterion read as 'no effect'."""
    gate = _ready_gate(tmp_path)
    with pytest.raises(ExperimentPreflightError, match="did not evaluate"):
        gate.finalize(
            verdict="NULL",
            grade=0,
            criteria={"generator_separation": float("nan")},
            decisive_criteria=["generator_separation"],
        )


def test_finalize_rejects_decisive_criteria_the_run_never_produced(tmp_path: Path):
    gate = _ready_gate(tmp_path)
    with pytest.raises(ExperimentPreflightError, match="not present in criteria"):
        gate.finalize(
            verdict="NULL",
            grade=0,
            criteria={"best_abs_pearson_r": 0.2},
            decisive_criteria=["never_computed"],
        )


def test_nan_criterion_is_allowed_when_not_decisive(tmp_path: Path):
    """A not-applicable criterion is legitimately NaN; only decisive ones bind."""
    gate = _ready_gate(tmp_path)
    receipt = gate.finalize(
        verdict="NULL",
        grade=0,
        criteria={
            "best_abs_pearson_r": 0.2,
            "state_reconstruction_r2": float("nan"),
        },
        decisive_criteria=["best_abs_pearson_r"],
    )
    assert validate_receipt(receipt) == []
    assert receipt["criteria_audit"]["not_applicable"] == ["state_reconstruction_r2"]


@pytest.mark.skip(reason="Mechanism pairing checks are coined_walk-specific; not shipped here.")
def test_mechanism_invisible_to_predictors_raises(tmp_path: Path):
    """A crossed design whose operator features are missing cannot discover."""
    gate = _gate(tmp_path, mechanism_pairing_key="instance_id")
    gate.check_plan(sizes=[6, 8, 10], n_per_size=60)
    rows = []
    for size in (6, 8, 10):
        for i in range(60):
            # One instance evaluated under two mechanisms, but every structural
            # column is instance-derived, so the mechanisms are indistinguishable.
            for generator in ("routing_steepest", "coin_theta_holonomy"):
                rows.append(
                    {
                        "instance_id": f"i{size}_{i}",
                        "size": size,
                        "generator": generator,
                        "matrix.Q.effective_rank": float(i),
                        "graph.Q.avg_degree": float(i) + 0.5,
                        TARGET: float(i % 7),
                    }
                )
    with pytest.raises(ExperimentPreflightError, match="not numerically visible"):
        gate.check_population(rows)


@pytest.mark.skip(reason="Mechanism pairing checks are coined_walk-specific; not shipped here.")
def test_mechanism_visible_when_operator_features_present(tmp_path: Path):
    gate = _gate(tmp_path, mechanism_pairing_key="instance_id")
    gate.check_plan(sizes=[6, 8, 10], n_per_size=60)
    rows = []
    for size in (6, 8, 10):
        for i in range(60):
            for k, generator in enumerate(("routing_steepest", "coin_theta_holonomy")):
                rows.append(
                    {
                        "instance_id": f"i{size}_{i}",
                        "size": size,
                        "generator": generator,
                        "matrix.Q.effective_rank": float(i),
                        "operator.coin_theta": float(k),
                        TARGET: float(i % 7),
                    }
                )
    gate.check_population(rows)
    visibility = gate._population["mechanism_visibility"]
    assert visibility["visible"] and visibility["paired"]


def test_finalize_requires_the_earlier_checks(tmp_path: Path):
    gate = _gate(tmp_path)
    with pytest.raises(ExperimentPreflightError, match="check_plan"):
        gate.finalize(verdict="NULL", grade=0)


def test_structural_helpers():
    rows = _rows(sizes=(6,), n_per_size=5)
    cols = structural_columns(rows)
    assert cols == ["hsp_sample.f.collision_rate", "landscape.diff_profile.mean"]
    assert distinct_structural_instances(rows, cols) == 5
    same = _rows(sizes=(6,), n_per_size=5, distinct=False)
    assert distinct_structural_instances(same, cols) == 1


def test_structural_columns_recognizes_repr_prefix():
    """`repr.best_complexity` (rde.representation's best-verified complexity) is a real
    input-derived predictor, same class as `landscape.*`/`hsp_sample.*` -- must be visible to
    the population-distinctness/mechanism-visibility checks, not silently dropped for lacking a
    whitelisted prefix (the exact failure mode `hsp_functions/domain.py` already documents for
    `hsp_sample.*` needing re-exposure under `landscape.*`).
    """
    rows = [{"repr.best_complexity": float(i), "metric.structure_strength": float(i)} for i in range(3)]
    assert structural_columns(rows) == ["repr.best_complexity"]


def test_finalize_accepts_deprecated_level_alias(tmp_path: Path):
    gate = _ready_gate(tmp_path)
    receipt = gate.finalize(verdict="NULL", level=0)
    assert receipt["grade"] == 0
    assert receipt.get("deprecated_finalize_level_alias") is True


def test_check_discovery_report_blocks_soft_failed_expression_ranker(tmp_path: Path):
    gate = _ready_gate(tmp_path)

    class _Report:
        stage_errors = [{"stage": "expression_ranker", "error": "KeyError: 'family_index'"}]

    with pytest.raises(ExperimentPreflightError, match="expression_ranker"):
        gate.check_discovery_report(_Report())
