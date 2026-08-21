"""Every experiment that claims a verdict must carry a valid receipt.

This is the enforcement backstop for
`../../src/rde/docs/experiment-playbook.md`: an experiment cannot ship a
`results.md` asserting a scientific verdict unless `ExperimentGate.finalize()`
wrote a structurally valid `receipt.json`. Skipping the gate therefore shows up
as a failing test rather than as a confident self-report.

Experiments that are explicitly retracted or are not RDE discovery runs opt out
with a `NO_RECEIPT` marker file explaining why.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rde.experiment.gate import RECEIPT_FILENAME, validate_receipt

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXPERIMENTS = _REPO_ROOT / "experiments"

# Verdict words that mean "this run reached a scientific conclusion".
_VERDICT_MARKERS = (
    "**verdict",
    "verdict:",
    "## verdict",
)


def _claims_a_verdict(results_md: Path) -> bool:
    text = results_md.read_text(encoding="utf-8", errors="ignore").lower()
    return any(marker in text for marker in _VERDICT_MARKERS)


def _experiment_dirs_claiming_verdicts() -> list[Path]:
    if not _EXPERIMENTS.is_dir():
        return []
    out: list[Path] = []
    for exp_dir in sorted(_EXPERIMENTS.iterdir()):
        if not exp_dir.is_dir():
            continue
        if (exp_dir / "NO_RECEIPT").is_file():
            continue
        results = exp_dir / "results.md"
        if results.is_file() and _claims_a_verdict(results):
            out.append(exp_dir)
    return out


@pytest.mark.parametrize(
    "exp_dir",
    _experiment_dirs_claiming_verdicts(),
    ids=lambda p: p.name,
)
def test_experiment_claiming_verdict_has_valid_receipt(exp_dir: Path):
    receipt_path = exp_dir / RECEIPT_FILENAME
    assert receipt_path.is_file(), (
        f"{exp_dir.name}/results.md claims a verdict but there is no "
        f"{RECEIPT_FILENAME}. Drive the run through rde.experiment.ExperimentGate, "
        "or add a NO_RECEIPT file explaining why the gate does not apply."
    )
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    problems = validate_receipt(payload)
    assert not problems, f"{exp_dir.name}: invalid receipt: {problems}"


def test_receipt_validator_rejects_a_shortcut_receipt():
    """The validator must reject the exact failure modes it exists to catch."""
    from rde.experiment.gate import DEFAULT_REQUIRED_PHASES, RECEIPT_VERSION

    good_phases = {p: {"work_units": 1, "work_kind": "x"} for p in DEFAULT_REQUIRED_PHASES}
    base = {
        "receipt_version": RECEIPT_VERSION,
        "domain_id": "d",
        "target": "metric.t",
        "preregistration_sha256": "abc",
        "verdict": "NULL",
        "phases": good_phases,
        "population": {
            "per_size": {"6": {"distinct_structural_instances": 100, "required_distinct": 60}},
            "mechanism_visibility": {"visible": True, "paired": True},
        },
        "leak_audit": {"raw_best_abs_r": 1.0, "clean_best_abs_r": 0.3},
        "criteria_audit": {"decisive": ["best_abs_pearson_r"], "vacuous": []},
    }
    assert validate_receipt(base) == []

    # One landscape reused across a grid.
    collapsed = json.loads(json.dumps(base))
    collapsed["population"]["per_size"]["6"]["distinct_structural_instances"] = 1
    assert any("distinct instances" in p for p in validate_receipt(collapsed))

    # Pipeline-only run: discovery phases never executed.
    screen_only = json.loads(json.dumps(base))
    screen_only["phases"] = {"population": {"work_units": 1, "work_kind": "rows"}}
    assert any("missing required phases" in p for p in validate_receipt(screen_only))

    # Predictors still reconstruct the target.
    leaky = json.loads(json.dumps(base))
    leaky["leak_audit"]["clean_best_abs_r"] = 1.0
    assert any("still leak" in p for p in validate_receipt(leaky))

    # No pre-registration recorded.
    unregistered = json.loads(json.dumps(base))
    unregistered["preregistration_sha256"] = ""
    assert any("preregistration" in p for p in validate_receipt(unregistered))

    # The mechanism under study reached discovery as a label only.
    invisible = json.loads(json.dumps(base))
    invisible["population"]["mechanism_visibility"]["visible"] = False
    assert any("not numerically visible" in p for p in validate_receipt(invisible))

    # The verdict rests on a criterion that never computed.
    on_nan = json.loads(json.dumps(base))
    on_nan["criteria_audit"]["vacuous"] = ["best_abs_pearson_r"]
    assert any("rests on unevaluated criteria" in p for p in validate_receipt(on_nan))

    # A version 1 receipt cannot assert the checks that did not exist yet.
    stale = json.loads(json.dumps(base))
    stale["receipt_version"] = 1
    assert any("receipt_version" in p for p in validate_receipt(stale))
