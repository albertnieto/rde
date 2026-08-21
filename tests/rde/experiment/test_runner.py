"""Gated experiment runner must record failures instead of traceback exits."""

from __future__ import annotations

import json
from pathlib import Path

from rde.experiment import ExperimentPreflightError, run_experiment_main, run_gated_experiment


def test_run_gated_experiment_records_gate_preflight(tmp_path: Path):
    exp_dir = tmp_path / "EXP-runner"
    exp_dir.mkdir()

    def _fail() -> int:
        raise ExperimentPreflightError("synthetic gate failure")

    code = run_gated_experiment(exp_dir, _fail)
    assert code == 1
    latest = exp_dir / "runs" / "run_failure_latest.json"
    assert latest.is_file()
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["stage"] == "gate_preflight"
    assert "synthetic gate failure" in payload["error"]


def test_run_gated_experiment_emits_run_failed_stdout(tmp_path: Path, capsys):
    exp_dir = tmp_path / "EXP-runner-stdout"
    exp_dir.mkdir()

    def _fail() -> int:
        raise ExperimentPreflightError("stdout failure line")

    code = run_gated_experiment(exp_dir, _fail)
    assert code == 1
    line = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["stage"] == "run_failed"
    assert payload["kind"] == "gate_preflight"
    assert payload["failure_stage"] == "gate_preflight"


def test_run_experiment_main_alias_matches_gated(tmp_path: Path):
    exp_dir = tmp_path / "EXP-alias"
    exp_dir.mkdir()

    def _ok() -> int:
        return 0

    assert run_experiment_main(exp_dir, _ok) == 0
    assert run_gated_experiment(exp_dir, _ok) == 0


def test_run_gated_experiment_records_unexpected_error(tmp_path: Path):
    exp_dir = tmp_path / "EXP-runner"
    exp_dir.mkdir()

    def _boom() -> int:
        raise RuntimeError("synthetic runtime failure")

    code = run_gated_experiment(exp_dir, _boom)
    assert code == 1
    payload = json.loads((exp_dir / "runs" / "run_failure_latest.json").read_text())
    assert payload["stage"] == "run_error"
    assert payload["error_type"] == "RuntimeError"
