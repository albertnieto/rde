"""Mechanical enforcement for RDE experiments (gate + receipt + run merging)."""

from __future__ import annotations

from rde.experiment.gate import (
    DEFAULT_REQUIRED_PHASES,
    RECEIPT_FILENAME,
    RECEIPT_VERSION,
    RECOVERY_REQUIRED_PHASES,
    ExperimentGate,
    ExperimentPreflightError,
    distinct_structural_instances,
    structural_columns,
    validate_receipt,
)
from rde.experiment.merge import (
    merge_runs_for_discovery,
    prepare_leak_clean_discovery,
    write_clean_discovery_run,
    write_run_subset,
)
from rde.experiment.runner import run_experiment_main, run_gated_experiment

__all__ = [
    "DEFAULT_REQUIRED_PHASES",
    "ExperimentGate",
    "ExperimentPreflightError",
    "RECEIPT_FILENAME",
    "RECEIPT_VERSION",
    "RECOVERY_REQUIRED_PHASES",
    "distinct_structural_instances",
    "merge_runs_for_discovery",
    "prepare_leak_clean_discovery",
    "run_experiment_main",
    "run_gated_experiment",
    "structural_columns",
    "validate_receipt",
    "write_clean_discovery_run",
    "write_run_subset",
]
