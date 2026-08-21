"""Top-level runner for RDE experiments — no silent tracebacks."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from rde.experiment.gate import ExperimentPreflightError
from rde.runtime.resilience import (
    emit_failure_line,
    failure_record,
    write_failure_artifact,
)


def run_experiment_main(
    experiment_dir: Path,
    main: Callable[[], int | None],
    *,
    run_dir: Path | None = None,
) -> int:
    """Run any ``experiments/EXP-*/run.py`` entry point with structured failures.

    Gate preflight violations and unexpected runtime errors both emit a durable
    JSON artifact under ``runs/`` plus a single ``{"stage": "run_failed", ...}``
    line on stdout. Successful completion returns the integer exit code from
    ``main`` (default 0).
    """
    run_dir = run_dir or (experiment_dir / "runs")
    try:
        code = main()
        return 0 if code is None else int(code)
    except ExperimentPreflightError as exc:
        payload = failure_record(stage="gate_preflight", exc=exc)
        path = write_failure_artifact(run_dir, payload)
        emit_failure_line(kind="gate_preflight", payload=payload, artifact=path)
        return 1
    except KeyboardInterrupt:
        payload = failure_record(stage="keyboard_interrupt", exc=KeyboardInterrupt())
        path = write_failure_artifact(run_dir, payload)
        emit_failure_line(kind="keyboard_interrupt", payload=payload, artifact=path)
        return 130
    except Exception as exc:  # noqa: BLE001 — experiment runner must never crash silently
        payload = failure_record(stage="run_error", exc=exc)
        path = write_failure_artifact(run_dir, payload)
        emit_failure_line(kind="run_error", payload=payload, artifact=path)
        return 1


# Backward-compatible name used by gated discovery experiments.
run_gated_experiment = run_experiment_main
