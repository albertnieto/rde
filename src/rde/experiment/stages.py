"""Structured experiment stage events (stdout + durable JSONL)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rde.io.events import log_json_line, log_stage_event


def log_machine_profile(profile_text: str) -> None:
    """Log the session machine profile block at experiment start."""
    from rde.io.events import log_progress

    for line in profile_text.rstrip("\n").splitlines():
        log_progress(line)


def record_population_run(run_dir: Path, summary: dict[str, Any]) -> None:
    """Record a population pipeline stage to stdout and ``runs/<run_id>.jsonl``."""
    log_stage_event(summary, jsonl_path=run_dir / f"{summary['run_id']}.jsonl")


def record_discovery_stage(
    run_dir: Path, name: str, *, jsonl_name: str = "discovery_stages.jsonl"
) -> None:
    """Record a discovery-loop stage to stdout and a durable JSONL file."""
    payload = {"stage": "discovery_stage", "name": str(name)}
    log_stage_event(payload, jsonl_path=run_dir / jsonl_name)


def record_gated_outcome(*, verdict: str, grade: int) -> None:
    """Emit the final gated-outcome JSON line to stdout."""
    log_json_line({"stage": "gated_outcome", "verdict": verdict, "grade": grade})
