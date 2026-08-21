"""run_discovery must return a partial report instead of raising."""

from __future__ import annotations

from pathlib import Path

from rde.discovery.loop import run_discovery


def test_run_discovery_returns_partial_report_on_fatal_load(tmp_path: Path, monkeypatch):
    def _boom(*_args, **_kwargs):
        raise RuntimeError("synthetic load failure")

    monkeypatch.setattr(
        "rde.discovery.context.DiscoveryContext.load",
        _boom,
    )
    report = run_discovery("missing_run", tmp_path, target="metric.y")
    assert report.run_id == "missing_run"
    assert report.stage_errors
    assert report.stage_errors[0]["stage"] == "run_discovery_fatal"
    assert "synthetic load failure" in report.stage_errors[0]["error"]
