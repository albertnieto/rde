"""Tests for RDE stdout logging helpers."""

from __future__ import annotations

import json

from rde.io.events import log_json_line, log_progress, log_stage_event


def test_log_progress_writes_flushed_stdout(capsys):
    log_progress("[pipeline test] [1/2] 50.0% elapsed=1s eta=1s detail")
    output = capsys.readouterr().out
    assert "[pipeline test] [1/2]" in output
    assert "eta=1s" in output


def test_log_json_line_emits_parseable_stdout(capsys):
    payload = {"stage": "run_failed", "kind": "gate_preflight", "error": "boom"}
    log_json_line(payload)
    line = capsys.readouterr().out.strip()
    assert json.loads(line) == payload


def test_log_stage_event_appends_jsonl_and_stdout(tmp_path, capsys):
    payload = {"stage": "population_run", "run_id": "exp_test", "rows": 3}
    path = tmp_path / "exp_test.jsonl"
    log_stage_event(payload, jsonl_path=path)
    assert path.read_text(encoding="utf-8").strip() == json.dumps(payload)
    assert json.loads(capsys.readouterr().out.strip()) == payload
