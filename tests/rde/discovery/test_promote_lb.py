"""Tests for lower-bound conjecture export."""

from __future__ import annotations

import json
from pathlib import Path

from rde.discovery.promote_lb import write_lower_bound_conjectures_jsonl


def test_write_lower_bound_conjectures_jsonl(tmp_path: Path):
    conjectures = [
        {
            "claim_type": "scoped_lower_bound_conjecture",
            "run_id": "test",
            "witness": "panel_aggregate",
        }
    ]
    out = write_lower_bound_conjectures_jsonl(conjectures, tmp_path / "lb.jsonl")
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["witness"] == "panel_aggregate"
