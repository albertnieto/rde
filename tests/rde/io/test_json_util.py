"""Tests for shared JSON artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rde.io.json_util import atomic_write, json_default, utc_now_iso, write_json


def test_utc_now_iso():
    assert utc_now_iso().endswith("+00:00")


def test_json_default_numpy_and_rejections():
    payload = {
        "int": np.int64(3),
        "float": np.float64(1.5),
        "arr": np.array([1.0, 2.0]),
    }
    encoded = json.loads(json.dumps(payload, default=json_default))
    assert encoded["int"] == 3
    assert encoded["float"] == 1.5
    assert encoded["arr"] == [1.0, 2.0]
    assert json_default(np.float64("nan")) is None

    with pytest.raises(TypeError, match="not JSON serializable"):
        json_default(object())


def test_write_json_and_atomic_write(tmp_path: Path):
    target = tmp_path / "nested" / "artifact.json"
    write_json(target, {"ok": True})
    assert target.read_text(encoding="utf-8").strip().startswith("{")

    atomic_path = tmp_path / "atomic.txt"
    atomic_write(atomic_path, lambda path: path.write_text("done", encoding="utf-8"))
    assert atomic_path.read_text(encoding="utf-8") == "done"
