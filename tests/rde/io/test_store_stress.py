"""Store write stress test (scaled-down million-row proxy)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rde.io.store import Store, count_jsonl_lines
from rde.testing.store_stress import stress_store_features, stress_write_jsonl


def test_jsonl_stress_write_10k(tmp_path: Path):
    path = tmp_path / "stress.jsonl"
    nbytes = stress_write_jsonl(path, 10_000, batch_size=512)
    assert count_jsonl_lines(path) == 10_000
    assert nbytes > 50_000


def test_store_stress_features_50k(tmp_path: Path):
    stats = stress_store_features(tmp_path, "stress50k", 50_000, batch_size=1024)
    assert stats["n_rows"] == 50_000
    store = Store(tmp_path)
    run_stats = store.run_stats("stress50k")
    assert run_stats["features_lines"] == 50_000
    store.close()


@pytest.mark.slow
def test_jsonl_stress_write_100k(tmp_path: Path):
    path = tmp_path / "stress_large.jsonl"
    nbytes = stress_write_jsonl(path, 100_000, batch_size=1024)
    assert count_jsonl_lines(path) == 100_000
    assert path.stat().st_size == nbytes


@pytest.mark.slow
def test_store_stress_features_1m(tmp_path: Path):
    stats = stress_store_features(tmp_path, "stress1m", 1_000_000, batch_size=2048)
    assert stats["n_rows"] == 1_000_000
