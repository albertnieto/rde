"""Tests for outcome assessment, operator learning, store stress via Store."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rde.analyze.outcome import assess_outcome
from rde.discovery.operator import learn_update_predictor
from rde.testing.store_stress import count_jsonl_lines, stress_store_features


def test_assess_outcome_level_1_on_linear_discovery():
    rows = []
    for size in (4, 8):
        for i in range(15):
            x = float(i + size)
            rows.append({"size": size, "feat": x, "generator": "g0", "target": 2 * x + 1})
    discovery = {
        "metric_candidates": [{"r_squared": 0.95, "extrapolation_r_squared": 0.9}],
        "latent": {"ridge_r_squared": 0.85, "target_correlations": {"latent_0": 0.9}},
        "symbolic": {"r_squared": 0.92},
    }
    result = assess_outcome(rows, "target", discovery=discovery)
    assert result.g1_met
    assert "best_expression_r_squared" in result.g1_triggers
    payload = result.to_payload()
    assert payload["grade"] == result.grade
    assert payload["g1_met"] is True
    assert "level" not in payload
    for k in range(6):
        assert f"g{k}_met" in payload
        assert f"level_{k}_met" not in payload
        assert f"level_{k}_triggers" not in payload


def test_assess_outcome_level_0_on_noise():
    rng = np.random.default_rng(0)
    rows = [
        {
            "size": 4 + (i % 2) * 4,
            "feat": float(rng.normal()),
            "generator": f"g{i % 2}",
            "target": float(rng.normal()),
        }
        for i in range(40)
    ]
    discovery = {
        "metric_candidates": [{"r_squared": 0.1}],
        "latent": {"target_correlations": {"latent_0": 0.05}, "ridge_r_squared": 0.1},
        "symbolic": {"r_squared": 0.05},
    }
    result = assess_outcome(rows, "target", discovery=discovery)
    assert result.g0_met or not result.g1_met


def test_learn_update_predictor():
    rows = [{"a": float(i), "b": float(i * 2), "update.step_ratio_mean": float(i * 0.01)} for i in range(20)]
    op = learn_update_predictor(rows, ["a", "b"])
    assert op.train_r_squared > 0.5


def test_stress_store_features(tmp_path: Path):
    stats = stress_store_features(tmp_path, "stress1", 5000, batch_size=256)
    assert stats["n_rows"] == 5000
    path = tmp_path / "runs" / "stress1" / "features.jsonl"
    assert count_jsonl_lines(path) == 5000


def test_outcome_cli(tmp_path: Path):
    from rde.cli import main

    stats = stress_store_features(tmp_path, "outcome_cli", 30, batch_size=8)
    assert stats["n_rows"] == 30
    assert (
        main(
            [
                "outcome",
                "--run-id",
                "outcome_cli",
                "--store-root",
                str(tmp_path),
                "--target",
                "metric.representation_complexity",
            ]
        )
        == 0
    )
