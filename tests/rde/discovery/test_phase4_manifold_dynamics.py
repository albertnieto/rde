"""Tests for Phase 4 manifold and r-dynamics step predictor."""

from __future__ import annotations

from rde.discovery.state_dynamics import predict_dynamics_step
from rde.discovery.manifold import manifold_from_rows


def _synthetic_rows():
    rows = []
    for size in (4, 6, 8):
        for i in range(5):
            for r in (1, 2, 4):
                rows.append(
                    {
                        "instance_id": f"s{size}-{i}",
                        "size": size,
                        "family_index": r,
                        "feat_a": float(size * r + i),
                        "feat_b": float(i + r),
                        "metric.representation_complexity": float(size + i + r * 0.1),
                    }
                )
    return rows


def test_manifold_from_rows():
    rows = _synthetic_rows()
    res = manifold_from_rows(rows, "metric.representation_complexity", feature_columns=["feat_a", "feat_b"])
    assert res.n_components == 2
    assert len(res.explained_variance) == 2


def test_dynamics_step_predictor():
    rows = _synthetic_rows()
    res = predict_dynamics_step(rows, "metric.representation_complexity", feature_columns=["feat_a", "feat_b"])
    assert res.n_pairs > 0
    assert res.train_r_squared > 0.5
