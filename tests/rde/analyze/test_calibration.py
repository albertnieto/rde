"""Tests for calibration and symbolic extrapolation metadata."""

from __future__ import annotations

import numpy as np

from rde.analyze.calibration import separation_score
from rde.discovery.symbolic import discover_equation, discover_latent_interpretation


def test_separation_score_distinguishes_groups():
    rows = []
    for g, base in enumerate((1.0, 10.0)):
        for i in range(20):
            rows.append({"generator": f"g{g}", "metric.representation_complexity": base + i * 0.01})
    score = separation_score(rows, label_key="generator")
    assert score > 10.0


def test_symbolic_fit_with_extrapolation_metadata():
    rows = []
    for size in (4, 8):
        for i in range(12):
            x = float(i + size)
            rows.append({"size": size, "feat": x, "target": 2 * x + 1})
    result = discover_equation(rows, "target", use_pysr=False, use_operon=False, extrapolation=True)
    assert result.r_squared > 0.99
    assert "extrapolation_r_squared" in result.metadata


def test_discover_latent_interpretation_fits_best_latent_dim():
    rows = [{"x": float(i), "y": float(i * i), "target": float(i)} for i in range(25)]
    codes = np.column_stack([[row["x"] for row in rows], [row["y"] for row in rows]])
    result = discover_latent_interpretation(
        rows,
        codes,
        main_target="target",
        feature_columns=["x", "y"],
        latent_columns=["latent.ae_0", "latent.ae_1"],
        use_pysr=False,
        use_operon=False,
    )
    assert result is not None
    assert result["latent_column"] == "latent.ae_0"
    assert result["r_squared"] > 0.95
    assert "x" in result["equation"]
