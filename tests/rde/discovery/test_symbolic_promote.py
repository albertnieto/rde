"""Tests for Phase 5 symbolic discovery and conjecture promotion."""

from __future__ import annotations

import re
import sys
from unittest.mock import patch

import numpy as np
import pytest

from rde.discovery.phase5 import run_phase5
from rde.discovery.promote import promote_top_conjectures
from rde.discovery.symbolic import (
    LatentSource,
    _pysr_fit,
    _pysr_restore_equation,
    _pysr_sanitize_variable_names,
    discover_latent_interpretations,
    discover_equation,
    run_symbolic_discovery,
    symbolic_backends,
)


def test_pysr_sanitize_variable_names_handles_dotted_features():
    features = [
        "metric.balanced_slice_rank",
        "derived.(log(A_coef) / compress.zlib_bits)",
        "dynamics.slice_rank_end",
    ]
    safe, mapping = _pysr_sanitize_variable_names(features)
    assert all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in safe)
    assert len(set(safe)) == len(features)
    assert mapping[safe[0]] == features[0]
    restored = _pysr_restore_equation(
        f"{safe[0]} + 2*{safe[2]}",
        mapping,
    )
    assert restored == f"{features[0]} + 2*{features[2]}"


def test_pysr_fit_sanitizes_variable_names_before_fit(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, list[str]] = {}

    class FakePySRRegressor:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y, variable_names=None):
            captured["variable_names"] = list(variable_names or [])

        def get_best(self):
            safe = captured["variable_names"][0]
            return {"equation": f"{safe} * 2", "loss": 0.01}

        def predict(self, X):
            return X[:, 0] * 2

    fake_pysr = type("pysr", (), {"PySRRegressor": FakePySRRegressor})
    monkeypatch.setitem(sys.modules, "pysr", fake_pysr)

    rows = [
        {
            "metric.balanced_slice_rank": float(i),
            "target": float(2 * i),
        }
        for i in range(1, 25)
    ]
    outcome = _pysr_fit(
        rows,
        "target",
        ["metric.balanced_slice_rank"],
        _return_outcome=True,
    )
    assert captured["variable_names"] == ["metric_balanced_slice_rank"]
    assert outcome.result is not None
    assert "metric.balanced_slice_rank" in outcome.result.equation
    assert outcome.result.method == "pysr"
    assert outcome.result.r_squared > 0.99


def test_symbolic_backends_reports_polynomial():
    info = symbolic_backends()
    assert info["polynomial_fallback"] is True
    assert info["physics_templates"] is True
    assert info["native_gp"] is True
    assert isinstance(info["ai_feynman"], bool)
    assert "pysr" in info
    assert "operon" in info


def test_discover_equation_records_skipped_unavailable_pysr():
    rows = [{"x": float(i), "target": float(i ** 2)} for i in range(1, 25)]
    fake_backends = {
        "polynomial_fallback": True,
        "physics_templates": True,
        "native_gp": True,
        "pysr": False,
        "operon": False,
        "ai_feynman": False,
        "errors": {"pysr": "not installed", "operon": "not installed"},
        "install_hint": "pip install -e '.[rde-symbolic]'",
    }
    with patch("rde.discovery.symbolic.symbolic_backends", return_value=fake_backends):
        result = discover_equation(
            rows,
            "target",
            features=["x"],
            use_pysr=True,
            use_operon=True,
            use_physics_templates=False,
            use_native_gp=False,
        )
    skipped = {s["engine"] for s in result.metadata.get("skipped_engines", [])}
    assert "pysr" in skipped
    assert "operon" in skipped
    assert result.r_squared > 0.9


def test_discover_latent_interpretations_top_k():
    rows = [{"x": float(i), "y": float(i * i), "target": float(i)} for i in range(25)]
    codes = np.column_stack([[row["x"] for row in rows], [row["y"] for row in rows]])
    interps = discover_latent_interpretations(
        rows,
        [LatentSource(codes=codes, columns=["latent.ae_0", "latent.ae_1"], source="autoencoder")],
        main_target="target",
        feature_columns=["x", "y"],
        top_latents=2,
        use_pysr=False,
        use_operon=False,
    )
    assert len(interps) == 2
    assert interps[0]["latent_source"] == "autoencoder"
    assert interps[0]["r_squared"] > 0.9


def test_run_symbolic_discovery_closes_latent_loop():
    rows = [{"x": float(i), "target": float(2 * i + 1)} for i in range(20)]
    codes = np.array([[row["x"] for row in rows]]).T
    result = run_symbolic_discovery(
        rows,
        "target",
        feature_columns=["x"],
        latent_sources=[LatentSource(codes=codes, columns=["latent.pca_0"], source="pca")],
        use_pysr=False,
        use_operon=False,
    )
    assert result.target_fit.r_squared > 0.99
    assert result.latent_interpretations
    assert result.latent_interpretations[0]["latent_source"] == "pca"


def test_run_symbolic_discovery_drops_low_finite_features():
    rows = [
        {"good": float(i), "all_nan": float("nan"), "target": float(2 * i + 1)}
        for i in range(24)
    ]
    result = run_symbolic_discovery(
        rows,
        "target",
        feature_columns=["all_nan", "good"],
        top_features=2,
        use_pysr=False,
        use_operon=False,
        use_native_gp=False,
    )
    assert result.target_fit.r_squared > 0.99
    assert result.target_fit.metadata["dropped_feature_columns"] == {"all_nan": 0}
    assert result.target_fit.metadata["feature_columns"] == ["good"]
    assert result.target_fit.metadata["feature_finite_counts"] == {"good": 24}


def test_discover_equation_records_insufficient_finite_rows():
    rows = [{"all_nan": float("nan"), "target": float(i)} for i in range(20)]
    result = discover_equation(
        rows,
        "target",
        features=["all_nan"],
        use_pysr=False,
        use_operon=False,
        use_physics_templates=False,
        use_native_gp=True,
        gp_generations=1,
        gp_population=4,
    )
    skipped = result.metadata["skipped_engines"]
    assert skipped == [
        {
            "engine": "native_gp",
            "reason": "insufficient_finite_rows:n=0;features=1",
        }
    ]


def test_discover_equation_records_fit_exception(monkeypatch: pytest.MonkeyPatch):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr("rde.discovery.programs.evolve_programs", fail)
    rows = [{"x": float(i), "target": float(2 * i + 1)} for i in range(20)]
    result = discover_equation(
        rows,
        "target",
        features=["x"],
        use_pysr=False,
        use_operon=False,
        use_physics_templates=False,
        use_native_gp=True,
        gp_generations=1,
        gp_population=4,
    )
    skipped = result.metadata["skipped_engines"]
    assert skipped[0]["engine"] == "native_gp"
    assert skipped[0]["reason"] == "fit_exception:RuntimeError: synthetic failure"


def test_discover_equation_includes_physics_templates():
    rows = [{"x": float(i), "target": float(i ** 2)} for i in range(1, 25)]
    result = discover_equation(rows, "target", features=["x"], use_pysr=False, use_operon=False)
    tried = result.metadata.get("candidates_tried", [])
    assert "polynomial_fallback" in tried
    assert "physics_templates" in tried
    assert result.r_squared > 0.95


def test_run_phase5_promotes_symbolic(tmp_path):
    rows = [
        {"matrix.Q.trace": float(i), "target": float(2 * i + 1), "size": 4} for i in range(20)
    ]
    report = run_phase5(
        "phase5_test",
        tmp_path,
        target="target",
        rows=rows,
        feature_columns=["matrix.Q.trace"],
        use_pysr=False,
        use_operon=False,
        promote=True,
    )
    assert report.symbolic["r_squared"] > 0.99
    assert report.promoted_conjectures
    assert report.promoted_conjectures[0]["source"] == "symbolic"


def test_promote_top_conjectures_merges_sources():
    metric = [
        {
            "expression": "log(a)",
            "target": "t",
            "r_squared": 0.8,
            "pearson_r": 0.9,
            "mdl": 2.0,
            "feature_columns": ["hsp_sample.f.collision_rate"],
        }
    ]
    symbolic = {
        "method": "polynomial_fallback",
        "equation": "2*hsp_sample.f.collision_rate + 1",
        "r_squared": 0.99,
        "feature_columns": ["hsp_sample.f.collision_rate"],
    }
    latent = [
        {
            "latent_column": "latent.ae_0",
            "latent_source": "autoencoder",
            "main_target": "t",
            "main_target_correlation": 0.95,
            "method": "polynomial_fallback",
            "equation": "hsp_sample.f.collision_rate",
            "r_squared": 0.98,
            "feature_columns": ["hsp_sample.f.collision_rate"],
        }
    ]
    promoted = promote_top_conjectures(
        target="t",
        metric_candidates=metric,
        symbolic=symbolic,
        latent_interpretations=latent,
        max_results=3,
        min_r_squared=0.5,
        leak_audit=True,
        domain_id="hsp_functions",
    )
    sources = {p["source"] for p in promoted}
    assert "symbolic" in sources
    assert "latent_interpretation" in sources
    assert len(promoted) <= 3
