"""Tests for Phase 6 — representation encoder, Walsh synthesis, rediscovery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rde.discovery.datasets import load_representation_batch
from rde.discovery.phase6 import _phase6_gpu_locked, run_phase6, write_phase6_report
from rde.discovery.rediscovery import (
    _score_seed_programs,
    assess_rediscovery,
    canonicalize_expression,
)
from rde.discovery.representation import discover_representation_encoder_decoder
from rde.discovery.walsh_synthesis import (
    _fit_walsh_support,
    _fit_walsh_supports_batch,
    _state_walsh_columns,
    run_walsh_synthesis,
    search_walsh_generating_functions,
)
from rde.discovery.datasets import RepresentationBatch, StatePairBatch
from rde.analyze.feature_table import FeatureTable
import rde.discovery.state_dynamics as state_dynamics
from rde.features.fourier import walsh_hadamard
from rde.runtime.pipeline import RunConfig, run_pipeline
from rde.core.plugins import build_registry


def test_canonicalize_expression_normalizes_whitespace():
    assert canonicalize_expression("  A + B ") == canonicalize_expression("a+b")


def test_seed_program_scoring_uses_one_population_batch(monkeypatch):
    rows = [{"x": float(i), "y": float(i + 1), "target": float(2 * i + 1)} for i in range(20)]
    calls: list[int] = []
    original = __import__(
        "rde.discovery.rediscovery",
        fromlist=["score_population"],
    ).score_population

    def wrapped(*args, **kwargs):
        calls.append(len(args[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr("rde.discovery.rediscovery.score_population", wrapped)
    result = _score_seed_programs(
        rows,
        "target",
        ["x", "y"],
        [
            ("x+y", "x+y", "gp", 1.0),
            ("y-x", "y-x", "symbolic", 1.0),
        ],
        min_fitness=0.0,
        eval_backend="numpy",
    )
    assert result is not None
    assert calls == [2]


def test_feature_table_take_preserves_columns_and_rows():
    rows = [
        {"x": 1.0, "target": 4.0},
        {"x": 2.0, "target": 5.0},
        {"x": 3.0, "target": 6.0},
    ]
    table = FeatureTable.from_rows(rows, columns=["x", "target"])
    subset = table.take(np.array([True, False, True]))
    assert subset.columns == table.columns
    np.testing.assert_array_equal(subset.data, np.array([[1.0, 4.0], [3.0, 6.0]]))
    assert subset.rows == [rows[0], rows[2]]


def test_descriptor_dynamics_pairs_rows_by_instance_and_index():
    rows = [
        {"instance_id": "b", "family_index": 2, "x": 20.0, "target": 200.0},
        {"instance_id": "a", "family_index": 1, "x": 10.0, "target": 100.0},
        {"instance_id": "a", "family_index": 2, "x": 11.0, "target": 110.0},
        {"instance_id": "b", "family_index": 1, "x": 19.0, "target": 190.0},
    ]
    X, y = state_dynamics._pair_rows(rows, "target", ["x"])
    np.testing.assert_array_equal(X, np.array([[10.0], [19.0]]))
    np.testing.assert_array_equal(y, np.array([110.0, 200.0]))


def test_descriptor_dynamics_ignores_rows_without_pair_keys():
    rows = [
        {"instance_id": "", "family_index": 1, "x": 1.0, "target": 10.0},
        {"instance_id": "", "family_index": 2, "x": 2.0, "target": 20.0},
        {"instance_id": "missing-index", "x": 3.0, "target": 30.0},
        {"instance_id": "missing-index", "family_index": 1, "x": 4.0, "target": 40.0},
        {"instance_id": "valid", "family_index": 1, "x": 5.0, "target": 50.0},
        {"instance_id": "valid", "family_index": 2, "x": 6.0, "target": 60.0},
    ]
    X, y = state_dynamics._pair_rows(rows, "target", ["x"])
    np.testing.assert_array_equal(X, np.array([[5.0]]))
    np.testing.assert_array_equal(y, np.array([60.0]))


def test_walsh_single_and_batch_support_fits_match():
    rng = np.random.default_rng(23)
    coefficients = np.array([1.5, -0.75])
    support = [1, 3]
    columns = rng.normal(size=(32, 6))
    target = 0.4 + columns[:, support] @ coefficients + 0.05 * rng.normal(size=32)

    single = _fit_walsh_support(columns, target, support)
    batch = _fit_walsh_supports_batch(columns, target, [support])[0]

    np.testing.assert_allclose(single[0], batch[1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(single[2:], batch[2:], rtol=1e-12, atol=1e-12)


def test_phase6_gpu_stages_are_process_locked():
    assert _phase6_gpu_locked("mlx", "numpy", False)
    assert _phase6_gpu_locked("numpy", "mlx", False)
    assert not _phase6_gpu_locked("numpy", "numpy", False)


def test_mlx_state_and_walsh_paths_match_numpy(tmp_path: Path, monkeypatch):
    pytest.importorskip("mlx")
    rng = np.random.default_rng(12)
    state_batch = StatePairBatch(
        prev_states=rng.normal(size=(5, 16)),
        next_states=rng.normal(size=(5, 16)),
        instance_ids=[str(i) for i in range(5)],
        r_indices=[1] * 5,
    )
    monkeypatch.setattr(
        state_dynamics,
        "load_state_pair_batch",
        lambda _run_id, _store_root: state_batch,
    )
    cpu_state = state_dynamics.predict_state_dynamics(
        "gpu_parity",
        tmp_path,
        checkpoint=False,
        compute_backend="numpy",
    )
    mlx_state = state_dynamics.predict_state_dynamics(
        "gpu_parity",
        tmp_path,
        checkpoint=False,
        compute_backend="mlx",
    )
    np.testing.assert_allclose(
        mlx_state.train_r_squared,
        cpu_state.train_r_squared,
        rtol=1e-5,
        atol=1e-6,
    )

    states = rng.random((3, 8))
    np.testing.assert_allclose(
        _state_walsh_columns(states, compute_backend="mlx"),
        _state_walsh_columns(states, compute_backend="numpy"),
        rtol=1e-5,
        atol=1e-6,
    )

    representation_batch = RepresentationBatch(
        instance_ids=["a", "a", "a", "b", "b", "b"],
        r_indices=np.array([1, 2, 3, 1, 2, 3]),
        sizes=np.full(6, 4),
        q_vectors=rng.normal(size=(6, 4)),
        state_vectors=rng.random((6, 8)),
    )
    cpu_rep = discover_representation_encoder_decoder(
        "gpu_parity",
        tmp_path,
        batch=representation_batch,
        checkpoint=False,
        compute_backend="numpy",
    )
    mlx_rep = discover_representation_encoder_decoder(
        "gpu_parity",
        tmp_path,
        batch=representation_batch,
        checkpoint=False,
        compute_backend="mlx",
    )
    np.testing.assert_allclose(
        mlx_rep.state_reconstruction_r2,
        cpu_rep.state_reconstruction_r2,
        rtol=1e-5,
        atol=1e-6,
    )


def test_walsh_generating_function_on_synthetic_rows():
    n = 8
    rows = []
    states = []
    for i in range(20):
        x = np.zeros(n)
        x[i % n] = 1.0
        coeffs = walsh_hadamard(x)
        rows.append({"target": float(coeffs[1])})
        states.append(x)
    result = search_walsh_generating_functions(
        rows,
        "target",
        state_vectors=np.array(states),
        max_support=3,
        max_candidates=200,
        seed=1,
    )
    assert result.top_programs
    assert abs(result.top_programs[0].pearson_r) > 0.9


def test_rediscovery_finds_repeated_linear_object():
    rows = [{"x": float(i), "y": float(i * 2), "size": 4 + (i % 2), "target": float(i + 1)} for i in range(40)]
    programs = [{"expression": "add(x,y)", "fitness": 0.95, "pearson_r": 0.99}]
    assessment = assess_rediscovery(
        rows,
        "target",
        programs=programs,
        variables=["x", "y"],
        n_splits=4,
        min_frequency=0.5,
        min_fitness=0.5,
        seed=0,
    )
    assert assessment.n_splits == 4
    assert assessment.dominant_object is not None


@pytest.mark.skip(reason="Requires qpfa_bound_trajectory (not in this distribution).")
def test_phase6_on_qpfa_run(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RDE_GP_PARALLEL", "1")
    reg = build_registry("qpfa_bound_trajectory", compute_backend="numpy")
    run_pipeline(
        RunConfig(
            domain_id="qpfa_bound_trajectory",
            n_instances=3,
            size=4,
            seed=11,
            indices=[1, 2, 4],
            store_root=tmp_path,
            run_id="phase6_qpfa",
            save_arrays=True,
        ),
        registry=reg,
    )

    batch = load_representation_batch("phase6_qpfa", tmp_path)
    assert batch.n_samples >= 3

    rep = discover_representation_encoder_decoder(
        "phase6_qpfa",
        tmp_path,
        checkpoint=True,
    )
    assert rep.n_samples >= 3
    assert rep.latent_dim >= 1
    assert np.isfinite(rep.state_reconstruction_r2)

    report = run_phase6(
        "phase6_qpfa",
        tmp_path,
        gp_generations=4,
        gp_population=16,
        walsh_max_candidates=300,
        rediscovery_splits=3,
        compute_backend="numpy",
    )
    assert report.n_rows == 9
    assert report.to_phase6_dict()["compute_backend"] == "numpy"
    assert report.walsh_synthesis is not None
    assert report.programs
    assert report.rediscovery is not None

    out = write_phase6_report(report, tmp_path / "phase6.json")
    payload = json.loads(out.read_text())
    assert payload["phase6"]["programs"]
    assert "representation" in payload["phase6"]


def test_walsh_synthesis_with_dreamcoder_programs():
    rows = [{"a": float(i), "b": float(i * 2), "fourier.array.sparsity": 0.1 * i, "target": float(i + 1)} for i in range(25)]
    result = run_walsh_synthesis(
        rows,
        "target",
        ["a", "b"],
        max_candidates=100,
        gp_generations=4,
        gp_population=12,
        seed=2,
    )
    assert result.dreamcoder_programs or result.top_programs
