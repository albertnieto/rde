"""Tests for Phase 4 completion — datasets, checkpoints, state dynamics, phase4 orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rde.analyze.feature_table import FeatureTable
from rde.discovery.checkpoint import load_latent_checkpoint, save_latent_checkpoint
from rde.discovery.datasets import (
    broadcast_instance_latents,
    load_instance_q_batch,
    load_state_pair_batch,
    q_upper_flat,
)
from rde.discovery.state_dynamics import predict_state_dynamics
from rde.discovery.latent import (
    correlate_latent_descriptors,
    discover_latent_from_run,
    load_pca_from_checkpoint,
    ridge_predict_columns,
    ridge_predict_multi,
)
from rde.discovery.impute import impute_matrix
from rde.discovery.phase4 import run_phase4, write_phase4_report
from rde.io.store import Store
from rde.runtime.pipeline import RunConfig, run_pipeline
from rde.core.plugins import build_registry


def test_q_upper_flat():
    q = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    flat = q_upper_flat(q)
    assert flat.shape == (6,)
    assert flat[0] == 1.0
    assert flat[1] == 2.0


def test_broadcast_instance_latents():
    rows = [
        {"instance_id": "a", "family_index": 1},
        {"instance_id": "a", "family_index": 2},
        {"instance_id": "b", "family_index": 1},
    ]
    codes = np.array([[1.0, 2.0], [3.0, 4.0]])
    out, cols = broadcast_instance_latents(rows, ["a", "b"], codes, ["z0", "z1"])
    assert out.shape == (3, 2)
    assert np.allclose(out[0], [1.0, 2.0])
    assert np.allclose(out[2], [3.0, 4.0])


def test_latent_checkpoint_roundtrip(tmp_path: Path):
    arrays = {"components": np.array([[1.0, 0.0], [0.0, 1.0]])}
    save_latent_checkpoint(tmp_path, "run1", "pca", arrays=arrays, metadata={"method": "pca"})
    loaded, meta = load_latent_checkpoint(tmp_path, "run1", "pca")
    assert loaded["components"].shape == (2, 2)
    assert meta["method"] == "pca"


def test_correlate_latent_descriptors():
    rows = [
        {"feat_a": 1.0, "metric.y": 2.0},
        {"feat_a": 2.0, "metric.y": 4.0},
        {"feat_a": 3.0, "metric.y": 6.0},
        {"feat_a": 4.0, "metric.y": 8.0},
    ]
    codes = np.array([[1.0], [2.0], [3.0], [4.0]])
    hits = correlate_latent_descriptors(codes, ["z0"], rows, target="metric.y", min_abs_r=0.5)
    assert hits
    assert hits[0]["descriptor_column"] in {"feat_a", "metric.y"}


def test_impute_matrix_replaces_missing_columns_vectorially():
    X = np.array(
        [
            [1.0, np.nan, np.nan],
            [3.0, 5.0, np.nan],
            [np.nan, 7.0, np.nan],
        ]
    )
    out = impute_matrix(X)
    np.testing.assert_allclose(
        out,
        np.array(
            [
                [1.0, 6.0, 0.0],
                [3.0, 5.0, 0.0],
                [2.0, 7.0, 0.0],
            ]
        ),
    )


def test_multi_target_ridge_matches_individual_fits():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(20, 4))
    targets = rng.normal(size=(20, 3))

    multi_coef, multi_r2 = ridge_predict_multi(X, targets)
    for i in range(targets.shape[1]):
        coef, r2 = ridge_predict_columns(X, targets[:, i])
        np.testing.assert_allclose(multi_coef[:, i], coef)
        np.testing.assert_allclose(multi_r2[i], r2)


def test_dual_state_ridge_prediction_matches_primal():
    rng = np.random.default_rng(5)
    prev = rng.normal(size=(5, 12))
    nxt = rng.normal(size=(5, 12))
    alpha = 1e-2

    primal = np.linalg.solve(
        prev.T @ prev + alpha * np.eye(prev.shape[1]),
        prev.T @ nxt,
    )
    dual_coefficients = np.linalg.solve(
        prev @ prev.T + alpha * np.eye(prev.shape[0]),
        nxt,
    )
    np.testing.assert_allclose(
        prev @ primal,
        nxt - alpha * dual_coefficients,
        rtol=1e-10,
        atol=1e-10,
    )


def test_feature_table_builds_numeric_matrix_with_missing_values():
    rows = [
        {"a": 1.0, "b": 2.0},
        {"a": 3.0},
        {"b": 6.0},
    ]

    table = FeatureTable.from_rows(rows, columns=["a", "b"])

    assert table.data.shape == (3, 2)
    np.testing.assert_allclose(
        table.data,
        np.array([[1.0, 2.0], [3.0, np.nan], [np.nan, 6.0]]),
        equal_nan=True,
    )


@pytest.mark.skip(reason="Requires qpfa_bound_trajectory Q primitive (not in this distribution).")
def test_phase4_on_qpfa_run(tmp_path: Path):
    reg = build_registry("qpfa_bound_trajectory", compute_backend="numpy")
    run_pipeline(
        RunConfig(
            domain_id="qpfa_bound_trajectory",
            n_instances=4,
            size=4,
            seed=7,
            indices=[1, 2, 4],
            store_root=tmp_path,
            run_id="phase4_qpfa",
            save_arrays=True,
        ),
        registry=reg,
    )

    q_batch = load_instance_q_batch("phase4_qpfa", tmp_path, target="metric.log_slice_rank")
    assert q_batch.q_vectors.shape[0] == 4

    state_batch = load_state_pair_batch("phase4_qpfa", tmp_path)
    assert state_batch.prev_states.shape[0] >= 4

    state_dyn = predict_state_dynamics("phase4_qpfa", tmp_path, checkpoint=True)
    assert state_dyn.n_pairs >= 4
    assert np.isfinite(state_dyn.train_r_squared)
    assert state_dyn.method in {"ridge_state", "dual_ridge_state"}

    pca = discover_latent_from_run("phase4_qpfa", tmp_path, checkpoint=True)
    assert pca.latent_dim >= 1
    _, meta = load_latent_checkpoint(tmp_path, "phase4_qpfa", "pca")
    assert meta["method"] == "pca"

    report = run_phase4("phase4_qpfa", tmp_path)
    assert report.n_rows == 12
    latent = report.to_latent_dict()
    assert "autoencoder" in latent
    assert "state_dynamics" in latent
    assert latent["state_dynamics"]["n_pairs"] >= 4

    out = write_phase4_report(report, tmp_path / "phase4.json")
    payload = json.loads(out.read_text())
    assert payload["latent"]["instance_q_autoencoder"]["n_instances"] == 4


def test_load_pca_from_checkpoint(tmp_path: Path):
    reg = build_registry("synthetic_poly")
    run_pipeline(
        RunConfig(
            domain_id="synthetic_poly",
            n_instances=5,
            size=6,
            seed=1,
            indices=[1, 2],
            store_root=tmp_path,
            run_id="pca_ckpt",
            generator_id="poly_narrow_coeff",
        ),
        registry=reg,
    )
    discover_latent_from_run("pca_ckpt", tmp_path, checkpoint=True)
    from rde.discovery.datasets import load_feature_matrix

    _, meta = load_latent_checkpoint(tmp_path, "pca_ckpt", "pca")
    use = meta["feature_columns"]
    X, cols, rows = load_feature_matrix("pca_ckpt", tmp_path, target="metric.representation_complexity")
    idx = [cols.index(c) for c in use]
    target = np.array([r.get("metric.representation_complexity", float("nan")) for r in rows])
    restored = load_pca_from_checkpoint("pca_ckpt", tmp_path, X[:, idx], target=target)
    assert restored is not None
    assert restored.latent_dim >= 1
