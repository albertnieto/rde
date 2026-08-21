"""Tests for discovery summary formatters."""

from __future__ import annotations

import numpy as np

from rde.discovery.autoencoder import AutoencoderResult, InstanceQResult
from rde.discovery.latent import LatentResult
from rde.discovery.loop import DiscoveryReport
from rde.discovery.manifold import ManifoldResult
from rde.discovery.phase4 import Phase4Report
from rde.discovery.phase5 import Phase5Report
from rde.discovery.report import (
    format_discovery_summary,
    format_phase4_summary,
    format_phase5_summary,
)
from rde.discovery.state_dynamics import DynamicsStepResult, StateDynamicsResult
from rde.discovery.trajectory import TrajectoryPredictorResult


def test_format_phase4_summary_includes_latent_blocks():
    report = Phase4Report(
        run_id="p4",
        target="metric.y",
        n_rows=100,
        checkpoint_dir="/tmp/ckpt",
        pca=LatentResult(
            latent_dim=2,
            explained_variance_ratio=[0.6, 0.3],
            latent_codes=np.zeros((10, 2)),
            target_correlations={"z0": 0.8, "z1": -0.2},
            reconstruction_error=0.01,
        ),
        feature_autoencoder=AutoencoderResult(
            hidden_dim=4,
            reconstruction_error=0.02,
            train_r_squared=0.9,
            target_correlation=0.7,
            method="random_features",
        ),
        instance_q_autoencoder=InstanceQResult(
            n_instances=5,
            q_dim=8,
            hidden_dim=3,
            reconstruction_error=0.03,
            train_r_squared=0.85,
            target_correlation=0.6,
            target_correlations={"z0": 0.6},
            method="random_features",
        ),
        trajectory=TrajectoryPredictorResult(
            train_r_squared=0.8,
            extrapolation_r_squared=0.5,
            n_train=40,
            n_test=10,
            feature_columns=["feat_a"],
        ),
        dynamics_descriptor=DynamicsStepResult(
            train_r_squared=0.75,
            n_pairs=30,
            feature_columns=["dyn_a"],
        ),
        dynamics_state=StateDynamicsResult(
            train_r_squared=0.7,
            n_pairs=25,
            state_dim=16,
            transition_rank=4,
            low_rank=True,
        ),
        manifold=ManifoldResult(
            n_components=2,
            explained_variance=[0.55, 0.25],
            target_correlation=0.4,
            embedding=[[0.0, 0.0], [1.0, 1.0]],
        ),
        latent_descriptor_correlations=[
            {"latent_column": "z0", "descriptor_column": "feat_a", "pearson_r": 0.9}
        ],
    )
    text = format_phase4_summary(report)
    assert "run_id=p4" in text
    assert "PCA:" in text
    assert "Feature AE" in text
    assert "Instance Q AE" in text
    assert "Trajectory:" in text
    assert "State dynamics:" in text
    assert "Manifold:" in text
    assert "Top latent ↔ descriptor:" in text


def test_format_phase5_and_discovery_summaries():
    phase5 = Phase5Report(
        run_id="p5",
        target="metric.y",
        n_rows=50,
        g1_met=True,
        symbolic={
            "method": "ridge",
            "r_squared": 0.88,
            "equation": "x + y",
            "skipped_engines": [{"engine": "pysr", "reason": "missing"}],
            "candidates_tried": [],
            "backends": {"numpy": True},
        },
        latent_interpretations=[
            {
                "latent_column": "z0",
                "latent_source": "pca",
                "r_squared": 0.7,
                "main_target_correlation": 0.65,
            }
        ],
        promoted_conjectures=[
            {"source": "symbolic", "expression": "a+b", "r_squared": 0.8}
        ],
        backends={"numpy": True},
    )
    p5_text = format_phase5_summary(phase5)
    assert "g1_met=True" in p5_text
    assert "Skipped engines:" in p5_text
    assert "Promoted:" in p5_text

    discovery = DiscoveryReport(
        run_id="disc",
        target="metric.y",
        n_rows=20,
        outcome_grade_hint=2,
        top_correlations=[{"column": "feat_a", "pearson_r": 0.9, "r_squared": 0.81}],
        metric_candidates=[{"expression": "x", "r_squared": 0.5, "level_hint": 1}],
        massive_catalog_skipped_no_arrays=True,
        parameters={"use_massive_catalog": True},
        symbolic={"method": "ridge", "equation": "x", "r_squared": 0.4},
        promoted_conjectures=[{"source": "expr", "expression": "x*y", "r_squared": 0.6}],
        cross_n_rule=[
            {
                "status": "ok",
                "stable": True,
                "candidate_source": "expr",
                "candidate_expression": "x",
                "slope_rule": {"equation": "n", "r_squared": 0.9},
            }
        ],
        coordinates={
            "method": "pca",
            "n_components": 2,
            "target_correlation": 0.3,
            "update_simplicity": 0.2,
        },
        phase6={
            "representation": {
                "n_samples": 10,
                "latent_dim": 3,
                "state_reconstruction_r2": 0.8,
                "latent_update_r2": 0.7,
                "g4_met": True,
            },
            "walsh_synthesis": {"top_programs": [{"expression": "walsh(x)", "pearson_r": 0.5}]},
            "rediscovery": {"status": "ok", "best_r2": 0.6},
        },
    )
    disc_text = format_discovery_summary(discovery)
    assert "Top correlations:" in disc_text
    assert "Massive descriptor catalog: skipped" in disc_text
    assert "Cross-N rule:" in disc_text
    assert "Coordinates:" in disc_text
    assert "Representation (Q,r)→z→Ψ:" in disc_text
    assert "Walsh synthesis:" in disc_text
