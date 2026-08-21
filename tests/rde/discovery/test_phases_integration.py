"""Integration tests for RDE Phases 1–6 modules."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rde.analyze.outcome import OutcomeAssessment
from rde.analyze.query import kmeans_clusters
from rde.discovery import run_discovery, write_discovery_report
from rde.discovery.context import DiscoveryContext
from rde.features.overlap import compute_overlap_descriptors
from rde.features.recurrence import estimate_recurrence_order
from rde.features.update import compute_update_descriptors
from rde.generators import get_generator, list_generators, register_builtin_generators
from rde.runtime.pipeline import RunConfig, run_pipeline
from rde.core.plugins import build_registry, list_domain_ids
from rde.core.schema import validate_feature_row, validate_features_file
from rde.io.store import Store


def test_phase1_generators_and_schema():
    register_builtin_generators()
    assert "synthetic_poly" in list_domain_ids()
    gens = list_generators("synthetic_poly")
    assert "poly_wide_coeff" in gens
    inst = get_generator("poly_wide_coeff").generate(2, 8, 0)
    assert len(inst) == 2
    assert inst[0].params["generator"] == "poly_wide_coeff"
    row = {
        "run_id": "r",
        "instance_id": "i",
        "domain_id": "d",
        "size": 4,
        "seed": 0,
        "family_index": 1,
        "slice_kind": "k",
        "descriptors": {},
        "metrics": {},
    }
    assert validate_feature_row(row) == []


def test_phase2_cross_slice_descriptors():
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    ov = compute_overlap_descriptors([1, 2], [v1, v2])
    assert ov["overlap.consecutive_mean"] == 0.0
    rec = estimate_recurrence_order([1, 2, 4], [v1, v2, v1])
    assert rec["recurrence.krylov_rank_full"] >= 2
    assert rec["recurrence.estimated_order"] == 2.0
    orth = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    rec_full = estimate_recurrence_order([1, 2, 3], orth)
    assert rec_full["recurrence.estimated_order"] == 3.0
    upd = compute_update_descriptors([1, 2], [v1, v2])
    assert upd["update.n_steps"] == 1.0


def test_phase2_kmeans():
    rows = [{"a": float(i), "b": float(i * 2)} for i in range(12)]
    out = kmeans_clusters(rows, ["a", "b"], k=3)
    assert len(out["labels"]) == 12


def test_phases_pipeline_and_discovery(tmp_path: Path):
    reg = build_registry("synthetic_poly")
    run_pipeline(
        RunConfig(
            domain_id="synthetic_poly",
            n_instances=6,
            size=8,
            seed=1,
            indices=[1, 2, 4],
            store_root=tmp_path,
            run_id="phase_test",
            generator_id="poly_narrow_coeff",
        ),
        registry=reg,
    )
    store = Store(tmp_path)
    rows = store.read_features("phase_test")
    store.close()
    assert len(rows) >= 6
    assert validate_features_file(rows) == []

    report = run_discovery(
        "phase_test", tmp_path, max_expr_candidates=100, gp_generations=5, use_pysr=False, expr_eval_backend="numpy"
    )
    out = write_discovery_report(report, tmp_path / "discovery.json")
    payload = json.loads(out.read_text())
    assert payload["n_rows"] >= 6
    assert "metric_candidates" in payload
    assert "latent" in payload
    assert "symbolic" in payload
    assert "latent_interpretation" in payload["symbolic"]
    assert "promoted_conjectures" in payload
    assert "programs" in payload
    assert "phase6" in payload


def test_discovery_wires_mlx_state_backend(monkeypatch, tmp_path: Path):
    import rde.discovery.loop as discovery_loop

    rows = [
        {"instance_id": f"i{i}", "family_index": 1, "size": 4, "x": float(i), "target": float(i + 1)}
        for i in range(8)
    ]
    context = DiscoveryContext(
        run_id="backend_wiring",
        store_root=tmp_path,
        target="target",
        rows=rows,
    )
    monkeypatch.setattr(
        discovery_loop.DiscoveryContext,
        "load",
        classmethod(lambda cls, run_id, store_root, *, target=None: context),
    )
    monkeypatch.setattr(discovery_loop, "correlate_with_target", lambda *args, **kwargs: [])
    monkeypatch.setattr(discovery_loop, "rank_descriptor_generators", lambda *args, **kwargs: [])

    class StubRanker:
        last_eval_backend = "numpy"
        last_eval_fallback_reason = None

        def __init__(self, **kwargs):
            del kwargs

        def rank_expressions_streaming(self, *args, **kwargs):
            del args, kwargs
            return []

        def to_records(self, hits):
            del hits
            return []

    monkeypatch.setattr(discovery_loop, "ConjectureRanker", StubRanker)
    phase4_calls: dict[str, object] = {}
    phase6_calls: dict[str, object] = {}

    class StubPhase4:
        pca = None
        feature_autoencoder = None
        instance_q_autoencoder = None

        def to_latent_dict(self):
            return {}

    def fake_phase4(*args, **kwargs):
        del args
        phase4_calls.update(kwargs)
        return StubPhase4()

    def fake_phase5(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(symbolic={})

    def fake_phase6(*args, **kwargs):
        del args
        phase6_calls.update(kwargs)
        return SimpleNamespace(
            to_phase6_dict=lambda: {
                "compute_backend": phase6_calls["compute_backend"],
                "programs": [],
                "coordinates": {},
                "operator": {},
            }
        )

    monkeypatch.setattr(discovery_loop, "run_phase4", fake_phase4)
    monkeypatch.setattr(discovery_loop, "run_phase5", fake_phase5)
    monkeypatch.setattr(discovery_loop, "run_phase6", fake_phase6)
    monkeypatch.setattr(discovery_loop, "promote_top_conjectures", lambda **kwargs: [])
    monkeypatch.setattr(
        "rde.analyze.leak_audit.audit_conjectures",
        lambda promoted, *, target: (promoted, {}),
    )
    monkeypatch.setattr(
        "rde.discovery.scaling_rule.fit_cross_n_generative_rule",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        discovery_loop,
        "search_linear_coordinates",
        lambda *args, **kwargs: SimpleNamespace(
            method="stub",
            n_components=0,
            target_correlation=float("nan"),
            update_simplicity=float("nan"),
        ),
    )
    monkeypatch.setattr(
        discovery_loop,
        "learn_update_predictor",
        lambda *args, **kwargs: SimpleNamespace(
            method="stub",
            train_r_squared=float("nan"),
            n_rows=0,
            feature_columns=[],
            update_key=kwargs.get("update_key"),
        ),
    )
    monkeypatch.setattr(
        discovery_loop,
        "assess_outcome",
        lambda *args, **kwargs: OutcomeAssessment(
            grade=0,
            g0_met=False,
            g1_met=False,
            g2_met=False,
            g3_met=False,
            g4_met=False,
            g5_met=False,
            target="target",
        ),
    )

    report = run_discovery(
        "backend_wiring",
        tmp_path,
        prefer_mlx_autoencoder=True,
        use_pysr=False,
        use_operon=False,
        use_native_gp=False,
        max_expr_candidates=1,
        gp_generations=1,
        gp_population=1,
    )
    assert phase4_calls["compute_backend"] == "mlx"
    assert phase6_calls["compute_backend"] == "mlx"
    assert report.phase6["compute_backend"] == "mlx"
