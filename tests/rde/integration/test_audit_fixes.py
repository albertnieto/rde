"""Tests for audit fixes: targets, CLI, cross-slice storage, recurrence."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from rde.analyze.query import flatten_features
from rde.features.recurrence import estimate_recurrence_order
from rde.generators import list_generators, register_builtin_generators
from rde.runtime.pipeline import RunConfig, run_pipeline
from rde.core.plugins import build_registry
from rde.runtime.targets import default_target_for_domain, resolve_target_for_run


def test_uniform_generators_not_registered():
    register_builtin_generators()
    gens = list_generators()
    assert not any(g.startswith("uniform_") for g in gens)


def test_default_target_per_domain():
    assert default_target_for_domain("hsp_functions") == "metric.structure_strength"
    assert default_target_for_domain("synthetic_poly") == "metric.representation_complexity"


def test_resolve_target_from_manifest(tmp_path: Path):
    reg = build_registry("synthetic_poly")
    run_pipeline(
        RunConfig(
            domain_id="synthetic_poly",
            n_instances=2,
            size=8,
            seed=0,
            indices=[1, 2],
            store_root=tmp_path,
            run_id="target_test",
            generator_id="poly_narrow_coeff",
        ),
        registry=reg,
    )
    assert resolve_target_for_run("target_test", tmp_path) == "metric.representation_complexity"


def test_cross_slice_in_instance_features_not_feature_rows(tmp_path: Path):
    reg = build_registry("synthetic_poly")
    run_pipeline(
        RunConfig(
            domain_id="synthetic_poly",
            n_instances=2,
            size=8,
            seed=1,
            indices=[1, 2, 4],
            store_root=tmp_path,
            run_id="dedup_test",
            generator_id="poly_narrow_coeff",
        ),
        registry=reg,
    )
    from rde.io.store import Store

    store = Store(tmp_path)
    inst = store.read_instance_features("dedup_test")[0]
    assert "dynamics.n_points" in inst["scalars"]
    feat = store.read_features("dedup_test")[0]
    assert "dynamics.n_points" not in feat["descriptors"]
    flat = flatten_features("dedup_test", tmp_path)
    assert flat[0]["dynamics.n_points"] == inst["scalars"]["dynamics.n_points"]


def test_recurrence_order_saturation():
    vectors = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    out = estimate_recurrence_order([1, 2, 3], vectors)
    assert out["recurrence.estimated_order"] == 3.0


def test_cli_list_and_backends():
    for cmd in (
        [sys.executable, "-m", "rde", "list", "domains"],
        [sys.executable, "-m", "rde", "list", "backends"],
        [sys.executable, "-m", "rde", "backends"],
    ):
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip()


def test_cli_validate_and_discover(tmp_path: Path):
    reg = build_registry("synthetic_poly")
    run_pipeline(
        RunConfig(
            domain_id="synthetic_poly",
            n_instances=4,
            size=8,
            seed=0,
            indices=[1, 2],
            store_root=tmp_path,
            run_id="cli_discover",
            generator_id="poly_narrow_coeff",
        ),
        registry=reg,
    )
    validate = subprocess.run(
        [sys.executable, "-m", "rde", "validate", "--run-id", "cli_discover", "--store-root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr

    discover = subprocess.run(
        [
            sys.executable,
            "-m",
            "rde",
            "discover",
            "--run-id",
            "cli_discover",
            "--store-root",
            str(tmp_path),
            "--no-pysr",
            "--max-candidates",
            "50",
            "--gp-generations",
            "3",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert discover.returncode == 0, discover.stderr
    conjectures = tmp_path / "discovery" / "conjectures.jsonl"
    assert conjectures.exists()
    lines = [json.loads(line) for line in conjectures.read_text().splitlines() if line.strip()]
    assert isinstance(lines, list)
